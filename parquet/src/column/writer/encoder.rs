// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#[cfg(feature = "arrow")]
use std::mem::MaybeUninit;

use bytes::Bytes;
use half::f16;

use crate::basic::{ConvertedType, Encoding, LogicalType, Type};
use crate::bloom_filter::Sbbf;
#[cfg(feature = "arrow")]
use crate::column::value::assume_init_prefix;
use crate::column::value::{Sink, gather_tiled};
use crate::column::writer::byte_array_encoder::{
    ByteArrayColumnEncoder, ByteBatchSource, ByteMinMax, ByteMinMaxOrder, ByteSink, ByteSinkTarget,
};
use crate::column::writer::{compare_greater_byte_array, is_nan_byte_array};
use crate::column::writer::{fallback_encoding, has_dictionary_support, update_max, update_min};
#[cfg(feature = "arrow")]
use crate::data_type::FixedLenByteArrayType;
use crate::data_type::private::ParquetValueType;
use crate::data_type::{BoolType, ByteArray, DataType, FixedLenByteArray, Int96};
use crate::encodings::encoding::{
    BoolColumnEncoder, ColumnEncode, Encoder, FlbaColumnEncoder, NumericColumnEncoder,
};
use crate::encodings::encoding::{BoolEncoder, NumericBatch, PackedBoolValues};
use crate::encodings::encoding::{FixedLenByteArrayEncoder, FixedLenByteArrayValues};
use crate::errors::{ParquetError, Result};
use crate::file::properties::{EnabledStatistics, WriterProperties};
use crate::geospatial::accumulator::{GeoStatsAccumulator, try_new_geo_stats_accumulator};
use crate::geospatial::statistics::GeospatialStatistics;
use crate::schema::types::{ColumnDescPtr, ColumnDescriptor};

/// The encoded data for a dictionary page
pub struct DictionaryPage {
    pub buf: Bytes,
    pub num_values: usize,
    pub is_sorted: bool,
}

/// The encoded values for a data page, with optional statistics
pub struct DataPageValues<T> {
    pub buf: Bytes,
    pub num_values: usize,
    pub encoding: Encoding,
    pub min_value: Option<T>,
    pub max_value: Option<T>,
    pub variable_length_bytes: Option<i64>,
}

/// A generic encoder of a column's values to data and dictionary pages, used by
/// [`super::GenericColumnWriter`].
pub trait ColumnValueEncoder {
    /// The underlying value type encoded by this encoder.
    type T: ParquetValueType;

    /// Create a new [`ColumnValueEncoder`]
    fn try_new(descr: &ColumnDescPtr, props: &WriterProperties) -> Result<Self>
    where
        Self: Sized;

    /// Returns the number of buffered values
    fn num_values(&self) -> usize;

    /// Returns true if this encoder has a dictionary page
    fn has_dictionary(&self) -> bool;

    /// Returns the estimated total memory usage of the encoder
    ///
    fn estimated_memory_size(&self) -> usize;

    /// Returns an estimate of the encoded size of dictionary page size in bytes, or `None` if no dictionary
    fn estimated_dict_page_size(&self) -> Option<usize>;

    /// Returns an estimate of the encoded data page size in bytes
    ///
    /// This should include:
    /// <already_written_encoded_byte_size> + <estimated_encoded_size_of_unflushed_bytes>
    fn estimated_data_page_size(&self) -> usize;

    /// Flush the dictionary page for this column chunk, if any. Subsequent
    /// writes use non-dictionary encoding.
    ///
    /// Note: [`Self::flush_data_page`] must be called first, as this will error if there
    /// are any pending page values
    fn flush_dict_page(&mut self) -> Result<Option<DictionaryPage>>;

    /// Flush the next data page for this column chunk
    fn flush_data_page(&mut self) -> Result<DataPageValues<Self::T>>;

    /// Flushes bloom filter if enabled and returns it, otherwise returns `None`. Subsequent writes
    /// will *not* be tracked by the bloom filter as it is empty since. This should be called once
    /// near the end of encoding.
    fn flush_bloom_filter(&mut self) -> Option<Sbbf>;

    /// Computes [`GeospatialStatistics`], if any, and resets internal state such that any internal
    /// accumulator is prepared to accumulate statistics for the next column chunk.
    fn flush_geospatial_statistics(&mut self) -> Option<Box<GeospatialStatistics>>;
}

/// Maps a sealed physical Rust value type to the encoder used for a particular
/// public [`DataType`] marker. The marker parameter makes downstream marker
/// types part of `ColumnWriterImpl<D>`'s type identity. The slice hook keeps
/// physical-value dispatch beside the corresponding encoder selection.
pub(crate) trait PhysicalValueDispatch: Sized {
    type Encoder<D>: ColumnEncode<D>
    where
        D: DataType<T = Self>;

    fn encode_slice<D>(enc: &mut ColumnValueEncoderImpl<D>, values: &[Self]) -> Result<()>
    where
        D: DataType<T = Self>;
}

/// Resolves the private encoder selected by [`DataType::T`].
pub(crate) type EncoderFor<D> = <<D as DataType>::T as PhysicalValueDispatch>::Encoder<D>;

macro_rules! impl_numeric_encoder_dispatch {
    ($value:ty) => {
        impl PhysicalValueDispatch for $value {
            type Encoder<D>
                = NumericColumnEncoder<D>
            where
                D: DataType<T = Self>;

            fn encode_slice<D>(enc: &mut ColumnValueEncoderImpl<D>, values: &[Self]) -> Result<()>
            where
                D: DataType<T = Self>,
            {
                if values.is_empty() {
                    Ok(())
                } else {
                    enc.commit(NumericBatch::Flat(values))
                }
            }
        }
    };
}

impl PhysicalValueDispatch for bool {
    type Encoder<D>
        = BoolColumnEncoder
    where
        D: DataType<T = Self>;

    fn encode_slice<D>(enc: &mut ColumnValueEncoderImpl<D>, values: &[Self]) -> Result<()>
    where
        D: DataType<T = Self>,
    {
        enc.commit(PackedBoolValues::from_bool_slice(values))
    }
}

impl_numeric_encoder_dispatch!(i32);
impl_numeric_encoder_dispatch!(i64);
impl_numeric_encoder_dispatch!(Int96);
impl_numeric_encoder_dispatch!(f32);
impl_numeric_encoder_dispatch!(f64);

impl PhysicalValueDispatch for FixedLenByteArray {
    type Encoder<D>
        = FlbaColumnEncoder
    where
        D: DataType<T = Self>;

    fn encode_slice<D>(enc: &mut ColumnValueEncoderImpl<D>, values: &[Self]) -> Result<()>
    where
        D: DataType<T = Self>,
    {
        encode_flba_slice(enc, values)
    }
}

impl PhysicalValueDispatch for ByteArray {
    type Encoder<D>
        = ByteArrayColumnEncoder
    where
        D: DataType<T = Self>;

    fn encode_slice<D>(enc: &mut ColumnValueEncoderImpl<D>, values: &[Self]) -> Result<()>
    where
        D: DataType<T = Self>,
    {
        encode_byte_slice(enc, values)
    }
}

/// Concrete [`ColumnValueEncoder`] implementation for one Parquet physical type.
///
/// This owns the physical encoder together with the state accumulated while a
/// column chunk is being written: dictionary adoption/fallback, value count,
/// statistics, bloom filter state, variable-length byte totals and geospatial
/// statistics. Slice-based writers call the trait methods directly; Arrow
/// numeric writers use the `arrow`-feature native path to feed values from Arrow
/// buffers without first materializing a dense Parquet buffer.
pub struct ColumnValueEncoderImpl<T: DataType> {
    /// The column's value encoder: a flat per-family enum whose `Dictionary` variant
    /// is a peer of the non-dictionary page encodings (see the `*ColumnEncoder`
    /// enums). Dictionary fallback (the dictionary overflowing its page-size limit)
    /// is a variant transition inside [`ColumnEncode::take_dict_page`].
    encoder: EncoderFor<T>,
    descr: ColumnDescPtr,
    num_values: usize,
    statistics_enabled: EnabledStatistics,
    min_value: Option<T::T>,
    max_value: Option<T::T>,
    bloom_filter: Option<Sbbf>,
    bloom_filter_target_fpp: f64,
    variable_length_bytes: Option<i64>,
    geo_stats_accumulator: Option<Box<dyn GeoStatsAccumulator>>,
    /// The non-dictionary page encoding selected from the writer properties.
    /// [`ColumnEncode::take_dict_page`] uses it to build the fallback encoder
    /// when dictionary encoding is abandoned.
    fallback_encoding: Encoding,
    /// Cross-batch scratch for the fixed-length byte-array [`Sink`]. Unused —
    /// and zero-initialized — for every other physical type.
    flba: FlbaWriteState,
}

/// Per-write-call scratch that the fixed-length byte-array [`Sink`] accumulates
/// across the batches of one write and folds into the column stats in
/// [`ColumnValueEncoderImpl::finish_flba`]. Kept owned (unlike byte-array's borrowed
/// `Option<&[u8]>`) because FLBA's `ObservedComputed` values live in transient
/// scratch tiles that cannot be borrowed past `commit`. The `min`/`max`
/// allocations are reused across writes; [`ColumnValueEncoderImpl::begin_flba`]
/// resets the flags.
#[derive(Default)]
struct FlbaWriteState {
    should_update_stats: bool,
    min: Vec<u8>,
    max: Vec<u8>,
    has_min: bool,
    has_max: bool,
}

/// Widest Arrow logical value computed into fixed-length bytes (decimal256).
#[cfg(feature = "arrow")]
pub(crate) const FLBA_COMPUTED_MAX_WIDTH: usize = 32;
/// Number of fixed-length values or run groups per stack batch.
#[cfg(feature = "arrow")]
pub(crate) const FLBA_BATCH_VALUES: usize = 64;

/// Packs generated FLBA values into a bounded stack batch while observing each
/// completed slot exactly once. Keeping the packer beside the sink makes it
/// impossible to commit an `ObservedComputed` batch without first updating
/// statistics and bloom state.
#[cfg(feature = "arrow")]
pub(crate) struct FlbaTilePacker<'a> {
    sink: &'a mut ColumnValueEncoderImpl<FixedLenByteArrayType>,
    tile: [u8; FLBA_BATCH_VALUES * FLBA_COMPUTED_MAX_WIDTH],
    width: usize,
    filled: usize,
}

#[cfg(feature = "arrow")]
impl<'a> FlbaTilePacker<'a> {
    #[inline]
    pub(crate) fn new(
        sink: &'a mut ColumnValueEncoderImpl<FixedLenByteArrayType>,
        width: usize,
    ) -> Self {
        Self {
            sink,
            tile: [0; FLBA_BATCH_VALUES * FLBA_COMPUTED_MAX_WIDTH],
            width,
            filled: 0,
        }
    }

    /// Fill and observe one value, flushing a full batch through the sole
    /// out-of-line handoff.
    #[inline]
    pub(crate) fn push(&mut self, fill: impl FnOnce(&mut [u8])) -> Result<()> {
        let offset = self.filled * self.width;
        let end = offset + self.width;
        fill(&mut self.tile[offset..end]);
        self.sink.observe_value(&self.tile[offset..end]);
        self.filled += 1;
        if self.filled == FLBA_BATCH_VALUES {
            let len = self.filled * self.width;
            self.sink
                .commit_observed_computed(&self.tile[..len], self.width, self.filled)?;
            self.filled = 0;
        }
        Ok(())
    }

    /// Commit the final partial batch. The downstream encoder also accepts an
    /// empty batch.
    #[inline]
    pub(crate) fn finish(self) -> Result<()> {
        let len = self.filled * self.width;
        self.sink
            .commit_observed_computed(&self.tile[..len], self.width, self.filled)
    }
}

/// Arrow packed-Boolean entry point.
impl ColumnValueEncoderImpl<BoolType> {
    #[cfg(feature = "arrow")]
    pub(crate) fn write_packed_bool(&mut self, values: PackedBoolValues<'_>) -> Result<()> {
        debug_assert!(!<BoolColumnEncoder as ColumnEncode<BoolType>>::is_dictionary(&self.encoder));
        // The encoder is the sink. The packed buffer and its selection descriptor
        // cross together, without expanding bits to Rust `bool`s.
        self.commit(values)
    }
}

/// The Boolean encoder is its family's [`Sink`]. `PackedBoolValues` retains any
/// dense, sparse, grouped, or unpacked-slice selection without
/// scalarising packed input. One commit counts true values in one pass, updates
/// statistics/bloom state, and encodes the selection. Arrow and slice input share
/// this handoff.
impl<'source, D: DataType<T = bool>> Sink<PackedBoolValues<'source>> for ColumnValueEncoderImpl<D> {
    #[inline(never)]
    fn commit(&mut self, values: PackedBoolValues<'source>) -> Result<()> {
        // INTERVAL has undefined sort order, so it must not emit min/max statistics.
        let should_update_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;
        let len = values.len();
        self.num_values += len;
        if len != 0 && (should_update_stats || self.bloom_filter.is_some()) {
            let true_count = values.true_count();
            if should_update_stats {
                let min = true_count == len;
                let max = true_count > 0;
                update_min(&self.descr, &min, &mut self.min_value);
                update_max(&self.descr, &max, &mut self.max_value);
            }
            if let Some(bloom) = self.bloom_filter.as_mut() {
                if true_count < len {
                    bloom.insert(&false);
                }
                if true_count > 0 {
                    bloom.insert(&true);
                }
            }
        }
        self.encoder.put_packed_bool(values)
    }
}

/// The fixed-length byte-array encoder is its family's [`Sink`]. It folds each
/// batch's stats/bloom straight into the encoder's own accumulator
/// ([`ColumnValueEncoderImpl::flba`]) and dictionary / PLAIN encoder. FLBA keeps
/// owned min/max state because computed values borrow a reused scratch tile and
/// cannot be retained across [`Sink::commit`] calls.
impl<D: DataType<T = FixedLenByteArray>> ColumnValueEncoderImpl<D> {
    /// Begin a fixed-length byte-array write of `len` values, resetting the
    /// cross-batch stats accumulator and reserving its encoder storage.
    pub(crate) fn begin_flba(&mut self, len: usize) {
        self.begin_flba_inner(len, true)
    }

    /// Begin a run-aware FLBA write. This intentionally skips reserving `len` dense
    /// dictionary indices because grouped input records `(index, count)` pairs
    /// instead. Grouped input only comes from Arrow.
    #[cfg_attr(not(feature = "arrow"), allow(dead_code))]
    pub(crate) fn begin_flba_run(&mut self, len: usize) {
        self.begin_flba_inner(len, false)
    }

    fn begin_flba_inner(&mut self, len: usize, reserve_dictionary_indices: bool) {
        self.num_values += len;
        match &mut self.encoder {
            FlbaColumnEncoder::Dictionary(dict_encoder) if reserve_dictionary_indices => {
                dict_encoder.reserve(len)
            }
            FlbaColumnEncoder::Dictionary(_) => {}
            encoder => encoder
                .reserve_fixed_len((self.descr.type_length().max(0) as usize).saturating_mul(len)),
        }
        // INTERVAL has undefined sort order, so it must not emit min/max statistics.
        self.flba.should_update_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;
        self.flba.min.clear();
        self.flba.max.clear();
        self.flba.has_min = false;
        self.flba.has_max = false;
    }

    /// Whether a dense `FixedSizeBinary` selection can be committed as one packed
    /// batch. Dictionary interning and geo accumulation require source-side
    /// per-value handling and therefore use gathered batches. Statistics and bloom
    /// state, when enabled, are folded downstream by [`Self::encode_dense`].
    #[cfg_attr(not(feature = "arrow"), allow(dead_code))]
    #[inline]
    pub(crate) fn flba_accepts_dense_batch(&self) -> bool {
        !<FlbaColumnEncoder as ColumnEncode<D>>::is_dictionary(&self.encoder)
            && self.geo_stats_accumulator.is_none()
    }

    // Per-value statistics and bloom folding stay inside the sparse and
    // computed FLBA loops.
    #[inline(always)]
    fn observe_value(&mut self, value: &[u8]) {
        if self.flba.should_update_stats && !is_nan_byte_array(&self.descr, value) {
            if !self.flba.has_min || compare_greater_byte_array(&self.descr, &self.flba.min, value)
            {
                self.flba.min.clear();
                self.flba.min.extend_from_slice(value);
                self.flba.has_min = true;
            }
            if !self.flba.has_max || compare_greater_byte_array(&self.descr, value, &self.flba.max)
            {
                self.flba.max.clear();
                self.flba.max.extend_from_slice(value);
                self.flba.has_max = true;
            }
        }
        if let Some(bloom) = self.bloom_filter.as_mut() {
            bloom.insert(value);
        }
    }

    /// Fold one value into page stats and bloom state, then intern or append it.
    #[inline]
    fn consume_value(&mut self, value: &[u8]) -> Result<()> {
        self.observe_value(value);
        match &mut self.encoder {
            FlbaColumnEncoder::Dictionary(dict) => {
                dict.put_value_bytes(value, || {
                    FixedLenByteArray::from(ByteArray::from(value.to_vec()))
                });
                Ok(())
            }
            other => other.append_fixed_len_value(value),
        }
    }

    /// Encode a bounded [`FlbaBatch::RunGroups`] tile. Each `(value, count)`
    /// folds stats/bloom and interns one dictionary entry, or expands `count`
    /// direct-encoded values. The producer crosses `commit` per bounded tile,
    /// not per selected group.
    #[inline(never)]
    fn encode_flba_run_groups(&mut self, values: &[&[u8]], counts: &[usize]) -> Result<()> {
        for (&value, &count) in values.iter().zip(counts) {
            if count == 0 {
                continue;
            }
            self.observe_value(value);
            match &mut self.encoder {
                FlbaColumnEncoder::Dictionary(dict) => {
                    dict.put_value_bytes_run(value, count, || {
                        FixedLenByteArray::from(ByteArray::from(value.to_vec()))
                    });
                }
                other => {
                    for _ in 0..count {
                        other.append_fixed_len_value(value)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Encode an observed computed batch of `count` fixed-width values. The
    /// producer already folded stats/bloom while filling the tile; this method
    /// performs only dictionary interning or one bulk encoder call, without a
    /// second observation pass.
    #[cfg(feature = "arrow")]
    #[inline(never)]
    fn encode_computed(&mut self, packed: &[u8], width: usize, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let values = FixedLenByteArrayValues::new(packed, width, count);
        match &mut self.encoder {
            FlbaColumnEncoder::Dictionary(dict) => {
                for value in values.iter() {
                    dict.put_value_bytes(value, || {
                        FixedLenByteArray::from(ByteArray::from(value.to_vec()))
                    });
                }
                Ok(())
            }
            other => other.put_fixed_len_byte_array(values),
        }
    }

    /// Encode one dense packed `FixedSizeBinary` batch borrowed from the source.
    /// Stats and bloom, when enabled, share one downstream observation pass before
    /// the packed values are handed to the active encoder in one bulk call. This
    /// path requires [`Self::flba_accepts_dense_batch`].
    #[inline(never)]
    fn encode_dense(&mut self, packed: &[u8], width: usize, count: usize) -> Result<()> {
        debug_assert!(!<FlbaColumnEncoder as ColumnEncode<D>>::is_dictionary(
            &self.encoder
        ));
        if count == 0 {
            return Ok(());
        }
        let values = FixedLenByteArrayValues::new(packed, width, count);
        let descr = &self.descr;
        let stats = self.flba.should_update_stats;
        if stats || self.bloom_filter.is_some() {
            let mut min: Option<&[u8]> = None;
            let mut max: Option<&[u8]> = None;
            let mut bloom = self.bloom_filter.as_mut();
            for value in values.iter() {
                if stats && !is_nan_byte_array(descr, value) {
                    if min.is_none_or(|c| compare_greater_byte_array(descr, c, value)) {
                        min = Some(value);
                    }
                    if max.is_none_or(|c| compare_greater_byte_array(descr, value, c)) {
                        max = Some(value);
                    }
                }
                if let Some(bloom) = bloom.as_deref_mut() {
                    bloom.insert(value);
                }
            }
            if let Some(min) = min {
                if !self.flba.has_min || compare_greater_byte_array(descr, &self.flba.min, min) {
                    self.flba.min.clear();
                    self.flba.min.extend_from_slice(min);
                    self.flba.has_min = true;
                }
            }
            if let Some(max) = max {
                if !self.flba.has_max || compare_greater_byte_array(descr, max, &self.flba.max) {
                    self.flba.max.clear();
                    self.flba.max.extend_from_slice(max);
                    self.flba.has_max = true;
                }
            }
        }
        match &mut self.encoder {
            FlbaColumnEncoder::Dictionary(_) => {
                unreachable!("dense FLBA path requires no dictionary")
            }
            other => other.put_fixed_len_byte_array(values),
        }
    }

    /// Gathered flat batch: borrowed `&[u8]` slices from sparse or dictionary
    /// input. Folds stats+bloom and encodes each value.
    #[inline(never)]
    pub(crate) fn encode_gathered(&mut self, tile: &[&[u8]]) -> Result<()> {
        for &value in tile {
            self.consume_value(value)?;
        }
        Ok(())
    }

    /// Encode low-level slice values while enforcing the descriptor width.
    /// Arrow fixed-size inputs carry their width in their array type and use
    /// [`Self::encode_gathered`] directly; only independently allocated
    /// [`FixedLenByteArray`] values require this check.
    #[inline(never)]
    fn encode_checked_gathered(&mut self, tile: &[&[u8]], expected: usize) -> Result<()> {
        for &value in tile {
            if value.len() != expected {
                return Err(general_err!(
                    "Mismatched FixedLenByteArray sizes: {} != {}",
                    value.len(),
                    expected
                ));
            }
            self.consume_value(value)?;
        }
        Ok(())
    }

    /// Commit a tile whose values were already observed by [`FlbaTilePacker`].
    #[cfg(feature = "arrow")]
    #[inline(always)]
    fn commit_observed_computed(&mut self, bytes: &[u8], width: usize, count: usize) -> Result<()> {
        self.commit(FlbaBatch::ObservedComputed(ObservedFlbaBatch {
            bytes,
            width,
            count,
        }))
    }

    /// Commit a gathered tile through the sole out-of-line batch handoff.
    #[inline(always)]
    pub(crate) fn flush_gathered(&mut self, tile: &[&[u8]]) -> Result<()> {
        self.commit(FlbaBatch::Gathered(tile))
    }

    /// Commit low-level slice values together with their required physical
    /// width. The width check and encoding happen downstream of the handoff.
    #[inline(always)]
    fn flush_checked_gathered(&mut self, tile: &[&[u8]], expected: usize) -> Result<()> {
        self.commit(FlbaBatch::CheckedGathered {
            values: tile,
            expected,
        })
    }

    /// Fold the write's accumulated raw min/max into the column stats. Called at the
    /// end of each fixed-length byte-array write (the encoder outlives the write, so
    /// this borrows `&mut self` rather than consuming a transient sink).
    pub(crate) fn finish_flba(&mut self) {
        if self.flba.has_min && self.flba.has_max {
            let (min, max) =
                raw_fixed_len_min_max_values(&self.descr, &self.flba.min, &self.flba.max);
            update_min(&self.descr, &min, &mut self.min_value);
            update_max(&self.descr, &max, &mut self.max_value);
        }
    }
}

/// One fixed-length byte-array batch handed to the concrete FLBA encoder. Its
/// variants preserve the shapes supplied by dense, computed, gathered, and
/// grouped inputs.
///
/// `CheckedGathered` is specific to the low-level slice path; `Dense` and
/// `ObservedComputed` come from Arrow.
#[cfg_attr(not(feature = "arrow"), allow(dead_code))]
pub(crate) enum FlbaBatch<'a> {
    /// Dense `FixedSizeBinary` values packed at `width`; stats/bloom are folded
    /// by `encode_dense` before its bulk encoder call.
    Dense {
        bytes: &'a [u8],
        width: usize,
        count: usize,
    },
    /// Computed values whose statistics and bloom state were folded during
    /// generation. Commit only encodes them.
    #[cfg(feature = "arrow")]
    ObservedComputed(ObservedFlbaBatch<'a>),
    /// Gathered per-value slices; stats/bloom folded and encoded per value
    /// (`encode_gathered`).
    Gathered(&'a [&'a [u8]]),
    /// Independently allocated low-level values, checked against the descriptor
    /// width before being observed or encoded.
    CheckedGathered {
        values: &'a [&'a [u8]],
        expected: usize,
    },
    /// A bounded tile of selected run groups. `values[i]` is repeated for
    /// `counts[i]` logical outputs; reordered selections may revisit a physical
    /// run in multiple groups. From Arrow only.
    RunGroups {
        values: &'a [&'a [u8]],
        counts: &'a [usize],
    },
}

/// A computed FLBA batch that has already been observed exactly once. Fields
/// are private so only [`FlbaTilePacker`] can construct one.
#[cfg(feature = "arrow")]
pub(crate) struct ObservedFlbaBatch<'a> {
    bytes: &'a [u8],
    width: usize,
    count: usize,
}

/// Encode one native FLBA batch. This is the sole out-of-line handoff from its
/// dense, gathered, run-group, and computed producers. The computed producer's
/// per-value [`ColumnValueEncoderImpl::observe_value`] call remains forced inline;
/// `ObservedComputed` therefore performs encoding only and never rescans the
/// tile.
impl<'batch, D: DataType<T = FixedLenByteArray>> Sink<FlbaBatch<'batch>>
    for ColumnValueEncoderImpl<D>
{
    #[inline(never)]
    fn commit(&mut self, values: FlbaBatch<'batch>) -> Result<()> {
        match values {
            FlbaBatch::Dense {
                bytes,
                width,
                count,
            } => self.encode_dense(bytes, width, count),
            #[cfg(feature = "arrow")]
            FlbaBatch::ObservedComputed(batch) => {
                self.encode_computed(batch.bytes, batch.width, batch.count)
            }
            FlbaBatch::Gathered(tile) => self.encode_gathered(tile),
            FlbaBatch::CheckedGathered { values, expected } => {
                self.encode_checked_gathered(values, expected)
            }
            FlbaBatch::RunGroups { values, counts } => self.encode_flba_run_groups(values, counts),
        }
    }
}

fn raw_fixed_len_min_max_values(
    descr: &ColumnDescriptor,
    min: &[u8],
    max: &[u8],
) -> (FixedLenByteArray, FixedLenByteArray) {
    if descr.logical_type_ref() == Some(&LogicalType::Float16) {
        let min = raw_float16_stat_zero(min, f16::NEG_ZERO);
        let max = raw_float16_stat_zero(max, f16::ZERO);
        return (min.to_vec().into(), max.to_vec().into());
    }

    (min.to_vec().into(), max.to_vec().into())
}

fn raw_float16_stat_zero(value: &[u8], replacement: f16) -> [u8; 2] {
    let value: [u8; 2] = value.try_into().unwrap();
    if f16::from_le_bytes(value) == f16::ZERO {
        replacement.to_le_bytes()
    } else {
        value
    }
}

/// The byte-array encoder drives one write-scoped [`ByteSink`]. Contiguous
/// offset-array spans cross it as native borrowed batches; non-contiguous
/// sources use bounded gathered batches. The sink retains borrowed min/max and
/// byte totals across every handoff and merges them once after the write.
impl<D: DataType<T = ByteArray>> ColumnValueEncoderImpl<D> {
    /// Encode a selected byte-value source, one value per logical output.
    pub(crate) fn write_byte_values<'a, S>(&mut self, values: S) -> Result<()>
    where
        S: ByteBatchSource<'a>,
    {
        self.write_byte_values_inner(values, false)
    }

    /// As [`Self::write_byte_values`], for a grouped cursor: run-collapsed
    /// dictionary encoding when eligible (dictionary present, no geo
    /// accumulator) interns one value per selected run group and buffers its
    /// `(index, count)` entry; otherwise the cursor is driven flat. PLAIN/DELTA
    /// and the geo accumulator require every logical value.
    // Grouped input only comes from Arrow.
    #[cfg(feature = "arrow")]
    pub(crate) fn write_byte_values_run_collapsed<'a, S>(&mut self, values: S) -> Result<()>
    where
        S: ByteBatchSource<'a>,
    {
        self.write_byte_values_inner(values, true)
    }

    fn write_byte_values_inner<'a>(
        &mut self,
        values: impl ByteBatchSource<'a>,
        run_end: bool,
    ) -> Result<()> {
        let len = values.len();
        let collect_stats = self.statistics_enabled != EnabledStatistics::None;
        let run_group_path = run_end
            && <ByteArrayColumnEncoder as ColumnEncode<D>>::is_dictionary(&self.encoder)
            && self.geo_stats_accumulator.is_none();

        let (min, max, bytes_written) = {
            let target = match &mut self.encoder {
                ByteArrayColumnEncoder::Dictionary(dict_encoder) => {
                    // A flat write materializes any pending run-buffered
                    // indices before the first new value and reserves once for
                    // the complete logical source.
                    if !run_group_path {
                        dict_encoder.reserve(len);
                    }
                    ByteSinkTarget::Dict(dict_encoder)
                }
                other => ByteSinkTarget::Fallback(other),
            };
            let mut sink = ByteSink {
                collect_stats,
                order: ByteMinMaxOrder::from_descr(&self.descr),
                min: None,
                max: None,
                bytes_written: 0,
                accumulator: self.geo_stats_accumulator.as_mut(),
                bloom: self.bloom_filter.as_mut(),
                target,
            };
            if run_group_path {
                values.drive_run_groups(&mut sink)?;
            } else {
                values.drive_flat(&mut sink)?;
            }
            (sink.min, sink.max, sink.bytes_written)
        };

        self.num_values += len;
        self.merge_page_minmax(min, max);
        // This feeds the offset index even when ordinary statistics are
        // disabled.
        *self.variable_length_bytes.get_or_insert(0) += bytes_written;
        Ok(())
    }

    /// Merge a page's observed min/max byte values into the running column min/max,
    /// using the column's byte order. Shared by the per-value byte path and the
    /// grouped fast path.
    fn merge_page_minmax(&mut self, min: Option<&[u8]>, max: Option<&[u8]>) {
        let order = ByteMinMaxOrder::from_descr(&self.descr);
        if let Some(min) = min {
            if self
                .min_value
                .as_ref()
                .is_none_or(|m| <ByteMinMax as MinMaxStrategy<'_>>::greater(order, m.data(), min))
            {
                self.min_value = Some(<ByteMinMax as MinMaxStrategy<'_>>::to_owned(min));
            }
        }
        if let Some(max) = max {
            if self
                .max_value
                .as_ref()
                .is_none_or(|m| <ByteMinMax as MinMaxStrategy<'_>>::greater(order, max, m.data()))
            {
                self.max_value = Some(<ByteMinMax as MinMaxStrategy<'_>>::to_owned(max));
            }
        }
    }
}

/// Slice `write_batch(&[ByteArray])`: drive the dense values through the same
/// `write_byte_values` engine as Arrow input. The values are separate
/// `ByteArray`s (not a contiguous packed run), so this takes the bounded
/// gathered path.
fn encode_byte_slice<D: DataType<T = ByteArray>>(
    enc: &mut ColumnValueEncoderImpl<D>,
    values: &[ByteArray],
) -> Result<()> {
    enc.write_byte_values(values)
}

/// The encoder is itself the numeric [`Sink`]: `commit` folds each batch's stats
/// and bloom bits and encodes it, straight into the encoder's own dictionary /
/// PLAIN encoder / statistics — no intermediate sink object.
///
/// Statistics and encoding share the same pass through each batch.
impl<'batch, T: DataType> Sink<NumericBatch<'batch, T::T>> for ColumnValueEncoderImpl<T>
where
    T::T: PhysicalValueDispatch<Encoder<T> = NumericColumnEncoder<T>>,
    T::T: Copy + 'static + for<'v> MinMaxStrategy<'v, Elem = T::T, Owned = T::T>,
{
    #[inline(never)]
    fn commit(&mut self, values: NumericBatch<'batch, T::T>) -> Result<()> {
        // INTERVAL has undefined sort order, so it must not emit min/max statistics.
        let should_update_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;
        let ctx = <T::T as MinMaxStrategy<'static>>::ctx(&self.descr);
        // Fold raw min/max per batch and normalize-merge into the column stats.
        let mut cmin: Option<T::T> = None;
        let mut cmax: Option<T::T> = None;
        // Flat and run-group encoding use separate per-shape functions.
        match values {
            NumericBatch::Flat(values) => {
                self.encode_flat(values, ctx, should_update_stats, &mut cmin, &mut cmax)?
            }
            #[cfg(feature = "arrow")]
            NumericBatch::RunGroups { values, counts } => self.encode_run_groups(
                values,
                counts,
                ctx,
                should_update_stats,
                &mut cmin,
                &mut cmax,
            )?,
        }
        self.merge_batch_stats(cmin, cmax);
        Ok(())
    }
}

#[allow(private_bounds)]
impl<T: DataType> ColumnValueEncoderImpl<T>
where
    T::T: PhysicalValueDispatch<Encoder<T> = NumericColumnEncoder<T>>,
    T::T: Copy + 'static + for<'v> MinMaxStrategy<'v, Elem = T::T, Owned = T::T>,
{
    /// Normalize-merge a batch's raw `(min, max)` into the running column stats.
    #[inline(never)]
    fn merge_batch_stats(&mut self, min: Option<T::T>, max: Option<T::T>) {
        if let Some(min) = min {
            update_min_normalized(&self.descr, &min, &mut self.min_value);
        }
        if let Some(max) = max {
            update_max_normalized(&self.descr, &max, &mut self.max_value);
        }
    }

    /// Encode one flat batch: one logical output per value. Kept out of line and
    /// separate from the run-group path so its hot min/max fold stays tight.
    #[inline(never)]
    fn encode_flat(
        &mut self,
        values: &[T::T],
        ctx: <T::T as MinMaxStrategy<'static>>::Ctx,
        should_update_stats: bool,
        cmin: &mut Option<T::T>,
        cmax: &mut Option<T::T>,
    ) -> Result<()> {
        let Self {
            encoder,
            num_values,
            bloom_filter,
            ..
        } = self;
        if should_update_stats {
            for &value in values {
                <T::T as MinMaxStrategy<'_>>::observe(ctx, value, cmin, cmax);
            }
        }
        if let Some(bloom) = bloom_filter.as_mut() {
            for &value in values {
                bloom.insert(&value);
            }
        }
        *num_values += values.len();
        match encoder {
            NumericColumnEncoder::Dictionary(dict) => {
                // `reserve` flushes any run-buffered indices
                // (`RunIndexBuffer::reserve` -> `materialize`), satisfying
                // `put_one`'s flushed-runs precondition; a separate flush here
                // is a redundant no-op.
                dict.reserve(values.len());
                for &value in values {
                    dict.put_one(&value);
                }
                Ok(())
            }
            other => <NumericColumnEncoder<T> as Encoder<T>>::put(other, values),
        }
    }

    /// Encode one run-group batch: `values[i]` spans `counts[i]` logical outputs.
    /// Selection planning already removed nulls, so every group is observed and
    /// every count is non-zero.
    #[cfg(feature = "arrow")]
    #[inline(never)]
    fn encode_run_groups(
        &mut self,
        values: &[T::T],
        counts: &[usize],
        ctx: <T::T as MinMaxStrategy<'static>>::Ctx,
        should_update_stats: bool,
        cmin: &mut Option<T::T>,
        cmax: &mut Option<T::T>,
    ) -> Result<()> {
        let Self {
            encoder,
            num_values,
            bloom_filter,
            ..
        } = self;
        match encoder {
            NumericColumnEncoder::Dictionary(dict) => {
                for (&value, &run_len) in values.iter().zip(counts) {
                    if should_update_stats {
                        <T::T as MinMaxStrategy<'_>>::observe(ctx, value, cmin, cmax);
                    }
                    if let Some(bloom) = bloom_filter.as_mut() {
                        bloom.insert(&value);
                    }
                    *num_values += run_len;
                    dict.put_value_run(&value, run_len);
                }
                Ok(())
            }
            other => {
                // PLAIN writes every value: expand each group to its logical
                // outputs, buffering into bounded batches for bulk `put`.
                let mut buf = [MaybeUninit::<T::T>::uninit(); 64];
                let mut filled = 0usize;
                for (&value, &run_len) in values.iter().zip(counts) {
                    if should_update_stats {
                        <T::T as MinMaxStrategy<'_>>::observe(ctx, value, cmin, cmax);
                    }
                    if let Some(bloom) = bloom_filter.as_mut() {
                        bloom.insert(&value);
                    }
                    *num_values += run_len;
                    for _ in 0..run_len {
                        buf[filled].write(value);
                        filled += 1;
                        if filled == buf.len() {
                            // SAFETY: every slot has just been initialized.
                            <NumericColumnEncoder<T> as Encoder<T>>::put(other, unsafe {
                                assume_init_prefix(&buf, filled)
                            })?;
                            filled = 0;
                        }
                    }
                }
                if filled > 0 {
                    // SAFETY: values are written sequentially through `filled`.
                    <NumericColumnEncoder<T> as Encoder<T>>::put(other, unsafe {
                        assume_init_prefix(&buf, filled)
                    })?;
                }
                Ok(())
            }
        }
    }
}

impl<T: DataType> ColumnValueEncoder for ColumnValueEncoderImpl<T> {
    type T = T::T;

    fn flush_bloom_filter(&mut self) -> Option<Sbbf> {
        let mut sbbf = self.bloom_filter.take()?;
        sbbf.fold_to_target_fpp(self.bloom_filter_target_fpp);
        Some(sbbf)
    }

    fn try_new(descr: &ColumnDescPtr, props: &WriterProperties) -> Result<Self> {
        let dict_supported = props.dictionary_enabled(descr.path())
            && has_dictionary_support(T::get_physical_type(), props);

        // The non-dictionary page encoding comes from writer properties so
        // `flush_dict_page` can build the fallback lazily.
        let fallback_encoding = props
            .encoding(descr.path())
            .unwrap_or_else(|| fallback_encoding(T::get_physical_type(), props));

        // Build the value encoder: the dictionary when supported, else the fallback
        // encoding. `new_column_encoder` eagerly validates the fallback encoding even
        // when starting on the dictionary, so an unsupported one fails fast at
        // construction (it is rebuilt on dictionary fallback — see `take_dict_page`).
        let encoder = <EncoderFor<T> as ColumnEncode<T>>::new_column_encoder(
            dict_supported,
            fallback_encoding,
            descr,
        )?;

        let statistics_enabled = props.statistics_enabled(descr.path());

        let (bloom_filter, bloom_filter_target_fpp) = create_bloom_filter(props, descr)?;

        let geo_stats_accumulator = try_new_geo_stats_accumulator(descr);

        Ok(Self {
            encoder,
            descr: descr.clone(),
            num_values: 0,
            statistics_enabled,
            bloom_filter,
            bloom_filter_target_fpp,
            min_value: None,
            max_value: None,
            variable_length_bytes: None,
            geo_stats_accumulator,
            fallback_encoding,
            flba: FlbaWriteState::default(),
        })
    }

    fn num_values(&self) -> usize {
        self.num_values
    }

    fn has_dictionary(&self) -> bool {
        self.encoder.is_dictionary()
    }

    fn estimated_memory_size(&self) -> usize {
        let encoder_size = self.encoder.memory_size();

        let bloom_filter_size = self
            .bloom_filter
            .as_ref()
            .map(|bf| bf.estimated_memory_size())
            .unwrap_or_default();

        // The running column min/max pin their values' heap bytes through page
        // flush — for byte arrays that can be two full values
        // (zero for the fixed-width scalars, whose `variable_length_bytes` is
        // `None`). The FLBA scratch accumulator is folded in the same way; it
        // stays empty for every other family.
        let stats_size = [&self.min_value, &self.max_value]
            .into_iter()
            .flatten()
            .map(|v| {
                <T::T as ParquetValueType>::variable_length_bytes(std::slice::from_ref(v))
                    .unwrap_or(0) as usize
            })
            .sum::<usize>()
            + self.flba.min.capacity()
            + self.flba.max.capacity();

        encoder_size + bloom_filter_size + stats_size
    }

    fn estimated_dict_page_size(&self) -> Option<usize> {
        self.encoder.dict_page_size()
    }

    fn estimated_data_page_size(&self) -> usize {
        self.encoder.data_page_size()
    }

    fn flush_dict_page(&mut self) -> Result<Option<DictionaryPage>> {
        if self.encoder.is_dictionary() && self.num_values != 0 {
            return Err(general_err!(
                "Must flush data pages before flushing dictionary"
            ));
        }
        // `take_dict_page` serializes the dictionary page and transitions the encoder
        // to the fallback encoding (dictionary fallback); `None` when not dictionary.
        Ok(self
            .encoder
            .take_dict_page(self.fallback_encoding, &self.descr)?
            .map(|(buf, num_values, is_sorted)| DictionaryPage {
                buf,
                num_values,
                is_sorted,
            }))
    }

    fn flush_data_page(&mut self) -> Result<DataPageValues<T::T>> {
        let (buf, encoding) = self.encoder.flush_data_page()?;

        Ok(DataPageValues {
            buf,
            encoding,
            num_values: std::mem::take(&mut self.num_values),
            min_value: self.min_value.take(),
            max_value: self.max_value.take(),
            variable_length_bytes: self.variable_length_bytes.take(),
        })
    }

    fn flush_geospatial_statistics(&mut self) -> Option<Box<GeospatialStatistics>> {
        self.geo_stats_accumulator.as_mut().map(|a| a.finish())?
    }
}

/// Slice `write_batch(&[FixedLenByteArray])`: gather each dense value's bytes into
/// bounded tiles and feed the same FLBA encoding/statistics path used by
/// Arrow input. Separate `FixedLenByteArray` values always take the gathered
/// path rather than a contiguous packed run.
fn encode_flba_slice<D: DataType<T = FixedLenByteArray>>(
    enc: &mut ColumnValueEncoderImpl<D>,
    values: &[FixedLenByteArray],
) -> Result<()> {
    let expected_width = (enc.fallback_encoding == Encoding::BYTE_STREAM_SPLIT)
        .then(|| enc.descr.type_length() as usize);
    enc.begin_flba(values.len());
    match expected_width {
        Some(expected) => gather_tiled::<64, _, _, _>(values, |batch| {
            enc.flush_checked_gathered(batch, expected)
        })?,
        None => gather_tiled::<64, _, _, _>(values, |batch| enc.flush_gathered(batch))?,
    }
    enc.finish_flba();
    Ok(())
}

#[inline]
fn int_is_unsigned(descr: &ColumnDescriptor) -> bool {
    if let Some(LogicalType::Integer(int)) = descr.logical_type_ref() {
        if !int.is_signed {
            return true;
        }
    }
    matches!(
        descr.converted_type(),
        ConvertedType::UINT_8
            | ConvertedType::UINT_16
            | ConvertedType::UINT_32
            | ConvertedType::UINT_64
    )
}

#[inline(always)]
fn int32_greater(unsigned: bool, a: i32, b: i32) -> bool {
    if unsigned {
        (a as u32) > (b as u32)
    } else {
        a > b
    }
}

#[inline(always)]
fn int64_greater(unsigned: bool, a: i64, b: i64) -> bool {
    if unsigned {
        (a as u64) > (b as u64)
    } else {
        a > b
    }
}

/// Min/max folding for numeric scalars and variable-width byte arrays. Numeric
/// strategies retain scalar values; the byte strategy retains sized `&[u8]`
/// handles to variable-width payloads without per-value allocation. `Owned` is
/// the materialized column statistic. FLBA uses a separate owned accumulator
/// because its computed values can borrow transient tiles.
///
/// Integer columns need descriptor-derived context because Parquet min/max for
/// unsigned logical types must compare the stored bits as unsigned values.
/// Float columns do not need descriptor context, but must skip NaN values when
/// computing min/max. [`Self::Ctx`] stores those decisions once per column so
/// [`Self::observe`] can stay small inside the per-value loop.
pub(crate) trait MinMaxStrategy<'v> {
    /// The per-value handle folded into statistics (owned scalar or borrowed bytes).
    type Elem: Copy;
    /// The materialized column statistic type.
    type Owned;
    /// Per-column comparison context, derived from the descriptor and reused
    /// for every value (e.g. integer signedness).
    type Ctx: Copy;

    /// Build the comparison context for this column.
    fn ctx(descr: &ColumnDescriptor) -> Self::Ctx;
    /// `a > b` under the column's logical order.
    fn greater(ctx: Self::Ctx, a: Self::Elem, b: Self::Elem) -> bool;
    /// True for values excluded from min/max (NaN floats); false by default.
    #[inline(always)]
    fn is_skippable(_ctx: Self::Ctx, _value: Self::Elem) -> bool {
        false
    }
    /// Materialize an accumulated element into the owned column statistic.
    fn to_owned(value: Self::Elem) -> Self::Owned;

    /// Fold one value into the running `(min, max)`.
    #[inline(always)]
    fn observe(
        ctx: Self::Ctx,
        value: Self::Elem,
        min: &mut Option<Self::Elem>,
        max: &mut Option<Self::Elem>,
    ) {
        if Self::is_skippable(ctx, value) {
            return;
        }
        if min.is_none_or(|current| Self::greater(ctx, current, value)) {
            *min = Some(value);
        }
        if max.is_none_or(|current| Self::greater(ctx, value, current)) {
            *max = Some(value);
        }
    }
}

impl<'v> MinMaxStrategy<'v> for i32 {
    type Elem = i32;
    type Owned = i32;
    type Ctx = bool;
    #[inline(always)]
    fn ctx(descr: &ColumnDescriptor) -> bool {
        int_is_unsigned(descr)
    }
    #[inline(always)]
    fn greater(unsigned: bool, a: i32, b: i32) -> bool {
        int32_greater(unsigned, a, b)
    }
    #[inline(always)]
    fn to_owned(v: i32) -> i32 {
        v
    }
}

impl<'v> MinMaxStrategy<'v> for i64 {
    type Elem = i64;
    type Owned = i64;
    type Ctx = bool;
    #[inline(always)]
    fn ctx(descr: &ColumnDescriptor) -> bool {
        int_is_unsigned(descr)
    }
    #[inline(always)]
    fn greater(unsigned: bool, a: i64, b: i64) -> bool {
        int64_greater(unsigned, a, b)
    }
    #[inline(always)]
    fn to_owned(v: i64) -> i64 {
        v
    }
}

impl<'v> MinMaxStrategy<'v> for f32 {
    type Elem = f32;
    type Owned = f32;
    type Ctx = ();
    #[inline(always)]
    fn ctx(_: &ColumnDescriptor) {}
    #[inline(always)]
    fn greater(_: (), a: f32, b: f32) -> bool {
        a > b
    }
    #[inline(always)]
    fn is_skippable(_: (), value: f32) -> bool {
        value.is_nan()
    }
    #[inline(always)]
    fn to_owned(v: f32) -> f32 {
        v
    }
}

impl<'v> MinMaxStrategy<'v> for f64 {
    type Elem = f64;
    type Owned = f64;
    type Ctx = ();
    #[inline(always)]
    fn ctx(_: &ColumnDescriptor) {}
    #[inline(always)]
    fn greater(_: (), a: f64, b: f64) -> bool {
        a > b
    }
    #[inline(always)]
    fn is_skippable(_: (), value: f64) -> bool {
        value.is_nan()
    }
    #[inline(always)]
    fn to_owned(v: f64) -> f64 {
        v
    }
}

impl<'v> MinMaxStrategy<'v> for Int96 {
    type Elem = Int96;
    type Owned = Int96;
    type Ctx = ();
    #[inline(always)]
    fn ctx(_: &ColumnDescriptor) {}
    #[inline(always)]
    fn greater(_: (), a: Int96, b: Int96) -> bool {
        // INT96 min/max use the timestamp `(days, nanos)` order (`Int96: Ord`),
        // matching the descriptor-driven `compare_greater` merge in `merge_batch_stats`.
        a > b
    }
    #[inline(always)]
    fn to_owned(v: Int96) -> Int96 {
        v
    }
}

/// Normalize floating-point zero for Parquet statistics. Irrespective of the
/// observed sign, a zero minimum is written as `-0.0` and a zero maximum as
/// `+0.0`.
#[inline]
fn replace_zero<T: ParquetValueType>(val: &T, descr: &ColumnDescriptor, replace: f32) -> T {
    match T::PHYSICAL_TYPE {
        Type::FLOAT if f32::from_le_bytes(val.as_bytes().try_into().unwrap()) == 0.0 => {
            T::try_from_le_slice(&f32::to_le_bytes(replace)).unwrap()
        }
        Type::DOUBLE if f64::from_le_bytes(val.as_bytes().try_into().unwrap()) == 0.0 => {
            T::try_from_le_slice(&f64::to_le_bytes(replace as f64)).unwrap()
        }
        Type::FIXED_LEN_BYTE_ARRAY
            if descr.logical_type_ref() == Some(LogicalType::Float16).as_ref()
                && f16::from_le_bytes(val.as_bytes().try_into().unwrap()) == f16::NEG_ZERO =>
        {
            T::try_from_le_slice(&f16::to_le_bytes(f16::from_f32(replace))).unwrap()
        }
        _ => val.clone(),
    }
}

#[inline]
fn update_min_normalized<T: ParquetValueType>(
    descr: &ColumnDescriptor,
    val: &T,
    min_value: &mut Option<T>,
) {
    let val = replace_zero(val, descr, -0.0);
    update_min(descr, &val, min_value);
}

#[inline]
fn update_max_normalized<T: ParquetValueType>(
    descr: &ColumnDescriptor,
    val: &T,
    max_value: &mut Option<T>,
) {
    let val = replace_zero(val, descr, 0.0);
    update_max(descr, &val, max_value);
}

/// Creates a bloom filter sized for the column's configured NDV, returning the filter
/// and the target FPP for folding.
pub(crate) fn create_bloom_filter(
    props: &WriterProperties,
    descr: &ColumnDescPtr,
) -> Result<(Option<Sbbf>, f64)> {
    match props.bloom_filter_properties(descr.path()) {
        Some(bf_props) => Ok((
            Some(Sbbf::new_with_ndv_fpp(bf_props.ndv(), bf_props.fpp())?),
            bf_props.fpp(),
        )),
        None => Ok((None, 0.0)),
    }
}

/// Plain-encoded byte cost of a single value of type `T::T`.
///
/// Derived from [`ParquetValueType::dict_encoding_size`] (which returns
/// `(per-value overhead, value-bytes)`) so we don't add a parallel
/// per-value-size hook to the trait. Mirrors the dispatch in
/// `KeyStorage::push` (`encodings/encoding/dict_encoder.rs`).
///
#[inline]
fn plain_encoded_byte_size<T: DataType>(value: &T::T) -> usize {
    let (overhead, bytes) = value.dict_encoding_size();
    match <T::T as ParquetValueType>::PHYSICAL_TYPE {
        // Plain BYTE_ARRAY = 4-byte length prefix + payload.
        Type::BYTE_ARRAY => overhead + bytes,
        // Plain FLBA = raw bytes only; `dict_encoding_size`'s length prefix
        // is irrelevant here, so the encoder passes `type_length` directly.
        Type::FIXED_LEN_BYTE_ARRAY => bytes,
        // Numeric/bool are short-circuited by the caller via
        // `mem::size_of`, so this is unreachable in practice; fall back to
        // `overhead` defensively.
        _ => overhead,
    }
}

/// How many leading present values fit in `byte_budget` bytes.
///
/// This returns a strict fit and may therefore return zero.
/// [`ByteBudgetChunker`](super::ByteBudgetChunker) retries that value on a
/// fresh page and forces one-value progress only when the value itself is
/// larger than an empty page's budget.
#[inline]
pub(crate) fn count_within_budget<'a, T: DataType>(
    byte_budget: usize,
    vals: impl Iterator<Item = &'a T::T>,
) -> usize
where
    T::T: 'a,
{
    let mut cum: usize = 0;
    let mut count = 0;
    for value in vals {
        let encoded = plain_encoded_byte_size::<T>(value);
        if encoded > byte_budget.saturating_sub(cum) {
            return count;
        }
        cum += encoded;
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::data_type::FixedLenByteArrayType;
    use crate::schema::types::{ColumnPath, Type as SchemaType};

    #[test]
    fn dense_flba_batches_merge_statistics() {
        let primitive = SchemaType::primitive_type_builder("col", Type::FIXED_LEN_BYTE_ARRAY)
            .with_length(1)
            .build()
            .unwrap();
        let descriptor = Arc::new(ColumnDescriptor::new(
            Arc::new(primitive),
            0,
            0,
            ColumnPath::from("col"),
        ));
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .build();
        let mut encoder =
            ColumnValueEncoderImpl::<FixedLenByteArrayType>::try_new(&descriptor, &properties)
                .unwrap();

        encoder.begin_flba(4);
        encoder
            .commit(FlbaBatch::Dense {
                bytes: b"az",
                width: 1,
                count: 2,
            })
            .unwrap();
        encoder
            .commit(FlbaBatch::Dense {
                bytes: b"my",
                width: 1,
                count: 2,
            })
            .unwrap();
        encoder.finish_flba();

        let page = encoder.flush_data_page().unwrap();
        assert_eq!(
            page.min_value,
            Some(FixedLenByteArray::from(ByteArray::from("a")))
        );
        assert_eq!(
            page.max_value,
            Some(FixedLenByteArray::from(ByteArray::from("z")))
        );
    }
}
