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

use crate::basic::Encoding;
use crate::bloom_filter::Sbbf;
use crate::column::writer::encoder::{
    ColumnValueEncoder, DataPageValues, DictionaryPage, create_bloom_filter,
};
use crate::column::writer::{ByteBudgetTarget, ColumnValueSource, Selected, ValueSelection};
use crate::data_type::{AsBytes, ByteArray, Int32Type};
use crate::encodings::encoding::{
    ChunkSink, DeltaBitPackEncoder, Encoder, RunEndValueIndices, RunEnds, ValueStream,
};
use crate::encodings::rle::RleEncoder;
use crate::errors::{ParquetError, Result};
use crate::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use crate::geospatial::accumulator::{GeoStatsAccumulator, try_new_geo_stats_accumulator};
use crate::geospatial::statistics::GeospatialStatistics;
use crate::schema::types::ColumnDescPtr;
use crate::util::bit_util::num_required_bits;
use crate::util::interner::{Interner, Storage};

use arrow_array::types::{
    ArrowDictionaryKeyType, BinaryType, ByteArrayType, LargeBinaryType, LargeUtf8Type, Utf8Type,
};
use arrow_array::{
    Array, BinaryArray, BinaryViewArray, DictionaryArray, FixedSizeBinaryArray, GenericByteArray,
    LargeBinaryArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow_buffer::{ArrowNativeType, Buffer, NullBuffer};
use arrow_schema::DataType;

macro_rules! with_dictionary_key_type {
    ($key_type:expr, |$key:ident| $body:block, $fallback:expr) => {
        match $key_type.as_ref() {
            DataType::UInt8 => {
                type $key = arrow_array::types::UInt8Type;
                $body
            }
            DataType::UInt16 => {
                type $key = arrow_array::types::UInt16Type;
                $body
            }
            DataType::UInt32 => {
                type $key = arrow_array::types::UInt32Type;
                $body
            }
            DataType::UInt64 => {
                type $key = arrow_array::types::UInt64Type;
                $body
            }
            DataType::Int8 => {
                type $key = arrow_array::types::Int8Type;
                $body
            }
            DataType::Int16 => {
                type $key = arrow_array::types::Int16Type;
                $body
            }
            DataType::Int32 => {
                type $key = arrow_array::types::Int32Type;
                $body
            }
            DataType::Int64 => {
                type $key = arrow_array::types::Int64Type;
                $body
            }
            _ => $fallback,
        }
    };
}

macro_rules! with_generic_byte_dictionary_value_type {
    ($value_type:expr, |$value:ident| $body:block, $fallback:expr) => {
        match $value_type.as_ref() {
            DataType::Utf8 => {
                type $value = Utf8Type;
                $body
            }
            DataType::LargeUtf8 => {
                type $value = LargeUtf8Type;
                $body
            }
            DataType::Binary => {
                type $value = BinaryType;
                $body
            }
            DataType::LargeBinary => {
                type $value = LargeBinaryType;
                $body
            }
            _ => $fallback,
        }
    };
}

/// A fallback encoder, i.e. non-dictionary, for [`ByteArray`]
struct FallbackEncoder {
    encoder: FallbackEncoderImpl,
    num_values: usize,
    variable_length_bytes: i64,
}

/// The fallback encoder in use
///
/// Note: DeltaBitPackEncoder is boxed as it is rather large
enum FallbackEncoderImpl {
    Plain {
        buffer: Vec<u8>,
    },
    DeltaLength {
        buffer: Vec<u8>,
        lengths: Box<DeltaBitPackEncoder<Int32Type>>,
    },
    Delta {
        buffer: Vec<u8>,
        last_value: Vec<u8>,
        prefix_lengths: Box<DeltaBitPackEncoder<Int32Type>>,
        suffix_lengths: Box<DeltaBitPackEncoder<Int32Type>>,
    },
}

impl FallbackEncoder {
    /// Create the fallback encoder for the given [`ColumnDescPtr`] and [`WriterProperties`]
    fn new(descr: &ColumnDescPtr, props: &WriterProperties) -> Result<Self> {
        // Set either main encoder or fallback encoder.
        let encoding =
            props
                .encoding(descr.path())
                .unwrap_or_else(|| match props.writer_version() {
                    WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
                    WriterVersion::PARQUET_2_0 => Encoding::DELTA_BYTE_ARRAY,
                });

        let encoder = match encoding {
            Encoding::PLAIN => FallbackEncoderImpl::Plain { buffer: vec![] },
            Encoding::DELTA_LENGTH_BYTE_ARRAY => FallbackEncoderImpl::DeltaLength {
                buffer: vec![],
                lengths: Box::new(DeltaBitPackEncoder::new()),
            },
            Encoding::DELTA_BYTE_ARRAY => FallbackEncoderImpl::Delta {
                buffer: vec![],
                last_value: vec![],
                prefix_lengths: Box::new(DeltaBitPackEncoder::new()),
                suffix_lengths: Box::new(DeltaBitPackEncoder::new()),
            },
            _ => {
                return Err(general_err!(
                    "unsupported encoding {} for byte array",
                    encoding
                ));
            }
        };

        Ok(Self {
            encoder,
            num_values: 0,
            variable_length_bytes: 0,
        })
    }

    /// Encode one tile of values, matching the concrete encoder once for the
    /// whole tile and reserving the PLAIN buffer before appending.
    #[inline(never)]
    fn encode_values(&mut self, values: &[&[u8]]) {
        self.num_values += values.len();
        match &mut self.encoder {
            FallbackEncoderImpl::Plain { buffer } => {
                let total: usize = values.iter().map(|v| v.len()).sum();
                buffer.reserve(total.saturating_add(values.len().saturating_mul(4)));
                for &value in values {
                    buffer.extend_from_slice((value.len() as u32).as_bytes());
                    buffer.extend_from_slice(value);
                    self.variable_length_bytes += value.len() as i64;
                }
            }
            FallbackEncoderImpl::DeltaLength { buffer, lengths } => {
                for &value in values {
                    lengths.put_i64(value.len() as i64).unwrap();
                    buffer.extend_from_slice(value);
                    self.variable_length_bytes += value.len() as i64;
                }
            }
            FallbackEncoderImpl::Delta {
                buffer,
                last_value,
                prefix_lengths,
                suffix_lengths,
            } => {
                for &value in values {
                    let mut prefix_length = 0;

                    while prefix_length < last_value.len()
                        && prefix_length < value.len()
                        && last_value[prefix_length] == value[prefix_length]
                    {
                        prefix_length += 1;
                    }

                    let suffix_length = value.len() - prefix_length;

                    last_value.clear();
                    last_value.extend_from_slice(value);

                    buffer.extend_from_slice(&value[prefix_length..]);
                    prefix_lengths.put_i64(prefix_length as i64).unwrap();
                    suffix_lengths.put_i64(suffix_length as i64).unwrap();
                    self.variable_length_bytes += value.len() as i64;
                }
            }
        }
    }

    /// Returns an estimate of the data page size in bytes
    ///
    /// This includes:
    /// <already_written_encoded_byte_size> + <estimated_encoded_size_of_unflushed_bytes>
    fn estimated_data_page_size(&self) -> usize {
        match &self.encoder {
            FallbackEncoderImpl::Plain { buffer, .. } => buffer.len(),
            FallbackEncoderImpl::DeltaLength { buffer, lengths } => {
                buffer.len() + lengths.estimated_data_encoded_size()
            }
            FallbackEncoderImpl::Delta {
                buffer,
                prefix_lengths,
                suffix_lengths,
                ..
            } => {
                buffer.len()
                    + prefix_lengths.estimated_data_encoded_size()
                    + suffix_lengths.estimated_data_encoded_size()
            }
        }
    }

    fn flush_data_page(
        &mut self,
        min_value: Option<ByteArray>,
        max_value: Option<ByteArray>,
    ) -> Result<DataPageValues<ByteArray>> {
        let (buf, encoding) = match &mut self.encoder {
            FallbackEncoderImpl::Plain { buffer } => (std::mem::take(buffer), Encoding::PLAIN),
            FallbackEncoderImpl::DeltaLength { buffer, lengths } => {
                let lengths = lengths.flush_buffer()?;

                let mut out = Vec::with_capacity(lengths.len() + buffer.len());
                out.extend_from_slice(&lengths);
                out.extend_from_slice(buffer);
                buffer.clear();
                (out, Encoding::DELTA_LENGTH_BYTE_ARRAY)
            }
            FallbackEncoderImpl::Delta {
                buffer,
                prefix_lengths,
                suffix_lengths,
                last_value,
            } => {
                let prefix_lengths = prefix_lengths.flush_buffer()?;
                let suffix_lengths = suffix_lengths.flush_buffer()?;

                let mut out =
                    Vec::with_capacity(prefix_lengths.len() + suffix_lengths.len() + buffer.len());
                out.extend_from_slice(&prefix_lengths);
                out.extend_from_slice(&suffix_lengths);
                out.extend_from_slice(buffer);
                buffer.clear();
                last_value.clear();
                (out, Encoding::DELTA_BYTE_ARRAY)
            }
        };

        // Capture value of variable_length_bytes and reset for next page
        let variable_length_bytes = Some(self.variable_length_bytes);
        self.variable_length_bytes = 0;

        Ok(DataPageValues {
            buf: buf.into(),
            num_values: std::mem::take(&mut self.num_values),
            encoding,
            min_value,
            max_value,
            variable_length_bytes,
        })
    }
}

/// [`Storage`] for the [`Interner`] used by [`DictEncoder`]
#[derive(Debug, Default)]
struct ByteArrayStorage {
    /// Encoded dictionary data
    page: Vec<u8>,

    values: Vec<std::ops::Range<usize>>,
}

impl Storage for ByteArrayStorage {
    type Key = u64;
    type Value = [u8];

    fn get(&self, idx: Self::Key) -> &Self::Value {
        &self.page[self.values[idx as usize].clone()]
    }

    fn push(&mut self, value: &Self::Value) -> Self::Key {
        let key = self.values.len();

        self.page.reserve(4 + value.len());
        self.page.extend_from_slice((value.len() as u32).as_bytes());

        let start = self.page.len();
        self.page.extend_from_slice(value);
        self.values.push(start..self.page.len());

        key as u64
    }

    #[allow(dead_code)] // not used in parquet_derive, so is dead there
    fn estimated_memory_size(&self) -> usize {
        self.page.capacity() * std::mem::size_of::<u8>()
            + self.values.capacity() * std::mem::size_of::<std::ops::Range<usize>>()
    }
}

/// A dictionary encoder for byte array data
#[derive(Debug, Default)]
struct DictEncoder {
    interner: Interner<ByteArrayStorage>,
    indices: Vec<u64>,
    /// Run-buffered indices `(index, count)` for run-end-encoded input — a run
    /// of `count` identical values shares one dictionary entry, so the index
    /// buffer is O(runs) not O(rows). Kept separate from `indices` so REE-only
    /// chunks stay compact; if dense values are appended later, the pending runs
    /// are materialized first.
    index_runs: Vec<(u64, u64)>,
    /// Logical value count buffered in `index_runs` (sum of run lengths).
    run_value_count: usize,
    variable_length_bytes: i64,
}

impl DictEncoder {
    fn reserve(&mut self, len: usize) {
        self.materialize_index_runs();
        self.indices.reserve(len);
    }

    #[inline]
    fn encode_value(&mut self, value: &[u8]) {
        let interned = self.interner.intern(value);
        self.indices.push(interned);
        self.variable_length_bytes += value.len() as i64;
    }

    /// Intern `value` and append it as a run of `count` repetitions — the
    /// run-end fast path, recording one `(index, count)` entry instead of
    /// `count` indices. `variable_length_bytes` still counts every logical
    /// occurrence so the page-size estimate matches the decoded payload.
    #[inline]
    fn encode_value_run(&mut self, value: &[u8], count: usize) {
        if count == 0 {
            return;
        }
        let interned = self.interner.intern(value);
        self.index_runs.push((interned, count as u64));
        self.run_value_count += count;
        self.variable_length_bytes += value.len() as i64 * count as i64;
    }

    /// Materialize pending run-buffered indices into the dense index buffer.
    /// This preserves append order when REE and dense batches are mixed in a
    /// column chunk. `variable_length_bytes` is already tracked when values are
    /// accepted, so only the index representation changes here.
    fn materialize_index_runs(&mut self) {
        if self.index_runs.is_empty() {
            return;
        }

        self.indices.reserve(self.run_value_count);
        for &(index, count) in &self.index_runs {
            self.indices
                .extend(std::iter::repeat_n(index, count as usize));
        }
        self.index_runs.clear();
        self.run_value_count = 0;
    }

    /// Intern one tile of values, keeping the hashing loop out of the caller's
    /// chunk-dispatch path.
    #[inline(never)]
    fn encode_values(&mut self, values: &[&[u8]]) {
        self.reserve(values.len());
        for &value in values {
            self.encode_value(value);
        }
    }

    fn num_values(&self) -> usize {
        self.indices.len() + self.run_value_count
    }

    fn bit_width(&self) -> u8 {
        let length = self.interner.storage().values.len();
        num_required_bits(length.saturating_sub(1) as u64)
    }

    fn estimated_memory_size(&self) -> usize {
        self.interner.estimated_memory_size()
            + self.indices.capacity() * std::mem::size_of::<u64>()
            + self.index_runs.capacity() * std::mem::size_of::<(u64, u64)>()
    }

    fn estimated_data_page_size(&self) -> usize {
        let bit_width = self.bit_width();
        1 + RleEncoder::max_buffer_size(bit_width, self.num_values())
    }

    fn estimated_dict_page_size(&self) -> usize {
        self.interner.storage().page.len()
    }

    fn flush_dict_page(self) -> DictionaryPage {
        let storage = self.interner.into_inner();

        DictionaryPage {
            buf: storage.page.into(),
            num_values: storage.values.len(),
            is_sorted: false,
        }
    }

    fn flush_data_page(
        &mut self,
        min_value: Option<ByteArray>,
        max_value: Option<ByteArray>,
    ) -> DataPageValues<ByteArray> {
        let num_values = self.num_values();
        let buffer_len = self.estimated_data_page_size();
        let mut buffer = Vec::with_capacity(buffer_len);
        buffer.push(self.bit_width());

        let mut encoder = RleEncoder::new_from_buf(self.bit_width(), buffer);
        for index in &self.indices {
            encoder.put(*index)
        }
        // Run-buffered indices: each run emits its index `count` times, which the
        // RLE encoder collapses back to one run in O(1).
        for &(index, count) in &self.index_runs {
            encoder.put_run(index, count as usize);
        }

        self.indices.clear();
        self.index_runs.clear();
        self.run_value_count = 0;

        // Capture value of variable_length_bytes and reset for next page
        let variable_length_bytes = Some(self.variable_length_bytes);
        self.variable_length_bytes = 0;

        DataPageValues {
            buf: encoder.consume().into(),
            num_values,
            encoding: Encoding::RLE_DICTIONARY,
            min_value,
            max_value,
            variable_length_bytes,
        }
    }
}

/// A [`ValueStream`] of `&[u8]` over the selected values of an offset-based byte
/// array (`Utf8`/`LargeUtf8`/`Binary`/`LargeBinary`).
///
/// Dense selections walk adjacent offset pairs with `windows(2)`. Variable
/// length values have no contiguous `&[&[u8]]` backing slice, so this stream
/// always yields gathered tiles.
struct OffsetByteValues<'a, T: ByteArrayType> {
    offsets: &'a [T::Offset],
    data: &'a [u8],
    selection: ValueSelection<'a>,
}

impl<'a, T: ByteArrayType> Clone for OffsetByteValues<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: ByteArrayType> Copy for OffsetByteValues<'a, T> {}

impl<'a, T: ByteArrayType> OffsetByteValues<'a, T> {
    #[inline]
    fn new(values: &'a GenericByteArray<T>, selection: ValueSelection<'a>) -> Self {
        Self {
            offsets: values.value_offsets(),
            data: values.value_data(),
            selection,
        }
    }
}

impl<'a, T: ByteArrayType> ValueStream<'a, &'a [u8]> for OffsetByteValues<'a, T> {
    // Variable-length values have no contiguous `&[&[u8]]` backing slice.
    type Bulk = [&'a [u8]];

    #[inline]
    fn len(self) -> usize {
        self.selection.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [&'a [u8]]> {
        None
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
        match self.selection {
            ValueSelection::Empty => Ok(()),
            ValueSelection::Dense { offset, len } => {
                let data = self.data;
                for w in self.offsets[offset..offset + len + 1].windows(2) {
                    f(&data[w[0].as_usize()..w[1].as_usize()])?;
                }
                Ok(())
            }
            ValueSelection::Sparse(indices) => {
                for &idx in indices {
                    f(dense_byte_value::<T>(self.offsets, self.data, idx))?;
                }
                Ok(())
            }
            ValueSelection::RunEnd(_) => {
                unreachable!("run-end byte columns are written via write_run_end")
            }
        }
    }
}

/// A [`ValueStream`] of `&[u8]` whose values are produced by a per-index closure
/// `get` — used for the layouts with O(1) value access (`Utf8View`/`BinaryView`,
/// `FixedSizeBinary`) and for dictionary-keyed byte arrays, where each row maps
/// through the keys to a dictionary value (or to `&[]` for a null key/value).
#[derive(Clone, Copy)]
struct MappedByteValues<'a, F> {
    selection: ValueSelection<'a>,
    get: F,
}

impl<'a, F> MappedByteValues<'a, F>
where
    F: Fn(usize) -> &'a [u8] + Copy,
{
    #[inline]
    fn new(selection: ValueSelection<'a>, get: F) -> Self {
        Self { selection, get }
    }
}

impl<'a, F> ValueStream<'a, &'a [u8]> for MappedByteValues<'a, F>
where
    F: Fn(usize) -> &'a [u8] + Copy + 'a,
{
    // Per-index gathers have no contiguous run to hand over.
    type Bulk = [&'a [u8]];

    #[inline]
    fn len(self) -> usize {
        self.selection.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [&'a [u8]]> {
        None
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
        self.selection.try_for_each(|idx| f((self.get)(idx)))
    }
}

/// A [`ValueStream`] of `&[u8]` over a run-end-encoded column's selected rows.
/// Drives the [`RunEndValueIndices`] cursor (O(rows + runs)) and maps each row's
/// physical run through `value_of`. Used by the non-dictionary run-end byte path
/// so it reuses the single run cursor instead of a `run_of` binary search per row.
#[derive(Clone, Copy)]
struct RunEndByteValues<'a, F> {
    runs: RunEndValueIndices<'a>,
    value_of: F,
}

impl<'a, F> ValueStream<'a, &'a [u8]> for RunEndByteValues<'a, F>
where
    F: Fn(usize) -> &'a [u8] + Copy + 'a,
{
    type Bulk = [&'a [u8]];

    #[inline]
    fn len(self) -> usize {
        self.runs.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [&'a [u8]]> {
        None
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
        let value_of = self.value_of;
        self.runs.try_for_each(|run| f(value_of(run)))
    }
}

pub struct ByteArrayEncoder {
    fallback: FallbackEncoder,
    dict_encoder: Option<DictEncoder>,
    statistics_enabled: EnabledStatistics,
    min_value: Option<ByteArray>,
    max_value: Option<ByteArray>,
    bloom_filter: Option<Sbbf>,
    bloom_filter_target_fpp: f64,
    geo_stats_accumulator: Option<Box<dyn GeoStatsAccumulator>>,
}

/// Where a [`ByteSink`] sends each encoded tile: interned into the dictionary, or
/// bulk-encoded through the fallback encoder. Both `encode_values` are infallible.
enum ByteSinkTarget<'e> {
    Dict(&'e mut DictEncoder),
    Fallback(&'e mut FallbackEncoder),
}

/// Consumes byte-array tiles: fold page stats, update the bloom filter, then
/// encode through either the dictionary or fallback encoder.
///
/// Plain min/max keep borrowed values during the walk and copy only when the
/// page-level extremum changes.
struct ByteSink<'a, 'e> {
    collect_stats: bool,
    min: Option<&'a [u8]>,
    max: Option<&'a [u8]>,
    accumulator: Option<&'e mut Box<dyn GeoStatsAccumulator>>,
    bloom: Option<&'e mut Sbbf>,
    target: ByteSinkTarget<'e>,
}

impl<'a> ChunkSink<[&'a [u8]]> for ByteSink<'a, '_> {
    #[inline]
    fn consume(&mut self, chunk: &[&'a [u8]]) -> Result<()> {
        if self.collect_stats {
            if let Some(accumulator) = self.accumulator.as_deref_mut() {
                for &v in chunk {
                    if accumulator.is_valid() {
                        accumulator.update_wkb(v);
                    }
                }
            } else {
                for &v in chunk {
                    if self.min.is_none_or(|current| current > v) {
                        self.min = Some(v);
                    }
                    if self.max.is_none_or(|current| current < v) {
                        self.max = Some(v);
                    }
                }
            }
        }
        // Insert the written values into the bloom filter.
        if let Some(bloom) = self.bloom.as_deref_mut() {
            for &v in chunk {
                bloom.insert(v);
            }
        }
        match &mut self.target {
            ByteSinkTarget::Dict(dict_encoder) => dict_encoder.encode_values(chunk),
            ByteSinkTarget::Fallback(fallback) => fallback.encode_values(chunk),
        }
        Ok(())
    }
}

impl ByteArrayEncoder {
    /// Encode an already-selected byte-value stream, folding page min/max and
    /// bloom-filter state in the same pass.
    fn write_byte_values<'a>(&mut self, values: impl ValueStream<'a, &'a [u8], Bulk = [&'a [u8]]>) {
        let len = values.len();
        let collect_stats = self.statistics_enabled != EnabledStatistics::None;

        // `ByteSink` encodes infallibly; the shared `ChunkSink` API is fallible
        // for other sinks.
        let (min, max) = {
            let target = match &mut self.dict_encoder {
                Some(dict_encoder) => {
                    dict_encoder.reserve(len);
                    ByteSinkTarget::Dict(dict_encoder)
                }
                None => ByteSinkTarget::Fallback(&mut self.fallback),
            };
            let mut sink = ByteSink {
                collect_stats,
                min: None,
                max: None,
                accumulator: self.geo_stats_accumulator.as_mut(),
                bloom: self.bloom_filter.as_mut(),
                target,
            };
            values
                .write_into(&mut sink)
                .expect("byte-array encode is infallible");
            (sink.min, sink.max)
        };

        self.merge_page_minmax(min, max);
    }

    /// Merge a page's observed min/max byte values into the running column
    /// min/max. Shared by the per-value byte path and the run-end fast path.
    fn merge_page_minmax(&mut self, min: Option<&[u8]>, max: Option<&[u8]>) {
        if let Some(min) = min {
            let min = ByteArray::from(min.to_vec());
            if self.min_value.as_ref().is_none_or(|m| m > &min) {
                self.min_value = Some(min);
            }
        }
        if let Some(max) = max {
            let max = ByteArray::from(max.to_vec());
            if self.max_value.as_ref().is_none_or(|m| m < &max) {
                self.max_value = Some(max);
            }
        }
    }

    /// Run-end fast path for the dictionary case: intern each run's byte value
    /// once and buffer one `(index, count)` entry, folding page min/max and the
    /// bloom filter per run. Returns `false` when not eligible — no dictionary
    /// (PLAIN/DELTA writes every value, so the run structure saves nothing), or
    /// a geospatial accumulator is active (it needs a per-value WKB update) — so
    /// the caller falls back to the per-value path.
    fn write_run_end_dict<'a>(
        &mut self,
        runs: RunEndValueIndices<'a>,
        value_of: impl Fn(usize) -> &'a [u8] + Copy,
    ) -> bool {
        if self.dict_encoder.is_none() || self.geo_stats_accumulator.is_some() {
            return false;
        }
        let collect_stats = self.statistics_enabled != EnabledStatistics::None;
        let (min, max) = {
            // `dict_encoder` and `bloom_filter` are disjoint fields, so both can
            // be borrowed across the run walk.
            let dict = self
                .dict_encoder
                .as_mut()
                .expect("dictionary encoder present");
            let mut bloom = self.bloom_filter.as_mut();
            let mut min: Option<&[u8]> = None;
            let mut max: Option<&[u8]> = None;
            let _: Result<(), ()> = runs.for_each_run(|run, count| {
                let value = value_of(run);
                if collect_stats {
                    if min.is_none_or(|m| m > value) {
                        min = Some(value);
                    }
                    if max.is_none_or(|m| m < value) {
                        max = Some(value);
                    }
                }
                if let Some(bloom) = bloom.as_deref_mut() {
                    bloom.insert(value);
                }
                dict.encode_value_run(value, count);
                Ok(())
            });
            // `min`/`max` borrow the run-values array (lifetime `'a`), not the
            // sink, so they outlive the walk — no eager copy needed here.
            (min, max)
        };
        self.merge_page_minmax(min, max);
        true
    }

    fn count_sparse_within_byte_budget(
        values: &dyn Array,
        indices: &[usize],
        byte_budget: usize,
    ) -> Option<usize> {
        // Two-stage walk for the simple offset-buffer byte array types:
        //   1. If indices are contiguous, compute the total payload in
        //      O(1) via a single subtraction on the offsets buffer.
        //      When the total fits the budget — the overwhelmingly
        //      common "small values" case — return immediately.
        //   2. Otherwise, walk per-value byte sizes from the offsets
        //      buffer (still cheap, no slice/UTF-8 construction) and
        //      exit at the first value that pushes the cumulative sum
        //      past the budget. This bounds skewed distributions: an
        //      outlier value is caught wherever it lands in the chunk.
        let count = match values.data_type() {
            DataType::Utf8 => count_within_budget_offsets(
                values.as_any().downcast_ref::<StringArray>().unwrap(),
                indices,
                byte_budget,
            ),
            DataType::LargeUtf8 => count_within_budget_offsets(
                values.as_any().downcast_ref::<LargeStringArray>().unwrap(),
                indices,
                byte_budget,
            ),
            DataType::Binary => count_within_budget_offsets(
                values.as_any().downcast_ref::<BinaryArray>().unwrap(),
                indices,
                byte_budget,
            ),
            DataType::LargeBinary => count_within_budget_offsets(
                values.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
                indices,
                byte_budget,
            ),
            // View arrays carry each value's length in the low 32 bits of
            // its u128 view word, so lengths are scannable without touching
            // any data buffer — and the common small-value case skips even
            // that scan via an O(1) conservative bound.
            DataType::Utf8View => {
                let array = values.as_any().downcast_ref::<StringViewArray>().unwrap();
                count_within_budget_views(
                    array.views(),
                    indices,
                    byte_budget,
                    max_view_value_len(array.data_buffers()),
                )
            }
            DataType::BinaryView => {
                let array = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                count_within_budget_views(
                    array.views(),
                    indices,
                    byte_budget,
                    max_view_value_len(array.data_buffers()),
                )
            }
            // The values in an arrow dictionary are already small and
            // deduplicated, so there is nothing to bound — treat every
            // chunk as fitting and stay on the batched path. (A per-value
            // walk through dict keys on every chunk also measured ~+30-80%
            // slower than `main`.)
            DataType::Dictionary(_, _) | DataType::FixedSizeBinary(_) => indices.len(),
            // Every other byte-array type `ByteArrayEncoder` is constructed for
            // has an explicit arm above, so nothing else can reach here.
            data_type => unreachable!("ByteArrayEncoder cannot be constructed for {data_type:?}"),
        };
        Some(count)
    }
}

impl ColumnValueEncoder for ByteArrayEncoder {
    type T = ByteArray;
    type Values = dyn Array;
    fn flush_bloom_filter(&mut self) -> Option<Sbbf> {
        let mut sbbf = self.bloom_filter.take()?;
        sbbf.fold_to_target_fpp(self.bloom_filter_target_fpp);
        Some(sbbf)
    }

    fn try_new(descr: &ColumnDescPtr, props: &WriterProperties) -> Result<Self>
    where
        Self: Sized,
    {
        let dictionary = props
            .dictionary_enabled(descr.path())
            .then(DictEncoder::default);

        let fallback = FallbackEncoder::new(descr, props)?;

        let (bloom_filter, bloom_filter_target_fpp) = create_bloom_filter(props, descr)?;

        let statistics_enabled = props.statistics_enabled(descr.path());

        let geo_stats_accumulator = try_new_geo_stats_accumulator(descr);

        Ok(Self {
            fallback,
            statistics_enabled,
            bloom_filter,
            bloom_filter_target_fpp,
            dict_encoder: dictionary,
            min_value: None,
            max_value: None,
            geo_stats_accumulator,
        })
    }

    fn num_values(&self) -> usize {
        match &self.dict_encoder {
            Some(encoder) => encoder.num_values(),
            None => self.fallback.num_values,
        }
    }

    fn has_dictionary(&self) -> bool {
        self.dict_encoder.is_some()
    }

    fn estimated_memory_size(&self) -> usize {
        let encoder_size = match &self.dict_encoder {
            Some(encoder) => encoder.estimated_memory_size(),
            // For the FallbackEncoder, these unflushed bytes are already encoded.
            // Therefore, the size should be the same as estimated_data_page_size.
            None => self.fallback.estimated_data_page_size(),
        };

        let bloom_filter_size = self
            .bloom_filter
            .as_ref()
            .map(|bf| bf.estimated_memory_size())
            .unwrap_or_default();

        let stats_size = self.min_value.as_ref().map(|v| v.len()).unwrap_or_default()
            + self.max_value.as_ref().map(|v| v.len()).unwrap_or_default();

        encoder_size + bloom_filter_size + stats_size
    }

    fn estimated_dict_page_size(&self) -> Option<usize> {
        Some(self.dict_encoder.as_ref()?.estimated_dict_page_size())
    }

    /// Returns an estimate of the data page size in bytes
    ///
    /// This includes:
    /// <already_written_encoded_byte_size> + <estimated_encoded_size_of_unflushed_bytes>
    fn estimated_data_page_size(&self) -> usize {
        match &self.dict_encoder {
            Some(encoder) => encoder.estimated_data_page_size(),
            None => self.fallback.estimated_data_page_size(),
        }
    }

    fn flush_dict_page(&mut self) -> Result<Option<DictionaryPage>> {
        match self.dict_encoder.take() {
            Some(encoder) => {
                if encoder.num_values() != 0 {
                    return Err(general_err!(
                        "Must flush data pages before flushing dictionary"
                    ));
                }

                Ok(Some(encoder.flush_dict_page()))
            }
            _ => Ok(None),
        }
    }

    fn flush_data_page(&mut self) -> Result<DataPageValues<ByteArray>> {
        let min_value = self.min_value.take();
        let max_value = self.max_value.take();

        match &mut self.dict_encoder {
            Some(encoder) => Ok(encoder.flush_data_page(min_value, max_value)),
            _ => self.fallback.flush_data_page(min_value, max_value),
        }
    }

    fn flush_geospatial_statistics(&mut self) -> Option<Box<GeospatialStatistics>> {
        self.geo_stats_accumulator.as_mut().map(|a| a.finish())?
    }
}

pub(crate) type ByteArraySource<'a> = Selected<'a, ByteArraySourceStorage<'a>>;

#[derive(Clone, Copy)]
pub(crate) struct ByteArraySourceStorage<'a> {
    values: &'a (dyn Array + 'static),
}

impl<'a> ByteArraySourceStorage<'a> {
    pub(crate) fn from_array(values: &'a (dyn Array + 'static)) -> Self {
        Self { values }
    }
}

impl<'a> ColumnValueSource<ByteArrayEncoder> for Selected<'a, ByteArraySourceStorage<'a>> {
    fn len(self) -> usize {
        Selected::len(self)
    }

    fn slice(self, offset: usize, len: usize) -> Self {
        Selected::slice(self, offset, len)
    }

    fn write_to(self, encoder: &mut ByteArrayEncoder) -> Result<()> {
        let values = self.storage().values;
        let selection = self.selection();
        if write_run_end(values, selection, encoder)? {
            return Ok(());
        }

        match selection {
            ValueSelection::Empty => Ok(()),
            // Dense and sparse both flow through the one chunked path. The
            // offset-array source specializes the dense walk (`windows(2)`)
            // internally, so there is no separate dense encode path.
            ValueSelection::Dense { .. } | ValueSelection::Sparse(_) => {
                write_selection(values, selection, encoder)
            }
            ValueSelection::RunEnd(_) => {
                unreachable!("run-end byte columns return early via write_run_end above")
            }
        }
    }

    fn count_within_byte_budget(self, budget: usize, target: ByteBudgetTarget) -> Option<usize> {
        let values = self.storage().values;
        if matches!(values.data_type(), DataType::RunEndEncoded(_, _)) {
            return count_run_end_within_byte_budget(values, self.selection(), budget, target);
        }
        match self.selection() {
            ValueSelection::Empty => None,
            // Mirror the dense/sparse split in `write_to` so the byte-budget
            // estimate matches the bytes actually written.
            ValueSelection::Dense { offset, len } => {
                Some(count_dense_within_byte_budget(values, offset, len, budget))
            }
            ValueSelection::Sparse(indices) => {
                ByteArrayEncoder::count_sparse_within_byte_budget(values, indices, budget)
            }
            // The run-end guard above returned before this match.
            ValueSelection::RunEnd(_) => None,
        }
    }
}

fn dense_byte_value<'a, T>(offsets: &[T::Offset], data: &'a [u8], idx: usize) -> &'a [u8]
where
    T: ByteArrayType,
{
    let start = offsets[idx].as_usize();
    let end = offsets[idx + 1].as_usize();
    &data[start..end]
}

fn downcast_dictionary<K>(values: &dyn Array) -> Result<&DictionaryArray<K>>
where
    K: ArrowDictionaryKeyType,
{
    values
        .as_any()
        .downcast_ref::<DictionaryArray<K>>()
        .ok_or_else(|| {
            ParquetError::General(format!(
                "Cannot downcast {} to dictionary",
                values.data_type()
            ))
        })
}

/// Dispatch a selected write to the typed byte-array source for `values`' layout
/// and hand it to [`ByteArrayEncoder::write_byte_values`]. Offset arrays build an
/// [`OffsetByteValues`] (dense walks `windows(2)`); the O(1)-access layouts and
/// dictionaries build a [`MappedByteValues`] over a per-index closure.
fn write_selection(
    values: &dyn Array,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()> {
    match values.data_type() {
        DataType::Utf8 => write_generic_byte_indices(
            values.as_any().downcast_ref::<StringArray>().unwrap(),
            selection,
            encoder,
        ),
        DataType::LargeUtf8 => write_generic_byte_indices(
            values.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            selection,
            encoder,
        ),
        DataType::Binary => write_generic_byte_indices(
            values.as_any().downcast_ref::<BinaryArray>().unwrap(),
            selection,
            encoder,
        ),
        DataType::LargeBinary => write_generic_byte_indices(
            values.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
            selection,
            encoder,
        ),
        DataType::Utf8View => write_string_view_indices(
            values.as_any().downcast_ref::<StringViewArray>().unwrap(),
            selection,
            encoder,
        ),
        DataType::BinaryView => write_binary_view_indices(
            values.as_any().downcast_ref::<BinaryViewArray>().unwrap(),
            selection,
            encoder,
        ),
        DataType::FixedSizeBinary(_) => write_fixed_size_binary_indices(
            values
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap(),
            selection,
            encoder,
        ),
        DataType::Dictionary(key, value) => match value.as_ref() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                with_generic_byte_dictionary_value_type!(
                    value,
                    |T| {
                        with_dictionary_key_type!(
                            key,
                            |K| {
                                write_generic_byte_dictionary_indices::<K, T>(
                                    values, selection, encoder,
                                )
                            },
                            unreachable!()
                        )
                    },
                    unreachable!()
                )
            }
            DataType::Utf8View => with_dictionary_key_type!(
                key,
                |K| { write_string_view_dictionary_indices::<K>(values, selection, encoder) },
                unreachable!()
            ),
            DataType::BinaryView => with_dictionary_key_type!(
                key,
                |K| { write_binary_view_dictionary_indices::<K>(values, selection, encoder) },
                unreachable!()
            ),
            DataType::FixedSizeBinary(_) => with_dictionary_key_type!(
                key,
                |K| { write_fixed_size_binary_dictionary_indices::<K>(values, selection, encoder) },
                unreachable!()
            ),
            d => unreachable!("cannot downcast {d} dictionary value to byte array"),
        },
        d => unreachable!("cannot downcast {d} to byte array"),
    }
}

fn write_generic_byte_indices<T>(
    values: &GenericByteArray<T>,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()>
where
    T: ByteArrayType,
{
    encoder.write_byte_values(OffsetByteValues::<T>::new(values, selection));
    Ok(())
}

/// If `values` is run-end encoded, stream its run values through the byte
/// dictionary path: each selected logical row maps to its physical run's value,
/// so the run values are interned into a dictionary (one RLE run per run)
/// and no dense byte array is materialized. Returns `true` when handled. The
/// non-null `selection` only references non-null run values, so the null check
/// in the mapping closure is defensive.
fn write_run_end<'a>(
    values: &'a dyn Array,
    selection: ValueSelection<'a>,
    encoder: &mut ByteArrayEncoder,
) -> Result<bool> {
    if !matches!(values.data_type(), DataType::RunEndEncoded(_, _)) {
        return Ok(false);
    }
    // Single source of truth for the RunArray<Int16/32/64> downcast (shared with
    // the numeric dispatch and level computation).
    let (run_ends, offset, run_values) = super::run_ends_of(values)?;

    match run_values.data_type() {
        DataType::Utf8 => {
            write_run_end_generic::<Utf8Type>(run_ends, offset, run_values, selection, encoder)
        }
        DataType::LargeUtf8 => {
            write_run_end_generic::<LargeUtf8Type>(run_ends, offset, run_values, selection, encoder)
        }
        DataType::Binary => {
            write_run_end_generic::<BinaryType>(run_ends, offset, run_values, selection, encoder)
        }
        DataType::LargeBinary => write_run_end_generic::<LargeBinaryType>(
            run_ends, offset, run_values, selection, encoder,
        ),
        DataType::Utf8View => {
            let arr = run_values
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            emit_run_end_byte_values(encoder, run_ends, offset, selection, move |run| {
                if arr.is_null(run) {
                    &[]
                } else {
                    arr.value(run).as_bytes()
                }
            });
        }
        DataType::BinaryView => {
            let arr = run_values
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap();
            emit_run_end_byte_values(encoder, run_ends, offset, selection, move |run| {
                if arr.is_null(run) {
                    &[]
                } else {
                    arr.value(run)
                }
            });
        }
        DataType::FixedSizeBinary(_) => {
            let arr = run_values
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            emit_run_end_byte_values(encoder, run_ends, offset, selection, move |run| {
                if arr.is_null(run) {
                    &[]
                } else {
                    arr.value(run)
                }
            });
        }
        d => {
            return Err(ParquetError::NYI(format!(
                "Run-end column with {d} run values is not supported"
            )));
        }
    }
    Ok(true)
}

/// Emit a run-end-encoded byte column from a per-run value accessor
/// `value_of(run) -> &[u8]`. Dictionary-encoded columns intern one entry per
/// run ([`ByteArrayEncoder::write_run_end_dict`]); otherwise each selected row
/// maps through its run via a [`RunEndByteValues`] cursor.
fn emit_run_end_byte_values<'a>(
    encoder: &mut ByteArrayEncoder,
    run_ends: RunEnds<'a>,
    offset: usize,
    selection: ValueSelection<'a>,
    value_of: impl Fn(usize) -> &'a [u8] + Copy + 'a,
) {
    let runs = RunEndValueIndices::new(run_ends, offset, selection);
    if encoder.write_run_end_dict(runs, value_of) {
        return;
    }
    // Non-dictionary fall-through (PLAIN/DELTA, or geo stats active): reuse the
    // same single run cursor instead of a `run_of` binary search per row.
    encoder.write_byte_values(RunEndByteValues { runs, value_of });
}

fn write_run_end_generic<'a, T>(
    run_ends: RunEnds<'a>,
    offset: usize,
    run_values: &'a dyn Array,
    selection: ValueSelection<'a>,
    encoder: &mut ByteArrayEncoder,
) where
    T: ByteArrayType,
{
    let arr = run_values
        .as_any()
        .downcast_ref::<GenericByteArray<T>>()
        .unwrap();
    let offsets = arr.value_offsets();
    let data = arr.value_data();
    emit_run_end_byte_values(encoder, run_ends, offset, selection, move |run| {
        if arr.is_null(run) {
            &[]
        } else {
            dense_byte_value::<T>(offsets, data, run)
        }
    });
}

fn write_string_view_indices(
    values: &StringViewArray,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()> {
    encoder.write_byte_values(MappedByteValues::new(selection, move |idx| {
        values.value(idx).as_bytes()
    }));
    Ok(())
}

fn write_binary_view_indices(
    values: &BinaryViewArray,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()> {
    encoder.write_byte_values(MappedByteValues::new(selection, move |idx| {
        values.value(idx)
    }));
    Ok(())
}

fn write_fixed_size_binary_indices(
    values: &FixedSizeBinaryArray,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()> {
    encoder.write_byte_values(MappedByteValues::new(selection, move |idx| {
        values.value(idx)
    }));
    Ok(())
}

fn write_generic_byte_dictionary_indices<K, T>(
    dictionary: &dyn Array,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()>
where
    K: ArrowDictionaryKeyType,
    T: ByteArrayType,
{
    let dictionary = downcast_dictionary::<K>(dictionary)?;
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<GenericByteArray<T>>()
        .unwrap();
    let keys = dictionary.keys();
    let offsets = values.value_offsets();
    let data = values.value_data();
    encoder.write_byte_values(MappedByteValues::new(selection, move |row| -> &[u8] {
        if keys.is_null(row) {
            return &[];
        }
        let key = keys.value(row).as_usize();
        if values.is_null(key) {
            &[]
        } else {
            dense_byte_value::<T>(offsets, data, key)
        }
    }));
    Ok(())
}

fn write_fixed_size_binary_dictionary_indices<K>(
    dictionary: &dyn Array,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()>
where
    K: ArrowDictionaryKeyType,
{
    let dictionary = downcast_dictionary::<K>(dictionary)?;
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();
    let keys = dictionary.keys();
    encoder.write_byte_values(MappedByteValues::new(selection, move |row| -> &[u8] {
        if keys.is_null(row) {
            return &[];
        }
        let key = keys.value(row).as_usize();
        if values.is_null(key) {
            &[]
        } else {
            values.value(key)
        }
    }));
    Ok(())
}

fn write_string_view_dictionary_indices<K>(
    dictionary: &dyn Array,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()>
where
    K: ArrowDictionaryKeyType,
{
    let dictionary = downcast_dictionary::<K>(dictionary)?;
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    let keys = dictionary.keys();
    encoder.write_byte_values(MappedByteValues::new(selection, move |row| -> &[u8] {
        if keys.is_null(row) {
            return &[];
        }
        let key = keys.value(row).as_usize();
        if values.is_null(key) {
            &[]
        } else {
            values.value(key).as_bytes()
        }
    }));
    Ok(())
}

fn write_binary_view_dictionary_indices<K>(
    dictionary: &dyn Array,
    selection: ValueSelection<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<()>
where
    K: ArrowDictionaryKeyType,
{
    let dictionary = downcast_dictionary::<K>(dictionary)?;
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .unwrap();
    let keys = dictionary.keys();
    encoder.write_byte_values(MappedByteValues::new(selection, move |row| -> &[u8] {
        if keys.is_null(row) {
            return &[];
        }
        let key = keys.value(row).as_usize();
        if values.is_null(key) {
            &[]
        } else {
            values.value(key)
        }
    }));
    Ok(())
}

fn count_run_end_within_byte_budget(
    values: &dyn Array,
    selection: ValueSelection<'_>,
    byte_budget: usize,
    target: ByteBudgetTarget,
) -> Option<usize> {
    // While dictionary encoding is active, REE byte values are interned once per
    // run, so logical decoded bytes are the wrong budget. Preserve the existing
    // dictionary-page behavior; after dictionary fallback or with dictionaries
    // disabled, `target` is DataPage and every logical value is written out.
    if target == ByteBudgetTarget::DictionaryPage {
        return None;
    }

    let Ok((run_ends, offset, run_values)) = super::run_ends_of(values) else {
        return None;
    };
    let runs = RunEndValueIndices::new(run_ends, offset, selection);
    match run_values.data_type() {
        DataType::Utf8 => {
            let array = run_values.as_any().downcast_ref::<StringArray>().unwrap();
            Some(count_run_end_with_lengths(runs, byte_budget, |run| {
                generic_byte_value_len(array, run)
            }))
        }
        DataType::LargeUtf8 => {
            let array = run_values
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap();
            Some(count_run_end_with_lengths(runs, byte_budget, |run| {
                generic_byte_value_len(array, run)
            }))
        }
        DataType::Binary => {
            let array = run_values.as_any().downcast_ref::<BinaryArray>().unwrap();
            Some(count_run_end_with_lengths(runs, byte_budget, |run| {
                generic_byte_value_len(array, run)
            }))
        }
        DataType::LargeBinary => {
            let array = run_values
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap();
            Some(count_run_end_with_lengths(runs, byte_budget, |run| {
                generic_byte_value_len(array, run)
            }))
        }
        DataType::Utf8View => {
            let array = run_values
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            Some(count_run_end_with_lengths(runs, byte_budget, |run| {
                view_byte_value_len(array.views(), array.nulls(), run)
            }))
        }
        DataType::BinaryView => {
            let array = run_values
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap();
            Some(count_run_end_with_lengths(runs, byte_budget, |run| {
                view_byte_value_len(array.views(), array.nulls(), run)
            }))
        }
        _ => None,
    }
}

fn count_run_end_with_lengths<'a>(
    runs: RunEndValueIndices<'a>,
    byte_budget: usize,
    value_len: impl Fn(usize) -> usize,
) -> usize {
    let mut count = 0usize;
    let mut cum = 0usize;
    let _: std::result::Result<(), ()> = runs.for_each_run(|run, run_count| {
        let per_value = value_len(run).saturating_add(std::mem::size_of::<u32>());
        let run_bytes = per_value.saturating_mul(run_count);
        if cum.saturating_add(run_bytes) > byte_budget {
            let remaining = byte_budget.saturating_sub(cum);
            count += ((remaining / per_value) + 1).min(run_count);
            return Err(());
        }
        cum += run_bytes;
        count += run_count;
        Ok(())
    });
    count
}

fn generic_byte_value_len<T: ByteArrayType>(array: &GenericByteArray<T>, idx: usize) -> usize {
    if array.is_null(idx) {
        0
    } else {
        let offsets = array.value_offsets();
        (offsets[idx + 1] - offsets[idx]).as_usize()
    }
}

fn view_byte_value_len(views: &[u128], nulls: Option<&NullBuffer>, idx: usize) -> usize {
    if nulls.is_some_and(|n| n.is_null(idx)) {
        0
    } else {
        (views[idx] as u32) as usize
    }
}

/// Upper bound on any single value's byte length in a view array.
fn max_view_value_len(buffers: &[Buffer]) -> usize {
    /// Bytes that fit inline in a u128 view word (the rest is len + prefix).
    const MAX_INLINE_VIEW_LEN: usize = 12;
    // An out-of-line view's data is a contiguous slice of exactly one data
    // buffer, so it cannot exceed the largest buffer; inline views hold at
    // most `MAX_INLINE_VIEW_LEN`. Loose (a value is usually far smaller than
    // a whole buffer) but O(number of buffers) and always sound.
    buffers
        .iter()
        .map(|b| b.len())
        .max()
        .unwrap_or(0)
        .max(MAX_INLINE_VIEW_LEN)
}

/// Number of leading `indices` whose cumulative plain-encoded size fits
/// `byte_budget` (boundary value included), for view arrays (`Utf8View`,
/// `BinaryView`).
fn count_within_budget_views(
    views: &[u128],
    indices: &[usize],
    byte_budget: usize,
    max_value_len: usize,
) -> usize {
    // Each plain-encoded BYTE_ARRAY value carries a 4-byte length prefix, so
    // the budget is compared against `value_len + size_of::<u32>()` — the
    // bytes actually written to the page, not just the payload.
    //
    // Stage 1: O(1) conservative bound. View arrays have no prefix-sum
    // offsets buffer, so the exact span subtraction used by
    // `count_within_budget_offsets` is unavailable; instead bound every
    // value by `max_value_len`. Skips the walk for the common small-value
    // case (what view arrays are built for, and where there is nothing to
    // bound).
    let per_value = max_value_len + std::mem::size_of::<u32>();
    if indices.len().saturating_mul(per_value) <= byte_budget {
        return indices.len();
    }
    // Stage 2: exact per-value scan, reading each length from the low 32
    // bits of its u128 view word (no data-buffer dereference).
    let mut cum: usize = 0;
    for (i, idx) in indices.iter().enumerate() {
        let len = (views[*idx] as u32) as usize;
        cum = cum.saturating_add(len + std::mem::size_of::<u32>());
        if cum > byte_budget {
            return i + 1;
        }
    }
    indices.len()
}

/// Number of leading `indices` whose cumulative plain-encoded size fits
/// `byte_budget` (boundary value included), for offset-buffer byte arrays
/// (`Utf8`/`LargeUtf8`/`Binary`/`LargeBinary`).
///
/// `indices` are assumed sorted ascending — they always are here, since
/// they come from `non_null_indices`, which is built in array order.
fn count_within_budget_offsets<T: ByteArrayType>(
    values: &GenericByteArray<T>,
    indices: &[usize],
    byte_budget: usize,
) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let n = indices.len();
    let first = indices[0];
    let last = indices[n - 1];
    let offsets = values.value_offsets();
    // Each plain-encoded value carries a 4-byte length prefix on the page.
    let prefix_overhead = std::mem::size_of::<u32>();

    // Stage 1: O(1) span upper bound. The span `offsets[last+1] -
    // offsets[first]` covers every array position in `[first, last]`, a
    // superset of `indices` — and the skipped positions in a nullable
    // column are nulls with zero offset delta, so the span still equals the
    // exact payload. If it fits the budget, every value fits. Covers the
    // common small-value case for both non-null and (sparse) nullable
    // columns.
    if last >= first {
        let payload = (offsets[last + 1] - offsets[first]).as_usize();
        if payload + n * prefix_overhead <= byte_budget {
            return n;
        }
    }

    // Stage 2: scan per-index lengths from the offsets buffer.
    let mut cum: usize = 0;
    for (i, idx) in indices.iter().enumerate() {
        let len = (offsets[idx + 1] - offsets[*idx]).as_usize() + prefix_overhead;
        cum = cum.saturating_add(len);
        if cum > byte_budget {
            return i + 1;
        }
    }
    n
}

/// Number of leading values in the dense range `[offset, offset + len)` whose
/// cumulative plain-encoded size fits `byte_budget` (boundary value included).
///
/// Dense counterpart of the sparse dispatch in
/// [`ByteArrayEncoder::count_sparse_within_byte_budget`], used for the
/// contiguous, all-valid selection produced for non-nullable byte columns.
fn count_dense_within_byte_budget(
    values: &dyn Array,
    offset: usize,
    len: usize,
    byte_budget: usize,
) -> usize {
    match values.data_type() {
        DataType::Utf8 => count_within_budget_offsets_dense(
            values.as_any().downcast_ref::<StringArray>().unwrap(),
            offset,
            len,
            byte_budget,
        ),
        DataType::LargeUtf8 => count_within_budget_offsets_dense(
            values.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            offset,
            len,
            byte_budget,
        ),
        DataType::Binary => count_within_budget_offsets_dense(
            values.as_any().downcast_ref::<BinaryArray>().unwrap(),
            offset,
            len,
            byte_budget,
        ),
        DataType::LargeBinary => count_within_budget_offsets_dense(
            values.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
            offset,
            len,
            byte_budget,
        ),
        DataType::Utf8View => {
            let array = values.as_any().downcast_ref::<StringViewArray>().unwrap();
            count_within_budget_views_dense(
                array.views(),
                offset,
                len,
                byte_budget,
                max_view_value_len(array.data_buffers()),
            )
        }
        DataType::BinaryView => {
            let array = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            count_within_budget_views_dense(
                array.views(),
                offset,
                len,
                byte_budget,
                max_view_value_len(array.data_buffers()),
            )
        }
        // Dictionary values are already small and deduplicated, so there is
        // nothing to bound — treat every chunk as fitting.
        DataType::Dictionary(_, _) | DataType::FixedSizeBinary(_) => len,
        data_type => unreachable!("ByteArrayEncoder cannot be constructed for {data_type:?}"),
    }
}

/// Dense-range counterpart of [`count_within_budget_offsets`].
fn count_within_budget_offsets_dense<T: ByteArrayType>(
    values: &GenericByteArray<T>,
    offset: usize,
    len: usize,
    byte_budget: usize,
) -> usize {
    if len == 0 {
        return 0;
    }
    let offsets = values.value_offsets();
    let prefix_overhead = std::mem::size_of::<u32>();
    // Stage 1: O(1) span over the contiguous range.
    let payload = (offsets[offset + len] - offsets[offset]).as_usize();
    if payload + len * prefix_overhead <= byte_budget {
        return len;
    }
    // Stage 2: per-value scan.
    let mut cum: usize = 0;
    for i in 0..len {
        let value_len =
            (offsets[offset + i + 1] - offsets[offset + i]).as_usize() + prefix_overhead;
        cum = cum.saturating_add(value_len);
        if cum > byte_budget {
            return i + 1;
        }
    }
    len
}

/// Dense-range counterpart of [`count_within_budget_views`].
fn count_within_budget_views_dense(
    views: &[u128],
    offset: usize,
    len: usize,
    byte_budget: usize,
    max_value_len: usize,
) -> usize {
    let per_value = max_value_len + std::mem::size_of::<u32>();
    if len.saturating_mul(per_value) <= byte_budget {
        return len;
    }
    let mut cum: usize = 0;
    for i in 0..len {
        let value_len = (views[offset + i] as u32) as usize;
        cum = cum.saturating_add(value_len + std::mem::size_of::<u32>());
        if cum > byte_budget {
            return i + 1;
        }
    }
    len
}
