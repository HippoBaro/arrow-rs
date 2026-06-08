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

use bytes::Bytes;
use half::f16;

use crate::basic::{ConvertedType, Encoding, LogicalType, Type};
use crate::bloom_filter::Sbbf;
use crate::column::writer::{
    compare_greater, fallback_encoding, has_dictionary_support, is_nan, update_max, update_min,
};
#[cfg(feature = "arrow")]
use crate::column::writer::{compare_greater_byte_array, is_nan_byte_array};
#[cfg(feature = "arrow")]
use crate::data_type::FixedLenByteArray;
use crate::data_type::private::ParquetValueType;
use crate::data_type::{
    BoolType, ByteArrayType, DataType, DoubleType, FixedLenByteArrayType, FloatType, Int32Type,
    Int64Type, Int96Type,
};
#[cfg(feature = "arrow")]
use crate::encodings::encoding::{
    BoolEncoder, DictionaryValueIndices, FixedLenByteArrayEncoder, FixedLenByteArrayValues,
    Int32Encoder, Int32Values, Int64Encoder, Int64Values, PackedBoolValues, ValueIndices,
};
use crate::encodings::encoding::{
    BoolEncoderObject, DictEncoder, Encoder, EncoderFactory, FixedLenByteArrayEncoderObject,
    Int32EncoderObject, Int64EncoderObject,
};
use crate::errors::{ParquetError, Result};
use crate::file::properties::{EnabledStatistics, WriterProperties};
use crate::geospatial::accumulator::{GeoStatsAccumulator, try_new_geo_stats_accumulator};
use crate::geospatial::statistics::GeospatialStatistics;
use crate::schema::types::{ColumnDescPtr, ColumnDescriptor};
#[cfg(feature = "arrow")]
use arrow_array::ArrayRef;

/// A collection of [`ParquetValueType`] encoded by a [`ColumnValueEncoder`]
pub trait ColumnValues {
    /// The number of values in this collection
    fn len(&self) -> usize;
}

#[cfg(feature = "arrow")]
impl ColumnValues for dyn arrow_array::Array {
    fn len(&self) -> usize {
        arrow_array::Array::len(self)
    }
}

impl<T: ParquetValueType> ColumnValues for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

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

/// A generic encoder of [`ColumnValues`] to data and dictionary pages used by
/// [super::GenericColumnWriter`]
pub trait ColumnValueEncoder {
    /// The underlying value type of [`Self::Values`]
    ///
    /// Note: this avoids needing to fully qualify `<Self::Values as ColumnValues>::T`
    type T: ParquetValueType;

    /// The values encoded by this encoder
    type Values: ColumnValues + ?Sized;

    /// Create a new [`ColumnValueEncoder`]
    fn try_new(descr: &ColumnDescPtr, props: &WriterProperties) -> Result<Self>
    where
        Self: Sized;

    /// Write the corresponding values to this [`ColumnValueEncoder`]
    fn write(&mut self, values: &Self::Values, offset: usize, len: usize) -> Result<()>;

    /// Write the values at the indexes in `indices` to this [`ColumnValueEncoder`]
    fn write_gather(&mut self, values: &Self::Values, indices: &[usize]) -> Result<()>;

    /// Returns the largest `k` such that the first `k` values in
    /// `values[offset..offset + len]` encode to at most `byte_budget`
    /// bytes — i.e. how many values fit in a single page byte budget.
    ///
    /// Returns `len` if every value fits. Returns at least 1 if a single
    /// value alone exceeds the budget, matching parquet's "at least one
    /// value per data page" rule.
    ///
    /// `None` means "no cheap estimate available"; the caller stays on
    /// the batched fast path and lets the post-write
    /// `should_add_data_page` check handle bounding.
    ///
    /// Implementations should short-circuit aggressively: the typical
    /// case is "everything fits, return `len`", and the next-most-common
    /// case is "one wide value, return 1." The variable-width walk only
    /// needs to be precise when the chunk is genuinely near the budget.
    fn count_values_within_byte_budget(
        _values: &Self::Values,
        _offset: usize,
        _len: usize,
        _byte_budget: usize,
    ) -> Option<usize> {
        None
    }

    /// As [`Self::count_values_within_byte_budget`] but using gather
    /// `indices` rather than a contiguous range. Returns the number of
    /// `indices` that fit, not the maximum index value.
    fn count_values_within_byte_budget_gather(
        _values: &Self::Values,
        _indices: &[usize],
        _byte_budget: usize,
    ) -> Option<usize> {
        None
    }

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

    /// Flush the dictionary page for this column chunk if any. Any subsequent calls to
    /// [`Self::write`] will not be dictionary encoded
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

/// Selects the encoder trait object used by [`ColumnValueEncoderImpl`] for a
/// Parquet physical type.
pub trait ColumnEncoderType: Sized {
    /// The trait object used to encode values of this physical type.
    type Encoder: EncoderFactory<Self> + ?Sized
    where
        Self: DataType;
}

impl ColumnEncoderType for BoolType {
    type Encoder = BoolEncoderObject;
}

impl ColumnEncoderType for Int32Type {
    type Encoder = Int32EncoderObject;
}

impl ColumnEncoderType for Int64Type {
    type Encoder = Int64EncoderObject;
}

impl ColumnEncoderType for FixedLenByteArrayType {
    type Encoder = FixedLenByteArrayEncoderObject;
}

impl ColumnEncoderType for Int96Type {
    type Encoder = dyn Encoder<Self>;
}

impl ColumnEncoderType for FloatType {
    type Encoder = dyn Encoder<Self>;
}

impl ColumnEncoderType for DoubleType {
    type Encoder = dyn Encoder<Self>;
}

impl ColumnEncoderType for ByteArrayType {
    type Encoder = dyn Encoder<Self>;
}

pub struct ColumnValueEncoderImpl<T: DataType, E: ?Sized = <T as ColumnEncoderType>::Encoder> {
    encoder: Box<E>,
    dict_encoder: Option<DictEncoder<T>>,
    descr: ColumnDescPtr,
    /// Per-column dictionary page-size limit (bytes). Used to decide up front
    /// whether an Arrow dictionary is small enough to adopt, matching the
    /// reactive fallback in `GenericColumnWriter::should_dict_fallback`.
    dictionary_page_size_limit: usize,
    num_values: usize,
    statistics_enabled: EnabledStatistics,
    min_value: Option<T::T>,
    max_value: Option<T::T>,
    bloom_filter: Option<Sbbf>,
    bloom_filter_target_fpp: f64,
    variable_length_bytes: Option<i64>,
    geo_stats_accumulator: Option<Box<dyn GeoStatsAccumulator>>,
}

impl<T: DataType, E: Encoder<T> + ?Sized> ColumnValueEncoderImpl<T, E> {
    fn min_max(&self, values: &[T::T], value_indices: Option<&[usize]>) -> Option<(T::T, T::T)> {
        match value_indices {
            Some(indices) => get_min_max(&self.descr, indices.iter().map(|x| &values[*x])),
            None => get_min_max(&self.descr, values.iter()),
        }
    }

    fn write_slice(&mut self, slice: &[T::T]) -> Result<()> {
        self.fold_stats(slice);

        match &mut self.dict_encoder {
            Some(encoder) => encoder.put(slice),
            _ => self.encoder.put(slice),
        }
    }

    /// Fold a slice of values into the column statistics, variable-length byte
    /// total, and bloom filter — everything [`Self::write_slice`] does except
    /// the actual encoding.
    fn fold_stats(&mut self, slice: &[T::T]) {
        if self.statistics_enabled != EnabledStatistics::None
            // INTERVAL, Geometry, and Geography have undefined sort order, so don't write min/max stats for them
            && self.descr.converted_type() != ConvertedType::INTERVAL
        {
            if let Some(accumulator) = self.geo_stats_accumulator.as_deref_mut() {
                update_geo_stats_accumulator(accumulator, slice.iter());
            } else if let Some((min, max)) = self.min_max(slice, None) {
                update_min(&self.descr, &min, &mut self.min_value);
                update_max(&self.descr, &max, &mut self.max_value);
            }

            if let Some(var_bytes) = T::T::variable_length_bytes(slice) {
                *self.variable_length_bytes.get_or_insert(0) += var_bytes;
            }
        }

        // encode the values into bloom filter if enabled
        if let Some(bloom_filter) = &mut self.bloom_filter {
            for value in slice {
                bloom_filter.insert(value);
            }
        }
    }

    /// Zero-hash Arrow dictionary adopt fast path, shared by the native
    /// fixed-width physical types (INT32/INT64/FLOAT/DOUBLE).
    ///
    /// `dict_values` are the Arrow dictionary's entries (already deduplicated,
    /// in key order, byte-identical to the parquet dictionary), `keys` the
    /// per-row indices into them, `identity` the values array (its `Arc`
    /// identity recognises the same dictionary across mini-batches), and
    /// `plain_dict_page_size` the exact byte size this dictionary would occupy
    /// as a PLAIN dictionary page (for fixed-width entries, `card * width`).
    /// That size, compared to the column's dictionary-page size limit, is the
    /// reactive dict-vs-PLAIN fallback decision (`should_dict_fallback`) made
    /// upfront — the byte-array encoder applies the identical rule with the
    /// variable-length page formula (`data_bytes + 4 * card`).
    ///
    /// Returns `Ok(true)` when the batch was fully handled: adopted verbatim
    /// (first dictionary, sequential copy, no hashing) then keys appended, or
    /// appended to an already-adopted dictionary.
    ///
    /// Returns `Ok(false)` when the caller must fall through to its normal
    /// path — either the dictionary is too wide to stay dictionary-encoded (the
    /// dictionary encoder has been dropped, so the fall-through builds a PLAIN
    /// column with no interning), or a different dictionary arrived (the dedup
    /// table has been seeded, so the fall-through interns and deduplicates
    /// against the already-adopted entries).
    #[cfg(feature = "arrow")]
    pub(crate) fn maybe_adopt_native_dict(
        &mut self,
        dict_values: &[T::T],
        keys: DictionaryValueIndices<'_>,
        identity: &ArrayRef,
        plain_dict_page_size: usize,
    ) -> Result<bool> {
        if self.dict_encoder.is_none() {
            return Ok(false);
        }

        let fresh = self.dict_encoder.as_ref().unwrap().num_entries() == 0;
        if fresh && plain_dict_page_size >= self.dictionary_page_size_limit {
            // Too wide to ever pay off as a dictionary: drop the dictionary
            // encoder now and let the caller build a PLAIN column, skipping all
            // interning (matches the legacy fallback's dict-vs-PLAIN decision,
            // but made up front instead of after wasted interning).
            self.dict_encoder = None;
            return Ok(false);
        }

        if self
            .dict_encoder
            .as_mut()
            .unwrap()
            .maybe_adopt_dictionary(dict_values, identity)
        {
            // The bloom filter is column-scoped, so insert each dictionary entry
            // exactly once, on first adoption.
            if fresh {
                if let Some(bloom_filter) = self.bloom_filter.as_mut() {
                    for value in dict_values {
                        bloom_filter.insert(value);
                    }
                }
            }

            self.num_values += keys.len();

            // Per-data-page min/max: fold the referenced value of each key into
            // the running min/max, which `flush_data_page` resets at each page
            // boundary. Folding per occurrence (rather than once over the whole
            // dictionary) keeps page-level statistics tight when a dictionary
            // spans multiple pages. The dictionary fits the page-size limit, so
            // this keyed read stays cache-resident.
            let update_stats = self.statistics_enabled != EnabledStatistics::None
                && self.descr.converted_type() != ConvertedType::INTERVAL;
            let descr = self.descr.as_ref();
            let dict_enc = self.dict_encoder.as_mut().unwrap();
            if update_stats {
                let mut page_min: Option<&T::T> = None;
                let mut page_max: Option<&T::T> = None;
                keys.try_for_each(|key| -> Result<()> {
                    let value = &dict_values[key];
                    if !is_nan(descr, value) {
                        if page_min.is_none_or(|current| compare_greater(descr, current, value)) {
                            page_min = Some(value);
                        }
                        if page_max.is_none_or(|current| compare_greater(descr, value, current)) {
                            page_max = Some(value);
                        }
                    }
                    dict_enc.push_index(key as u64);
                    Ok(())
                })?;
                if let Some(min) = page_min {
                    update_min(descr, min, &mut self.min_value);
                }
                if let Some(max) = page_max {
                    update_max(descr, max, &mut self.max_value);
                }
            } else {
                keys.try_for_each(|key| -> Result<()> {
                    dict_enc.push_index(key as u64);
                    Ok(())
                })?;
            }
            Ok(true)
        } else {
            // A different dictionary arrived: leave adopt mode and intern from
            // here on. The caller's fall-through dedups against the adopted
            // entries.
            self.dict_encoder
                .as_mut()
                .unwrap()
                .seed_dedup_from_storage();
            Ok(false)
        }
    }
}

#[allow(dead_code)]
impl ColumnValueEncoderImpl<BoolType, BoolEncoderObject> {
    #[cfg(feature = "arrow")]
    pub(crate) fn write_packed_bool(&mut self, values: PackedBoolValues<'_>) -> Result<()> {
        self.num_values += values.len();

        if !values.is_empty() {
            let should_update_stats = self.statistics_enabled != EnabledStatistics::None
                && self.descr.converted_type() != ConvertedType::INTERVAL;

            if should_update_stats || self.bloom_filter.is_some() {
                let true_count = values.true_count();

                if should_update_stats {
                    let min = true_count == values.len();
                    let max = true_count > 0;
                    update_min(&self.descr, &min, &mut self.min_value);
                    update_max(&self.descr, &max, &mut self.max_value);
                }

                if let Some(bloom_filter) = &mut self.bloom_filter {
                    if true_count < values.len() {
                        bloom_filter.insert(&false);
                    }
                    if true_count > 0 {
                        bloom_filter.insert(&true);
                    }
                }
            }
        }

        debug_assert!(self.dict_encoder.is_none());
        self.encoder.put_packed_bool(values)
    }
}

#[allow(dead_code)]
impl ColumnValueEncoderImpl<FixedLenByteArrayType, FixedLenByteArrayEncoderObject> {
    #[cfg(feature = "arrow")]
    pub(crate) fn supports_raw_fixed_len_byte_array(&self) -> bool {
        self.dict_encoder.is_none()
            && self.geo_stats_accumulator.is_none()
            && matches!(
                self.encoder.encoding(),
                Encoding::PLAIN | Encoding::BYTE_STREAM_SPLIT
            )
    }

    #[cfg(feature = "arrow")]
    pub(crate) fn write_raw_fixed_len_byte_array(
        &mut self,
        values: FixedLenByteArrayValues<'_>,
    ) -> Result<()> {
        self.num_values += values.len();

        if !values.is_empty() {
            let should_update_stats = self.statistics_enabled != EnabledStatistics::None
                && self.descr.converted_type() != ConvertedType::INTERVAL;

            if should_update_stats {
                if let Some((min, max)) = self.raw_fixed_len_min_max(values) {
                    update_min(&self.descr, &min, &mut self.min_value);
                    update_max(&self.descr, &max, &mut self.max_value);
                }
            }

            if let Some(bloom_filter) = &mut self.bloom_filter {
                for value in values.iter() {
                    bloom_filter.insert(value);
                }
            }
        }

        debug_assert!(self.dict_encoder.is_none());
        self.encoder.put_fixed_len_byte_array(values)
    }

    #[cfg(feature = "arrow")]
    fn raw_fixed_len_min_max(
        &self,
        values: FixedLenByteArrayValues<'_>,
    ) -> Option<(FixedLenByteArray, FixedLenByteArray)> {
        let mut min: Option<&[u8]> = None;
        let mut max: Option<&[u8]> = None;

        for value in values.iter() {
            if is_nan_byte_array(&self.descr, value) {
                continue;
            }

            if min.is_none_or(|current| compare_greater_byte_array(&self.descr, current, value)) {
                min = Some(value);
            }
            if max.is_none_or(|current| compare_greater_byte_array(&self.descr, value, current)) {
                max = Some(value);
            }
        }

        Some(raw_fixed_len_min_max_values(&self.descr, min?, max?))
    }
}

#[cfg(feature = "arrow")]
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

#[cfg(feature = "arrow")]
fn raw_float16_stat_zero(value: &[u8], replacement: f16) -> [u8; 2] {
    let value: [u8; 2] = value.try_into().unwrap();
    if f16::from_le_bytes(value) == f16::ZERO {
        replacement.to_le_bytes()
    } else {
        value
    }
}

#[allow(dead_code)]
impl ColumnValueEncoderImpl<Int32Type, Int32EncoderObject> {
    #[cfg(feature = "arrow")]
    pub(crate) fn write_int32_values(
        &mut self,
        values: Int32Values<'_>,
        dict_identity: Option<&ArrayRef>,
    ) -> Result<()> {
        // Zero-hash dictionary adopt fast path for i32-backed, dictionary-encoded
        // input (see [`Self::maybe_adopt_native_dict`]). On `false`, fall through
        // to the legacy path below — which builds a PLAIN column (dictionary too
        // wide, encoder dropped) or interns and deduplicates against the adopted
        // entries (a different dictionary arrived).
        if let (Some(identity), Int32Values::I32(dict_values, ValueIndices::Dictionary(keys))) =
            (dict_identity, values)
        {
            // PLAIN dict-page size = card * width (fixed-width entries).
            let plain_dict_page_size = dict_values.len().saturating_mul(std::mem::size_of::<i32>());
            if self.maybe_adopt_native_dict(dict_values, keys, identity, plain_dict_page_size)? {
                return Ok(());
            }
        }

        if self.dict_encoder.is_some() || values.is_sparse() {
            self.num_values += values.len();
            let materialized = values.materialize();
            return self.write_slice(&materialized);
        }

        self.num_values += values.len();

        if values.is_empty() {
            return self.encoder.put_int32_values(values);
        }

        let should_update_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;

        if let Some(bloom_filter) = &mut self.bloom_filter {
            values.for_each(|value| bloom_filter.insert(&value));
        }

        // Dictionary-backed values require a scattered keyed read. Fuse the
        // min/max computation into the single encode pass so it rides in that
        // gather's cache-miss shadow rather than taking a second keyed walk.
        // (Dense values are contiguous; a sequential min/max scan is already
        // cheap, so they stay on the separate path below.)
        if should_update_stats && values.is_dictionary() {
            let unsigned = int_is_unsigned(&self.descr);
            let mut min: Option<i32> = None;
            let mut max: Option<i32> = None;
            self.encoder
                .put_int32_values_observed(values, &mut |value| {
                    if min.is_none_or(|current| int32_greater(unsigned, current, value)) {
                        min = Some(value);
                    }
                    if max.is_none_or(|current| int32_greater(unsigned, value, current)) {
                        max = Some(value);
                    }
                })?;
            if let Some(min) = min {
                update_min(&self.descr, &min, &mut self.min_value);
            }
            if let Some(max) = max {
                update_max(&self.descr, &max, &mut self.max_value);
            }
            return Ok(());
        }

        if should_update_stats {
            if let Some((min, max)) = int32_min_max(&self.descr, values) {
                update_min(&self.descr, &min, &mut self.min_value);
                update_max(&self.descr, &max, &mut self.max_value);
            }
        }

        self.encoder.put_int32_values(values)
    }
}

#[allow(dead_code)]
impl ColumnValueEncoderImpl<Int64Type, Int64EncoderObject> {
    #[cfg(feature = "arrow")]
    pub(crate) fn write_int64_values(
        &mut self,
        values: Int64Values<'_>,
        dict_identity: Option<&ArrayRef>,
    ) -> Result<()> {
        // Zero-hash dictionary adopt fast path for i64-backed, dictionary-encoded
        // input (see [`Self::maybe_adopt_native_dict`]). On `false`, fall through
        // to the legacy path below.
        if let (Some(identity), Int64Values::I64(dict_values, ValueIndices::Dictionary(keys))) =
            (dict_identity, values)
        {
            // PLAIN dict-page size = card * width (fixed-width entries).
            let plain_dict_page_size = dict_values.len().saturating_mul(std::mem::size_of::<i64>());
            if self.maybe_adopt_native_dict(dict_values, keys, identity, plain_dict_page_size)? {
                return Ok(());
            }
        }

        if self.dict_encoder.is_some() || values.is_sparse() {
            self.num_values += values.len();
            let materialized = values.materialize();
            return self.write_slice(&materialized);
        }

        self.num_values += values.len();

        if values.is_empty() {
            return self.encoder.put_int64_values(values);
        }

        let should_update_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;

        if let Some(bloom_filter) = &mut self.bloom_filter {
            values.for_each(|value| bloom_filter.insert(&value));
        }

        // Dictionary-backed values require a scattered keyed read. Fuse the
        // min/max computation into the single encode pass so it rides in that
        // gather's cache-miss shadow rather than taking a second keyed walk.
        // (Dense values are contiguous; a sequential min/max scan is already
        // cheap, so they stay on the separate path below.)
        if should_update_stats && values.is_dictionary() {
            let unsigned = int_is_unsigned(&self.descr);
            let mut min: Option<i64> = None;
            let mut max: Option<i64> = None;
            self.encoder
                .put_int64_values_observed(values, &mut |value| {
                    if min.is_none_or(|current| int64_greater(unsigned, current, value)) {
                        min = Some(value);
                    }
                    if max.is_none_or(|current| int64_greater(unsigned, value, current)) {
                        max = Some(value);
                    }
                })?;
            if let Some(min) = min {
                update_min(&self.descr, &min, &mut self.min_value);
            }
            if let Some(max) = max {
                update_max(&self.descr, &max, &mut self.max_value);
            }
            return Ok(());
        }

        if should_update_stats {
            if let Some((min, max)) = int64_min_max(&self.descr, values) {
                update_min(&self.descr, &min, &mut self.min_value);
                update_max(&self.descr, &max, &mut self.max_value);
            }
        }

        self.encoder.put_int64_values(values)
    }
}

impl<T: DataType, E: EncoderFactory<T> + ?Sized> ColumnValueEncoder
    for ColumnValueEncoderImpl<T, E>
{
    type T = T::T;

    type Values = [T::T];

    fn flush_bloom_filter(&mut self) -> Option<Sbbf> {
        let mut sbbf = self.bloom_filter.take()?;
        sbbf.fold_to_target_fpp(self.bloom_filter_target_fpp);
        Some(sbbf)
    }

    fn try_new(descr: &ColumnDescPtr, props: &WriterProperties) -> Result<Self> {
        let dict_supported = props.dictionary_enabled(descr.path())
            && has_dictionary_support(T::get_physical_type(), props);
        let dict_encoder = dict_supported.then(|| DictEncoder::new(descr.clone()));
        let dictionary_page_size_limit = props.column_dictionary_page_size_limit(descr.path());

        // Set either main encoder or fallback encoder.
        let encoder = E::get_encoder(
            props
                .encoding(descr.path())
                .unwrap_or_else(|| fallback_encoding(T::get_physical_type(), props)),
            descr,
        )?;

        let statistics_enabled = props.statistics_enabled(descr.path());

        let (bloom_filter, bloom_filter_target_fpp) = create_bloom_filter(props, descr)?;

        let geo_stats_accumulator = try_new_geo_stats_accumulator(descr);

        Ok(Self {
            encoder,
            dict_encoder,
            descr: descr.clone(),
            dictionary_page_size_limit,
            num_values: 0,
            statistics_enabled,
            bloom_filter,
            bloom_filter_target_fpp,
            min_value: None,
            max_value: None,
            variable_length_bytes: None,
            geo_stats_accumulator,
        })
    }

    fn write(&mut self, values: &[T::T], offset: usize, len: usize) -> Result<()> {
        self.num_values += len;

        let slice = values.get(offset..offset + len).ok_or_else(|| {
            general_err!(
                "Expected to write {} values, but have only {}",
                len,
                values.len() - offset
            )
        })?;

        self.write_slice(slice)
    }

    fn write_gather(&mut self, values: &Self::Values, indices: &[usize]) -> Result<()> {
        self.num_values += indices.len();
        let slice: Vec<_> = indices.iter().map(|idx| values[*idx].clone()).collect();
        self.write_slice(&slice)
    }

    fn count_values_within_byte_budget(
        values: &[T::T],
        offset: usize,
        len: usize,
        byte_budget: usize,
    ) -> Option<usize> {
        // Clamp so that a caller-supplied `len` that overruns the input
        // (e.g. a level/value mismatch the encoder will reject later)
        // returns an estimate instead of panicking here.
        let end = (offset + len).min(values.len());
        let start = offset.min(end);
        count_within_budget::<T>(
            end - start,
            byte_budget,
            values[start..end].iter().map(Some),
        )
    }

    fn count_values_within_byte_budget_gather(
        values: &[T::T],
        indices: &[usize],
        byte_budget: usize,
    ) -> Option<usize> {
        // `values.get` yields `None` for an out-of-range index (defensive
        // against a level/value mismatch the encoder rejects later); such a
        // position is counted but contributes no bytes.
        count_within_budget::<T>(
            indices.len(),
            byte_budget,
            indices.iter().map(|&i| values.get(i)),
        )
    }

    fn num_values(&self) -> usize {
        self.num_values
    }

    fn has_dictionary(&self) -> bool {
        self.dict_encoder.is_some()
    }

    fn estimated_memory_size(&self) -> usize {
        let encoder_size = self.encoder.estimated_memory_size();

        let dict_encoder_size = self
            .dict_encoder
            .as_ref()
            .map(|encoder| encoder.estimated_memory_size())
            .unwrap_or_default();

        let bloom_filter_size = self
            .bloom_filter
            .as_ref()
            .map(|bf| bf.estimated_memory_size())
            .unwrap_or_default();

        encoder_size + dict_encoder_size + bloom_filter_size
    }

    fn estimated_dict_page_size(&self) -> Option<usize> {
        Some(self.dict_encoder.as_ref()?.dict_encoded_size())
    }

    fn estimated_data_page_size(&self) -> usize {
        match &self.dict_encoder {
            Some(encoder) => encoder.estimated_data_encoded_size(),
            _ => self.encoder.estimated_data_encoded_size(),
        }
    }

    fn flush_dict_page(&mut self) -> Result<Option<DictionaryPage>> {
        match self.dict_encoder.take() {
            Some(encoder) => {
                if self.num_values != 0 {
                    return Err(general_err!(
                        "Must flush data pages before flushing dictionary"
                    ));
                }

                let buf = encoder.write_dict()?;

                Ok(Some(DictionaryPage {
                    buf,
                    num_values: encoder.num_entries(),
                    is_sorted: encoder.is_sorted(),
                }))
            }
            _ => Ok(None),
        }
    }

    fn flush_data_page(&mut self) -> Result<DataPageValues<T::T>> {
        let (buf, encoding) = match &mut self.dict_encoder {
            Some(encoder) => (encoder.write_indices()?, Encoding::RLE_DICTIONARY),
            _ => (self.encoder.flush_buffer()?, self.encoder.encoding()),
        };

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

#[cfg(feature = "arrow")]
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

#[cfg(feature = "arrow")]
#[inline(always)]
fn int32_greater(unsigned: bool, a: i32, b: i32) -> bool {
    if unsigned {
        (a as u32) > (b as u32)
    } else {
        a > b
    }
}

#[cfg(feature = "arrow")]
#[inline(always)]
fn int64_greater(unsigned: bool, a: i64, b: i64) -> bool {
    if unsigned {
        (a as u64) > (b as u64)
    } else {
        a > b
    }
}

#[cfg(feature = "arrow")]
fn int32_min_max(descr: &ColumnDescriptor, values: Int32Values<'_>) -> Option<(i32, i32)> {
    let unsigned = int_is_unsigned(descr);
    let mut min = None;
    let mut max = None;

    values.for_each(|value| {
        if min.is_none_or(|current| int32_greater(unsigned, current, value)) {
            min = Some(value);
        }
        if max.is_none_or(|current| int32_greater(unsigned, value, current)) {
            max = Some(value);
        }
    });

    Some((min?, max?))
}

#[cfg(feature = "arrow")]
fn int64_min_max(descr: &ColumnDescriptor, values: Int64Values<'_>) -> Option<(i64, i64)> {
    let unsigned = int_is_unsigned(descr);
    let mut min = None;
    let mut max = None;

    values.for_each(|value| {
        if min.is_none_or(|current| int64_greater(unsigned, current, value)) {
            min = Some(value);
        }
        if max.is_none_or(|current| int64_greater(unsigned, value, current)) {
            max = Some(value);
        }
    });

    Some((min?, max?))
}

fn get_min_max<'a, T, I>(descr: &ColumnDescriptor, mut iter: I) -> Option<(T, T)>
where
    T: ParquetValueType + 'a,
    I: Iterator<Item = &'a T>,
{
    let first = loop {
        let next = iter.next()?;
        if !is_nan(descr, next) {
            break next;
        }
    };

    let mut min = first;
    let mut max = first;
    for val in iter {
        if is_nan(descr, val) {
            continue;
        }
        if compare_greater(descr, min, val) {
            min = val;
        }
        if compare_greater(descr, val, max) {
            max = val;
        }
    }

    // Float/Double statistics have special case for zero.
    //
    // If computed min is zero, whether negative or positive,
    // the spec states that the min should be written as -0.0
    // (negative zero)
    //
    // For max, it has similar logic but will be written as 0.0
    // (positive zero)
    let min = replace_zero(min, descr, -0.0);
    let max = replace_zero(max, descr, 0.0);

    Some((min, max))
}

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

fn update_geo_stats_accumulator<'a, T, I>(bounder: &mut dyn GeoStatsAccumulator, iter: I)
where
    T: ParquetValueType + 'a,
    I: Iterator<Item = &'a T>,
{
    if bounder.is_valid() {
        for val in iter {
            bounder.update_wkb(val.as_bytes());
        }
    }
}

/// Plain-encoded byte cost of a single value of type `T::T`.
///
/// Derived from [`ParquetValueType::dict_encoding_size`] (which returns
/// `(per-value overhead, value-bytes)`) so we don't add a parallel
/// per-value-size hook to the trait. Mirrors the dispatch in
/// `KeyStorage::push` (`encodings/encoding/dict_encoder.rs`).
///
/// Placed at the end of the module deliberately. Inserting it above the
/// `ColumnValueEncoder` trait shifts the trait and `ColumnValueEncoderImpl`
/// within the compiled module enough to perturb downstream code placement,
/// which measurably regresses unrelated arrow-writer string benchmarks
/// (~5-9% on `string` / `string_and_binary_view`). Defining it last keeps
/// the hot encoder code at the offsets it has on `main`.
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

/// How many leading values fit in `byte_budget` bytes, shared by the two
/// `ColumnValueEncoder::count_values_within_byte_budget*` methods (one walks a
/// contiguous slice, the other gathers by index).
///
/// `n` is the answer when everything fits; `vals` yields each candidate value,
/// or `None` for a position that should still be counted but contributes no
/// bytes (an out-of-range gather index). The boundary value that crosses the
/// budget is included in the count so the caller's page-flush check trips on
/// this mini-batch rather than leaving a sliver for the next page; this also
/// catches a lone outlier wherever it lands among small values.
///
/// Defined at the end of the module alongside `plain_encoded_byte_size` for
/// the same reason — see that function's note on code placement and the
/// `string` / `string_and_binary_view` benchmarks.
#[inline]
fn count_within_budget<'a, T: DataType>(
    n: usize,
    byte_budget: usize,
    vals: impl Iterator<Item = Option<&'a T::T>>,
) -> Option<usize>
where
    T::T: 'a,
{
    // Fixed-size physical types have a constant per-value byte cost, so the
    // answer is one division — no walk needed.
    let phys = <T::T as ParquetValueType>::PHYSICAL_TYPE;
    if phys != Type::BYTE_ARRAY && phys != Type::FIXED_LEN_BYTE_ARRAY {
        let per = std::mem::size_of::<T::T>().max(1);
        return Some((byte_budget / per).max(1).min(n));
    }
    // Variable-width: accumulate, exit at the first value past the budget.
    let mut cum: usize = 0;
    for (i, v) in vals.enumerate() {
        if let Some(v) = v {
            cum = cum.saturating_add(plain_encoded_byte_size::<T>(v));
        }
        if cum > byte_budget {
            return Some(i + 1);
        }
    }
    Some(n)
}
