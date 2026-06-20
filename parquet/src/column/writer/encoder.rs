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
use crate::column::writer::{ColumnWriter, ColumnWriterImpl};
use crate::column::writer::{
    compare_greater, fallback_encoding, has_dictionary_support, is_nan, update_max, update_min,
};
use crate::data_type::private::ParquetValueType;
use crate::data_type::{
    BoolType, ByteArrayType, DataType, DoubleType, FixedLenByteArrayType, FloatType, Int32Type,
    Int64Type, Int96Type,
};
#[cfg(feature = "arrow")]
use crate::encodings::encoding::{
    ChunkSink, DoubleValues, FloatValues, Int32Values, Int64Values, ValueStream,
};
use crate::encodings::encoding::{
    DictEncoder, DoubleEncoderObject, Encoder, EncoderFactory, FloatEncoderObject,
    Int32EncoderObject, Int64EncoderObject,
};
use crate::errors::{ParquetError, Result};
use crate::file::properties::{EnabledStatistics, WriterProperties};
use crate::geospatial::accumulator::{GeoStatsAccumulator, try_new_geo_stats_accumulator};
use crate::geospatial::statistics::GeospatialStatistics;
use crate::schema::types::{ColumnDescPtr, ColumnDescriptor};

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

/// Selects the encoder trait object used by `ColumnValueEncoderImpl` for a
/// Parquet physical type.
pub trait ColumnEncoderType: DataType + Sized {
    /// The trait object used to encode values of this physical type.
    type Encoder: EncoderFactory<Self> + ?Sized;

    /// Returns the underlying [`ColumnWriterImpl`] for the given [`ColumnWriter`].
    fn get_column_writer(column_writer: ColumnWriter<'_>) -> Option<ColumnWriterImpl<'_, Self>>;

    /// Returns a reference to the underlying [`ColumnWriterImpl`] for the given [`ColumnWriter`].
    fn get_column_writer_ref<'a, 'b: 'a>(
        column_writer: &'b ColumnWriter<'a>,
    ) -> Option<&'b ColumnWriterImpl<'a, Self>>;

    /// Returns a mutable reference to the underlying [`ColumnWriterImpl`] for the given
    /// [`ColumnWriter`].
    fn get_column_writer_mut<'a, 'b: 'a>(
        column_writer: &'a mut ColumnWriter<'b>,
    ) -> Option<&'a mut ColumnWriterImpl<'b, Self>>;
}

macro_rules! make_encoder_type {
    ($name:ident, $writer_ident:ident, $encoder:ty) => {
        impl ColumnEncoderType for $name {
            type Encoder = $encoder;

            fn get_column_writer(
                column_writer: ColumnWriter<'_>,
            ) -> Option<ColumnWriterImpl<'_, Self>> {
                match column_writer {
                    ColumnWriter::$writer_ident(w) => Some(w),
                    _ => None,
                }
            }

            fn get_column_writer_ref<'a, 'b: 'a>(
                column_writer: &'b ColumnWriter<'a>,
            ) -> Option<&'b ColumnWriterImpl<'a, Self>> {
                match column_writer {
                    ColumnWriter::$writer_ident(w) => Some(w),
                    _ => None,
                }
            }

            fn get_column_writer_mut<'a, 'b: 'a>(
                column_writer: &'a mut ColumnWriter<'b>,
            ) -> Option<&'a mut ColumnWriterImpl<'b, Self>> {
                match column_writer {
                    ColumnWriter::$writer_ident(w) => Some(w),
                    _ => None,
                }
            }
        }
    };
}

make_encoder_type!(BoolType, BoolColumnWriter, dyn Encoder<Self>);
make_encoder_type!(Int32Type, Int32ColumnWriter, Int32EncoderObject);
make_encoder_type!(Int64Type, Int64ColumnWriter, Int64EncoderObject);
make_encoder_type!(
    FixedLenByteArrayType,
    FixedLenByteArrayColumnWriter,
    dyn Encoder<Self>
);
make_encoder_type!(Int96Type, Int96ColumnWriter, dyn Encoder<Self>);
make_encoder_type!(FloatType, FloatColumnWriter, FloatEncoderObject);
make_encoder_type!(DoubleType, DoubleColumnWriter, DoubleEncoderObject);
make_encoder_type!(ByteArrayType, ByteArrayColumnWriter, dyn Encoder<Self>);

pub struct ColumnValueEncoderImpl<
    T: ColumnEncoderType,
    E: ?Sized = <T as ColumnEncoderType>::Encoder,
> {
    encoder: Box<E>,
    dict_encoder: Option<DictEncoder<T>>,
    descr: ColumnDescPtr,
    num_values: usize,
    statistics_enabled: EnabledStatistics,
    min_value: Option<T::T>,
    max_value: Option<T::T>,
    bloom_filter: Option<Sbbf>,
    bloom_filter_target_fpp: f64,
    variable_length_bytes: Option<i64>,
    geo_stats_accumulator: Option<Box<dyn GeoStatsAccumulator>>,
}

impl<T: ColumnEncoderType, E: Encoder<T> + ?Sized> ColumnValueEncoderImpl<T, E> {
    fn min_max(&self, values: &[T::T], value_indices: Option<&[usize]>) -> Option<(T::T, T::T)> {
        match value_indices {
            Some(indices) => get_min_max(&self.descr, indices.iter().map(|x| &values[*x])),
            None => get_min_max(&self.descr, values.iter()),
        }
    }

    fn write_slice(&mut self, slice: &[T::T]) -> Result<()> {
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

        match &mut self.dict_encoder {
            Some(encoder) => encoder.put(slice),
            _ => self.encoder.put(slice),
        }
    }
}

/// Shared chunked walk for numeric encoders. Dictionary adoption is handled by
/// the native entry points before they fall back to this intern/PLAIN path.
#[cfg(feature = "arrow")]
#[allow(private_bounds)]
impl<T, E> ColumnValueEncoderImpl<T, E>
where
    T: ColumnEncoderType,
    E: Encoder<T> + ?Sized,
    T::T: StatFold,
{
    #[inline]
    pub(crate) fn fold_value_stream<'a, S: ValueStream<'a, T::T, Bulk = [T::T]>>(
        &mut self,
        values: S,
    ) -> Result<()>
    where
        T::T: 'a,
    {
        let should_update_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;
        let ctx = <T::T as StatFold>::ctx(&self.descr);
        self.num_values += values.len();

        // Drive the selected values into a sink that folds stats and bloom
        // state before encoding each chunk.
        let (min, max) = {
            let target = match self.dict_encoder.as_mut() {
                Some(dict) => NumericSinkTarget::Dict(dict),
                None => NumericSinkTarget::Plain(&mut *self.encoder),
            };
            let mut sink = NumericSink {
                should_update_stats,
                ctx,
                min: None,
                max: None,
                bloom: self.bloom_filter.as_mut(),
                target,
            };
            values.write_into(&mut sink)?;
            (sink.min, sink.max)
        };

        if let Some(min) = min {
            update_min_normalized(&self.descr, &min, &mut self.min_value);
        }
        if let Some(max) = max {
            update_max_normalized(&self.descr, &max, &mut self.max_value);
        }
        Ok(())
    }
}

/// Where a [`NumericSink`] sends each encoded chunk: interned into the column's
/// dictionary, or bulk-`put` into the fallback encoder.
#[cfg(feature = "arrow")]
enum NumericSinkTarget<'a, T: DataType, E: ?Sized> {
    Dict(&'a mut DictEncoder<T>),
    Plain(&'a mut E),
}

/// Consumes numeric chunks: fold min/max, update the bloom filter, then encode.
#[cfg(feature = "arrow")]
struct NumericSink<'a, T: DataType, E: ?Sized>
where
    T::T: StatFold,
{
    should_update_stats: bool,
    ctx: <T::T as StatFold>::Ctx,
    min: Option<T::T>,
    max: Option<T::T>,
    bloom: Option<&'a mut Sbbf>,
    target: NumericSinkTarget<'a, T, E>,
}

#[cfg(feature = "arrow")]
impl<T: DataType, E: Encoder<T> + ?Sized> ChunkSink<[T::T]> for NumericSink<'_, T, E>
where
    T::T: StatFold,
{
    // Deliberately out-of-line. `ValueStream::write_into`'s gather path runs a
    // per-value closure that fills an N=64 tile and calls `consume` only when the
    // tile is full. Inlining `consume` (stats fold + bloom + dict/plain `put`)
    // into that closure bloats it past the inliner's threshold, so the closure is
    // emitted out-of-line and *called per value* — spilling the tile cursor/buffer
    // to its captured environment every value. Cutting the inline here keeps the
    // closure tiny so it folds into the gather loop (cursor in registers, no
    // per-value call); `consume` then runs once per tile, amortized over 64 values.
    #[inline(never)]
    fn consume(&mut self, chunk: &[T::T]) -> Result<()> {
        if self.should_update_stats {
            for &value in chunk {
                <T::T as StatFold>::observe(self.ctx, value, &mut self.min, &mut self.max);
            }
        }
        if let Some(bloom) = self.bloom.as_deref_mut() {
            for &value in chunk {
                bloom.insert(&value);
            }
        }
        match &mut self.target {
            NumericSinkTarget::Dict(dict) => dict.put(chunk),
            NumericSinkTarget::Plain(encoder) => encoder.put(chunk),
        }
    }
}

/// Generates `ColumnValueEncoderImpl::write_{int32,int64,float,double}_values`.
/// The value stream handles selection and casts; [`StatFold`] handles min/max
/// comparison.
macro_rules! numeric_values_method {
    ($fn:ident, $values_ty:ty) => {
        #[cfg(feature = "arrow")]
        pub(crate) fn $fn(&mut self, values: $values_ty) -> Result<()> {
            self.fold_value_stream(values)
        }
    };
}

impl ColumnValueEncoderImpl<Int32Type, Int32EncoderObject> {
    numeric_values_method!(write_int32_values, Int32Values<'_>);
}

impl ColumnValueEncoderImpl<Int64Type, Int64EncoderObject> {
    numeric_values_method!(write_int64_values, Int64Values<'_>);
}

impl ColumnValueEncoderImpl<FloatType, FloatEncoderObject> {
    numeric_values_method!(write_float_values, FloatValues<'_>);
}

impl ColumnValueEncoderImpl<DoubleType, DoubleEncoderObject> {
    numeric_values_method!(write_double_values, DoubleValues<'_>);
}

impl<T: ColumnEncoderType, E: EncoderFactory<T> + ?Sized> ColumnValueEncoder
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

/// Per-physical-type min/max folding for the fused stats walks.
///
/// Integer types carry signedness in the context; float types skip NaN and
/// compare with the natural order.
#[cfg(feature = "arrow")]
pub(crate) trait StatFold: Copy {
    /// Per-column comparison context, derived from the descriptor once and
    /// reused for every value (e.g. integer signedness).
    type Ctx: Copy;
    fn ctx(descr: &ColumnDescriptor) -> Self::Ctx;
    /// `a > b` under the column's logical order.
    fn greater(ctx: Self::Ctx, a: Self, b: Self) -> bool;
    /// True for values excluded from min/max (NaN floats); always false for ints.
    fn is_skippable(value: Self) -> bool;
    /// Fold one value into the running `(min, max)`.
    #[inline(always)]
    fn observe(ctx: Self::Ctx, value: Self, min: &mut Option<Self>, max: &mut Option<Self>) {
        if Self::is_skippable(value) {
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

#[cfg(feature = "arrow")]
impl StatFold for i32 {
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
    fn is_skippable(_: i32) -> bool {
        false
    }
}

#[cfg(feature = "arrow")]
impl StatFold for i64 {
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
    fn is_skippable(_: i64) -> bool {
        false
    }
}

#[cfg(feature = "arrow")]
impl StatFold for f32 {
    type Ctx = ();
    #[inline(always)]
    fn ctx(_: &ColumnDescriptor) {}
    #[inline(always)]
    fn greater(_: (), a: f32, b: f32) -> bool {
        a > b
    }
    #[inline(always)]
    fn is_skippable(value: f32) -> bool {
        value.is_nan()
    }
}

#[cfg(feature = "arrow")]
impl StatFold for f64 {
    type Ctx = ();
    #[inline(always)]
    fn ctx(_: &ColumnDescriptor) {}
    #[inline(always)]
    fn greater(_: (), a: f64, b: f64) -> bool {
        a > b
    }
    #[inline(always)]
    fn is_skippable(value: f64) -> bool {
        value.is_nan()
    }
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

#[inline]
#[cfg(feature = "arrow")]
fn update_min_normalized<T: ParquetValueType>(
    descr: &ColumnDescriptor,
    val: &T,
    min_value: &mut Option<T>,
) {
    let val = replace_zero(val, descr, -0.0);
    update_min(descr, &val, min_value);
}

#[inline]
#[cfg(feature = "arrow")]
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
