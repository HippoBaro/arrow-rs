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
use crate::column::writer::{ColumnValueSource, Selected, ValueSelectionRef};
use crate::data_type::{AsBytes, ByteArray, Int32Type};
use crate::encodings::encoding::{DeltaBitPackEncoder, Encoder};
use crate::encodings::rle::RleEncoder;
use crate::errors::{ParquetError, Result};
use crate::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use crate::geospatial::accumulator::{GeoStatsAccumulator, try_new_geo_stats_accumulator};
use crate::geospatial::statistics::GeospatialStatistics;
use crate::schema::types::ColumnDescPtr;
use crate::util::bit_util::num_required_bits;
use crate::util::interner::{Interner, Storage};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowDictionaryKeyType, BinaryType, ByteArrayType, LargeBinaryType, LargeUtf8Type, Utf8Type,
};
use arrow_array::{
    Array, ArrayAccessor, ArrayRef, BinaryArray, BinaryViewArray, DictionaryArray,
    FixedSizeBinaryArray, GenericByteArray, LargeBinaryArray, LargeStringArray, PrimitiveArray,
    StringArray, StringViewArray,
};
use arrow_buffer::{ArrowNativeType, Buffer};
use arrow_schema::DataType;

macro_rules! downcast_dict_impl {
    ($array:ident, $key:ident, $val:ident, $op:expr $(, $arg:expr)*) => {{
        $op($array
            .as_any()
            .downcast_ref::<DictionaryArray<arrow_array::types::$key>>()
            .unwrap()
            .downcast_dict::<$val>()
            .unwrap()$(, $arg)*)
    }};
}

macro_rules! downcast_dict_op {
    ($key_type:expr, $val:ident, $array:ident, $op:expr $(, $arg:expr)*) => {
        match $key_type.as_ref() {
            DataType::UInt8 => downcast_dict_impl!($array, UInt8Type, $val, $op$(, $arg)*),
            DataType::UInt16 => downcast_dict_impl!($array, UInt16Type, $val, $op$(, $arg)*),
            DataType::UInt32 => downcast_dict_impl!($array, UInt32Type, $val, $op$(, $arg)*),
            DataType::UInt64 => downcast_dict_impl!($array, UInt64Type, $val, $op$(, $arg)*),
            DataType::Int8 => downcast_dict_impl!($array, Int8Type, $val, $op$(, $arg)*),
            DataType::Int16 => downcast_dict_impl!($array, Int16Type, $val, $op$(, $arg)*),
            DataType::Int32 => downcast_dict_impl!($array, Int32Type, $val, $op$(, $arg)*),
            DataType::Int64 => downcast_dict_impl!($array, Int64Type, $val, $op$(, $arg)*),
            _ => unreachable!(),
        }
    };
}

macro_rules! downcast_op {
    ($data_type:expr, $array:ident, $op:expr $(, $arg:expr)*) => {
        match $data_type {
            DataType::Utf8 => $op($array.as_any().downcast_ref::<StringArray>().unwrap()$(, $arg)*),
            DataType::LargeUtf8 => {
                $op($array.as_any().downcast_ref::<LargeStringArray>().unwrap()$(, $arg)*)
            }
            DataType::Utf8View => $op($array.as_any().downcast_ref::<StringViewArray>().unwrap()$(, $arg)*),
            DataType::Binary => {
                $op($array.as_any().downcast_ref::<BinaryArray>().unwrap()$(, $arg)*)
            }
            DataType::LargeBinary => {
                $op($array.as_any().downcast_ref::<LargeBinaryArray>().unwrap()$(, $arg)*)
            }
            DataType::FixedSizeBinary(_) => {
                $op($array.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap()$(, $arg)*)
            }
            DataType::BinaryView => {
                $op($array.as_any().downcast_ref::<BinaryViewArray>().unwrap()$(, $arg)*)
            }
            DataType::Dictionary(key, value) => match value.as_ref() {
                DataType::Utf8 => downcast_dict_op!(key, StringArray, $array, $op$(, $arg)*),
                DataType::LargeUtf8 => {
                    downcast_dict_op!(key, LargeStringArray, $array, $op$(, $arg)*)
                }
                DataType::Binary => downcast_dict_op!(key, BinaryArray, $array, $op$(, $arg)*),
                DataType::LargeBinary => {
                    downcast_dict_op!(key, LargeBinaryArray, $array, $op$(, $arg)*)
                }
                DataType::FixedSizeBinary(_) => {
                    downcast_dict_op!(key, FixedSizeBinaryArray, $array, $op$(, $arg)*)
                }
                d => unreachable!("cannot downcast {} dictionary value to byte array", d),
            },
            d => unreachable!("cannot downcast {} to byte array", d),
        }
    };
}

macro_rules! downcast_key_dictionary_impl {
    ($array:ident, $key:ident, $value_type:ty, $selection:expr, $encoder:ident) => {
        try_encode_generic_byte_dictionary::<arrow_array::types::$key, $value_type>(
            $array
                .as_any()
                .downcast_ref::<DictionaryArray<arrow_array::types::$key>>()
                .unwrap(),
            $selection,
            $encoder,
        )
    };
}

macro_rules! downcast_key_dictionary_op {
    ($key_type:expr, $value_type:ty, $array:ident, $selection:expr, $encoder:ident) => {
        match $key_type.as_ref() {
            DataType::UInt8 => {
                downcast_key_dictionary_impl!($array, UInt8Type, $value_type, $selection, $encoder)
            }
            DataType::UInt16 => {
                downcast_key_dictionary_impl!($array, UInt16Type, $value_type, $selection, $encoder)
            }
            DataType::UInt32 => {
                downcast_key_dictionary_impl!($array, UInt32Type, $value_type, $selection, $encoder)
            }
            DataType::UInt64 => {
                downcast_key_dictionary_impl!($array, UInt64Type, $value_type, $selection, $encoder)
            }
            DataType::Int8 => {
                downcast_key_dictionary_impl!($array, Int8Type, $value_type, $selection, $encoder)
            }
            DataType::Int16 => {
                downcast_key_dictionary_impl!($array, Int16Type, $value_type, $selection, $encoder)
            }
            DataType::Int32 => {
                downcast_key_dictionary_impl!($array, Int32Type, $value_type, $selection, $encoder)
            }
            DataType::Int64 => {
                downcast_key_dictionary_impl!($array, Int64Type, $value_type, $selection, $encoder)
            }
            _ => Ok(false),
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

    /// Encode `values` to the in-progress page
    fn encode<T>(&mut self, values: T, indices: impl ExactSizeIterator<Item = usize>)
    where
        T: ArrayAccessor + Copy,
        T::Item: AsRef<[u8]>,
    {
        self.num_values += indices.len();
        match &mut self.encoder {
            FallbackEncoderImpl::Plain { buffer } => {
                for idx in indices {
                    let value = values.value(idx);
                    let value = value.as_ref();
                    buffer.extend_from_slice((value.len() as u32).as_bytes());
                    buffer.extend_from_slice(value);
                    self.variable_length_bytes += value.len() as i64;
                }
            }
            FallbackEncoderImpl::DeltaLength { buffer, lengths } => {
                for idx in indices {
                    let value = values.value(idx);
                    let value = value.as_ref();
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
                for idx in indices {
                    let value = values.value(idx);
                    let value = value.as_ref();
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

    /// Encode a contiguous range from an offset-based byte array
    fn encode_dense<T>(&mut self, values: &GenericByteArray<T>, offset: usize, len: usize)
    where
        T: ByteArrayType,
    {
        self.num_values += len;
        let offsets = values.value_offsets();
        let end = offset + len;
        let total_bytes = offsets[end].as_usize() - offsets[offset].as_usize();

        match &mut self.encoder {
            FallbackEncoderImpl::Plain { buffer } => {
                buffer.reserve(total_bytes.saturating_add(len.saturating_mul(4)));
                for value in dense_byte_values(values, offset, len) {
                    buffer.extend_from_slice((value.len() as u32).as_bytes());
                    buffer.extend_from_slice(value);
                    self.variable_length_bytes += value.len() as i64;
                }
            }
            FallbackEncoderImpl::DeltaLength { buffer, lengths } => {
                buffer.reserve(total_bytes);
                for value in dense_byte_values(values, offset, len) {
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
                buffer.reserve(total_bytes);
                for value in dense_byte_values(values, offset, len) {
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
    variable_length_bytes: i64,
    /// Monotonic data-page counter, bumped on each [`Self::flush_data_page`].
    /// Used with [`ByteDictAdoptState::Adopted`]'s `stat_gen` to fold each
    /// dictionary entry's value into min/max exactly once **per page** it
    /// appears in.
    page_gen: u32,
    /// State of the zero-hash Arrow dictionary adopt fast path.
    adopt: ByteDictAdoptState,
}

/// Tracks the zero-hash Arrow dictionary adopt fast path (mirrors the generic
/// `DictEncoder`'s adopt machinery for the bespoke byte-array encoder).
#[derive(Debug, Default)]
enum ByteDictAdoptState {
    /// No values yet; the next Arrow dictionary may be adopted verbatim.
    #[default]
    Fresh,
    /// Built by adopting one Arrow dictionary verbatim. `identity` is that
    /// dictionary's values array (held so `Arc::ptr_eq` stays sound across
    /// batches); `stat_gen[i]` is the `page_gen` at which entry `i` was last
    /// folded into the page min/max (`u32::MAX` = never), making per-page
    /// statistics exact without a per-occurrence value read.
    Adopted {
        identity: ArrayRef,
        stat_gen: Vec<u32>,
    },
    /// Interning mode (dedup hash table active); never returns to adopting.
    Legacy,
}

impl DictEncoder {
    /// Encode `values` to the in-progress page
    fn encode<T>(&mut self, values: T, indices: impl ExactSizeIterator<Item = usize>)
    where
        T: ArrayAccessor + Copy,
        T::Item: AsRef<[u8]>,
    {
        self.indices.reserve(indices.len());

        for idx in indices {
            let value = values.value(idx);
            let interned = self.interner.intern(value.as_ref());
            self.indices.push(interned);
            self.variable_length_bytes += value.as_ref().len() as i64;
        }
    }

    /// Encode a contiguous range from an offset-based byte array
    fn encode_dense<T>(&mut self, values: &GenericByteArray<T>, offset: usize, len: usize)
    where
        T: ByteArrayType,
    {
        self.indices.reserve(len);

        for value in dense_byte_values(values, offset, len) {
            let interned = self.interner.intern(value);
            self.indices.push(interned);
            self.variable_length_bytes += value.len() as i64;
        }
    }

    fn bit_width(&self) -> u8 {
        let length = self.interner.storage().values.len();
        num_required_bits(length.saturating_sub(1) as u64)
    }

    fn estimated_memory_size(&self) -> usize {
        let stat_gen_size = match &self.adopt {
            ByteDictAdoptState::Adopted { stat_gen, .. } => {
                stat_gen.capacity() * std::mem::size_of::<u32>()
            }
            _ => 0,
        };

        self.interner.estimated_memory_size()
            + self.indices.capacity() * std::mem::size_of::<u64>()
            + stat_gen_size
    }

    fn estimated_data_page_size(&self) -> usize {
        let bit_width = self.bit_width();
        1 + RleEncoder::max_buffer_size(bit_width, self.indices.len())
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
        let num_values = self.indices.len();
        let buffer_len = self.estimated_data_page_size();
        let mut buffer = Vec::with_capacity(buffer_len);
        buffer.push(self.bit_width());

        let mut encoder = RleEncoder::new_from_buf(self.bit_width(), buffer);
        for index in &self.indices {
            encoder.put(*index)
        }

        self.indices.clear();

        // Start a new page generation so per-page min/max re-folds each entry
        // referenced in the next page (see `ByteDictAdoptState::Adopted`).
        self.page_gen = self.page_gen.wrapping_add(1);

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

pub struct ByteArrayEncoder {
    fallback: FallbackEncoder,
    dict_encoder: Option<DictEncoder>,
    statistics_enabled: EnabledStatistics,
    min_value: Option<ByteArray>,
    max_value: Option<ByteArray>,
    bloom_filter: Option<Sbbf>,
    bloom_filter_target_fpp: f64,
    geo_stats_accumulator: Option<Box<dyn GeoStatsAccumulator>>,
    /// The column's dictionary-page size limit. An incoming Arrow dictionary
    /// whose PLAIN-encoded page would meet or exceed this is too wide to adopt
    /// (it would fall back to PLAIN anyway), so the encoder drops dictionary
    /// encoding upfront rather than building a dictionary it will discard.
    dictionary_page_size_limit: usize,
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

        let dictionary_page_size_limit = props.column_dictionary_page_size_limit(descr.path());

        Ok(Self {
            fallback,
            statistics_enabled,
            bloom_filter,
            bloom_filter_target_fpp,
            dict_encoder: dictionary,
            min_value: None,
            max_value: None,
            geo_stats_accumulator,
            dictionary_page_size_limit,
        })
    }

    fn write(&mut self, values: &Self::Values, offset: usize, len: usize) -> Result<()> {
        if try_encode_dictionary(values, ValueSelectionRef::Dense { offset, len }, self)? {
            return Ok(());
        }

        write_dense(values, offset, len, self)
    }

    fn write_gather(&mut self, values: &Self::Values, indices: &[usize]) -> Result<()> {
        if try_encode_dictionary(values, ValueSelectionRef::Sparse(indices), self)? {
            return Ok(());
        }

        downcast_op!(
            values.data_type(),
            values,
            encode,
            indices.iter().copied(),
            self
        );
        Ok(())
    }

    fn count_values_within_byte_budget_gather(
        values: &Self::Values,
        indices: &[usize],
        byte_budget: usize,
    ) -> Option<usize> {
        // `ByteArrayEncoder` only ever writes via `write_gather`, so this
        // is the relevant method.
        //
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

    fn num_values(&self) -> usize {
        match &self.dict_encoder {
            Some(encoder) => encoder.indices.len(),
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
                if !encoder.indices.is_empty() {
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
        if let Some(materialized) = materialize_dictionary_with_selected_nulls(values, selection)? {
            return ByteArraySource::new(
                ByteArraySourceStorage::from_array(materialized.as_ref()),
                selection,
            )
            .write_to(encoder);
        }

        if try_encode_dictionary(values, selection, encoder)? {
            return Ok(());
        }

        match selection {
            ValueSelectionRef::Empty => Ok(()),
            ValueSelectionRef::Dense { offset, len } => write_dense(values, offset, len, encoder),
            ValueSelectionRef::Sparse(indices) => {
                downcast_op!(
                    values.data_type(),
                    values,
                    encode,
                    indices.iter().copied(),
                    encoder
                );
                Ok(())
            }
        }
    }

    fn count_within_byte_budget(self, budget: usize) -> Option<usize> {
        let values = self.storage().values;
        match self.selection() {
            ValueSelectionRef::Empty => None,
            // Mirror the dense/gather split in `write_to` so the byte-budget
            // estimate matches the bytes actually written.
            ValueSelectionRef::Dense { offset, len } => {
                Some(count_dense_within_byte_budget(values, offset, len, budget))
            }
            ValueSelectionRef::Sparse(indices) => {
                ByteArrayEncoder::count_values_within_byte_budget_gather(values, indices, budget)
            }
        }
    }
}

fn materialize_dictionary_with_selected_nulls(
    values: &dyn Array,
    selection: ValueSelectionRef<'_>,
) -> Result<Option<ArrayRef>> {
    let DataType::Dictionary(_, _) = values.data_type() else {
        return Ok(None);
    };

    let logical_nulls = values.logical_nulls();
    if !super::value_indices_include_nulls(logical_nulls.as_ref(), super::value_indices(selection))
    {
        return Ok(None);
    }

    let dictionary = values.as_any_dictionary();
    Ok(Some(arrow_select::take::take(
        dictionary.values(),
        dictionary.keys(),
        None,
    )?))
}

fn dense_byte_value<'a, T>(offsets: &[T::Offset], data: &'a [u8], idx: usize) -> &'a [u8]
where
    T: ByteArrayType,
{
    let start = offsets[idx].as_usize();
    let end = offsets[idx + 1].as_usize();
    &data[start..end]
}

fn try_encode_dictionary(
    values: &dyn Array,
    selection: ValueSelectionRef<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<bool> {
    if encoder.dict_encoder.is_none() {
        return Ok(false);
    }

    let DataType::Dictionary(key_type, value_type) = values.data_type() else {
        return Ok(false);
    };

    if selection.len() == 0 {
        return Ok(true);
    }

    // A sparse selection is built from the leaf's valid (non-null) positions
    // (see `levels::write_leaf`), so it can never reference a null key slot;
    // skip the O(n) walk. Only a dense selection (e.g. a required field whose
    // dictionary still carries null key slots) needs the guard, and there it is
    // a cheap popcount over the range.
    let key_selection = super::value_indices(selection);
    if !key_selection.is_sparse()
        && super::value_indices_include_nulls(values.nulls(), key_selection)
    {
        return Ok(false);
    }

    match value_type.as_ref() {
        DataType::Utf8 => {
            downcast_key_dictionary_op!(key_type, Utf8Type, values, selection, encoder)
        }
        DataType::LargeUtf8 => {
            downcast_key_dictionary_op!(key_type, LargeUtf8Type, values, selection, encoder)
        }
        DataType::Binary => {
            downcast_key_dictionary_op!(key_type, BinaryType, values, selection, encoder)
        }
        DataType::LargeBinary => {
            downcast_key_dictionary_op!(key_type, LargeBinaryType, values, selection, encoder)
        }
        _ => Ok(false),
    }
}

fn try_encode_generic_byte_dictionary<K, T>(
    dictionary: &DictionaryArray<K>,
    selection: ValueSelectionRef<'_>,
    encoder: &mut ByteArrayEncoder,
) -> Result<bool>
where
    K: ArrowDictionaryKeyType,
    T: ByteArrayType,
{
    let Some(values) = dictionary
        .values()
        .as_any()
        .downcast_ref::<GenericByteArray<T>>()
    else {
        return Ok(false);
    };

    // A dictionary that carries null value slots cannot be adopted verbatim
    // (PLAIN dictionary pages hold no nulls); fall back to interning.
    if values.null_count() != 0 {
        return Ok(false);
    }

    Ok(encode_dictionary(
        encoder,
        dictionary.values(),
        dictionary.keys(),
        values,
        selection,
    ))
}

fn for_each_selected(selection: ValueSelectionRef<'_>, mut f: impl FnMut(usize)) {
    match selection {
        ValueSelectionRef::Empty => {}
        ValueSelectionRef::Dense { offset, len } => {
            for idx in offset..offset + len {
                f(idx);
            }
        }
        ValueSelectionRef::Sparse(indices) => {
            for &idx in indices {
                f(idx);
            }
        }
    }
}

/// Zero-hash adopt fast path for an Arrow byte-array dictionary.
///
/// On the first dictionary seen by an empty encoder, its values are transcoded
/// verbatim into the PLAIN dictionary page (Arrow key `k` → parquet index `k`, a
/// straight sequential copy with no hashing) and the keys are pushed as indices
/// directly. Per-data-page min/max is folded exactly via a per-page generation
/// stamp — each entry contributes to every page it appears in, read once per
/// page rather than once per occurrence. This mirrors the generic numeric adopt
/// path, but keeps the per-page gen stamp (rather than folding per occurrence)
/// because byte comparisons are far more expensive than integer ones.
///
/// Returns `false` (caller falls back to per-occurrence interning) when adopting
/// is unsafe or unprofitable:
/// * a geometry column (needs per-row WKB accumulation),
/// * a dictionary whose PLAIN page would reach the column's dictionary-page size
///   limit — too wide to stay dictionary-encoded, so the encoder is dropped and
///   the column goes straight to PLAIN (the same cardinality bypass as ints),
/// * a prior plain/interned write already populated the encoder, or
/// * a *different* dictionary arrives — its entries seed the dedup table first
///   so the interner can continue without inserting duplicates.
fn encode_dictionary<T, K>(
    encoder: &mut ByteArrayEncoder,
    dictionary_values: &ArrayRef,
    keys: &PrimitiveArray<K>,
    values: &GenericByteArray<T>,
    selection: ValueSelectionRef<'_>,
) -> bool
where
    T: ByteArrayType,
    K: ArrowDictionaryKeyType,
{
    // Geometry columns accumulate per-row WKB statistics; they cannot adopt.
    if encoder.geo_stats_accumulator.is_some() {
        return false;
    }

    let offsets = values.value_offsets();
    let data = values.value_data();

    // Exact byte size this dictionary would occupy as a PLAIN dictionary page:
    // the referenced value bytes plus a 4-byte length prefix per entry (the
    // variable-length analogue of the fixed-width encoder's `card * width`; see
    // `ColumnValueEncoderImpl::maybe_adopt_native_dict`). The referenced span is
    // taken from the offsets (not `data.len()`) so a sliced dictionary array is
    // measured exactly rather than over-counted.
    let referenced_bytes = offsets[values.len()].as_usize() - offsets[0].as_usize();
    let plain_dict_page_size = referenced_bytes + 4 * values.len();

    enum Step {
        /// Empty encoder, dictionary fits: transcode its values verbatim.
        Adopt,
        /// The already-adopted dictionary again: just push keys.
        Append,
        /// A different dictionary: seed dedup from adopted entries, then intern.
        SeedFallback,
        /// A prior intern/legacy write populated the encoder: intern.
        LegacyFallback,
        /// Empty but too wide to adopt: drop dictionary encoding (go PLAIN).
        DropToPlain,
    }

    let step = {
        let dict_encoder = encoder.dict_encoder.as_ref().unwrap();
        match &dict_encoder.adopt {
            ByteDictAdoptState::Legacy => Step::LegacyFallback,
            ByteDictAdoptState::Adopted { identity, .. } => {
                if Arc::ptr_eq(identity, dictionary_values) {
                    Step::Append
                } else {
                    Step::SeedFallback
                }
            }
            ByteDictAdoptState::Fresh => {
                if !dict_encoder.interner.storage().values.is_empty() {
                    // A plain/interned write already populated the encoder; its
                    // dedup table is live, so keep interning.
                    Step::LegacyFallback
                } else if plain_dict_page_size >= encoder.dictionary_page_size_limit {
                    Step::DropToPlain
                } else {
                    Step::Adopt
                }
            }
        }
    };

    let fresh_adopt = matches!(step, Step::Adopt);

    match step {
        Step::LegacyFallback => {
            encoder.dict_encoder.as_mut().unwrap().adopt = ByteDictAdoptState::Legacy;
            return false;
        }
        Step::DropToPlain => {
            // Too wide to stay dictionary-encoded: drop the encoder so the column
            // is written as PLAIN with no interning attempted (the int bypass).
            encoder.dict_encoder = None;
            return false;
        }
        Step::SeedFallback => {
            let dict_encoder = encoder.dict_encoder.as_mut().unwrap();
            let entries = dict_encoder.interner.storage().values.len() as u64;
            dict_encoder.interner.seed_dedup(0..entries);
            dict_encoder.adopt = ByteDictAdoptState::Legacy;
            return false;
        }
        Step::Adopt => {
            let dict_encoder = encoder.dict_encoder.as_mut().unwrap();
            for i in 0..values.len() {
                dict_encoder
                    .interner
                    .push_value(dense_byte_value::<T>(offsets, data, i));
            }
            dict_encoder.adopt = ByteDictAdoptState::Adopted {
                identity: dictionary_values.clone(),
                stat_gen: vec![u32::MAX; values.len()],
            };
        }
        Step::Append => {}
    }

    // On a fresh adoption fold every entry into the bloom filter (a borrow of
    // `encoder` disjoint from the dict encoder mutated above).
    if fresh_adopt {
        if let Some(bloom_filter) = encoder.bloom_filter.as_mut() {
            for i in 0..values.len() {
                bloom_filter.insert(dense_byte_value::<T>(offsets, data, i));
            }
        }
    }

    // Push the selected keys as parquet indices (key == index for an adopted
    // dictionary) and fold exact per-page min/max.
    let collect_min_max = encoder.statistics_enabled != EnabledStatistics::None;
    let page_gen = encoder.dict_encoder.as_ref().unwrap().page_gen;
    let mut batch_min: Option<&[u8]> = None;
    let mut batch_max: Option<&[u8]> = None;
    {
        let DictEncoder {
            indices,
            variable_length_bytes,
            adopt,
            ..
        } = encoder.dict_encoder.as_mut().unwrap();
        let ByteDictAdoptState::Adopted { stat_gen, .. } = adopt else {
            unreachable!("Adopt/Append leaves the encoder in the Adopted state");
        };

        indices.reserve(selection.len());
        for_each_selected(selection, |idx| {
            let key = keys.value(idx).as_usize();
            // Fold this entry into the page min/max once per page it appears in
            // (not once globally): an entry referenced across several data pages
            // must contribute to each page's stats, else later pages prune
            // unsoundly.
            if collect_min_max && stat_gen[key] != page_gen {
                stat_gen[key] = page_gen;
                let value = dense_byte_value::<T>(offsets, data, key);
                if batch_min.is_none_or(|min| min > value) {
                    batch_min = Some(value);
                }
                if batch_max.is_none_or(|max| max < value) {
                    batch_max = Some(value);
                }
            }
            indices.push(key as u64);
            // Per-occurrence logical byte length, from the (small) offsets buffer.
            *variable_length_bytes +=
                (offsets[key + 1].as_usize() - offsets[key].as_usize()) as i64;
        });
    }

    if let Some(min) = batch_min {
        let min = ByteArray::from(min.to_vec());
        if encoder.min_value.as_ref().is_none_or(|m| m > &min) {
            encoder.min_value = Some(min);
        }
    }
    if let Some(max) = batch_max {
        let max = ByteArray::from(max.to_vec());
        if encoder.max_value.as_ref().is_none_or(|m| m < &max) {
            encoder.max_value = Some(max);
        }
    }

    true
}

/// Encodes the provided `values` and `indices` to `encoder`
///
/// This is a free function so it can be used with `downcast_op!`
fn encode<T, I>(values: T, indices: I, encoder: &mut ByteArrayEncoder)
where
    T: ArrayAccessor + Copy,
    T::Item: Copy + Ord + AsRef<[u8]>,
    I: ExactSizeIterator<Item = usize> + Clone,
{
    if encoder.statistics_enabled != EnabledStatistics::None {
        if let Some(accumulator) = encoder.geo_stats_accumulator.as_mut() {
            update_geo_stats_accumulator(accumulator.as_mut(), values, indices.clone());
        } else if let Some((min, max)) = compute_min_max(values, indices.clone()) {
            if encoder.min_value.as_ref().is_none_or(|m| m > &min) {
                encoder.min_value = Some(min);
            }

            if encoder.max_value.as_ref().is_none_or(|m| m < &max) {
                encoder.max_value = Some(max);
            }
        }
    }

    // encode the values into bloom filter if enabled
    if let Some(bloom_filter) = &mut encoder.bloom_filter {
        for idx in indices.clone() {
            bloom_filter.insert(values.value(idx).as_ref());
        }
    }

    match &mut encoder.dict_encoder {
        Some(dict_encoder) => dict_encoder.encode(values, indices),
        None => encoder.fallback.encode(values, indices),
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
/// Dense counterpart of the gather dispatch in
/// [`ByteArrayEncoder::count_values_within_byte_budget_gather`], used for the
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

/// Dispatches a contiguous range write to `encode_dense` for offset-based byte
/// arrays, falling back to `encode` for array types that don't expose a
/// value_offsets/value_data layout (Dictionary, FixedSizeBinary, BinaryView,
/// Utf8View).
fn write_dense(
    values: &dyn Array,
    offset: usize,
    len: usize,
    encoder: &mut ByteArrayEncoder,
) -> Result<()> {
    match values.data_type() {
        DataType::Utf8 => encode_dense(
            values.as_any().downcast_ref::<StringArray>().unwrap(),
            offset,
            len,
            encoder,
        ),
        DataType::LargeUtf8 => encode_dense(
            values.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            offset,
            len,
            encoder,
        ),
        DataType::Binary => encode_dense(
            values.as_any().downcast_ref::<BinaryArray>().unwrap(),
            offset,
            len,
            encoder,
        ),
        DataType::LargeBinary => encode_dense(
            values.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
            offset,
            len,
            encoder,
        ),
        _ => {
            downcast_op!(
                values.data_type(),
                values,
                encode,
                offset..offset + len,
                encoder
            );
            Ok(())
        }
    }
}

/// Encodes a contiguous range from an offset-based byte array
fn encode_dense<T>(
    values: &GenericByteArray<T>,
    offset: usize,
    len: usize,
    encoder: &mut ByteArrayEncoder,
) -> Result<()>
where
    T: ByteArrayType,
{
    if len == 0 {
        return Ok(());
    }

    if encoder.statistics_enabled != EnabledStatistics::None {
        if let Some(accumulator) = encoder.geo_stats_accumulator.as_mut() {
            update_geo_stats_accumulator_values(
                accumulator.as_mut(),
                dense_byte_values(values, offset, len),
            );
        } else if let Some((min, max)) =
            compute_min_max_values(dense_byte_values(values, offset, len))
        {
            if encoder.min_value.as_ref().is_none_or(|m| m > &min) {
                encoder.min_value = Some(min);
            }

            if encoder.max_value.as_ref().is_none_or(|m| m < &max) {
                encoder.max_value = Some(max);
            }
        }
    }

    if let Some(bloom_filter) = &mut encoder.bloom_filter {
        for value in dense_byte_values(values, offset, len) {
            bloom_filter.insert(value);
        }
    }

    match &mut encoder.dict_encoder {
        Some(dict_encoder) => dict_encoder.encode_dense(values, offset, len),
        None => encoder.fallback.encode_dense(values, offset, len),
    }

    Ok(())
}

fn dense_byte_values<T>(
    values: &GenericByteArray<T>,
    offset: usize,
    len: usize,
) -> impl ExactSizeIterator<Item = &[u8]>
where
    T: ByteArrayType,
{
    // Walk the offsets buffer with `windows(2)`: each value's `[start, end]`
    // pair is read from a fixed-length-2 sub-slice, so the per-value offset
    // reads carry no bounds checks and the loop stays tight. Indexing
    // `offsets[idx]` / `offsets[idx + 1]` separately each iteration (over a
    // `Range`) defeats that elision and is measurably slower for small values.
    let data = values.value_data();
    values.value_offsets()[offset..offset + len + 1]
        .windows(2)
        .map(move |w| {
            let start = w[0].as_usize();
            let end = w[1].as_usize();
            &data[start..end]
        })
}

/// Computes the min and max for the provided array and indices
///
/// This is a free function so it can be used with `downcast_op!`
fn compute_min_max<T>(
    array: T,
    mut valid: impl Iterator<Item = usize>,
) -> Option<(ByteArray, ByteArray)>
where
    T: ArrayAccessor,
    T::Item: Copy + Ord + AsRef<[u8]>,
{
    let first_idx = valid.next()?;

    let first_val = array.value(first_idx);
    let mut min = first_val;
    let mut max = first_val;
    for idx in valid {
        let val = array.value(idx);
        min = min.min(val);
        max = max.max(val);
    }
    Some((min.as_ref().to_vec().into(), max.as_ref().to_vec().into()))
}

fn compute_min_max_values<T>(mut values: impl Iterator<Item = T>) -> Option<(ByteArray, ByteArray)>
where
    T: Copy + Ord + AsRef<[u8]>,
{
    let first_val = values.next()?;
    let mut min = first_val;
    let mut max = first_val;
    for val in values {
        min = min.min(val);
        max = max.max(val);
    }
    Some((min.as_ref().to_vec().into(), max.as_ref().to_vec().into()))
}

/// Updates geospatial statistics for the provided array and indices
fn update_geo_stats_accumulator<T>(
    bounder: &mut dyn GeoStatsAccumulator,
    array: T,
    valid: impl Iterator<Item = usize>,
) where
    T: ArrayAccessor,
    T::Item: Copy + Ord + AsRef<[u8]>,
{
    if bounder.is_valid() {
        for idx in valid {
            let val = array.value(idx);
            bounder.update_wkb(val.as_ref());
        }
    }
}

fn update_geo_stats_accumulator_values<T>(
    bounder: &mut dyn GeoStatsAccumulator,
    values: impl Iterator<Item = T>,
) where
    T: AsRef<[u8]>,
{
    if bounder.is_valid() {
        for value in values {
            bounder.update_wkb(value.as_ref());
        }
    }
}
