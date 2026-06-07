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

//! The byte-array encoder engine: dictionary + fallback (PLAIN/DELTA) encoding of
//! `&[u8]` values, with statistics, bloom, and geospatial accumulation.
//!
//! The Arrow bridge borrows contiguous offset-array spans directly and gathers
//! non-contiguous values into bounded `&[u8]` tiles. Both representations cross
//! the same concrete batch seam; source-layout iteration stays private here.

use crate::basic::{ConvertedType, Encoding, LogicalType};
use crate::bloom_filter::Sbbf;
use crate::column::value::{Sink, ValueCursor, gather_run_groups_tiled, gather_tiled};
use crate::column::writer::encoder::MinMaxStrategy;
use crate::column::writer::{compare_greater_byte_array_decimals, compare_greater_f16};
use crate::data_type::private::byte_array_length;
use crate::data_type::{AsBytes, ByteArray, ByteArrayType, DataType, Int32Type};
use crate::encodings::encoding::ColumnEncode;
use crate::encodings::encoding::{DeltaBitPackEncoder, Encoder, RunIndexBuffer};
use crate::encodings::rle::RleEncoder;
use crate::errors::{ParquetError, Result};
use crate::geospatial::accumulator::GeoStatsAccumulator;
use crate::schema::types::{ColumnDescPtr, ColumnDescriptor};
use crate::util::bit_util::num_required_bits;
use crate::util::interner::{Interner, Storage};
use bytes::Bytes;

/// Observation performed alongside byte-array encoding.
///
/// A named generic protocol lets the target combine observation with encoding
/// without nesting another captured closure at the call site.
trait ByteValueObserver<'a> {
    #[cfg(feature = "arrow")]
    fn observe(&mut self, value: &'a [u8]);

    fn observe_batch<'batch>(&mut self, values: FlatByteBatch<'batch, 'a>) -> Result<()>
    where
        'a: 'batch;
}

impl<'a> ByteValueObserver<'a> for () {
    #[cfg(feature = "arrow")]
    #[inline(always)]
    fn observe(&mut self, _value: &'a [u8]) {}

    #[inline(always)]
    fn observe_batch<'batch>(&mut self, _values: FlatByteBatch<'batch, 'a>) -> Result<()>
    where
        'a: 'batch,
    {
        Ok(())
    }
}

/// Internal for-each operation. Named consumers let the layout loop inline the
/// operation body without LLVM outlining an anonymous closure per value.
trait ByteValueConsumer<'source> {
    fn consume(&mut self, value: &'source [u8]) -> Result<()>;
}

struct PlainSize {
    payload: usize,
}

impl ByteValueConsumer<'_> for PlainSize {
    #[inline(always)]
    fn consume(&mut self, value: &[u8]) -> Result<()> {
        byte_array_length(value.len())?;
        self.payload = self.payload.saturating_add(value.len());
        Ok(())
    }
}

struct PlainEncode<'state> {
    buffer: &'state mut Vec<u8>,
}

impl ByteValueConsumer<'_> for PlainEncode<'_> {
    #[inline(always)]
    fn consume(&mut self, value: &[u8]) -> Result<()> {
        self.buffer
            .extend_from_slice((value.len() as u32).as_bytes());
        self.buffer.extend_from_slice(value);
        Ok(())
    }
}

struct DeltaLengthEncode<'state> {
    buffer: &'state mut Vec<u8>,
    lengths: &'state mut DeltaBitPackEncoder<Int32Type>,
    total: i64,
}

impl ByteValueConsumer<'_> for DeltaLengthEncode<'_> {
    #[inline(always)]
    fn consume(&mut self, value: &[u8]) -> Result<()> {
        self.lengths.put_i64(value.len() as i64)?;
        self.buffer.extend_from_slice(value);
        self.total += value.len() as i64;
        Ok(())
    }
}

struct DeltaEncode<'state> {
    buffer: &'state mut Vec<u8>,
    last_value: &'state mut Vec<u8>,
    prefix_lengths: &'state mut DeltaBitPackEncoder<Int32Type>,
    suffix_lengths: &'state mut DeltaBitPackEncoder<Int32Type>,
    total: i64,
}

impl ByteValueConsumer<'_> for DeltaEncode<'_> {
    #[inline(always)]
    fn consume(&mut self, value: &[u8]) -> Result<()> {
        let mut prefix_length = 0;
        while prefix_length < self.last_value.len()
            && prefix_length < value.len()
            && self.last_value[prefix_length] == value[prefix_length]
        {
            prefix_length += 1;
        }

        let suffix_length = value.len() - prefix_length;
        self.last_value.clear();
        self.last_value.extend_from_slice(value);
        self.buffer.extend_from_slice(&value[prefix_length..]);
        self.prefix_lengths.put_i64(prefix_length as i64)?;
        self.suffix_lengths.put_i64(suffix_length as i64)?;
        self.total += value.len() as i64;
        Ok(())
    }
}

#[cfg(feature = "arrow")]
struct ObserveAndConsume<'state, O, C> {
    observer: &'state mut O,
    consumer: &'state mut C,
}

#[cfg(feature = "arrow")]
impl<'source, O, C> ByteValueConsumer<'source> for ObserveAndConsume<'_, O, C>
where
    O: ByteValueObserver<'source>,
    C: ByteValueConsumer<'source>,
{
    #[inline(always)]
    fn consume(&mut self, value: &'source [u8]) -> Result<()> {
        self.observer.observe(value);
        self.consumer.consume(value)
    }
}

struct DictionaryEncode<'state> {
    encoder: &'state mut ByteArrayDictEncoder,
    total: i64,
}

impl<'source> ByteValueConsumer<'source> for DictionaryEncode<'_> {
    #[inline(always)]
    fn consume(&mut self, value: &'source [u8]) -> Result<()> {
        self.encoder.encode_value(value)?;
        self.total += value.len() as i64;
        Ok(())
    }
}

/// How byte-array min/max statistics are ordered. Determined from the column
/// descriptor. `Unsigned` (plain lexicographic) is the case for every Arrow byte
/// column (Utf8/Binary/views); the signed variants only arise for low-level
/// `BYTE_ARRAY` columns carrying a `DECIMAL`/`FLOAT16` logical type.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ByteMinMaxOrder {
    Unsigned,
    Decimal,
    Float16,
}

impl ByteMinMaxOrder {
    pub(crate) fn from_descr(descr: &ColumnDescriptor) -> Self {
        if matches!(descr.logical_type_ref(), Some(LogicalType::Decimal(_)))
            || descr.converted_type() == ConvertedType::DECIMAL
        {
            ByteMinMaxOrder::Decimal
        } else if matches!(descr.logical_type_ref(), Some(LogicalType::Float16)) {
            ByteMinMaxOrder::Float16
        } else {
            ByteMinMaxOrder::Unsigned
        }
    }

    /// `true` when `a > b` under this column's byte order.
    #[inline]
    pub(crate) fn greater(self, a: &[u8], b: &[u8]) -> bool {
        match self {
            ByteMinMaxOrder::Unsigned => a > b,
            ByteMinMaxOrder::Decimal => compare_greater_byte_array_decimals(a, b),
            ByteMinMaxOrder::Float16 => compare_greater_f16(a, b),
        }
    }
}

/// [`MinMaxStrategy`] marker for variable-width byte arrays. The sized `&[u8]`
/// handles retain borrowed payloads as min/max without per-value allocation; the
/// materialized column statistic is a [`ByteArray`]. Ordering comes from the
/// column's [`ByteMinMaxOrder`].
pub(crate) struct ByteMinMax;

impl<'v> MinMaxStrategy<'v> for ByteMinMax {
    type Elem = &'v [u8];
    type Owned = ByteArray;
    type Ctx = ByteMinMaxOrder;

    #[inline(always)]
    fn ctx(descr: &ColumnDescriptor) -> ByteMinMaxOrder {
        ByteMinMaxOrder::from_descr(descr)
    }
    #[inline(always)]
    fn greater(order: ByteMinMaxOrder, a: &[u8], b: &[u8]) -> bool {
        order.greater(a, b)
    }
    #[inline(always)]
    fn to_owned(v: &[u8]) -> ByteArray {
        ByteArray::from(v.to_vec())
    }
}

/// The byte-array value encoder: the contiguous-page dictionary, or one of the
/// non-dictionary fallback encodings (PLAIN / DELTA_LENGTH_BYTE_ARRAY /
/// DELTA_BYTE_ARRAY). The variable-width analog of the numeric/FLBA flat
/// `*ColumnEncoder` enums — `Dictionary` is a peer variant that spans the column
/// chunk and emits a dictionary page.
/// Value count and variable-length byte totals are tracked by the enclosing
/// column encoder.
///
/// Note: DeltaBitPackEncoder is boxed as it is rather large.
#[doc(hidden)]
pub enum ByteArrayColumnEncoder {
    Dictionary(ByteArrayDictEncoder),
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

impl ByteArrayColumnEncoder {
    /// Build a non-dictionary fallback encoder for `encoding`.
    fn from_encoding(encoding: Encoding, _descr: &ColumnDescPtr) -> Result<Self> {
        Ok(match encoding {
            Encoding::PLAIN => Self::Plain { buffer: vec![] },
            Encoding::DELTA_LENGTH_BYTE_ARRAY => Self::DeltaLength {
                buffer: vec![],
                lengths: Box::new(DeltaBitPackEncoder::new()),
            },
            Encoding::DELTA_BYTE_ARRAY => Self::Delta {
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
        })
    }

    /// Encode one byte-value batch into the fallback encoder, returning the
    /// total variable-length byte count (sum of value lengths).
    ///
    /// This thin dispatcher inlines through the sink adapter; the selected
    /// target helper below is the one out-of-line boundary for the batch.
    #[inline(always)]
    fn encode_values<'batch, 'source: 'batch>(
        &mut self,
        values: FlatByteBatch<'batch, 'source>,
    ) -> Result<i64> {
        self.encode_values_observed(values, &mut ())
    }

    /// Encode and observe one batch. Native spans fuse both operations in one
    /// traversal; bounded gathered batches are replayed as two tight loops.
    #[inline(always)]
    fn encode_values_observed<'batch, 'source: 'batch, O>(
        &mut self,
        values: FlatByteBatch<'batch, 'source>,
        observer: &mut O,
    ) -> Result<i64>
    where
        O: ByteValueObserver<'source>,
    {
        match self {
            Self::Dictionary(_) => {
                unreachable!("dictionary variant is not routed through the fallback encoder")
            }
            Self::Plain { buffer } => encode_plain_values(buffer, values, observer),
            Self::DeltaLength { buffer, lengths } => {
                encode_delta_length_values(buffer, lengths, values, observer)
            }
            Self::Delta {
                buffer,
                last_value,
                prefix_lengths,
                suffix_lengths,
            } => encode_delta_values(
                buffer,
                last_value,
                prefix_lengths,
                suffix_lengths,
                values,
                observer,
            ),
        }
    }
}

/// Each fallback target owns one out-of-line batch operation. The native-batch
/// representation match and its value loop inline into this boundary, avoiding
/// both an oversized all-target commit and a scalar callback.
#[inline(always)]
fn prepare_plain_values<'batch, 'source: 'batch>(
    buffer: &mut Vec<u8>,
    values: FlatByteBatch<'batch, 'source>,
) -> Result<i64> {
    // Dense small-offset sources provide an exact O(1) reservation and
    // intrinsically valid u32 lengths. Other sources retain the full
    // validation pass before the buffer is mutated.
    let total = if let Some((payload, encoded)) = values.exact_plain_size() {
        buffer.reserve(encoded);
        payload
    } else {
        let mut size = PlainSize { payload: 0 };
        values.try_for_each(&mut size)?;
        buffer.reserve(size.payload.saturating_add(values.len().saturating_mul(4)));
        size.payload
    };
    Ok(total as i64)
}

#[inline(never)]
fn encode_plain_values<'batch, 'source: 'batch, O>(
    buffer: &mut Vec<u8>,
    values: FlatByteBatch<'batch, 'source>,
    observer: &mut O,
) -> Result<i64>
where
    O: ByteValueObserver<'source>,
{
    let total = prepare_plain_values(buffer, values)?;
    let mut encode = PlainEncode { buffer };
    values.try_for_each_observed(observer, &mut encode)?;
    Ok(total)
}

#[inline(never)]
fn encode_delta_length_values<'batch, 'source: 'batch, O>(
    buffer: &mut Vec<u8>,
    lengths: &mut DeltaBitPackEncoder<Int32Type>,
    values: FlatByteBatch<'batch, 'source>,
    observer: &mut O,
) -> Result<i64>
where
    O: ByteValueObserver<'source>,
{
    let mut encode = DeltaLengthEncode {
        buffer,
        lengths,
        total: 0,
    };
    values.try_for_each_observed(observer, &mut encode)?;
    Ok(encode.total)
}

#[inline(never)]
fn encode_delta_values<'batch, 'source: 'batch, O>(
    buffer: &mut Vec<u8>,
    last_value: &mut Vec<u8>,
    prefix_lengths: &mut DeltaBitPackEncoder<Int32Type>,
    suffix_lengths: &mut DeltaBitPackEncoder<Int32Type>,
    values: FlatByteBatch<'batch, 'source>,
    observer: &mut O,
) -> Result<i64>
where
    O: ByteValueObserver<'source>,
{
    let mut encode = DeltaEncode {
        buffer,
        last_value,
        prefix_lengths,
        suffix_lengths,
        total: 0,
    };
    values.try_for_each_observed(observer, &mut encode)?;
    Ok(encode.total)
}

impl Encoder<ByteArrayType> for ByteArrayColumnEncoder {
    fn put(&mut self, values: &[ByteArray]) -> Result<()> {
        gather_tiled::<BYTE_ARRAY_BATCH_VALUES, _, _, _>(values, |values| {
            self.encode_values(FlatByteBatch::Gathered(values))
                .map(|_| ())
        })
    }

    fn encoding(&self) -> Encoding {
        match self {
            Self::Dictionary(_) => {
                unreachable!("dictionary variant is not routed through Encoder")
            }
            Self::Plain { .. } => Encoding::PLAIN,
            Self::DeltaLength { .. } => Encoding::DELTA_LENGTH_BYTE_ARRAY,
            Self::Delta { .. } => Encoding::DELTA_BYTE_ARRAY,
        }
    }

    /// Returns an estimate of the data page size in bytes
    ///
    /// This includes:
    /// <already_written_encoded_byte_size> + <estimated_encoded_size_of_unflushed_bytes>
    fn estimated_data_encoded_size(&self) -> usize {
        match self {
            Self::Dictionary(_) => {
                unreachable!("dictionary variant is not routed through Encoder")
            }
            Self::Plain { buffer, .. } => buffer.len(),
            Self::DeltaLength { buffer, lengths } => {
                buffer.len() + lengths.estimated_data_encoded_size()
            }
            Self::Delta {
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

    fn estimated_memory_size(&self) -> usize {
        match self {
            Self::Dictionary(_) => {
                unreachable!("dictionary variant is not routed through Encoder")
            }
            Self::Plain { buffer } => buffer.capacity(),
            Self::DeltaLength { buffer, lengths } => {
                buffer.capacity() + lengths.estimated_memory_size()
            }
            Self::Delta {
                buffer,
                last_value,
                prefix_lengths,
                suffix_lengths,
            } => {
                buffer.capacity()
                    + last_value.capacity()
                    + prefix_lengths.estimated_memory_size()
                    + suffix_lengths.estimated_memory_size()
            }
        }
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        let buf = match self {
            Self::Dictionary(_) => {
                unreachable!("dictionary variant is not routed through Encoder")
            }
            Self::Plain { buffer } => std::mem::take(buffer),
            Self::DeltaLength { buffer, lengths } => {
                let lengths = lengths.flush_buffer()?;

                let mut out = Vec::with_capacity(lengths.len() + buffer.len());
                out.extend_from_slice(&lengths);
                out.extend_from_slice(buffer);
                buffer.clear();
                out
            }
            Self::Delta {
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
                out
            }
        };
        Ok(buf.into())
    }
}

impl<D: DataType<T = ByteArray>> ColumnEncode<D> for ByteArrayColumnEncoder {
    fn new_column_encoder(
        dict_supported: bool,
        fallback_encoding: Encoding,
        descr: &ColumnDescPtr,
    ) -> Result<Self> {
        if dict_supported {
            // Eagerly validate the fallback encoding and initialize the dictionary.
            Self::from_encoding(fallback_encoding, descr)?;
            Ok(Self::Dictionary(ByteArrayDictEncoder::default()))
        } else {
            Self::from_encoding(fallback_encoding, descr)
        }
    }

    fn is_dictionary(&self) -> bool {
        matches!(self, Self::Dictionary(_))
    }

    fn take_dict_page(
        &mut self,
        fallback_encoding: Encoding,
        descr: &ColumnDescPtr,
    ) -> Result<Option<(Bytes, usize, bool)>> {
        if !<Self as ColumnEncode<D>>::is_dictionary(self) {
            return Ok(None);
        }
        let fallback = Self::from_encoding(fallback_encoding, descr)?;
        let Self::Dictionary(dict) = std::mem::replace(self, fallback) else {
            unreachable!("is_dictionary checked above");
        };
        let num_values = dict.num_entries();
        let is_sorted = dict.is_sorted();
        let buf = dict.write_dict()?;
        Ok(Some((buf, num_values, is_sorted)))
    }

    fn flush_data_page(&mut self) -> Result<(Bytes, Encoding)> {
        match self {
            Self::Dictionary(dict) => Ok((dict.write_indices()?, Encoding::RLE_DICTIONARY)),
            other => Ok((
                <Self as Encoder<ByteArrayType>>::flush_buffer(other)?,
                <Self as Encoder<ByteArrayType>>::encoding(other),
            )),
        }
    }

    fn dict_page_size(&self) -> Option<usize> {
        match self {
            Self::Dictionary(dict) => Some(dict.dict_encoded_size()),
            _ => None,
        }
    }

    fn data_page_size(&self) -> usize {
        match self {
            Self::Dictionary(dict) => dict.estimated_data_encoded_size(),
            other => <Self as Encoder<ByteArrayType>>::estimated_data_encoded_size(other),
        }
    }

    fn memory_size(&self) -> usize {
        match self {
            Self::Dictionary(dict) => dict.estimated_memory_size(),
            other => <Self as Encoder<ByteArrayType>>::estimated_memory_size(other),
        }
    }
}

/// [`Storage`] for the [`Interner`] used by [`ByteArrayDictEncoder`].
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
        let len = u32::try_from(value.len()).expect("byte-array length validated before interning");

        self.page.reserve(4 + value.len());
        self.page.extend_from_slice(len.as_bytes());

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
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct ByteArrayDictEncoder {
    interner: Interner<ByteArrayStorage>,
    /// The buffered data-page indices (dense and run-buffered), shared with the
    /// generic dictionary encoder.
    indices: RunIndexBuffer,
}

impl ByteArrayDictEncoder {
    pub(crate) fn reserve(&mut self, len: usize) {
        self.indices.reserve(len);
    }

    // Keep interning inside the selected dictionary batch loop. Leaving this
    // as a scalar call is disproportionately expensive for gathered input.
    #[inline(always)]
    fn encode_value(&mut self, value: &[u8]) -> Result<()> {
        byte_array_length(value.len())?;
        let interned = self.interner.intern(value);
        self.indices.push(interned);
        Ok(())
    }

    /// Intern `value` and append it as a run of `count` repetitions — the
    /// run-end fast path, recording one `(index, count)` entry instead of
    /// `count` indices.
    #[inline]
    fn encode_value_run(&mut self, value: &[u8], count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        byte_array_length(value.len())?;
        let interned = self.interner.intern(value);
        self.indices.push_run(interned, count);
        Ok(())
    }

    /// Observe, then intern, one byte-value batch.
    ///
    /// Dictionary hashing carries enough state that fusing it with observation
    /// expands the hot loop and increases register pressure. Replaying the
    /// borrowed batch keeps both loops tight without changing the batch seam.
    #[inline(never)]
    fn encode_values_observed<'batch, 'source: 'batch>(
        &mut self,
        values: FlatByteBatch<'batch, 'source>,
        observer: &mut ByteSinkObserver<'source, '_>,
    ) -> Result<i64> {
        observer.observe_batch(values)?;

        let mut encode = DictionaryEncode {
            encoder: self,
            total: 0,
        };
        values.try_for_each(&mut encode)?;
        Ok(encode.total)
    }

    fn bit_width(&self) -> u8 {
        let length = self.interner.storage().values.len();
        num_required_bits(length.saturating_sub(1) as u64)
    }

    fn num_entries(&self) -> usize {
        self.interner.storage().values.len()
    }

    fn is_sorted(&self) -> bool {
        false
    }

    fn dict_encoded_size(&self) -> usize {
        self.interner.storage().page.len()
    }

    fn estimated_data_encoded_size(&self) -> usize {
        let bit_width = self.bit_width();
        1 + RleEncoder::max_buffer_size(bit_width, self.indices.num_values())
    }

    fn estimated_memory_size(&self) -> usize {
        self.interner.estimated_memory_size() + self.indices.estimated_memory_size()
    }

    fn write_dict(self) -> Result<Bytes> {
        // The dictionary page IS the interner's contiguous page: move it out
        // rather than cloning up to a full dictionary-page-limit of bytes.
        Ok(Bytes::from(self.interner.into_inner().page))
    }

    fn write_indices(&mut self) -> Result<Bytes> {
        self.indices.write_indices(self.bit_width())
    }
}

/// One flat byte-value batch handed to the column encoder.
///
/// Dense offset arrays retain their native borrowed representation. Everything
/// else is normalized to a bounded slice of borrowed values before crossing the
/// sink boundary.
#[derive(Clone, Copy)]
pub(crate) enum FlatByteBatch<'batch, 'source> {
    #[cfg(feature = "arrow")]
    Offset32 {
        offsets: &'source [i32],
        data: &'source [u8],
    },
    #[cfg(feature = "arrow")]
    Offset64 {
        offsets: &'source [i64],
        data: &'source [u8],
    },
    Gathered(&'batch [&'source [u8]]),
}

impl<'batch, 'source: 'batch> FlatByteBatch<'batch, 'source> {
    #[inline(always)]
    pub(crate) fn len(self) -> usize {
        match self {
            #[cfg(feature = "arrow")]
            Self::Offset32 { offsets, .. } => offsets.len().saturating_sub(1),
            #[cfg(feature = "arrow")]
            Self::Offset64 { offsets, .. } => offsets.len().saturating_sub(1),
            Self::Gathered(values) => values.len(),
        }
    }

    /// Returns `(payload_bytes, encoded_bytes)` when it is both O(1) to compute
    /// and the representation guarantees every value length fits a Parquet
    /// `u32` prefix.
    #[inline(always)]
    pub(crate) fn exact_plain_size(self) -> Option<(usize, usize)> {
        match self {
            #[cfg(feature = "arrow")]
            Self::Offset32 { offsets, .. } => {
                let payload = (*offsets.last()? - *offsets.first()?) as usize;
                Some((
                    payload,
                    payload.saturating_add(self.len().saturating_mul(std::mem::size_of::<u32>())),
                ))
            }
            _ => None,
        }
    }

    /// Visit logical values with one representation match per batch. The named
    /// consumer is forced into the selected loop; the encoder operation around
    /// this dispatcher is the sole out-of-line boundary.
    #[inline(always)]
    fn try_for_each<C>(self, consumer: &mut C) -> Result<()>
    where
        C: ByteValueConsumer<'source>,
    {
        match self {
            #[cfg(feature = "arrow")]
            Self::Offset32 { offsets, data } => walk_offset32(offsets, data, consumer),
            #[cfg(feature = "arrow")]
            Self::Offset64 { offsets, data } => walk_offset64(offsets, data, consumer),
            Self::Gathered(values) => walk_gathered(values, consumer),
        }
    }

    /// Observe and consume one batch. A gathered tile is cheap to replay, so
    /// its observation and encoding loops stay independently optimizable.
    /// Native offset spans may be arbitrarily large and retain one fused walk.
    #[inline(always)]
    fn try_for_each_observed<C, O>(self, observer: &mut O, consumer: &mut C) -> Result<()>
    where
        C: ByteValueConsumer<'source>,
        O: ByteValueObserver<'source>,
    {
        match self {
            Self::Gathered(values) => {
                observer.observe_batch(Self::Gathered(values))?;
                walk_gathered(values, consumer)
            }
            #[cfg(feature = "arrow")]
            values => values.try_for_each(&mut ObserveAndConsume { observer, consumer }),
        }
    }
}

#[cfg(feature = "arrow")]
#[inline(always)]
fn walk_offset32<'source, C>(
    offsets: &'source [i32],
    data: &'source [u8],
    consumer: &mut C,
) -> Result<()>
where
    C: ByteValueConsumer<'source>,
{
    for window in offsets.windows(2) {
        consumer.consume(&data[window[0] as usize..window[1] as usize])?;
    }
    Ok(())
}

#[cfg(feature = "arrow")]
#[inline(always)]
fn walk_offset64<'source, C>(
    offsets: &'source [i64],
    data: &'source [u8],
    consumer: &mut C,
) -> Result<()>
where
    C: ByteValueConsumer<'source>,
{
    for window in offsets.windows(2) {
        consumer.consume(&data[window[0] as usize..window[1] as usize])?;
    }
    Ok(())
}

#[inline(always)]
fn walk_gathered<'batch, 'source: 'batch, C>(
    values: &'batch [&'source [u8]],
    consumer: &mut C,
) -> Result<()>
where
    C: ByteValueConsumer<'source>,
{
    for &value in values {
        consumer.consume(value)?;
    }
    Ok(())
}

/// One bounded grouped byte-value tile handed to the dictionary encoder.
#[derive(Clone, Copy)]
pub(crate) struct GroupedByteBatch<'batch, 'source> {
    values: &'batch [&'source [u8]],
    counts: &'batch [usize],
}

/// Number of non-contiguous byte values gathered per sink handoff.
pub(crate) const BYTE_ARRAY_BATCH_VALUES: usize = 64;

/// A byte-value producer. The default path gathers non-contiguous values into
/// bounded reference batches; Arrow offset arrays override this to lend
/// contiguous ranges directly.
pub(crate) trait ByteBatchSource<'source>: ValueCursor<&'source [u8]> {
    #[inline(always)]
    fn drive_flat(self, sink: &mut ByteSink<'source, '_>) -> Result<()> {
        gather_tiled::<BYTE_ARRAY_BATCH_VALUES, _, _, _>(self, |values| {
            sink.commit(FlatByteBatch::Gathered(values))
        })
    }

    #[inline(always)]
    fn drive_run_groups(self, sink: &mut ByteSink<'source, '_>) -> Result<()> {
        gather_run_groups_tiled::<BYTE_ARRAY_BATCH_VALUES, _, _>(self, |values, counts| {
            sink.commit(GroupedByteBatch { values, counts })
        })
    }
}

/// Destination selected once for a complete byte-value write.
pub(crate) enum ByteSinkTarget<'encoder> {
    Dict(&'encoder mut ByteArrayDictEncoder),
    Fallback(&'encoder mut ByteArrayColumnEncoder),
}

/// Write-scoped byte state retained across native spans and gathered tiles.
///
/// Borrowed min/max values are copied into the column statistics only after the
/// complete source succeeds, avoiding per-tile allocations on sparse input.
pub(crate) struct ByteSink<'source, 'encoder> {
    pub(crate) collect_stats: bool,
    pub(crate) order: ByteMinMaxOrder,
    pub(crate) min: Option<&'source [u8]>,
    pub(crate) max: Option<&'source [u8]>,
    pub(crate) bytes_written: i64,
    pub(crate) accumulator: Option<&'encoder mut Box<dyn GeoStatsAccumulator>>,
    pub(crate) bloom: Option<&'encoder mut Sbbf>,
    pub(crate) target: ByteSinkTarget<'encoder>,
}

struct ByteSinkObserver<'source, 'state> {
    collect_stats: bool,
    order: ByteMinMaxOrder,
    min: &'state mut Option<&'source [u8]>,
    max: &'state mut Option<&'source [u8]>,
    accumulator: Option<&'state mut (dyn GeoStatsAccumulator + 'static)>,
    bloom: Option<&'state mut Sbbf>,
}

struct ObserveMinMax<'source, 'state> {
    order: ByteMinMaxOrder,
    min: &'state mut Option<&'source [u8]>,
    max: &'state mut Option<&'source [u8]>,
}

impl<'source> ByteValueConsumer<'source> for ObserveMinMax<'source, '_> {
    #[inline(always)]
    fn consume(&mut self, value: &'source [u8]) -> Result<()> {
        <ByteMinMax as MinMaxStrategy<'_>>::observe(self.order, value, self.min, self.max);
        Ok(())
    }
}

struct ObserveGeo<'state> {
    accumulator: &'state mut (dyn GeoStatsAccumulator + 'static),
}

impl<'source> ByteValueConsumer<'source> for ObserveGeo<'_> {
    #[inline(always)]
    fn consume(&mut self, value: &'source [u8]) -> Result<()> {
        if self.accumulator.is_valid() {
            self.accumulator.update_wkb(value);
        }
        Ok(())
    }
}

struct ObserveBloom<'state> {
    bloom: &'state mut Sbbf,
}

impl<'source> ByteValueConsumer<'source> for ObserveBloom<'_> {
    #[inline(always)]
    fn consume(&mut self, value: &'source [u8]) -> Result<()> {
        self.bloom.insert(value);
        Ok(())
    }
}

impl<'source> ByteSinkObserver<'source, '_> {
    #[inline(always)]
    fn observe_batch<'batch>(&mut self, values: FlatByteBatch<'batch, 'source>) -> Result<()>
    where
        'source: 'batch,
    {
        if self.collect_stats {
            if let Some(accumulator) = self.accumulator.as_deref_mut() {
                values.try_for_each(&mut ObserveGeo { accumulator })?;
            } else {
                values.try_for_each(&mut ObserveMinMax {
                    order: self.order,
                    min: self.min,
                    max: self.max,
                })?;
            }
        }
        if let Some(bloom) = self.bloom.as_deref_mut() {
            values.try_for_each(&mut ObserveBloom { bloom })?;
        }
        Ok(())
    }
}

impl<'source> ByteValueObserver<'source> for ByteSinkObserver<'source, '_> {
    #[cfg(feature = "arrow")]
    #[inline(always)]
    fn observe(&mut self, value: &'source [u8]) {
        if self.collect_stats {
            if let Some(accumulator) = self.accumulator.as_mut() {
                if accumulator.is_valid() {
                    accumulator.update_wkb(value);
                }
            } else {
                <ByteMinMax as MinMaxStrategy<'_>>::observe(self.order, value, self.min, self.max);
            }
        }
        if let Some(bloom) = self.bloom.as_mut() {
            bloom.insert(value);
        }
    }

    #[inline(always)]
    fn observe_batch<'batch>(&mut self, values: FlatByteBatch<'batch, 'source>) -> Result<()>
    where
        'source: 'batch,
    {
        ByteSinkObserver::observe_batch(self, values)
    }
}

impl<'batch, 'source: 'batch> Sink<FlatByteBatch<'batch, 'source>> for ByteSink<'source, '_> {
    #[inline(always)]
    fn commit(&mut self, values: FlatByteBatch<'batch, 'source>) -> Result<()> {
        let Self {
            collect_stats,
            order,
            min,
            max,
            bytes_written,
            accumulator,
            bloom,
            target,
        } = self;
        let mut observer = ByteSinkObserver {
            collect_stats: *collect_stats,
            order: *order,
            min,
            max,
            accumulator: accumulator
                .as_deref_mut()
                .map(|accumulator| accumulator.as_mut()),
            bloom: bloom.as_deref_mut(),
        };
        let written = match target {
            ByteSinkTarget::Dict(encoder) => encoder.encode_values_observed(values, &mut observer),
            ByteSinkTarget::Fallback(encoder) => {
                encoder.encode_values_observed(values, &mut observer)
            }
        }?;
        *bytes_written += written;
        Ok(())
    }
}

impl<'source> ByteSink<'source, '_> {
    #[inline(never)]
    fn commit_grouped(&mut self, values: &[&'source [u8]], counts: &[usize]) -> Result<()> {
        debug_assert!(self.accumulator.is_none());
        if self.collect_stats {
            let order = self.order;
            for &value in values {
                <ByteMinMax as MinMaxStrategy<'_>>::observe(
                    order,
                    value,
                    &mut self.min,
                    &mut self.max,
                );
            }
        }
        if let Some(bloom) = self.bloom.as_deref_mut() {
            for &value in values {
                bloom.insert(value);
            }
        }

        let ByteSinkTarget::Dict(encoder) = &mut self.target else {
            unreachable!("grouped byte batch emitted without a dictionary encoder")
        };
        for (&value, &count) in values.iter().zip(counts) {
            encoder.encode_value_run(value, count)?;
            self.bytes_written = self
                .bytes_written
                .saturating_add((value.len() as i64).saturating_mul(count as i64));
        }
        Ok(())
    }
}

impl<'batch, 'source: 'batch> Sink<GroupedByteBatch<'batch, 'source>> for ByteSink<'source, '_> {
    #[inline(always)]
    fn commit(&mut self, values: GroupedByteBatch<'batch, 'source>) -> Result<()> {
        self.commit_grouped(values.values, values.counts)
    }
}

/// Independently allocated byte values supplied by the low-level slice API are
/// already a dense source descriptor.
impl<'a, T: AsBytes> ValueCursor<&'a [u8]> for &'a [T] {
    #[inline]
    fn len(self) -> usize {
        <[T]>::len(self)
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
        for value in self {
            f(value.as_bytes())?;
        }
        Ok(())
    }
}

impl<'a, T: AsBytes> ByteBatchSource<'a> for &'a [T] {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_memory_size_accounts_for_retained_allocations() {
        let plain = ByteArrayColumnEncoder::Plain {
            buffer: Vec::with_capacity(101),
        };
        let ByteArrayColumnEncoder::Plain { buffer } = &plain else {
            unreachable!()
        };
        assert_eq!(
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_memory_size(&plain),
            buffer.capacity()
        );

        let delta_length = ByteArrayColumnEncoder::DeltaLength {
            buffer: Vec::with_capacity(103),
            lengths: Box::new(DeltaBitPackEncoder::new()),
        };
        let ByteArrayColumnEncoder::DeltaLength { buffer, lengths } = &delta_length else {
            unreachable!()
        };
        assert_eq!(
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_memory_size(
                &delta_length
            ),
            buffer.capacity() + lengths.estimated_memory_size()
        );

        let mut delta = ByteArrayColumnEncoder::Delta {
            buffer: Vec::with_capacity(107),
            last_value: Vec::with_capacity(109),
            prefix_lengths: Box::new(DeltaBitPackEncoder::new()),
            suffix_lengths: Box::new(DeltaBitPackEncoder::new()),
        };
        let ByteArrayColumnEncoder::Delta {
            buffer,
            last_value,
            prefix_lengths,
            suffix_lengths,
        } = &delta
        else {
            unreachable!()
        };
        assert_eq!(
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_memory_size(&delta),
            buffer.capacity()
                + last_value.capacity()
                + prefix_lengths.estimated_memory_size()
                + suffix_lengths.estimated_memory_size()
        );

        let values = [
            ByteArray::from("common-prefix-a"),
            ByteArray::from("common-prefix-b"),
        ];
        delta.put(&values).unwrap();
        let encoded =
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_data_encoded_size(&delta);
        assert!(
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_memory_size(&delta)
                > encoded
        );

        <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::flush_buffer(&mut delta).unwrap();
        assert_eq!(
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_data_encoded_size(&delta),
            0
        );
        assert!(
            <ByteArrayColumnEncoder as Encoder<ByteArrayType>>::estimated_memory_size(&delta) > 0
        );
    }
}
