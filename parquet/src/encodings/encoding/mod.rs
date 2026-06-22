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

//! Contains all supported encoders for Parquet.

use std::{cmp, marker::PhantomData};

use crate::basic::*;
#[cfg(feature = "arrow")]
use crate::column::writer::ValueSelection;
use crate::data_type::private::ParquetValueType;
use crate::data_type::*;
use crate::encodings::rle::RleEncoder;
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;
use crate::util::bit_util::{BitWriter, num_required_bits};

#[cfg(feature = "arrow")]
use crate::util::bit_util::get_bit;
#[cfg(feature = "arrow")]
use arrow_buffer::bit_chunk_iterator::UnalignedBitChunk;
#[cfg(feature = "arrow")]
use arrow_buffer::{ArrowNativeType, NullBuffer, i256};
use byte_stream_split_encoder::{ByteStreamSplitEncoder, VariableWidthByteStreamSplitEncoder};
use bytes::Bytes;
pub use dict_encoder::DictEncoder;

mod byte_stream_split_encoder;
mod dict_encoder;

// ----------------------------------------------------------------------
// Encoders

/// An Parquet encoder for the data type `T`.
///
/// Currently this allocates internal buffers for the encoded values. After done putting
/// values, caller should call `flush_buffer()` to get an immutable buffer pointer.
pub trait Encoder<T: DataType>: Send {
    /// Encodes data from `values`.
    fn put(&mut self, values: &[T::T]) -> Result<()>;

    /// Encodes data from `values`, which contains spaces for null values, that is
    /// identified by `valid_bits`.
    ///
    /// Returns the number of non-null values encoded.
    #[cfg(test)]
    fn put_spaced(&mut self, values: &[T::T], valid_bits: &[u8]) -> Result<usize> {
        let num_values = values.len();
        let mut buffer = Vec::with_capacity(num_values);
        // TODO: this is pretty inefficient. Revisit in future.
        for (i, item) in values.iter().enumerate().take(num_values) {
            if crate::util::bit_util::get_bit(valid_bits, i) {
                buffer.push(item.clone());
            }
        }
        self.put(&buffer[..])?;
        Ok(buffer.len())
    }

    /// Returns the encoding type of this encoder.
    fn encoding(&self) -> Encoding;

    /// Returns an estimate of the encoded data, in bytes.
    /// Method call must be O(1).
    fn estimated_data_encoded_size(&self) -> usize;

    /// Returns an estimate of the memory use of this encoder, in bytes
    fn estimated_memory_size(&self) -> usize;

    /// Flushes the underlying byte buffer that's being processed by this encoder, and
    /// return the immutable copy of it. This will also reset the internal state.
    fn flush_buffer(&mut self) -> Result<Bytes>;
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum DictionaryValueIndices<'a> {
    I8(&'a [i8], ValueSelection<'a>),
    I16(&'a [i16], ValueSelection<'a>),
    I32(&'a [i32], ValueSelection<'a>),
    I64(&'a [i64], ValueSelection<'a>),
    U8(&'a [u8], ValueSelection<'a>),
    U16(&'a [u16], ValueSelection<'a>),
    U32(&'a [u32], ValueSelection<'a>),
    U64(&'a [u64], ValueSelection<'a>),
}

#[cfg(feature = "arrow")]
impl<'a> DictionaryValueIndices<'a> {
    pub(crate) fn i8(keys: &'a [i8], selection: ValueSelection<'a>) -> Self {
        Self::I8(keys, selection)
    }

    pub(crate) fn i16(keys: &'a [i16], selection: ValueSelection<'a>) -> Self {
        Self::I16(keys, selection)
    }

    pub(crate) fn i32(keys: &'a [i32], selection: ValueSelection<'a>) -> Self {
        Self::I32(keys, selection)
    }

    pub(crate) fn i64(keys: &'a [i64], selection: ValueSelection<'a>) -> Self {
        Self::I64(keys, selection)
    }

    pub(crate) fn u8(keys: &'a [u8], selection: ValueSelection<'a>) -> Self {
        Self::U8(keys, selection)
    }

    pub(crate) fn u16(keys: &'a [u16], selection: ValueSelection<'a>) -> Self {
        Self::U16(keys, selection)
    }

    pub(crate) fn u32(keys: &'a [u32], selection: ValueSelection<'a>) -> Self {
        Self::U32(keys, selection)
    }

    pub(crate) fn u64(keys: &'a [u64], selection: ValueSelection<'a>) -> Self {
        Self::U64(keys, selection)
    }

    pub(crate) fn len(self) -> usize {
        match self {
            Self::I8(_, selection)
            | Self::I16(_, selection)
            | Self::I32(_, selection)
            | Self::I64(_, selection)
            | Self::U8(_, selection)
            | Self::U16(_, selection)
            | Self::U32(_, selection)
            | Self::U64(_, selection) => selection.len(),
        }
    }

    fn slice(self, offset: usize, len: usize) -> Self {
        match self {
            Self::I8(keys, selection) => Self::I8(keys, selection.slice(offset, len)),
            Self::I16(keys, selection) => Self::I16(keys, selection.slice(offset, len)),
            Self::I32(keys, selection) => Self::I32(keys, selection.slice(offset, len)),
            Self::I64(keys, selection) => Self::I64(keys, selection.slice(offset, len)),
            Self::U8(keys, selection) => Self::U8(keys, selection.slice(offset, len)),
            Self::U16(keys, selection) => Self::U16(keys, selection.slice(offset, len)),
            Self::U32(keys, selection) => Self::U32(keys, selection.slice(offset, len)),
            Self::U64(keys, selection) => Self::U64(keys, selection.slice(offset, len)),
        }
    }

    #[inline(always)]
    fn index_at(self, idx: usize) -> usize {
        match self {
            Self::I8(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::I16(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::I32(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::I64(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::U8(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::U16(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::U32(keys, selection) => keys[selection.index_at(idx)].as_usize(),
            Self::U64(keys, selection) => keys[selection.index_at(idx)].as_usize(),
        }
    }

    #[inline]
    pub(crate) fn try_for_each<E>(
        self,
        mut f: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Self::I8(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::I16(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::I32(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::I64(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::U8(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::U16(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::U32(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
            Self::U64(keys, selection) => selection.try_for_each(|idx| f(keys[idx].as_usize())),
        }
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DefaultedDictionaryValueIndices<'a> {
    indices: DictionaryValueIndices<'a>,
    key_nulls: Option<&'a NullBuffer>,
    value_nulls: Option<&'a NullBuffer>,
}

#[cfg(feature = "arrow")]
impl<'a> DefaultedDictionaryValueIndices<'a> {
    pub(crate) fn new(
        indices: DictionaryValueIndices<'a>,
        key_nulls: Option<&'a NullBuffer>,
        value_nulls: Option<&'a NullBuffer>,
    ) -> Self {
        Self {
            indices,
            key_nulls,
            value_nulls,
        }
    }

    pub(crate) fn len(self) -> usize {
        self.indices.len()
    }

    pub(crate) fn slice(self, offset: usize, len: usize) -> Self {
        Self {
            indices: self.indices.slice(offset, len),
            key_nulls: self.key_nulls,
            value_nulls: self.value_nulls,
        }
    }

    #[inline]
    pub(crate) fn try_for_each<E>(
        self,
        mut f: impl FnMut(Option<usize>) -> Result<(), E>,
    ) -> Result<(), E> {
        match self.indices {
            DictionaryValueIndices::I8(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::I16(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::I32(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::I64(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::U8(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::U16(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::U32(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
            DictionaryValueIndices::U64(keys, selection) => {
                self.try_for_each_key(keys, selection, &mut f)
            }
        }
    }

    #[inline]
    fn try_for_each_key<K, E>(
        self,
        keys: &'a [K],
        selection: ValueSelection<'a>,
        f: &mut impl FnMut(Option<usize>) -> Result<(), E>,
    ) -> Result<(), E>
    where
        K: ArrowNativeType,
    {
        // Hoist the loop-invariant null-buffer presence out of the per-row loop
        // by specializing on which of (key_nulls, value_nulls) are set. The
        // common defaulted case (key nulls only) then skips the value-null arm,
        // and the no-nulls case degenerates to a plain keyed gather.
        match (self.key_nulls, self.value_nulls) {
            (None, None) => selection.try_for_each(|row| f(Some(keys[row].as_usize()))),
            (Some(key_nulls), None) => selection.try_for_each(|row| {
                if key_nulls.is_null(row) {
                    f(None)
                } else {
                    f(Some(keys[row].as_usize()))
                }
            }),
            (None, Some(value_nulls)) => selection.try_for_each(|row| {
                let idx = keys[row].as_usize();
                if idx < value_nulls.len() && value_nulls.is_null(idx) {
                    f(None)
                } else {
                    f(Some(idx))
                }
            }),
            (Some(key_nulls), Some(value_nulls)) => selection.try_for_each(|row| {
                if key_nulls.is_null(row) {
                    return f(None);
                }
                let idx = keys[row].as_usize();
                if idx < value_nulls.len() && value_nulls.is_null(idx) {
                    f(None)
                } else {
                    f(Some(idx))
                }
            }),
        }
    }
}

/// Borrowed packed boolean values.
///
/// Bits are addressed in the same least-significant-bit-first order used by
/// Arrow boolean buffers and Parquet boolean encodings.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "arrow")]
pub(crate) struct PackedBoolValues<'a> {
    bytes: &'a [u8],
    selection: PackedBoolSelection<'a>,
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "arrow")]
enum PackedBoolSelection<'a> {
    Dense {
        bit_offset: usize,
        len: usize,
    },
    Sparse {
        bit_offset: usize,
        indices: &'a [usize],
    },
    Indexed {
        bit_offset: usize,
        indices: ValueIndices<'a>,
    },
    Defaulted {
        value_bits: &'a [u8],
        value_bit_offset: usize,
        indices: DefaultedDictionaryValueIndices<'a>,
    },
}

#[cfg(feature = "arrow")]
impl<'a> PackedBoolValues<'a> {
    pub(crate) fn new(bytes: &'a [u8], bit_offset: usize, len: usize) -> Self {
        Self {
            bytes,
            selection: PackedBoolSelection::Dense { bit_offset, len },
        }
    }

    pub(crate) fn new_sparse(bytes: &'a [u8], bit_offset: usize, indices: &'a [usize]) -> Self {
        Self {
            bytes,
            selection: PackedBoolSelection::Sparse {
                bit_offset,
                indices,
            },
        }
    }

    pub(crate) fn new_indexed(
        bytes: &'a [u8],
        bit_offset: usize,
        indices: ValueIndices<'a>,
    ) -> Self {
        Self {
            bytes,
            selection: PackedBoolSelection::Indexed {
                bit_offset,
                indices,
            },
        }
    }

    /// Selects bits from a boolean dictionary's `value_bits` through defaulted
    /// keyed indices, yielding `false` for any null key or null dictionary value.
    pub(crate) fn new_defaulted(
        value_bits: &'a [u8],
        value_bit_offset: usize,
        indices: DefaultedDictionaryValueIndices<'a>,
    ) -> Self {
        // The byte source lives in the `Defaulted` selection; `bytes` is unused.
        Self {
            bytes: value_bits,
            selection: PackedBoolSelection::Defaulted {
                value_bits,
                value_bit_offset,
                indices,
            },
        }
    }

    pub(crate) fn dense(self) -> Option<(&'a [u8], usize, usize)> {
        match self.selection {
            PackedBoolSelection::Dense { bit_offset, len } => Some((self.bytes, bit_offset, len)),
            PackedBoolSelection::Sparse { .. }
            | PackedBoolSelection::Indexed { .. }
            | PackedBoolSelection::Defaulted { .. } => None,
        }
    }

    pub(crate) fn len(self) -> usize {
        match self.selection {
            PackedBoolSelection::Dense { len, .. } => len,
            PackedBoolSelection::Sparse { indices, .. } => indices.len(),
            PackedBoolSelection::Indexed { indices, .. } => indices.len(),
            PackedBoolSelection::Defaulted { indices, .. } => indices.len(),
        }
    }

    pub(crate) fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Push the packed boolean run directly to a sink. Empty runs are preserved
    /// because all-null pages still need to reach the boolean encoder.
    #[inline]
    pub(crate) fn write_into<S: ChunkSink<PackedBoolValues<'a>>>(self, sink: &mut S) -> Result<()> {
        sink.consume(&self)
    }

    /// Resolve the selection once, then yield each selected bit.
    #[inline]
    fn for_each(self, mut f: impl FnMut(bool)) {
        let bytes = self.bytes;
        match self.selection {
            PackedBoolSelection::Dense { bit_offset, len } => {
                for i in 0..len {
                    f(get_bit(bytes, bit_offset + i));
                }
            }
            PackedBoolSelection::Sparse {
                bit_offset,
                indices,
            } => {
                for &idx in indices {
                    f(get_bit(bytes, bit_offset + idx));
                }
            }
            PackedBoolSelection::Indexed {
                bit_offset,
                indices,
            } => {
                indices.for_each(|idx| f(get_bit(bytes, bit_offset + idx)));
            }
            PackedBoolSelection::Defaulted {
                value_bits,
                value_bit_offset,
                indices,
            } => {
                let _ = indices.try_for_each(|idx| -> Result<(), ()> {
                    f(idx.is_some_and(|idx| get_bit(value_bits, value_bit_offset + idx)));
                    Ok(())
                });
            }
        }
    }

    #[inline]
    fn put_indexed_packed(self, bit_writer: &mut BitWriter) {
        // Stream the selected bits once, packing them LSB-first into words.
        let mut word: u64 = 0;
        let mut bits: usize = 0;
        self.for_each(|b| {
            word |= (b as u64) << bits;
            bits += 1;
            if bits == 64 {
                bit_writer.put_value(word, 64);
                word = 0;
                bits = 0;
            }
        });
        if bits > 0 {
            bit_writer.put_value(word, bits);
        }
    }

    pub(crate) fn true_count(self) -> usize {
        match self.selection {
            // Dense: count set bits in the contiguous range in bulk (popcount)
            // instead of testing each bit individually.
            PackedBoolSelection::Dense { bit_offset, len } => {
                UnalignedBitChunk::new(self.bytes, bit_offset, len).count_ones()
            }
            // Sparse/Indexed/Defaulted: selected bits are non-contiguous, so
            // resolve the selection once and count via `for_each` (no popcount).
            PackedBoolSelection::Sparse { .. }
            | PackedBoolSelection::Indexed { .. }
            | PackedBoolSelection::Defaulted { .. } => {
                let mut count = 0;
                self.for_each(|b| count += b as usize);
                count
            }
        }
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum ValueIndices<'a> {
    Empty,
    Dense { offset: usize, len: usize },
    Sparse(&'a [usize]),
    Dictionary(DictionaryValueIndices<'a>),
}

#[cfg(feature = "arrow")]
impl<'a> ValueIndices<'a> {
    pub(crate) fn dictionary(indices: DictionaryValueIndices<'a>) -> Self {
        Self::Dictionary(indices)
    }

    pub(crate) fn len(self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Dense { len, .. } => len,
            Self::Sparse(indices) => indices.len(),
            Self::Dictionary(indices) => indices.len(),
        }
    }

    pub(crate) fn is_sparse(self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    pub(crate) fn slice(self, offset: usize, len: usize) -> Self {
        match self {
            Self::Empty => {
                debug_assert_eq!(offset, 0);
                debug_assert_eq!(len, 0);
                Self::Empty
            }
            Self::Dense {
                offset: base,
                len: _selection_len,
            } => Self::Dense {
                offset: base + offset,
                len,
            },
            Self::Sparse(indices) => Self::Sparse(&indices[offset..offset + len]),
            Self::Dictionary(indices) => Self::Dictionary(indices.slice(offset, len)),
        }
    }

    #[inline(always)]
    pub(crate) fn index_at(self, idx: usize) -> usize {
        debug_assert!(idx < self.len());
        match self {
            Self::Empty => unreachable!("empty indices have no values"),
            Self::Dense { offset, .. } => offset + idx,
            Self::Sparse(indices) => indices[idx],
            Self::Dictionary(indices) => indices.index_at(idx),
        }
    }

    #[inline]
    pub(crate) fn try_for_each<E>(
        self,
        mut f: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Self::Empty => Ok(()),
            Self::Dense { offset, len } => {
                for idx in offset..offset + len {
                    f(idx)?;
                }
                Ok(())
            }
            Self::Sparse(indices) => {
                for &idx in indices {
                    f(idx)?;
                }
                Ok(())
            }
            Self::Dictionary(indices) => indices.try_for_each(f),
        }
    }

    #[inline]
    pub(crate) fn for_each(self, mut f: impl FnMut(usize)) {
        let _ = self.try_for_each(|idx| -> Result<(), ()> {
            f(idx);
            Ok(())
        });
    }
}

#[cfg(feature = "arrow")]
/// Consumes chunks produced by [`ValueStream::write_into`].
///
/// `C` is the chunk type, not necessarily the scalar value type: numeric
/// streams push `[T]`, and other value families push their own packed chunk
/// representation.
pub(crate) trait ChunkSink<C: ?Sized> {
    fn consume(&mut self, chunk: &C) -> Result<()>;
}

/// Shared iteration surface over a selected stream of physical values `T`.
///
/// Implementations provide length, optional dense bulk access, and per-value
/// iteration. The default [`Self::write_into`] uses the bulk path when
/// available, otherwise gathers small tiles and sends them to a [`ChunkSink`].
#[cfg(feature = "arrow")]
pub(crate) trait ValueStream<'a, T: Copy + Default + 'a>: Copy + 'a {
    /// The contiguous chunk type used by [`Self::bulk`]. For numeric streams
    /// this is `[T]`; for dense fixed-length byte arrays it is packed `[u8]`.
    type Bulk: ?Sized;

    fn len(self) -> usize;

    /// The whole contiguous run when this stream can expose one.
    fn bulk(self) -> Option<&'a Self::Bulk>;

    fn try_for_each<E>(self, f: impl FnMut(T) -> Result<(), E>) -> Result<(), E>;

    /// Push the selected values into `sink`, using one bulk chunk when possible
    /// and fixed-size gathered tiles otherwise.
    #[inline]
    fn write_into<S: ChunkSink<[T]> + ChunkSink<Self::Bulk>>(self, sink: &mut S) -> Result<()> {
        if let Some(bulk) = self.bulk() {
            if self.len() != 0 {
                sink.consume(bulk)?;
            }
            return Ok(());
        }
        const N: usize = 64;
        let mut buf = [T::default(); N];
        let mut filled = 0;
        self.try_for_each(|v| -> Result<()> {
            buf[filled] = v;
            filled += 1;
            if filled == N {
                // `&buf[..]` (not `&buf`): with both `ChunkSink<[T]>` and
                // `ChunkSink<Self::Bulk>` in scope the array→slice coercion is no
                // longer inferred, so name the `[T]` slice explicitly.
                sink.consume(&buf[..])?;
                filled = 0;
            }
            Ok(())
        })?;
        if filled > 0 {
            sink.consume(&buf[..filled])?;
        }
        Ok(())
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Int32Values<'a> {
    I32(&'a [i32], ValueIndices<'a>),
    I8(&'a [i8], ValueIndices<'a>),
    I16(&'a [i16], ValueIndices<'a>),
    U8(&'a [u8], ValueIndices<'a>),
    U16(&'a [u16], ValueIndices<'a>),
    Date64Days(&'a [i64], ValueIndices<'a>),
    I64Cast(&'a [i64], ValueIndices<'a>),
    I128Cast(&'a [i128], ValueIndices<'a>),
    I256Cast(&'a [i256], ValueIndices<'a>),
}

#[cfg(feature = "arrow")]
impl<'a> Int32Values<'a> {
    pub(crate) fn i32(values: &'a [i32], indices: ValueIndices<'a>) -> Self {
        Self::I32(values, indices)
    }

    pub(crate) fn i8(values: &'a [i8], indices: ValueIndices<'a>) -> Self {
        Self::I8(values, indices)
    }

    pub(crate) fn i16(values: &'a [i16], indices: ValueIndices<'a>) -> Self {
        Self::I16(values, indices)
    }

    pub(crate) fn u8(values: &'a [u8], indices: ValueIndices<'a>) -> Self {
        Self::U8(values, indices)
    }

    pub(crate) fn u16(values: &'a [u16], indices: ValueIndices<'a>) -> Self {
        Self::U16(values, indices)
    }

    pub(crate) fn date64_days(values: &'a [i64], indices: ValueIndices<'a>) -> Self {
        Self::Date64Days(values, indices)
    }

    pub(crate) fn i64_cast(values: &'a [i64], indices: ValueIndices<'a>) -> Self {
        Self::I64Cast(values, indices)
    }

    pub(crate) fn i128_cast(values: &'a [i128], indices: ValueIndices<'a>) -> Self {
        Self::I128Cast(values, indices)
    }

    pub(crate) fn i256_cast(values: &'a [i256], indices: ValueIndices<'a>) -> Self {
        Self::I256Cast(values, indices)
    }
}

#[cfg(feature = "arrow")]
impl<'a> ValueStream<'a, i32> for Int32Values<'a> {
    type Bulk = [i32];

    #[inline]
    fn len(self) -> usize {
        match self {
            Self::I32(_, indices)
            | Self::I8(_, indices)
            | Self::I16(_, indices)
            | Self::U8(_, indices)
            | Self::U16(_, indices)
            | Self::Date64Days(_, indices)
            | Self::I64Cast(_, indices)
            | Self::I128Cast(_, indices) => indices.len(),
            Self::I256Cast(_, indices) => indices.len(),
        }
    }

    #[inline]
    fn bulk(self) -> Option<&'a [i32]> {
        match self {
            Self::I32(values, ValueIndices::Dense { offset, len }) => {
                Some(&values[offset..offset + len])
            }
            _ => None,
        }
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(i32) -> Result<(), E>) -> Result<(), E> {
        match self {
            Self::I32(values, indices) => indices.try_for_each(|idx| f(values[idx])),
            Self::I8(values, indices) => indices.try_for_each(|idx| f(values[idx] as i32)),
            Self::I16(values, indices) => indices.try_for_each(|idx| f(values[idx] as i32)),
            Self::U8(values, indices) => indices.try_for_each(|idx| f(values[idx] as i32)),
            Self::U16(values, indices) => indices.try_for_each(|idx| f(values[idx] as i32)),
            Self::Date64Days(values, indices) => {
                indices.try_for_each(|idx| f((values[idx] / 86_400_000) as i32))
            }
            Self::I64Cast(values, indices) => indices.try_for_each(|idx| f(values[idx] as i32)),
            Self::I128Cast(values, indices) => indices.try_for_each(|idx| f(values[idx] as i32)),
            Self::I256Cast(values, indices) => {
                indices.try_for_each(|idx| f(values[idx].as_i128() as i32))
            }
        }
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Int64Values<'a> {
    I64(&'a [i64], ValueIndices<'a>),
    I128Cast(&'a [i128], ValueIndices<'a>),
    I256Cast(&'a [i256], ValueIndices<'a>),
}

#[cfg(feature = "arrow")]
impl<'a> Int64Values<'a> {
    pub(crate) fn i64(values: &'a [i64], indices: ValueIndices<'a>) -> Self {
        Self::I64(values, indices)
    }

    pub(crate) fn i128_cast(values: &'a [i128], indices: ValueIndices<'a>) -> Self {
        Self::I128Cast(values, indices)
    }

    pub(crate) fn i256_cast(values: &'a [i256], indices: ValueIndices<'a>) -> Self {
        Self::I256Cast(values, indices)
    }
}

#[cfg(feature = "arrow")]
impl<'a> ValueStream<'a, i64> for Int64Values<'a> {
    type Bulk = [i64];

    #[inline]
    fn len(self) -> usize {
        match self {
            Self::I64(_, indices) => indices.len(),
            Self::I128Cast(_, indices) => indices.len(),
            Self::I256Cast(_, indices) => indices.len(),
        }
    }

    #[inline]
    fn bulk(self) -> Option<&'a [i64]> {
        match self {
            Self::I64(values, ValueIndices::Dense { offset, len }) => {
                Some(&values[offset..offset + len])
            }
            _ => None,
        }
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(i64) -> Result<(), E>) -> Result<(), E> {
        match self {
            Self::I64(values, indices) => indices.try_for_each(|idx| f(values[idx])),
            Self::I128Cast(values, indices) => indices.try_for_each(|idx| f(values[idx] as i64)),
            Self::I256Cast(values, indices) => {
                indices.try_for_each(|idx| f(values[idx].as_i128() as i64))
            }
        }
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FloatValues<'a> {
    values: &'a [f32],
    indices: ValueIndices<'a>,
}

#[cfg(feature = "arrow")]
impl<'a> FloatValues<'a> {
    pub(crate) fn new(values: &'a [f32], indices: ValueIndices<'a>) -> Self {
        Self { values, indices }
    }
}

#[cfg(feature = "arrow")]
impl<'a> ValueStream<'a, f32> for FloatValues<'a> {
    type Bulk = [f32];

    #[inline]
    fn len(self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [f32]> {
        match self.indices {
            ValueIndices::Dense { offset, len } => Some(&self.values[offset..offset + len]),
            ValueIndices::Empty | ValueIndices::Sparse(_) | ValueIndices::Dictionary(_) => None,
        }
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(f32) -> Result<(), E>) -> Result<(), E> {
        self.indices.try_for_each(|idx| f(self.values[idx]))
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DoubleValues<'a> {
    values: &'a [f64],
    indices: ValueIndices<'a>,
}

#[cfg(feature = "arrow")]
impl<'a> DoubleValues<'a> {
    pub(crate) fn new(values: &'a [f64], indices: ValueIndices<'a>) -> Self {
        Self { values, indices }
    }
}

#[cfg(feature = "arrow")]
impl<'a> ValueStream<'a, f64> for DoubleValues<'a> {
    type Bulk = [f64];

    #[inline]
    fn len(self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [f64]> {
        match self.indices {
            ValueIndices::Dense { offset, len } => Some(&self.values[offset..offset + len]),
            ValueIndices::Empty | ValueIndices::Sparse(_) | ValueIndices::Dictionary(_) => None,
        }
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(f64) -> Result<(), E>) -> Result<(), E> {
        self.indices.try_for_each(|idx| f(self.values[idx]))
    }
}

/// Borrowed fixed-length byte-array values.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "arrow")]
pub(crate) struct FixedLenByteArrayValues<'a> {
    bytes: &'a [u8],
    type_length: usize,
    indices: ValueIndices<'a>,
}

#[cfg(feature = "arrow")]
impl<'a> FixedLenByteArrayValues<'a> {
    pub(crate) fn new(bytes: &'a [u8], type_length: usize, len: usize) -> Self {
        assert_eq!(
            bytes.len(),
            type_length
                .checked_mul(len)
                .expect("fixed-length byte-array values length overflow")
        );
        Self {
            bytes,
            type_length,
            indices: ValueIndices::Dense { offset: 0, len },
        }
    }

    pub(crate) fn new_selected(
        bytes: &'a [u8],
        type_length: usize,
        indices: ValueIndices<'a>,
    ) -> Self {
        if type_length == 0 {
            assert!(
                bytes.is_empty(),
                "zero-width fixed-length byte-array values must not have data bytes"
            );
        } else {
            assert_eq!(
                bytes.len() % type_length,
                0,
                "fixed-length byte-array values length must be a multiple of type length"
            );
        }

        Self {
            bytes,
            type_length,
            indices,
        }
    }

    pub(crate) fn dense_bytes(self) -> Option<&'a [u8]> {
        match self.indices {
            ValueIndices::Dense { offset, len } => {
                let start = offset * self.type_length;
                let end = start + len * self.type_length;
                Some(&self.bytes[start..end])
            }
            _ => None,
        }
    }

    pub(crate) fn type_length(self) -> usize {
        self.type_length
    }

    pub(crate) fn len(self) -> usize {
        self.indices.len()
    }

    pub(crate) fn iter(self) -> FixedLenByteArrayValueIter<'a> {
        FixedLenByteArrayValueIter {
            bytes: self.bytes,
            type_length: self.type_length,
            indices: self.indices,
            offset: 0,
        }
    }
}

/// Dense fixed-length byte-array values are already packed in PLAIN wire
/// layout, so [`Self::Bulk`] is the raw byte run. Sparse selections still yield
/// gathered `&[u8]` values.
#[cfg(feature = "arrow")]
impl<'a> ValueStream<'a, &'a [u8]> for FixedLenByteArrayValues<'a> {
    type Bulk = [u8];

    #[inline]
    fn len(self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [u8]> {
        self.dense_bytes()
    }

    #[inline]
    fn try_for_each<E>(self, f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
        self.iter().try_for_each(f)
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "arrow")]
pub(crate) struct FixedLenByteArrayValueIter<'a> {
    bytes: &'a [u8],
    type_length: usize,
    indices: ValueIndices<'a>,
    offset: usize,
}

#[cfg(feature = "arrow")]
impl<'a> Iterator for FixedLenByteArrayValueIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.indices.len() {
            return None;
        }

        let idx = self.indices.index_at(self.offset);
        self.offset += 1;

        if self.type_length == 0 {
            return Some(&self.bytes[..0]);
        }

        let start = idx * self.type_length;
        let end = start + self.type_length;
        Some(&self.bytes[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.offset;
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "arrow")]
impl ExactSizeIterator for FixedLenByteArrayValueIter<'_> {}

/// [`ValueStream`] for a dictionary selection that substitutes a default value
/// for null keys or null dictionary values.
#[cfg(feature = "arrow")]
#[derive(Clone, Copy)]
pub(crate) struct DefaultedValues<'a, T, F> {
    indices: DefaultedDictionaryValueIndices<'a>,
    map: F,
    _marker: std::marker::PhantomData<fn() -> T>,
}

#[cfg(feature = "arrow")]
impl<'a, T, F> DefaultedValues<'a, T, F>
where
    T: Copy + Default + 'a,
    F: Fn(Option<usize>) -> T + Copy,
{
    #[inline]
    pub(crate) fn new(indices: DefaultedDictionaryValueIndices<'a>, map: F) -> Self {
        Self {
            indices,
            map,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "arrow")]
impl<'a, T, F> ValueStream<'a, T> for DefaultedValues<'a, T, F>
where
    T: Copy + Default + 'a,
    F: Fn(Option<usize>) -> T + Copy + 'a,
{
    // A defaulted dictionary is always a keyed gather (each row maps through the
    // keys and substitutes a default for nulls), so there is never a contiguous
    // run to hand over — it always tiles.
    type Bulk = [T];

    #[inline]
    fn len(self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn bulk(self) -> Option<&'a [T]> {
        None
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(T) -> Result<(), E>) -> Result<(), E> {
        self.indices.try_for_each(|opt| f((self.map)(opt)))
    }
}

/// Encodes packed boolean values.
///
/// This is a storage-shape fast path for Arrow boolean buffers.
#[doc(hidden)]
#[cfg(feature = "arrow")]
pub(crate) trait BoolEncoder: Encoder<BoolType> {
    fn put_packed_bool(&mut self, _values: PackedBoolValues<'_>) -> Result<()> {
        Err(general_err!(
            "Packed boolean values are not supported by this encoder"
        ))
    }
}

/// Encodes raw fixed-length byte-array values.
///
/// This is a storage-shape fast path for Arrow fixed-width buffers.
#[doc(hidden)]
#[cfg(feature = "arrow")]
pub(crate) trait FixedLenByteArrayEncoder: Encoder<FixedLenByteArrayType> {
    fn put_fixed_len_byte_array(&mut self, values: FixedLenByteArrayValues<'_>) -> Result<()> {
        // Generic fallback for encoders without a raw fast path — e.g.
        // DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY, selected for FLBA under
        // PARQUET_2_0 — by materializing each value and routing through the
        // standard `Encoder::put`. The `supports_raw_fixed_len_byte_array` gate
        // in the Arrow writer keeps the bulk fast path (PLAIN / BYTE_STREAM_SPLIT,
        // which override this method) off this route, so this only carries the
        // per-value stream path for the remaining encodings.
        for value in values.iter() {
            let value = FixedLenByteArray::from(ByteArray::from(value.to_vec()));
            self.put(std::slice::from_ref(&value))?;
        }
        Ok(())
    }

    /// Reserve room for `additional_bytes` of appended fixed-length values. No-op
    /// unless the encoder buffers raw bytes contiguously (PLAIN, BYTE_STREAM_SPLIT),
    /// letting the streaming write size its buffer exactly once from the known value
    /// count.
    #[cfg(feature = "arrow")]
    fn reserve_fixed_len(&mut self, _additional_bytes: usize) {}

    /// Append a single fixed-length value's bytes into the encoder's own state, in
    /// the single streaming pass that drives sparse / computed FLBA columns. Each
    /// encoder folds the value into its native representation directly (PLAIN/BSS
    /// append to their byte buffer; DELTA front-codes / records the length), so no
    /// contiguous column buffer is materialized. The default is a correct fallback
    /// for encodings not used on the FLBA value path.
    #[cfg(feature = "arrow")]
    fn append_fixed_len_value(&mut self, value: &[u8]) -> Result<()> {
        self.put_fixed_len_byte_array(FixedLenByteArrayValues::new(value, value.len(), 1))
    }
}

/// Gets a encoder for the particular data type `T` and encoding `encoding`. Memory usage
/// for the encoder instance is tracked by `mem_tracker`.
pub fn get_encoder<T: DataType>(
    encoding: Encoding,
    descr: &ColumnDescPtr,
) -> Result<Box<dyn Encoder<T>>> {
    let encoder: Box<dyn Encoder<T>> = match encoding {
        Encoding::PLAIN => Box::new(PlainEncoder::new()),
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
            return Err(general_err!(
                "Cannot initialize this encoding through this function"
            ));
        }
        Encoding::RLE => Box::new(RleValueEncoder::new()),
        Encoding::DELTA_BINARY_PACKED => Box::new(DeltaBitPackEncoder::new()),
        Encoding::DELTA_LENGTH_BYTE_ARRAY => Box::new(DeltaLengthByteArrayEncoder::new()),
        Encoding::DELTA_BYTE_ARRAY => Box::new(DeltaByteArrayEncoder::new()),
        Encoding::BYTE_STREAM_SPLIT => match T::get_physical_type() {
            Type::FIXED_LEN_BYTE_ARRAY => Box::new(VariableWidthByteStreamSplitEncoder::new(
                descr.type_length(),
            )),
            _ => Box::new(ByteStreamSplitEncoder::new()),
        },
        e => return Err(nyi_err!("Encoding {} is not supported", e)),
    };
    Ok(encoder)
}

/// Builds the encoder trait object held by a `ColumnValueEncoderImpl`.
///
/// The blanket implementation for `dyn Encoder<T>` defers to [`get_encoder`],
/// giving the existing dynamic-dispatch behavior. Physical-type-specialized
/// implementations override it where a faster path is available.
#[doc(hidden)]
pub trait EncoderFactory<T: DataType>: Encoder<T> {
    fn get_encoder(encoding: Encoding, descr: &ColumnDescPtr) -> Result<Box<Self>>;
}

impl<T: DataType> EncoderFactory<T> for dyn Encoder<T> {
    fn get_encoder(encoding: Encoding, descr: &ColumnDescPtr) -> Result<Box<Self>> {
        get_encoder::<T>(encoding, descr)
    }
}

/// Builds a fixed-set encoder-object enum for one physical type plus its
/// object-safe [`Encoder`] forwarding.
macro_rules! encoder_object {
    ($name:ident, $ty:ty, $bss:ty) => {
        #[doc(hidden)]
        pub enum $name {
            Plain(PlainEncoder<$ty>),
            Rle(RleValueEncoder<$ty>),
            // The DELTA family wraps one or more (large) `DeltaBitPackEncoder`s;
            // box them so the enum stays small. These are cold fallback paths.
            DeltaBinaryPacked(Box<DeltaBitPackEncoder<$ty>>),
            DeltaLengthByteArray(Box<DeltaLengthByteArrayEncoder<$ty>>),
            DeltaByteArray(Box<DeltaByteArrayEncoder<$ty>>),
            ByteStreamSplit($bss),
        }

        impl Encoder<$ty> for $name {
            fn put(&mut self, values: &[<$ty as DataType>::T]) -> Result<()> {
                match self {
                    Self::Plain(e) => e.put(values),
                    Self::Rle(e) => e.put(values),
                    Self::DeltaBinaryPacked(e) => e.put(values),
                    Self::DeltaLengthByteArray(e) => e.put(values),
                    Self::DeltaByteArray(e) => e.put(values),
                    Self::ByteStreamSplit(e) => e.put(values),
                }
            }

            fn encoding(&self) -> Encoding {
                match self {
                    Self::Plain(e) => e.encoding(),
                    Self::Rle(e) => e.encoding(),
                    Self::DeltaBinaryPacked(e) => e.encoding(),
                    Self::DeltaLengthByteArray(e) => e.encoding(),
                    Self::DeltaByteArray(e) => e.encoding(),
                    Self::ByteStreamSplit(e) => e.encoding(),
                }
            }

            fn estimated_data_encoded_size(&self) -> usize {
                match self {
                    Self::Plain(e) => e.estimated_data_encoded_size(),
                    Self::Rle(e) => e.estimated_data_encoded_size(),
                    Self::DeltaBinaryPacked(e) => e.estimated_data_encoded_size(),
                    Self::DeltaLengthByteArray(e) => e.estimated_data_encoded_size(),
                    Self::DeltaByteArray(e) => e.estimated_data_encoded_size(),
                    Self::ByteStreamSplit(e) => e.estimated_data_encoded_size(),
                }
            }

            fn estimated_memory_size(&self) -> usize {
                match self {
                    Self::Plain(e) => e.estimated_memory_size(),
                    Self::Rle(e) => e.estimated_memory_size(),
                    Self::DeltaBinaryPacked(e) => e.estimated_memory_size(),
                    Self::DeltaLengthByteArray(e) => e.estimated_memory_size(),
                    Self::DeltaByteArray(e) => e.estimated_memory_size(),
                    Self::ByteStreamSplit(e) => e.estimated_memory_size(),
                }
            }

            fn flush_buffer(&mut self) -> Result<Bytes> {
                match self {
                    Self::Plain(e) => e.flush_buffer(),
                    Self::Rle(e) => e.flush_buffer(),
                    Self::DeltaBinaryPacked(e) => e.flush_buffer(),
                    Self::DeltaLengthByteArray(e) => e.flush_buffer(),
                    Self::DeltaByteArray(e) => e.flush_buffer(),
                    Self::ByteStreamSplit(e) => e.flush_buffer(),
                }
            }
        }
    };
}

encoder_object!(
    BoolEncoderObject,
    BoolType,
    ByteStreamSplitEncoder<BoolType>
);
encoder_object!(
    FixedLenByteArrayEncoderObject,
    FixedLenByteArrayType,
    VariableWidthByteStreamSplitEncoder<FixedLenByteArrayType>
);

/// Numeric encoder-object enum, generic over the physical type `T`.
#[doc(hidden)]
pub enum NumericEncoderObject<T: DataType> {
    Plain(PlainEncoder<T>),
    Rle(RleValueEncoder<T>),
    // The DELTA family wraps one or more (large) `DeltaBitPackEncoder`s; box
    // them so the enum stays small. These are cold fallback paths.
    DeltaBinaryPacked(Box<DeltaBitPackEncoder<T>>),
    DeltaLengthByteArray(Box<DeltaLengthByteArrayEncoder<T>>),
    DeltaByteArray(Box<DeltaByteArrayEncoder<T>>),
    ByteStreamSplit(ByteStreamSplitEncoder<T>),
}

impl<T: DataType> Encoder<T> for NumericEncoderObject<T> {
    fn put(&mut self, values: &[<T as DataType>::T]) -> Result<()> {
        match self {
            Self::Plain(e) => e.put(values),
            Self::Rle(e) => e.put(values),
            Self::DeltaBinaryPacked(e) => e.put(values),
            Self::DeltaLengthByteArray(e) => e.put(values),
            Self::DeltaByteArray(e) => e.put(values),
            Self::ByteStreamSplit(e) => e.put(values),
        }
    }

    fn encoding(&self) -> Encoding {
        match self {
            Self::Plain(e) => e.encoding(),
            Self::Rle(e) => e.encoding(),
            Self::DeltaBinaryPacked(e) => e.encoding(),
            Self::DeltaLengthByteArray(e) => e.encoding(),
            Self::DeltaByteArray(e) => e.encoding(),
            Self::ByteStreamSplit(e) => e.encoding(),
        }
    }

    fn estimated_data_encoded_size(&self) -> usize {
        match self {
            Self::Plain(e) => e.estimated_data_encoded_size(),
            Self::Rle(e) => e.estimated_data_encoded_size(),
            Self::DeltaBinaryPacked(e) => e.estimated_data_encoded_size(),
            Self::DeltaLengthByteArray(e) => e.estimated_data_encoded_size(),
            Self::DeltaByteArray(e) => e.estimated_data_encoded_size(),
            Self::ByteStreamSplit(e) => e.estimated_data_encoded_size(),
        }
    }

    fn estimated_memory_size(&self) -> usize {
        match self {
            Self::Plain(e) => e.estimated_memory_size(),
            Self::Rle(e) => e.estimated_memory_size(),
            Self::DeltaBinaryPacked(e) => e.estimated_memory_size(),
            Self::DeltaLengthByteArray(e) => e.estimated_memory_size(),
            Self::DeltaByteArray(e) => e.estimated_memory_size(),
            Self::ByteStreamSplit(e) => e.estimated_memory_size(),
        }
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        match self {
            Self::Plain(e) => e.flush_buffer(),
            Self::Rle(e) => e.flush_buffer(),
            Self::DeltaBinaryPacked(e) => e.flush_buffer(),
            Self::DeltaLengthByteArray(e) => e.flush_buffer(),
            Self::DeltaByteArray(e) => e.flush_buffer(),
            Self::ByteStreamSplit(e) => e.flush_buffer(),
        }
    }
}

/// Per-physical-type aliases used by the column writer.
pub type Int32EncoderObject = NumericEncoderObject<Int32Type>;
pub type Int64EncoderObject = NumericEncoderObject<Int64Type>;
pub type FloatEncoderObject = NumericEncoderObject<FloatType>;
pub type DoubleEncoderObject = NumericEncoderObject<DoubleType>;

#[cfg(feature = "arrow")]
impl BoolEncoder for BoolEncoderObject {
    fn put_packed_bool(&mut self, values: PackedBoolValues<'_>) -> Result<()> {
        match self {
            Self::Plain(e) => e.put_packed_bool(values),
            Self::Rle(e) => e.put_packed_bool(values),
            Self::DeltaBinaryPacked(e) => e.put_packed_bool(values),
            Self::DeltaLengthByteArray(e) => e.put_packed_bool(values),
            Self::DeltaByteArray(e) => e.put_packed_bool(values),
            Self::ByteStreamSplit(e) => e.put_packed_bool(values),
        }
    }
}

#[cfg(feature = "arrow")]
impl FixedLenByteArrayEncoder for FixedLenByteArrayEncoderObject {
    fn put_fixed_len_byte_array(&mut self, values: FixedLenByteArrayValues<'_>) -> Result<()> {
        match self {
            Self::Plain(e) => e.put_fixed_len_byte_array(values),
            Self::Rle(e) => e.put_fixed_len_byte_array(values),
            Self::DeltaBinaryPacked(e) => e.put_fixed_len_byte_array(values),
            Self::DeltaLengthByteArray(e) => e.put_fixed_len_byte_array(values),
            Self::DeltaByteArray(e) => e.put_fixed_len_byte_array(values),
            Self::ByteStreamSplit(e) => e.put_fixed_len_byte_array(values),
        }
    }

    #[cfg(feature = "arrow")]
    fn reserve_fixed_len(&mut self, additional_bytes: usize) {
        match self {
            Self::Plain(e) => e.reserve_fixed_len(additional_bytes),
            Self::Rle(e) => e.reserve_fixed_len(additional_bytes),
            Self::DeltaBinaryPacked(e) => e.reserve_fixed_len(additional_bytes),
            Self::DeltaLengthByteArray(e) => e.reserve_fixed_len(additional_bytes),
            Self::DeltaByteArray(e) => e.reserve_fixed_len(additional_bytes),
            Self::ByteStreamSplit(e) => e.reserve_fixed_len(additional_bytes),
        }
    }

    #[cfg(feature = "arrow")]
    fn append_fixed_len_value(&mut self, value: &[u8]) -> Result<()> {
        match self {
            Self::Plain(e) => e.append_fixed_len_value(value),
            Self::Rle(e) => e.append_fixed_len_value(value),
            Self::DeltaBinaryPacked(e) => e.append_fixed_len_value(value),
            Self::DeltaLengthByteArray(e) => e.append_fixed_len_value(value),
            Self::DeltaByteArray(e) => e.append_fixed_len_value(value),
            Self::ByteStreamSplit(e) => e.append_fixed_len_value(value),
        }
    }
}

/// The encoding -> concrete-variant selection shared by every encoder-object
/// enum's `EncoderFactory` impl (`BoolEncoderObject`,
/// `FixedLenByteArrayEncoderObject`, `NumericEncoderObject<T>`). It mirrors
/// `get_encoder` — which builds the `Box<dyn Encoder>` fallback for the
/// non-static-dispatch types (`Int96`, `ByteArray`) — but constructs concrete
/// `Self::` variants for static dispatch. `$bss` is the `BYTE_STREAM_SPLIT` arm,
/// the only part that differs between the enums
/// (`FixedLenByteArrayEncoderObject` uses the variable-width encoder, which
/// needs `descr`). The `return`s in the dictionary/unsupported arms exit the
/// caller's `get_encoder`, so this is only valid inside that method.
macro_rules! encoder_object_from_encoding {
    ($encoding:expr, $bss:expr $(,)?) => {
        match $encoding {
            Encoding::PLAIN => Self::Plain(PlainEncoder::new()),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                return Err(general_err!(
                    "Cannot initialize this encoding through this function"
                ));
            }
            Encoding::RLE => Self::Rle(RleValueEncoder::new()),
            Encoding::DELTA_BINARY_PACKED => Self::DeltaBinaryPacked(Box::default()),
            Encoding::DELTA_LENGTH_BYTE_ARRAY => Self::DeltaLengthByteArray(Box::default()),
            Encoding::DELTA_BYTE_ARRAY => Self::DeltaByteArray(Box::default()),
            Encoding::BYTE_STREAM_SPLIT => $bss,
            e => return Err(nyi_err!("Encoding {} is not supported", e)),
        }
    };
}

impl EncoderFactory<BoolType> for BoolEncoderObject {
    fn get_encoder(encoding: Encoding, _descr: &ColumnDescPtr) -> Result<Box<Self>> {
        Ok(Box::new(encoder_object_from_encoding!(
            encoding,
            Self::ByteStreamSplit(ByteStreamSplitEncoder::new())
        )))
    }
}

impl EncoderFactory<FixedLenByteArrayType> for FixedLenByteArrayEncoderObject {
    fn get_encoder(encoding: Encoding, descr: &ColumnDescPtr) -> Result<Box<Self>> {
        Ok(Box::new(encoder_object_from_encoding!(
            encoding,
            Self::ByteStreamSplit(VariableWidthByteStreamSplitEncoder::new(
                descr.type_length()
            ))
        )))
    }
}

impl<T: DataType> EncoderFactory<T> for NumericEncoderObject<T> {
    fn get_encoder(encoding: Encoding, _descr: &ColumnDescPtr) -> Result<Box<Self>> {
        Ok(Box::new(encoder_object_from_encoding!(
            encoding,
            Self::ByteStreamSplit(ByteStreamSplitEncoder::new())
        )))
    }
}

// ----------------------------------------------------------------------
// Plain encoding

/// Plain encoding that supports all types.
/// Values are encoded back to back.
/// The plain encoding is used whenever a more efficient encoding can not be used.
/// It stores the data in the following format:
/// - BOOLEAN - 1 bit per value, 0 is false; 1 is true.
/// - INT32 - 4 bytes per value, stored as little-endian.
/// - INT64 - 8 bytes per value, stored as little-endian.
/// - FLOAT - 4 bytes per value, stored as IEEE little-endian.
/// - DOUBLE - 8 bytes per value, stored as IEEE little-endian.
/// - BYTE_ARRAY - 4 byte length stored as little endian, followed by bytes.
/// - FIXED_LEN_BYTE_ARRAY - just the bytes are stored.
pub struct PlainEncoder<T: DataType> {
    buffer: Vec<u8>,
    bit_writer: BitWriter,
    _phantom: PhantomData<T>,
}

impl<T: DataType> Default for PlainEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType> PlainEncoder<T> {
    /// Creates new plain encoder.
    pub fn new() -> Self {
        Self {
            buffer: vec![],
            bit_writer: BitWriter::new(256),
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for PlainEncoder<T> {
    // Performance Note:
    // As far as can be seen these functions are rarely called and as such we can hint to the
    // compiler that they dont need to be folded into hot locations in the final output.
    #[cold]
    fn encoding(&self) -> Encoding {
        Encoding::PLAIN
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.buffer.len() + self.bit_writer.bytes_written()
    }

    #[inline]
    fn flush_buffer(&mut self) -> Result<Bytes> {
        self.buffer
            .extend_from_slice(self.bit_writer.flush_buffer());
        self.bit_writer.clear();
        Ok(std::mem::take(&mut self.buffer).into())
    }

    #[inline]
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        T::T::encode(values, &mut self.buffer, &mut self.bit_writer)?;
        Ok(())
    }

    /// Return the estimated memory size of this encoder.
    fn estimated_memory_size(&self) -> usize {
        self.buffer.capacity() * std::mem::size_of::<u8>() + self.bit_writer.estimated_memory_size()
    }
}

#[cfg(feature = "arrow")]
impl BoolEncoder for PlainEncoder<BoolType> {
    #[inline]
    fn put_packed_bool(&mut self, values: PackedBoolValues<'_>) -> Result<()> {
        if let Some((bytes, bit_offset, len)) = values.dense() {
            self.bit_writer.put_bits(bytes, bit_offset, len);
        } else {
            values.put_indexed_packed(&mut self.bit_writer);
        }
        Ok(())
    }
}

#[cfg(feature = "arrow")]
impl FixedLenByteArrayEncoder for PlainEncoder<FixedLenByteArrayType> {
    #[inline]
    fn put_fixed_len_byte_array(&mut self, values: FixedLenByteArrayValues<'_>) -> Result<()> {
        match values.dense_bytes() {
            Some(bytes) => self.buffer.extend_from_slice(bytes),
            None => values
                .iter()
                .for_each(|value| self.buffer.extend_from_slice(value)),
        }
        Ok(())
    }

    /// PLAIN stores fixed-length values back-to-back, so each streamed value is
    /// appended straight into the page buffer — one copy, no intermediate buffer.
    #[cfg(feature = "arrow")]
    #[inline]
    fn reserve_fixed_len(&mut self, additional_bytes: usize) {
        self.buffer.reserve(additional_bytes);
    }

    #[cfg(feature = "arrow")]
    #[inline]
    fn append_fixed_len_value(&mut self, value: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(value);
        Ok(())
    }
}

// ----------------------------------------------------------------------
// RLE encoding

const DEFAULT_RLE_BUFFER_LEN: usize = 1024;

/// RLE/Bit-Packing hybrid encoding for values.
/// Currently is used only for data pages v2 and supports boolean types.
pub struct RleValueEncoder<T: DataType> {
    // Buffer with raw values that we collect,
    // when flushing buffer they are encoded using RLE encoder
    encoder: Option<RleEncoder>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> Default for RleValueEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType> RleValueEncoder<T> {
    /// Creates new rle value encoder.
    pub fn new() -> Self {
        Self {
            encoder: None,
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for RleValueEncoder<T> {
    #[inline]
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        ensure_phys_ty!(Type::BOOLEAN, "RleValueEncoder only supports BoolType");

        let rle_encoder = self.encoder.get_or_insert_with(|| {
            let mut buffer = Vec::with_capacity(DEFAULT_RLE_BUFFER_LEN);
            // Reserve space for length
            buffer.extend_from_slice(&[0; 4]);
            RleEncoder::new_from_buf(1, buffer)
        });

        for value in values {
            let value = value.as_u64()?;
            rle_encoder.put(value)
        }
        Ok(())
    }

    // Performance Note:
    // As far as can be seen these functions are rarely called and as such we can hint to the
    // compiler that they dont need to be folded into hot locations in the final output.
    #[cold]
    fn encoding(&self) -> Encoding {
        Encoding::RLE
    }

    #[inline]
    fn estimated_data_encoded_size(&self) -> usize {
        match self.encoder {
            Some(ref enc) => enc.len(),
            None => 0,
        }
    }

    #[inline]
    fn flush_buffer(&mut self) -> Result<Bytes> {
        ensure_phys_ty!(Type::BOOLEAN, "RleValueEncoder only supports BoolType");
        let rle_encoder = self
            .encoder
            .take()
            .expect("RLE value encoder is not initialized");

        // Flush all encoder buffers and raw values
        let mut buf = rle_encoder.consume();
        assert!(buf.len() >= 4, "should have had padding inserted");

        // Note that buf does not have any offset, all data is encoded bytes
        let len = (buf.len() - 4) as i32;
        buf[..4].copy_from_slice(&len.to_le_bytes());

        Ok(buf.into())
    }

    /// return the estimated memory size of this encoder.
    fn estimated_memory_size(&self) -> usize {
        self.encoder
            .as_ref()
            .map_or(0, |enc| enc.estimated_memory_size())
    }
}

#[cfg(feature = "arrow")]
impl BoolEncoder for RleValueEncoder<BoolType> {
    #[inline]
    fn put_packed_bool(&mut self, values: PackedBoolValues<'_>) -> Result<()> {
        let rle_encoder = self.encoder.get_or_insert_with(|| {
            let mut buffer = Vec::with_capacity(DEFAULT_RLE_BUFFER_LEN);
            // Reserve space for length
            buffer.extend_from_slice(&[0; 4]);
            RleEncoder::new_from_buf(1, buffer)
        });

        values.for_each(|b| rle_encoder.put(b as u64));
        Ok(())
    }
}

#[cfg(feature = "arrow")]
impl FixedLenByteArrayEncoder for RleValueEncoder<FixedLenByteArrayType> {}

// ----------------------------------------------------------------------
// DELTA_BINARY_PACKED encoding

const MAX_PAGE_HEADER_WRITER_SIZE: usize = 32;
const DEFAULT_BIT_WRITER_SIZE: usize = 1024 * 1024;
const DEFAULT_NUM_MINI_BLOCKS: usize = 4;

/// Delta bit packed encoder.
/// Consists of a header followed by blocks of delta encoded values binary packed.
///
/// Delta-binary-packing:
/// ```shell
///   [page-header] [block 1], [block 2], ... [block N]
/// ```
///
/// Each page header consists of:
/// ```shell
///   [block size] [number of miniblocks in a block] [total value count] [first value]
/// ```
///
/// Each block consists of:
/// ```shell
///   [min delta] [list of bitwidths of miniblocks] [miniblocks]
/// ```
///
/// Current implementation writes values in `put` method, multiple calls to `put` to
/// existing block or start new block if block size is exceeded. Calling `flush_buffer`
/// writes out all data and resets internal state, including page header.
///
/// Supports only INT32 and INT64.
pub struct DeltaBitPackEncoder<T: DataType> {
    page_header_writer: BitWriter,
    bit_writer: BitWriter,
    total_values: usize,
    first_value: i64,
    current_value: i64,
    block_size: usize,
    mini_block_size: usize,
    num_mini_blocks: usize,
    values_in_block: usize,
    deltas: Vec<i64>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> Default for DeltaBitPackEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType> DeltaBitPackEncoder<T> {
    /// Creates new delta bit packed encoder.
    pub fn new() -> Self {
        Self::assert_supported_type();

        // Size miniblocks so that they can be efficiently decoded
        let mini_block_size = match T::T::PHYSICAL_TYPE {
            Type::INT32 => 32,
            Type::INT64 => 64,
            _ => unreachable!(),
        };

        let num_mini_blocks = DEFAULT_NUM_MINI_BLOCKS;
        let block_size = mini_block_size * num_mini_blocks;
        assert_eq!(block_size % 128, 0);

        DeltaBitPackEncoder {
            page_header_writer: BitWriter::new(MAX_PAGE_HEADER_WRITER_SIZE),
            bit_writer: BitWriter::new(DEFAULT_BIT_WRITER_SIZE),
            total_values: 0,
            first_value: 0,
            current_value: 0, // current value to keep adding deltas
            block_size,       // can write fewer values than block size for last block
            mini_block_size,
            num_mini_blocks,
            values_in_block: 0, // will be at most block_size
            deltas: vec![0; block_size],
            _phantom: PhantomData,
        }
    }

    /// Writes page header for blocks, this method is invoked when we are done encoding
    /// values. It is also okay to encode when no values have been provided
    fn write_page_header(&mut self) {
        // We ignore the result of each 'put' operation, because
        // MAX_PAGE_HEADER_WRITER_SIZE is chosen to fit all header values and
        // guarantees that writes will not fail.

        // Write the size of each block
        self.page_header_writer.put_vlq_int(self.block_size as u64);
        // Write the number of mini blocks
        self.page_header_writer
            .put_vlq_int(self.num_mini_blocks as u64);
        // Write the number of all values (including non-encoded first value)
        self.page_header_writer
            .put_vlq_int(self.total_values as u64);
        // Write first value
        self.page_header_writer.put_zigzag_vlq_int(self.first_value);
    }

    // Write current delta buffer (<= 'block size' values) into bit writer
    #[inline(never)]
    fn flush_block_values(&mut self) -> Result<()> {
        if self.values_in_block == 0 {
            return Ok(());
        }

        let mut min_delta = i64::MAX;
        for i in 0..self.values_in_block {
            min_delta = cmp::min(min_delta, self.deltas[i]);
        }

        // Write min delta
        self.bit_writer.put_zigzag_vlq_int(min_delta);

        // Slice to store bit width for each mini block
        let offset = self.bit_writer.skip(self.num_mini_blocks);

        for i in 0..self.num_mini_blocks {
            // Find how many values we need to encode - either block size or whatever
            // values left
            let n = cmp::min(self.mini_block_size, self.values_in_block);
            if n == 0 {
                // Decoders should be agnostic to the padding value, we therefore use 0xFF
                // when running tests. However, not all implementations may handle this correctly
                // so pad with 0 when not running tests
                let pad_value = cfg!(test).then(|| 0xFF).unwrap_or(0);
                for j in i..self.num_mini_blocks {
                    self.bit_writer.write_at(offset + j, pad_value);
                }
                break;
            }

            // Compute the max delta in current mini block
            let mut max_delta = i64::MIN;
            for j in 0..n {
                max_delta = cmp::max(max_delta, self.deltas[i * self.mini_block_size + j]);
            }

            // Compute bit width to store (max_delta - min_delta)
            let bit_width = num_required_bits(self.subtract_u64(max_delta, min_delta)) as usize;
            self.bit_writer.write_at(offset + i, bit_width as u8);

            // Encode values in current mini block using min_delta and bit_width
            for j in 0..n {
                let packed_value =
                    self.subtract_u64(self.deltas[i * self.mini_block_size + j], min_delta);
                self.bit_writer.put_value(packed_value, bit_width);
            }

            // Pad the last block (n < mini_block_size)
            for _ in n..self.mini_block_size {
                self.bit_writer.put_value(0, bit_width);
            }

            self.values_in_block -= n;
        }

        assert_eq!(
            self.values_in_block, 0,
            "Expected 0 values in block, found {}",
            self.values_in_block
        );
        Ok(())
    }

    #[inline]
    #[cfg(feature = "arrow")]
    pub(crate) fn put_i64(&mut self, value: i64) -> Result<()> {
        if self.total_values == 0 {
            self.first_value = value;
            self.current_value = value;
            self.total_values = 1;
            return Ok(());
        }

        self.total_values += 1;
        self.deltas[self.values_in_block] = self.subtract(value, self.current_value);
        self.current_value = value;
        self.values_in_block += 1;
        if self.values_in_block == self.block_size {
            self.flush_block_values()?;
        }
        Ok(())
    }

    #[inline]
    fn put_i64_values(&mut self, len: usize, mut value_at: impl FnMut(usize) -> i64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        let mut idx = if self.total_values == 0 {
            self.first_value = value_at(0);
            self.current_value = self.first_value;
            1
        } else {
            0
        };
        self.total_values += len;

        while idx < len {
            let value = value_at(idx);
            self.deltas[self.values_in_block] = self.subtract(value, self.current_value);
            self.current_value = value;
            idx += 1;
            self.values_in_block += 1;
            if self.values_in_block == self.block_size {
                self.flush_block_values()?;
            }
        }
        Ok(())
    }
}

// Implementation is shared between Int32Type and Int64Type,
// see `DeltaBitPackEncoderConversion` below for specifics.
impl<T: DataType> Encoder<T> for DeltaBitPackEncoder<T> {
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        self.put_i64_values(values.len(), |idx| {
            values[idx].as_i64().expect(DELTA_BIT_PACK_TYPE_ERROR)
        })
    }

    // Performance Note:
    // As far as can be seen these functions are rarely called and as such we can hint to the
    // compiler that they dont need to be folded into hot locations in the final output.
    #[cold]
    fn encoding(&self) -> Encoding {
        Encoding::DELTA_BINARY_PACKED
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.bit_writer.bytes_written()
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        // Write remaining values
        self.flush_block_values()?;
        // Write page header with total values
        self.write_page_header();

        let mut buffer = Vec::new();
        buffer.extend_from_slice(self.page_header_writer.flush_buffer());
        buffer.extend_from_slice(self.bit_writer.flush_buffer());

        // Reset state
        self.page_header_writer.clear();
        self.bit_writer.clear();
        self.total_values = 0;
        self.first_value = 0;
        self.current_value = 0;
        self.values_in_block = 0;

        Ok(buffer.into())
    }

    /// return the estimated memory size of this encoder.
    fn estimated_memory_size(&self) -> usize {
        self.page_header_writer.estimated_memory_size()
            + self.bit_writer.estimated_memory_size()
            + self.deltas.capacity() * std::mem::size_of::<i64>()
            + std::mem::size_of::<Self>()
    }
}

#[cfg(feature = "arrow")]
impl BoolEncoder for DeltaBitPackEncoder<BoolType> {}
#[cfg(feature = "arrow")]
impl FixedLenByteArrayEncoder for DeltaBitPackEncoder<FixedLenByteArrayType> {}

/// Helper trait to define specific conversions and subtractions when computing deltas
trait DeltaBitPackEncoderConversion<T: DataType> {
    // Method should panic if type is not supported, otherwise no-op
    fn assert_supported_type();

    fn subtract(&self, left: i64, right: i64) -> i64;

    fn subtract_u64(&self, left: i64, right: i64) -> u64;
}

const DELTA_BIT_PACK_TYPE_ERROR: &str =
    "DeltaBitPackDecoder only supports Int32Type, UInt32Type, Int64Type, and UInt64Type";

impl<T: DataType> DeltaBitPackEncoderConversion<T> for DeltaBitPackEncoder<T> {
    #[inline]
    fn assert_supported_type() {
        ensure_phys_ty!(Type::INT32 | Type::INT64, "{}", DELTA_BIT_PACK_TYPE_ERROR);
    }

    #[inline]
    fn subtract(&self, left: i64, right: i64) -> i64 {
        // It is okay for values to overflow, wrapping_sub wrapping around at the boundary
        match T::get_physical_type() {
            Type::INT32 => (left as i32).wrapping_sub(right as i32) as i64,
            Type::INT64 => left.wrapping_sub(right),
            _ => panic!("{}", DELTA_BIT_PACK_TYPE_ERROR),
        }
    }

    #[inline]
    fn subtract_u64(&self, left: i64, right: i64) -> u64 {
        match T::get_physical_type() {
            // Conversion of i32 -> u32 -> u64 is to avoid non-zero left most bytes in int repr
            Type::INT32 => (left as i32).wrapping_sub(right as i32) as u32 as u64,
            Type::INT64 => left.wrapping_sub(right) as u64,
            _ => panic!("{}", DELTA_BIT_PACK_TYPE_ERROR),
        }
    }
}

// ----------------------------------------------------------------------
// DELTA_LENGTH_BYTE_ARRAY encoding

/// Encoding for byte arrays to separate the length values and the data.
/// The lengths are encoded using DELTA_BINARY_PACKED encoding, data is
/// stored as raw bytes.
pub struct DeltaLengthByteArrayEncoder<T: DataType> {
    // length encoder
    len_encoder: DeltaBitPackEncoder<Int32Type>,
    // concatenated value bytes, appended directly (no per-value `ByteArray`
    // allocation); the lengths in `len_encoder` delimit them.
    data: Vec<u8>,
    // data size in bytes of encoded values
    encoded_size: usize,
    _phantom: PhantomData<T>,
}

impl<T: DataType> Default for DeltaLengthByteArrayEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType> DeltaLengthByteArrayEncoder<T> {
    /// Creates new delta length byte array encoder.
    pub fn new() -> Self {
        Self {
            len_encoder: DeltaBitPackEncoder::new(),
            data: vec![],
            encoded_size: 0,
            _phantom: PhantomData,
        }
    }

    /// Append a batch of byte slices already borrowed from a source buffer:
    /// feed their lengths to the length encoder, then copy the bytes into the
    /// contiguous data buffer — with no per-value `ByteArray` allocation. Used
    /// by the fixed-width DELTA_BYTE_ARRAY suffix path.
    #[cfg(feature = "arrow")]
    fn put_byte_slices(&mut self, slices: &[&[u8]]) -> Result<()> {
        let lengths: Vec<i32> = slices.iter().map(|s| s.len() as i32).collect();
        self.len_encoder.put(&lengths)?;
        for s in slices {
            self.encoded_size += s.len();
            self.data.extend_from_slice(s);
        }
        Ok(())
    }
}

impl<T: DataType> Encoder<T> for DeltaLengthByteArrayEncoder<T> {
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        ensure_phys_ty!(
            Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY,
            "DeltaLengthByteArrayEncoder only supports ByteArrayType"
        );

        let val_it = || {
            values
                .iter()
                .map(|x| x.as_any().downcast_ref::<ByteArray>().unwrap())
        };

        let lengths: Vec<i32> = val_it().map(|byte_array| byte_array.len() as i32).collect();
        self.len_encoder.put(&lengths)?;
        for byte_array in val_it() {
            self.encoded_size += byte_array.len();
            self.data.extend_from_slice(byte_array.data());
        }

        Ok(())
    }

    // Performance Note:
    // As far as can be seen these functions are rarely called and as such we can hint to the
    // compiler that they dont need to be folded into hot locations in the final output.
    #[cold]
    fn encoding(&self) -> Encoding {
        Encoding::DELTA_LENGTH_BYTE_ARRAY
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.len_encoder.estimated_data_encoded_size() + self.encoded_size
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        ensure_phys_ty!(
            Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY,
            "DeltaLengthByteArrayEncoder only supports ByteArrayType"
        );

        let mut total_bytes = vec![];
        let lengths = self.len_encoder.flush_buffer()?;
        total_bytes.extend_from_slice(&lengths);
        total_bytes.extend_from_slice(&self.data);
        self.data.clear();
        self.encoded_size = 0;

        Ok(total_bytes.into())
    }

    /// return the estimated memory size of this encoder.
    fn estimated_memory_size(&self) -> usize {
        self.len_encoder.estimated_memory_size() + self.data.len() + std::mem::size_of::<Self>()
    }
}

#[cfg(feature = "arrow")]
impl BoolEncoder for DeltaLengthByteArrayEncoder<BoolType> {}
#[cfg(feature = "arrow")]
impl FixedLenByteArrayEncoder for DeltaLengthByteArrayEncoder<FixedLenByteArrayType> {
    /// Native bulk DELTA_LENGTH_BYTE_ARRAY for fixed-width values: collect all
    /// lengths and feed the length encoder once, then append each value's bytes
    /// straight from the Arrow buffer — no per-value `Encoder::put` round-trip.
    fn put_fixed_len_byte_array(&mut self, values: FixedLenByteArrayValues<'_>) -> Result<()> {
        let lengths: Vec<i32> = values.iter().map(|v| v.len() as i32).collect();
        self.len_encoder.put(&lengths)?;
        for value in values.iter() {
            self.encoded_size += value.len();
            self.data.extend_from_slice(value);
        }
        Ok(())
    }

    /// Record one value's length and append its bytes.
    #[cfg(feature = "arrow")]
    #[inline]
    fn append_fixed_len_value(&mut self, value: &[u8]) -> Result<()> {
        self.len_encoder.put(&[value.len() as i32])?;
        self.encoded_size += value.len();
        self.data.extend_from_slice(value);
        Ok(())
    }
}

// ----------------------------------------------------------------------
// DELTA_BYTE_ARRAY encoding

/// Encoding for byte arrays, prefix lengths are encoded using DELTA_BINARY_PACKED
/// encoding, followed by suffixes with DELTA_LENGTH_BYTE_ARRAY encoding.
pub struct DeltaByteArrayEncoder<T: DataType> {
    prefix_len_encoder: DeltaBitPackEncoder<Int32Type>,
    suffix_writer: DeltaLengthByteArrayEncoder<ByteArrayType>,
    previous: Vec<u8>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> Default for DeltaByteArrayEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType> DeltaByteArrayEncoder<T> {
    /// Creates new delta byte array encoder.
    pub fn new() -> Self {
        Self {
            prefix_len_encoder: DeltaBitPackEncoder::new(),
            suffix_writer: DeltaLengthByteArrayEncoder::new(),
            previous: vec![],
            _phantom: PhantomData,
        }
    }
}

/// Length of the byte prefix shared by `previous` and `current` — the
/// DELTA_BYTE_ARRAY front-coding match length.
#[inline]
fn common_prefix_len(previous: &[u8], current: &[u8]) -> usize {
    let max = cmp::min(previous.len(), current.len());
    let mut n = 0;
    while n < max && previous[n] == current[n] {
        n += 1;
    }
    n
}

impl<T: DataType> Encoder<T> for DeltaByteArrayEncoder<T> {
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        let mut prefix_lengths: Vec<i32> = vec![];
        let mut suffixes: Vec<ByteArray> = vec![];

        let values = values
            .iter()
            .map(|x| x.as_any())
            .map(|x| match T::get_physical_type() {
                Type::BYTE_ARRAY => x.downcast_ref::<ByteArray>().unwrap(),
                Type::FIXED_LEN_BYTE_ARRAY => x.downcast_ref::<FixedLenByteArray>().unwrap(),
                _ => panic!(
                    "DeltaByteArrayEncoder only supports ByteArrayType and FixedLenByteArrayType"
                ),
            });

        for byte_array in values {
            let current = byte_array.data();
            let match_len = common_prefix_len(&self.previous, current);
            prefix_lengths.push(match_len as i32);
            suffixes.push(byte_array.slice(match_len, byte_array.len() - match_len));
            // Update previous for the next prefix
            self.previous.clear();
            self.previous.extend_from_slice(current);
        }
        self.prefix_len_encoder.put(&prefix_lengths)?;
        self.suffix_writer.put(&suffixes)?;

        Ok(())
    }

    // Performance Note:
    // As far as can be seen these functions are rarely called and as such we can hint to the
    // compiler that they dont need to be folded into hot locations in the final output.
    #[cold]
    fn encoding(&self) -> Encoding {
        Encoding::DELTA_BYTE_ARRAY
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.prefix_len_encoder.estimated_data_encoded_size()
            + self.suffix_writer.estimated_data_encoded_size()
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        match T::get_physical_type() {
            Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY => {
                // TODO: investigate if we can merge lengths and suffixes
                // without copying data into new vector.
                let mut total_bytes = vec![];
                // Insert lengths ...
                let lengths = self.prefix_len_encoder.flush_buffer()?;
                total_bytes.extend_from_slice(&lengths);
                // ... followed by suffixes
                let suffixes = self.suffix_writer.flush_buffer()?;
                total_bytes.extend_from_slice(&suffixes);

                self.previous.clear();
                Ok(total_bytes.into())
            }
            _ => panic!(
                "DeltaByteArrayEncoder only supports ByteArrayType and FixedLenByteArrayType"
            ),
        }
    }

    /// return the estimated memory size of this encoder.
    fn estimated_memory_size(&self) -> usize {
        self.prefix_len_encoder.estimated_memory_size()
            + self.suffix_writer.estimated_memory_size()
            + (self.previous.capacity() * std::mem::size_of::<u8>())
    }
}

#[cfg(feature = "arrow")]
impl BoolEncoder for DeltaByteArrayEncoder<BoolType> {}
#[cfg(feature = "arrow")]
impl FixedLenByteArrayEncoder for DeltaByteArrayEncoder<FixedLenByteArrayType> {
    /// Front-code a dense fixed-width run directly from the Arrow buffer.
    fn put_fixed_len_byte_array(&mut self, values: FixedLenByteArrayValues<'_>) -> Result<()> {
        let mut prefix_lengths: Vec<i32> = Vec::with_capacity(values.len());
        // Suffixes are borrowed directly from the Arrow buffer — no per-value
        // `ByteArray`/`to_vec` materialization; `put_byte_slices` copies them
        // once into the suffix writer's contiguous buffer.
        let mut suffixes: Vec<&[u8]> = Vec::with_capacity(values.len());
        for current in values.iter() {
            let match_len = common_prefix_len(&self.previous, current);
            prefix_lengths.push(match_len as i32);
            suffixes.push(&current[match_len..]);
            self.previous.clear();
            self.previous.extend_from_slice(current);
        }
        self.prefix_len_encoder.put(&prefix_lengths)?;
        self.suffix_writer.put_byte_slices(&suffixes)?;
        Ok(())
    }

    /// Front-code one streamed fixed-width value.
    #[cfg(feature = "arrow")]
    #[inline]
    fn append_fixed_len_value(&mut self, value: &[u8]) -> Result<()> {
        let match_len = common_prefix_len(&self.previous, value);
        self.prefix_len_encoder.put(&[match_len as i32])?;
        self.suffix_writer.put_byte_slices(&[&value[match_len..]])?;
        self.previous.clear();
        self.previous.extend_from_slice(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::encodings::decoding::{Decoder, DictDecoder, PlainDecoder, get_decoder};
    use crate::schema::types::{ColumnDescPtr, ColumnDescriptor, ColumnPath, Type as SchemaType};
    use crate::util::bit_util;
    use crate::util::test_common::rand_gen::{RandGen, random_bytes};

    const TEST_SET_SIZE: usize = 1024;

    /// A [`ChunkSink`] that simply collects every value handed to it, so a test
    /// can assert which values a [`ValueStream`] selects/converts via `write_into`
    /// and in what order.
    #[cfg(feature = "arrow")]
    #[derive(Default)]
    struct CollectSink<T> {
        values: Vec<T>,
    }

    #[cfg(feature = "arrow")]
    impl<T: Clone> ChunkSink<[T]> for CollectSink<T> {
        fn consume(&mut self, chunk: &[T]) -> Result<()> {
            self.values.extend_from_slice(chunk);
            Ok(())
        }
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn write_into_int32_dense_pushes_whole_slice() {
        let mut sink = CollectSink::<i32>::default();
        let values = [1, 2, 3, 4];

        Int32Values::i32(&values, ValueIndices::Dense { offset: 1, len: 2 })
            .write_into(&mut sink)
            .unwrap();

        assert_eq!(sink.values, [2, 3]);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn write_into_int32_gathers_converted_values() {
        let mut sink = CollectSink::<i32>::default();
        let values = [1i8, 2, 3, 4];

        Int32Values::i8(&values, ValueIndices::Dense { offset: 1, len: 2 })
            .write_into(&mut sink)
            .unwrap();

        assert_eq!(sink.values, [2, 3]);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn write_into_int64_dense_pushes_whole_slice() {
        let mut sink = CollectSink::<i64>::default();
        let values = [1, 2, 3, 4];

        Int64Values::i64(&values, ValueIndices::Dense { offset: 1, len: 2 })
            .write_into(&mut sink)
            .unwrap();

        assert_eq!(sink.values, [2, 3]);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn write_into_int64_gathers_converted_values() {
        let mut sink = CollectSink::<i64>::default();
        let values = [1i128, 2, 3, 4];

        Int64Values::i128_cast(&values, ValueIndices::Dense { offset: 1, len: 2 })
            .write_into(&mut sink)
            .unwrap();

        assert_eq!(sink.values, [2, 3]);
    }

    #[test]
    fn test_get_encoders() {
        // supported encodings
        create_and_check_encoder::<Int32Type>(0, Encoding::PLAIN, None);
        create_and_check_encoder::<Int32Type>(0, Encoding::DELTA_BINARY_PACKED, None);
        create_and_check_encoder::<Int32Type>(0, Encoding::DELTA_LENGTH_BYTE_ARRAY, None);
        create_and_check_encoder::<Int32Type>(0, Encoding::DELTA_BYTE_ARRAY, None);
        create_and_check_encoder::<BoolType>(0, Encoding::RLE, None);

        // error when initializing
        create_and_check_encoder::<Int32Type>(
            0,
            Encoding::RLE_DICTIONARY,
            Some(general_err!(
                "Cannot initialize this encoding through this function"
            )),
        );
        create_and_check_encoder::<Int32Type>(
            0,
            Encoding::PLAIN_DICTIONARY,
            Some(general_err!(
                "Cannot initialize this encoding through this function"
            )),
        );

        // unsupported
        #[allow(deprecated)]
        create_and_check_encoder::<Int32Type>(
            0,
            Encoding::BIT_PACKED,
            Some(nyi_err!("Encoding BIT_PACKED is not supported")),
        );
    }

    #[test]
    fn test_bool() {
        BoolType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        BoolType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        BoolType::test(Encoding::RLE, TEST_SET_SIZE, -1);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_plain_encoder_sparse_packed_bool() {
        let input = [
            0b1010_1101,
            0b0111_0010,
            0b1100_1001,
            0b0011_1110,
            0b0101_0101,
            0b1000_1111,
            0b1111_0000,
            0b0001_1011,
            0b1011_0110,
            0b0100_1001,
            0b1110_0011,
            0b0010_1100,
            0b1001_0111,
            0b0110_1010,
        ];
        let indices: Vec<_> = (0..93).filter(|idx| idx % 3 != 1).collect();

        let mut encoder = PlainEncoder::<BoolType>::new();
        encoder
            .put_packed_bool(PackedBoolValues::new_sparse(&input, 5, &indices))
            .unwrap();
        let encoded = encoder.flush_buffer().unwrap();

        assert_eq!(encoded.len(), indices.len().div_ceil(8));
        for (out_idx, &input_idx) in indices.iter().enumerate() {
            assert_eq!(
                bit_util::get_bit(&encoded, out_idx),
                bit_util::get_bit(&input, 5 + input_idx),
                "mismatch at output bit {out_idx}"
            );
        }
    }

    #[test]
    fn test_i32() {
        Int32Type::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        Int32Type::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        Int32Type::test(Encoding::DELTA_BINARY_PACKED, TEST_SET_SIZE, -1);
        Int32Type::test(Encoding::BYTE_STREAM_SPLIT, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_i64() {
        Int64Type::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        Int64Type::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        Int64Type::test(Encoding::DELTA_BINARY_PACKED, TEST_SET_SIZE, -1);
        Int64Type::test(Encoding::BYTE_STREAM_SPLIT, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_i96() {
        Int96Type::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        Int96Type::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_float() {
        FloatType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        FloatType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        FloatType::test(Encoding::BYTE_STREAM_SPLIT, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_double() {
        DoubleType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        DoubleType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        DoubleType::test(Encoding::BYTE_STREAM_SPLIT, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_byte_array() {
        ByteArrayType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        ByteArrayType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        ByteArrayType::test(Encoding::DELTA_LENGTH_BYTE_ARRAY, TEST_SET_SIZE, -1);
        ByteArrayType::test(Encoding::DELTA_BYTE_ARRAY, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_fixed_len_byte_array() {
        FixedLenByteArrayType::test(Encoding::PLAIN, TEST_SET_SIZE, 100);
        FixedLenByteArrayType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, 100);
        FixedLenByteArrayType::test(Encoding::DELTA_BYTE_ARRAY, TEST_SET_SIZE, 100);
        FixedLenByteArrayType::test(Encoding::BYTE_STREAM_SPLIT, TEST_SET_SIZE, 100);
    }

    #[test]
    fn test_dict_encoded_size() {
        fn run_test<T: DataType>(type_length: i32, values: &[T::T], expected_size: usize) {
            let mut encoder = create_test_dict_encoder::<T>(type_length);
            assert_eq!(encoder.dict_encoded_size(), 0);
            encoder.put(values).unwrap();
            assert_eq!(encoder.dict_encoded_size(), expected_size);
            // We do not reset encoded size of the dictionary keys after flush_buffer
            encoder.flush_buffer().unwrap();
            assert_eq!(encoder.dict_encoded_size(), expected_size);
        }

        // Only 2 variations of values 1 byte each
        run_test::<BoolType>(-1, &[true, false, true, false, true], 2);
        run_test::<Int32Type>(-1, &[1i32, 2i32, 3i32, 4i32, 5i32], 20);
        run_test::<Int64Type>(-1, &[1i64, 2i64, 3i64, 4i64, 5i64], 40);
        run_test::<FloatType>(-1, &[1f32, 2f32, 3f32, 4f32, 5f32], 20);
        run_test::<DoubleType>(-1, &[1f64, 2f64, 3f64, 4f64, 5f64], 40);
        // Int96: len + reference
        run_test::<Int96Type>(
            -1,
            &[Int96::from(vec![1, 2, 3]), Int96::from(vec![2, 3, 4])],
            24,
        );
        run_test::<ByteArrayType>(-1, &[ByteArray::from("abcd"), ByteArray::from("efj")], 15);
        run_test::<FixedLenByteArrayType>(
            2,
            &[ByteArray::from("ab").into(), ByteArray::from("bc").into()],
            4,
        );
    }

    #[test]
    fn test_estimated_data_encoded_size() {
        fn run_test<T: DataType>(
            encoding: Encoding,
            type_length: i32,
            values: &[T::T],
            initial_size: usize,
            max_size: usize,
            flush_size: usize,
        ) {
            let mut encoder = match encoding {
                Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY => {
                    Box::new(create_test_dict_encoder::<T>(type_length))
                }
                _ => create_test_encoder::<T>(type_length, encoding),
            };
            assert_eq!(encoder.estimated_data_encoded_size(), initial_size);

            encoder.put(values).unwrap();
            assert_eq!(encoder.estimated_data_encoded_size(), max_size);

            encoder.flush_buffer().unwrap();
            assert_eq!(encoder.estimated_data_encoded_size(), flush_size);
        }

        // PLAIN
        run_test::<Int32Type>(Encoding::PLAIN, -1, &[123; 1024], 0, 4096, 0);

        // DICTIONARY
        // NOTE: The final size is almost the same because the dictionary entries are
        // preserved after encoded values have been written.
        run_test::<Int32Type>(Encoding::RLE_DICTIONARY, -1, &[123, 1024], 0, 2, 0);

        // DELTA_BINARY_PACKED
        run_test::<Int32Type>(Encoding::DELTA_BINARY_PACKED, -1, &[123; 1024], 0, 35, 0);

        // RLE
        let mut values = vec![];
        values.extend_from_slice(&[true; 16]);
        values.extend_from_slice(&[false; 16]);
        run_test::<BoolType>(Encoding::RLE, -1, &values, 0, 6, 0);

        // DELTA_LENGTH_BYTE_ARRAY
        run_test::<ByteArrayType>(
            Encoding::DELTA_LENGTH_BYTE_ARRAY,
            -1,
            &[ByteArray::from("ab"), ByteArray::from("abc")],
            0,
            5, // only value bytes, length encoder is not flushed yet
            0,
        );

        // DELTA_BYTE_ARRAY
        run_test::<ByteArrayType>(
            Encoding::DELTA_BYTE_ARRAY,
            -1,
            &[ByteArray::from("ab"), ByteArray::from("abc")],
            0,
            3, // only suffix bytes, length encoder is not flushed yet
            0,
        );

        // BYTE_STREAM_SPLIT
        run_test::<FloatType>(Encoding::BYTE_STREAM_SPLIT, -1, &[0.1, 0.2], 0, 8, 0);
    }

    #[test]
    fn test_byte_stream_split_example_f32() {
        // Test data from https://github.com/apache/parquet-format/blob/2a481fe1aad64ff770e21734533bb7ef5a057dac/Encodings.md#byte-stream-split-byte_stream_split--9
        let mut encoder = create_test_encoder::<FloatType>(0, Encoding::BYTE_STREAM_SPLIT);
        let mut decoder = create_test_decoder::<FloatType>(0, Encoding::BYTE_STREAM_SPLIT);

        let input = vec![
            f32::from_le_bytes([0xAA, 0xBB, 0xCC, 0xDD]),
            f32::from_le_bytes([0x00, 0x11, 0x22, 0x33]),
            f32::from_le_bytes([0xA3, 0xB4, 0xC5, 0xD6]),
        ];

        encoder.put(&input).unwrap();
        let encoded = encoder.flush_buffer().unwrap();

        assert_eq!(
            encoded,
            Bytes::from(vec![
                0xAA_u8, 0x00, 0xA3, 0xBB, 0x11, 0xB4, 0xCC, 0x22, 0xC5, 0xDD, 0x33, 0xD6
            ])
        );

        let mut decoded = vec![0.0; input.len()];
        decoder.set_data(encoded, input.len()).unwrap();
        decoder.get(&mut decoded).unwrap();

        assert_eq!(decoded, input);
    }

    // See: https://github.com/sunchao/parquet-rs/issues/47
    #[test]
    fn test_issue_47() {
        let mut encoder = create_test_encoder::<ByteArrayType>(0, Encoding::DELTA_BYTE_ARRAY);
        let mut decoder = create_test_decoder::<ByteArrayType>(0, Encoding::DELTA_BYTE_ARRAY);

        let input = vec![
            ByteArray::from("aa"),
            ByteArray::from("aaa"),
            ByteArray::from("aa"),
            ByteArray::from("aaa"),
        ];

        let mut output = vec![ByteArray::default(); input.len()];

        let mut result = put_and_get(&mut encoder, &mut decoder, &input[..2], &mut output[..2]);
        assert!(
            result.is_ok(),
            "first put_and_get() failed with: {}",
            result.unwrap_err()
        );
        result = put_and_get(&mut encoder, &mut decoder, &input[2..], &mut output[2..]);
        assert!(
            result.is_ok(),
            "second put_and_get() failed with: {}",
            result.unwrap_err()
        );
        assert_eq!(output, input);
    }

    trait EncodingTester<T: DataType> {
        fn test(enc: Encoding, total: usize, type_length: i32) {
            let result = match enc {
                Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY => {
                    Self::test_dict_internal(total, type_length)
                }
                enc => Self::test_internal(enc, total, type_length),
            };

            assert!(
                result.is_ok(),
                "Expected result to be OK but got err:\n {}",
                result.unwrap_err()
            );
        }

        fn test_internal(enc: Encoding, total: usize, type_length: i32) -> Result<()>;

        fn test_dict_internal(total: usize, type_length: i32) -> Result<()>;
    }

    impl<T: DataType + RandGen<T>> EncodingTester<T> for T {
        fn test_internal(enc: Encoding, total: usize, type_length: i32) -> Result<()> {
            let mut encoder = create_test_encoder::<T>(type_length, enc);
            let mut decoder = create_test_decoder::<T>(type_length, enc);
            let mut values = <T as RandGen<T>>::gen_vec(type_length, total);
            let mut result_data = vec![T::T::default(); total];

            // Test put/get spaced.
            let num_bytes = bit_util::ceil(total as i64, 8);
            let valid_bits = random_bytes(num_bytes as usize);
            let values_written = encoder.put_spaced(&values[..], &valid_bits[..])?;
            let data = encoder.flush_buffer()?;
            decoder.set_data(data, values_written)?;
            let _ = decoder.get_spaced(
                &mut result_data[..],
                values.len() - values_written,
                &valid_bits[..],
            )?;

            // Check equality
            for i in 0..total {
                if bit_util::get_bit(&valid_bits[..], i) {
                    assert_eq!(result_data[i], values[i]);
                } else {
                    assert_eq!(result_data[i], T::T::default());
                }
            }

            let mut actual_total = put_and_get(
                &mut encoder,
                &mut decoder,
                &values[..],
                &mut result_data[..],
            )?;
            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            // Encode more data after flush and test with decoder

            values = <T as RandGen<T>>::gen_vec(type_length, total);
            actual_total = put_and_get(
                &mut encoder,
                &mut decoder,
                &values[..],
                &mut result_data[..],
            )?;
            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            Ok(())
        }

        fn test_dict_internal(total: usize, type_length: i32) -> Result<()> {
            let mut encoder = create_test_dict_encoder::<T>(type_length);
            let mut values = <T as RandGen<T>>::gen_vec(type_length, total);
            encoder.put(&values[..])?;

            let mut data = encoder.flush_buffer()?;
            let mut decoder = create_test_dict_decoder::<T>();
            let mut dict_decoder = PlainDecoder::<T>::new(type_length);
            dict_decoder.set_data(encoder.write_dict()?, encoder.num_entries())?;
            decoder.set_dict(Box::new(dict_decoder))?;
            let mut result_data = vec![T::T::default(); total];
            decoder.set_data(data, total)?;
            let mut actual_total = decoder.get(&mut result_data)?;

            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            // Encode more data after flush and test with decoder

            values = <T as RandGen<T>>::gen_vec(type_length, total);
            encoder.put(&values[..])?;
            data = encoder.flush_buffer()?;

            let mut dict_decoder = PlainDecoder::<T>::new(type_length);
            dict_decoder.set_data(encoder.write_dict()?, encoder.num_entries())?;
            decoder.set_dict(Box::new(dict_decoder))?;
            decoder.set_data(data, total)?;
            actual_total = decoder.get(&mut result_data)?;

            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            Ok(())
        }
    }

    fn put_and_get<T: DataType>(
        encoder: &mut Box<dyn Encoder<T>>,
        decoder: &mut Box<dyn Decoder<T>>,
        input: &[T::T],
        output: &mut [T::T],
    ) -> Result<usize> {
        encoder.put(input)?;
        let data = encoder.flush_buffer()?;
        decoder.set_data(data, input.len())?;
        decoder.get(output)
    }

    fn create_and_check_encoder<T: DataType>(
        type_length: i32,
        encoding: Encoding,
        err: Option<ParquetError>,
    ) {
        let desc = create_test_col_desc_ptr(type_length, T::get_physical_type());
        let encoder = get_encoder::<T>(encoding, &desc);
        match err {
            Some(parquet_error) => {
                assert_eq!(
                    encoder.err().unwrap().to_string(),
                    parquet_error.to_string()
                )
            }
            None => assert_eq!(encoder.unwrap().encoding(), encoding),
        }
    }

    // Creates test column descriptor.
    fn create_test_col_desc_ptr(type_len: i32, t: Type) -> ColumnDescPtr {
        let ty = SchemaType::primitive_type_builder("t", t)
            .with_length(type_len)
            .build()
            .unwrap();
        Arc::new(ColumnDescriptor::new(
            Arc::new(ty),
            0,
            0,
            ColumnPath::new(vec![]),
        ))
    }

    fn create_test_encoder<T: DataType>(type_len: i32, enc: Encoding) -> Box<dyn Encoder<T>> {
        let desc = create_test_col_desc_ptr(type_len, T::get_physical_type());
        get_encoder(enc, &desc).unwrap()
    }

    fn create_test_decoder<T: DataType>(type_len: i32, enc: Encoding) -> Box<dyn Decoder<T>> {
        let desc = create_test_col_desc_ptr(type_len, T::get_physical_type());
        get_decoder(desc, enc).unwrap()
    }

    fn create_test_dict_encoder<T: DataType>(type_len: i32) -> DictEncoder<T> {
        let desc = create_test_col_desc_ptr(type_len, T::get_physical_type());
        DictEncoder::<T>::new(desc)
    }

    fn create_test_dict_decoder<T: DataType>() -> DictDecoder<T> {
        DictDecoder::<T>::new()
    }
}
