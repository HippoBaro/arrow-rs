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

// ----------------------------------------------------------------------
// Dictionary encoding

use bytes::Bytes;

use crate::basic::{Encoding, Type};
use crate::data_type::DataType;
use crate::data_type::private::ParquetValueType;
use crate::encodings::encoding::{Encoder, PlainEncoder};
use crate::encodings::rle::RleEncoder;
use crate::errors::Result;
use crate::schema::types::ColumnDescPtr;
use crate::util::bit_util::num_required_bits;
use crate::util::interner::{Interner, Storage};
#[cfg(feature = "arrow")]
use arrow_array::ArrayRef;
#[cfg(feature = "arrow")]
use std::sync::Arc;

#[derive(Debug)]
struct KeyStorage<T: DataType> {
    uniques: Vec<T::T>,

    /// size of unique values (keys) in the dictionary, in bytes.
    size_in_bytes: usize,

    type_length: usize,
}

impl<T: DataType> Storage for KeyStorage<T> {
    type Key = u64;
    type Value = T::T;

    fn get(&self, idx: Self::Key) -> &Self::Value {
        &self.uniques[idx as usize]
    }

    fn push(&mut self, value: &Self::Value) -> Self::Key {
        let (base_size, num_elements) = value.dict_encoding_size();

        let unique_size = match T::get_physical_type() {
            Type::BYTE_ARRAY => base_size + num_elements,
            Type::FIXED_LEN_BYTE_ARRAY => self.type_length,
            _ => base_size,
        };
        self.size_in_bytes += unique_size;

        let key = self.uniques.len() as u64;
        self.uniques.push(value.clone());
        key
    }

    fn estimated_memory_size(&self) -> usize {
        let uniques_heap_bytes = match T::get_physical_type() {
            Type::FIXED_LEN_BYTE_ARRAY => self.type_length * self.uniques.len(),
            _ => <Self::Value as ParquetValueType>::variable_length_bytes(&self.uniques)
                .unwrap_or(0) as usize,
        };
        self.uniques.capacity() * std::mem::size_of::<T::T>() + uniques_heap_bytes
    }
}

/// Dictionary encoder.
/// The dictionary encoding builds a dictionary of values encountered in a given column.
/// The dictionary page is written first, before the data pages of the column chunk.
///
/// Dictionary page format: the entries in the dictionary - in dictionary order -
/// using the plain encoding.
///
/// Data page format: the bit width used to encode the entry ids stored as 1 byte
/// (max bit width = 32), followed by the values encoded using RLE/Bit packed described
/// above (with the given bit width).
pub struct DictEncoder<T: DataType> {
    interner: Interner<KeyStorage<T>>,

    /// The buffered indices
    indices: Vec<u64>,

    /// Tracks the Arrow zero-hash dictionary adopt fast path (see
    /// [`Self::maybe_adopt_dictionary`]).
    #[cfg(feature = "arrow")]
    adopt: AdoptState,
}

/// State of the Arrow zero-hash dictionary adopt fast path.
#[cfg(feature = "arrow")]
enum AdoptState {
    /// No values yet; the next Arrow dictionary can be adopted verbatim.
    Fresh,
    /// Built by adopting one Arrow dictionary verbatim. The `ArrayRef` is held
    /// to keep that allocation alive, so `Arc::ptr_eq` identity checks against
    /// later batches stay sound (a freed buffer could otherwise be reused at
    /// the same address).
    Adopted(ArrayRef),
    /// Interning mode (dedup hash table active); never returns to adopting.
    Legacy,
}

impl<T: DataType> DictEncoder<T> {
    /// Creates new dictionary encoder.
    pub fn new(desc: ColumnDescPtr) -> Self {
        let storage = KeyStorage {
            uniques: vec![],
            size_in_bytes: 0,
            type_length: desc.type_length() as usize,
        };

        Self {
            interner: Interner::new(storage),
            indices: vec![],
            #[cfg(feature = "arrow")]
            adopt: AdoptState::Fresh,
        }
    }

    /// Returns true if dictionary entries are sorted, false otherwise.
    pub fn is_sorted(&self) -> bool {
        // Sorting is not supported currently.
        false
    }

    /// Returns number of unique values (keys) in the dictionary.
    pub fn num_entries(&self) -> usize {
        self.interner.storage().uniques.len()
    }

    /// Returns size of unique values (keys) in the dictionary, in bytes.
    pub fn dict_encoded_size(&self) -> usize {
        self.interner.storage().size_in_bytes
    }

    /// Writes out the dictionary values with PLAIN encoding in a byte buffer, and return
    /// the result.
    pub fn write_dict(&self) -> Result<Bytes> {
        let mut plain_encoder = PlainEncoder::<T>::new();
        plain_encoder.put(&self.interner.storage().uniques)?;
        plain_encoder.flush_buffer()
    }

    /// Writes out the dictionary values with RLE encoding in a byte buffer, and return
    /// the result.
    pub fn write_indices(&mut self) -> Result<Bytes> {
        let buffer_len = self.estimated_data_encoded_size();
        let mut buffer = Vec::with_capacity(buffer_len);
        buffer.push(self.bit_width());

        // Write bit width in the first byte
        let mut encoder = RleEncoder::new_from_buf(self.bit_width(), buffer);
        for index in &self.indices {
            encoder.put(*index)
        }
        self.indices.clear();
        Ok(encoder.consume().into())
    }

    fn put_one(&mut self, value: &T::T) {
        self.indices.push(self.interner.intern(value));
    }

    /// Try to adopt an Arrow dictionary's `values` (already deduplicated and in
    /// key order) as the parquet dictionary page verbatim, so Arrow key `k`
    /// maps to parquet index `k` with no hashing and a sequential copy.
    ///
    /// `identity` is the values array; its `Arc` identity recognises the same
    /// dictionary across mini-batches (and is held to keep the allocation alive
    /// so the comparison is sound).
    ///
    /// Returns `true` when the caller may append this dictionary's keys
    /// directly: either it is the first dictionary (adopted now) or the very
    /// same array adopted earlier. Returns `false` when a *different*
    /// dictionary arrives — the caller must then call
    /// [`Self::seed_dedup_from_storage`] and fall back to interning.
    #[cfg(feature = "arrow")]
    pub fn maybe_adopt_dictionary(&mut self, values: &[T::T], identity: &ArrayRef) -> bool {
        match &self.adopt {
            AdoptState::Legacy => return false,
            AdoptState::Adopted(adopted) => return Arc::ptr_eq(adopted, identity),
            AdoptState::Fresh => {}
        }

        // Adopt verbatim only when the encoder is empty. A prior plain
        // (interned) write to the same column populates the dictionary while
        // leaving the state `Fresh`; adopting then would map Arrow key `k` to
        // the wrong parquet index. Switch to interning instead — its dedup
        // table is already populated by that prior write.
        if self.num_entries() != 0 {
            self.adopt = AdoptState::Legacy;
            return false;
        }

        for value in values {
            self.interner.push_value(value);
        }
        self.adopt = AdoptState::Adopted(identity.clone());
        true
    }

    /// Leave adopt mode: rebuild the dedup table from the values already in
    /// storage so subsequent interning deduplicates against them instead of
    /// duplicating them. Idempotent.
    #[cfg(feature = "arrow")]
    pub fn seed_dedup_from_storage(&mut self) {
        if matches!(self.adopt, AdoptState::Legacy) {
            return;
        }
        self.adopt = AdoptState::Legacy;
        let num_entries = self.num_entries() as u64;
        self.interner.seed_dedup(0..num_entries);
    }

    /// Append a pre-resolved dictionary index (an Arrow key) directly.
    #[cfg(feature = "arrow")]
    pub fn push_index(&mut self, index: u64) {
        self.indices.push(index);
    }

    #[inline]
    fn bit_width(&self) -> u8 {
        num_required_bits(self.num_entries().saturating_sub(1) as u64)
    }
}

impl<T: DataType> Encoder<T> for DictEncoder<T> {
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        self.indices.reserve(values.len());
        for i in values {
            self.put_one(i)
        }
        Ok(())
    }

    // Performance Note:
    // As far as can be seen these functions are rarely called and as such we can hint to the
    // compiler that they dont need to be folded into hot locations in the final output.
    fn encoding(&self) -> Encoding {
        Encoding::PLAIN_DICTIONARY
    }

    /// Returns an estimate of the data page size in bytes
    ///
    /// This includes:
    /// <already_written_encoded_byte_size> + <estimated_encoded_size_of_unflushed_bytes>
    fn estimated_data_encoded_size(&self) -> usize {
        let bit_width = self.bit_width();
        RleEncoder::max_buffer_size(bit_width, self.indices.len())
    }

    fn flush_buffer(&mut self) -> Result<Bytes> {
        self.write_indices()
    }

    /// Returns the estimated total memory usage
    ///
    /// For this encoder, the indices are unencoded bytes (refer to [`Self::write_indices`]).
    fn estimated_memory_size(&self) -> usize {
        self.interner.estimated_memory_size() + self.indices.capacity() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::data_type::{
        ByteArray, ByteArrayType, FixedLenByteArray, FixedLenByteArrayType, Int32Type,
    };
    use crate::encodings::encoding::Encoder;
    use crate::schema::types::{ColumnDescriptor, ColumnPath, Type as SchemaType};

    fn make_col_desc<T: DataType>() -> ColumnDescPtr {
        make_col_desc_with_length::<T>(-1)
    }

    fn make_col_desc_with_length<T: DataType>(type_length: i32) -> ColumnDescPtr {
        let ty = SchemaType::primitive_type_builder("col", T::get_physical_type())
            .with_length(type_length)
            .build()
            .unwrap();
        Arc::new(ColumnDescriptor::new(
            Arc::new(ty),
            0,
            0,
            ColumnPath::new(vec![]),
        ))
    }

    #[test]
    fn test_estimated_memory_size_primitive_with_duplicates() {
        let mut encoder = DictEncoder::<Int32Type>::new(make_col_desc::<Int32Type>());
        let empty_size = encoder.estimated_memory_size();

        // 3 distinct values, repeated to produce 9 indices total.
        encoder.put(&[1, 2, 3, 1, 2, 3, 1, 2, 3]).unwrap();

        let size = encoder.estimated_memory_size();

        // Must account for the 3 unique dictionary entries.
        let dict_entry_size = 3 * std::mem::size_of::<i32>();
        assert!(
            size >= empty_size + dict_entry_size,
            "memory size {size} should grow by at least the dict storage ({dict_entry_size} bytes)"
        );

        // Must also account for the 9 buffered indices.
        let indices_size = 9 * std::mem::size_of::<u64>();
        assert!(
            size >= empty_size + dict_entry_size + indices_size,
            "memory size {size} should include indices ({indices_size} bytes)"
        );
    }

    #[test]
    fn test_estimated_memory_size_primitive_all_distinct() {
        let mut encoder = DictEncoder::<Int32Type>::new(make_col_desc::<Int32Type>());
        let empty_size = encoder.estimated_memory_size();

        let values: Vec<i32> = (0..100).collect();
        encoder.put(&values).unwrap();

        let size = encoder.estimated_memory_size();

        // Must account for the 100 unique dictionary entries.
        let dict_entry_size = 100 * std::mem::size_of::<i32>();
        assert!(
            size >= empty_size + dict_entry_size,
            "memory size {size} should grow by at least the dict storage ({dict_entry_size} bytes)"
        );

        // Must also account for the 100 buffered indices.
        let indices_size = 100 * std::mem::size_of::<u64>();
        assert!(
            size >= empty_size + dict_entry_size + indices_size,
            "memory size {size} should include indices ({indices_size} bytes)"
        );
    }

    #[test]
    fn test_estimated_memory_size_byte_array_with_duplicates() {
        let mut encoder = DictEncoder::<ByteArrayType>::new(make_col_desc::<ByteArrayType>());
        let empty_size = encoder.estimated_memory_size();

        // 3 distinct byte strings ("foo", "bar", "baz" — 3 bytes each), repeated to produce
        // 9 indices total.
        let vals: Vec<ByteArray> = [
            "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz",
        ]
        .iter()
        .map(|s| ByteArray::from(*s))
        .collect();
        encoder.put(&vals).unwrap();

        let size = encoder.estimated_memory_size();

        // Must account for the 3 unique dictionary entries, including their heap-allocated bytes.
        let dict_entry_size = 3 * std::mem::size_of::<ByteArray>() + 3 * 3; // 3 values × 3 bytes each
        assert!(
            size >= empty_size + dict_entry_size,
            "memory size {size} should grow by at least the dict storage ({dict_entry_size} bytes)"
        );

        // Must also account for the 9 buffered indices.
        let indices_size = 9 * std::mem::size_of::<u64>();
        assert!(
            size >= empty_size + dict_entry_size + indices_size,
            "memory size {size} should include indices ({indices_size} bytes)"
        );
    }

    #[test]
    fn test_estimated_memory_size_byte_array_all_distinct() {
        let mut encoder = DictEncoder::<ByteArrayType>::new(make_col_desc::<ByteArrayType>());
        let empty_size = encoder.estimated_memory_size();

        // 100 distinct values: "0".."9" (1 byte each) and "10".."99" (2 bytes each).
        let values: Vec<ByteArray> = (0..100_u32)
            .map(|i| ByteArray::from(i.to_string().into_bytes()))
            .collect();
        let bytes_total: usize = values.iter().map(|v| v.len()).sum(); // 10×1 + 90×2 = 190
        encoder.put(&values).unwrap();

        let size = encoder.estimated_memory_size();

        // Must account for the 100 unique dictionary entries, including their heap-allocated bytes.
        let dict_entry_size = 100 * std::mem::size_of::<ByteArray>() + bytes_total;
        assert!(
            size >= empty_size + dict_entry_size,
            "memory size {size} should grow by at least the dict storage ({dict_entry_size} bytes)"
        );

        // Must also account for the 100 buffered indices.
        let indices_size = 100 * std::mem::size_of::<u64>();
        assert!(
            size >= empty_size + dict_entry_size + indices_size,
            "memory size {size} should include indices ({indices_size} bytes)"
        );
    }

    #[test]
    fn test_estimated_memory_size_fixed_len_byte_array_with_duplicates() {
        const TYPE_LEN: usize = 3;
        let mut encoder = DictEncoder::<FixedLenByteArrayType>::new(make_col_desc_with_length::<
            FixedLenByteArrayType,
        >(TYPE_LEN as i32));
        let empty_size = encoder.estimated_memory_size();

        // 3 distinct 3-byte values, repeated to produce 9 indices total.
        let vals = [
            b"foo", b"bar", b"baz", b"foo", b"bar", b"baz", b"foo", b"bar", b"baz",
        ]
        .iter()
        .map(|b| FixedLenByteArray::from(b.to_vec()))
        .collect::<Vec<_>>();
        encoder.put(&vals).unwrap();

        let size = encoder.estimated_memory_size();

        // Must account for the 3 unique dictionary entries: struct overhead plus the
        // fixed-length bytes allocated per entry.
        let dict_entry_size = 3 * std::mem::size_of::<FixedLenByteArray>() + 3 * TYPE_LEN;
        assert!(
            size >= empty_size + dict_entry_size,
            "memory size {size} should grow by at least the dict storage ({dict_entry_size} bytes)"
        );

        // Must also account for the 9 buffered indices.
        let indices_size = 9 * std::mem::size_of::<u64>();
        assert!(
            size >= empty_size + dict_entry_size + indices_size,
            "memory size {size} should include indices ({indices_size} bytes)"
        );
    }

    #[test]
    fn test_estimated_memory_size_fixed_len_byte_array_all_distinct() {
        const TYPE_LEN: usize = 3;
        let mut encoder = DictEncoder::<FixedLenByteArrayType>::new(make_col_desc_with_length::<
            FixedLenByteArrayType,
        >(TYPE_LEN as i32));
        let empty_size = encoder.estimated_memory_size();

        // 100 distinct 3-byte values: zero-padded big-endian u8 indices.
        let values = (0..100_u8)
            .map(|i| FixedLenByteArray::from(vec![0u8, 0u8, i]))
            .collect::<Vec<_>>();
        encoder.put(&values).unwrap();

        let size = encoder.estimated_memory_size();

        // Must account for the 100 unique dictionary entries: struct overhead plus the
        // fixed-length bytes allocated per entry.
        let dict_entry_size = 100 * std::mem::size_of::<FixedLenByteArray>() + 100 * TYPE_LEN;
        assert!(
            size >= empty_size + dict_entry_size,
            "memory size {size} should grow by at least the dict storage ({dict_entry_size} bytes)"
        );

        // Must also account for the 100 buffered indices.
        let indices_size = 100 * std::mem::size_of::<u64>();
        assert!(
            size >= empty_size + dict_entry_size + indices_size,
            "memory size {size} should include indices ({indices_size} bytes)"
        );
    }

    #[test]
    fn test_estimated_memory_size_includes_interner_dedup_table() {
        // The dedup `HashTable` in `Interner` is preallocated with
        // `DEFAULT_DEDUP_CAPACITY` slots at construction, independent of any
        // values pushed.
        let encoder = DictEncoder::<Int32Type>::new(make_col_desc::<Int32Type>());

        let size = encoder.estimated_memory_size();

        assert!(
            size > 0,
            "memory size should include the preallocated dedup hash table"
        );
    }

    #[test]
    fn test_estimated_memory_size_accounts_for_indices_capacity() {
        // Exercises the `indices.capacity()` (not `.len()`) accounting.
        // After a flush, `indices` is cleared but its capacity is retained; pushing a
        // smaller batch afterwards leaves capacity strictly greater than length.
        let mut encoder = DictEncoder::<Int32Type>::new(make_col_desc::<Int32Type>());

        let big: Vec<i32> = vec![0; 64];
        encoder.put(&big).unwrap();
        let _ = encoder.flush_buffer().unwrap();

        let flushed_size = encoder.estimated_memory_size();

        // Push a single value — indices.len() == 1 but indices.capacity() >= 64.
        // No change on the key storage since the value is already interned.
        encoder.put(&[0]).unwrap();

        let size = encoder.estimated_memory_size();

        assert_eq!(
            size, flushed_size,
            "memory size should include retained indices capacity",
        );
    }

    #[test]
    fn test_estimated_memory_size_accounts_for_uniques_capacity() {
        let mut encoder = DictEncoder::<Int32Type>::new(make_col_desc::<Int32Type>());

        let values: Vec<i32> = (0..64).collect();
        encoder.put(&values).unwrap();
        // Flush indices so they don't mask the uniques accounting in the lower bound.
        let _ = encoder.flush_buffer().unwrap();

        let size1 = encoder.estimated_memory_size();

        // Push more values to trigger uniques capacity growth.
        // The pre-allocated dedup hash table is unlikely to be resized.
        let values: Vec<i32> = (64..128).collect();
        encoder.put(&values).unwrap();
        // Flush indices so they don't mask the uniques accounting in the lower bound.
        let _ = encoder.flush_buffer().unwrap();

        let size2 = encoder.estimated_memory_size();

        let min_uniques_bytes = 64 * std::mem::size_of::<i32>();
        assert!(
            size2 >= size1 + min_uniques_bytes,
            "memory size {size2} should grow from {size1} by allocated uniques capacity \
             (at least {min_uniques_bytes} bytes)"
        );
    }
}
