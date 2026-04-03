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

use crate::arrow::array_reader::ArrayReader;
use crate::column::reader::run_level_buffer::{AlignedRunIter, RunCursor, levels_to_runs};
use crate::errors::ParquetError;
use crate::errors::Result;
use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, builder::BooleanBufferBuilder,
    new_empty_array,
};
use arrow_buffer::Buffer;
use arrow_buffer::ToByteSlice;
use arrow_data::{ArrayData, transform::MutableArrayData};
use arrow_schema::DataType as ArrowType;
use std::any::Any;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

/// Implementation of list array reader.
pub struct ListArrayReader<OffsetSize: OffsetSizeTrait> {
    item_reader: Box<dyn ArrayReader>,
    data_type: ArrowType,
    /// The definition level at which this list is not null
    def_level: i16,
    /// The repetition level that corresponds to a new value in this array
    rep_level: i16,
    /// If this list is nullable
    nullable: bool,
    /// When true, the child reader produces unpadded (compact) values and
    /// consume_batch builds the list directly without MutableArrayData.
    unpadded_child: bool,
    _marker: PhantomData<OffsetSize>,
}

impl<OffsetSize: OffsetSizeTrait> ListArrayReader<OffsetSize> {
    /// Construct list array reader.
    pub fn new(
        item_reader: Box<dyn ArrayReader>,
        data_type: ArrowType,
        def_level: i16,
        rep_level: i16,
        nullable: bool,
    ) -> Self {
        Self {
            item_reader,
            data_type,
            def_level,
            rep_level,
            nullable,
            unpadded_child: false,
            _marker: PhantomData,
        }
    }

    /// Enable unpadded child mode. When set, the child reader is expected to
    /// produce compact (non-null-padded) arrays, and consume_batch will build
    /// the list directly without the MutableArrayData filtering pass.
    pub fn set_unpadded_child(&mut self, unpadded: bool) {
        self.unpadded_child = unpadded;
    }

    /// Consume batch when the child reader produces padded (null-expanded) values.
    /// This is the original code path using MutableArrayData to filter out padding.
    fn consume_batch_padded(&mut self) -> Result<ArrayRef> {
        let next_batch_array = self.item_reader.consume_batch()?;
        if next_batch_array.is_empty() {
            return Ok(new_empty_array(&self.data_type));
        }

        let def_level_runs = self.item_reader.get_def_level_runs();

        let rep_levels = self
            .item_reader
            .get_rep_levels()
            .ok_or_else(|| general_err!("item_reader rep levels are None."))?;

        if OffsetSize::from_usize(next_batch_array.len()).is_none() {
            return Err(general_err!(
                "offset of {} would overflow list array",
                next_batch_array.len()
            ));
        }

        if !rep_levels.is_empty() && rep_levels[0] != 0 {
            return Err(general_err!("first repetition level of batch must be 0"));
        }

        let mut list_offsets: Vec<OffsetSize> = Vec::with_capacity(next_batch_array.len() + 1);

        let mut validity = self
            .nullable
            .then(|| BooleanBufferBuilder::new(next_batch_array.len()));

        let mut cur_offset = 0;
        let mut filter_start = None;
        let mut skipped = 0;

        let data = next_batch_array.to_data();
        let mut child_data_builder =
            MutableArrayData::new(vec![&data], false, next_batch_array.len());

        // Get def level runs (or convert from flat) and rep level runs
        let def_level_runs_owned;
        let def_runs = if let Some(runs) = def_level_runs {
            runs
        } else {
            let flat = self
                .item_reader
                .get_def_levels()
                .ok_or_else(|| general_err!("item_reader def levels are None."))?;
            def_level_runs_owned = levels_to_runs(flat);
            &def_level_runs_owned
        };
        let rep_runs = levels_to_runs(rep_levels);

        for (d, r, count) in AlignedRunIter::new(def_runs, &rep_runs) {
            let count = count as usize;
            if r > self.rep_level {
                // Inner repetition — already handled by child
                if d < self.def_level {
                    return Err(general_err!(
                        "Encountered repetition level too large for definition level"
                    ));
                }
            } else if r == self.rep_level {
                // Continuation within current list
                cur_offset += count;
            } else {
                // Record boundaries (r < rep_level)
                if d >= self.def_level {
                    // Non-null lists: each of the `count` boundaries starts
                    // a new list with one value from the padded child.
                    filter_start.get_or_insert(cur_offset + skipped);
                    for _ in 0..count {
                        list_offsets.push(OffsetSize::from_usize(cur_offset).unwrap());
                        cur_offset += 1;
                    }
                    if let Some(v) = validity.as_mut() {
                        v.append_n(count, true);
                    }
                } else {
                    // Null/empty lists — bulk operation
                    if let Some(start) = filter_start.take() {
                        child_data_builder.extend(0, start, cur_offset + skipped);
                    }
                    let offset = OffsetSize::from_usize(cur_offset).unwrap();
                    list_offsets.extend(std::iter::repeat_n(offset, count));
                    if let Some(v) = validity.as_mut() {
                        v.append_n(count, d + 1 == self.def_level);
                    }
                    skipped += count;
                }
            }
        }

        list_offsets.push(OffsetSize::from_usize(cur_offset).unwrap());

        let child_data = if skipped == 0 {
            next_batch_array.to_data()
        } else {
            if let Some(start) = filter_start.take() {
                child_data_builder.extend(0, start, cur_offset + skipped)
            }

            child_data_builder.freeze()
        };

        if cur_offset != child_data.len() {
            return Err(general_err!("Failed to reconstruct list from level data"));
        }

        let value_offsets = Buffer::from(list_offsets.to_byte_slice());

        let mut data_builder = ArrayData::builder(self.get_data_type().clone())
            .len(list_offsets.len() - 1)
            .add_buffer(value_offsets)
            .add_child_data(child_data);

        if let Some(builder) = validity {
            assert_eq!(builder.len(), list_offsets.len() - 1);
            data_builder = data_builder.null_bit_buffer(Some(builder.into()))
        }

        let list_data = unsafe { data_builder.build_unchecked() };

        let result_array = GenericListArray::<OffsetSize>::from(list_data);
        Ok(Arc::new(result_array))
    }

    /// Consume batch when the child reader produces unpadded (compact) values.
    ///
    /// The compact child array contains only values where def == max_def_level.
    /// Entries for null lists are absent, and entries for null items within
    /// non-null lists are also absent. This method reconstructs the list
    /// structure using def/rep levels, re-inserting null item entries via a
    /// MutableArrayData sized to the compact array (much smaller than the
    /// fully-padded size for sparse columns).
    fn consume_batch_unpadded(&mut self) -> Result<ArrayRef> {
        let compact_array = self.item_reader.consume_batch()?;

        let def_level_runs = self.item_reader.get_def_level_runs();

        let rep_levels = self
            .item_reader
            .get_rep_levels()
            .ok_or_else(|| general_err!("item_reader rep levels are None."))?;

        if rep_levels.is_empty() {
            return Ok(new_empty_array(&self.data_type));
        }

        if rep_levels[0] != 0 {
            return Err(general_err!("first repetition level of batch must be 0"));
        }

        // Build a run cursor for def levels. If runs aren't available,
        // fall back to materializing from get_def_levels().
        let def_level_runs_owned;
        let def_runs = if let Some(runs) = def_level_runs {
            runs
        } else {
            let flat = self
                .item_reader
                .get_def_levels()
                .ok_or_else(|| general_err!("item_reader def levels are None."))?;
            def_level_runs_owned =
                crate::column::reader::run_level_buffer::levels_to_runs(flat);
            &def_level_runs_owned
        };

        // Max definition level from runs — O(runs) not O(rows).
        let max_def_level = def_runs
            .iter()
            .map(|(v, _)| *v)
            .max()
            .unwrap_or(self.def_level);

        // Whether items within non-null lists can themselves be null.
        let items_nullable = max_def_level > self.def_level;

        let mut list_offsets: Vec<OffsetSize> = Vec::new();
        let mut validity = self
            .nullable
            .then(|| BooleanBufferBuilder::new(rep_levels.len()));

        // Convert rep levels to runs for aligned iteration
        let rep_runs = levels_to_runs(rep_levels);

        let child_data;

        if !items_nullable {
            // Fast path using aligned run pairs — O(runs) for sparse data.
            //
            // For each aligned (def_val, rep_val, count) triple:
            // - rep=0, def=0: `count` null records → bulk push offsets + bulk false bits
            // - rep=0, def>=def_level: new non-null list (count=1 since next entry changes rep)
            // - rep>0, def>=def_level: `count` continuation values → offset += count
            // - rep>0, def<def_level: impossible for !items_nullable
            let mut cur_offset: usize = 0;

            for (d, r, count) in AlignedRunIter::new(def_runs, &rep_runs) {
                let count = count as usize;
                if r < self.rep_level {
                    // Record boundaries: `count` new records all with the same
                    // def level `d` and rep level `r` (which is < rep_level,
                    // so each is a new list).
                    if d >= self.def_level {
                        // `count` non-null lists, each starting with one value
                        for _ in 0..count {
                            list_offsets.push(
                                OffsetSize::from_usize(cur_offset)
                                    .ok_or_else(|| general_err!("offset overflow"))?,
                            );
                            cur_offset += 1;
                        }
                        if let Some(v) = validity.as_mut() {
                            v.append_n(count, true);
                        }
                    } else {
                        // `count` null or empty lists — bulk push identical
                        // offsets and bulk append validity bits.
                        let offset = OffsetSize::from_usize(cur_offset)
                            .ok_or_else(|| general_err!("offset overflow"))?;
                        list_offsets.extend(std::iter::repeat_n(offset, count));
                        if let Some(v) = validity.as_mut() {
                            v.append_n(count, d + 1 == self.def_level);
                        }
                    }
                } else if d >= self.def_level {
                    // Continuation values within a non-null list
                    cur_offset += count;
                }
            }

            list_offsets.push(
                OffsetSize::from_usize(cur_offset)
                    .ok_or_else(|| general_err!("offset overflow"))?,
            );

            assert_eq!(
                cur_offset,
                compact_array.len(),
                "not all compact values consumed: used {cur_offset}, have {}",
                compact_array.len()
            );

            child_data = compact_array.to_data();
        } else {
            // Slow path: items within non-null lists can be null.
            // Still uses per-element RunCursor since MutableArrayData
            // extend calls need precise index tracking.
            let mut cur_offset: usize = 0;
            let mut compact_idx: usize = 0;
            let mut def_cursor = RunCursor::new(def_runs);

            let compact_data = compact_array.to_data();
            let mut child_builder =
                MutableArrayData::new(vec![&compact_data], true, compact_array.len());

            let mut run_start: Option<usize> = None;

            for r in rep_levels {
                let d = def_cursor.next();
                let is_list_boundary = *r < self.rep_level;
                let list_exists = d >= self.def_level;
                let value_exists = d >= max_def_level;

                if is_list_boundary {
                    if let Some(start) = run_start.take() {
                        child_builder.extend(0, start, compact_idx);
                    }

                    list_offsets.push(
                        OffsetSize::from_usize(cur_offset)
                            .ok_or_else(|| general_err!("offset overflow"))?,
                    );

                    if list_exists {
                        if let Some(v) = validity.as_mut() {
                            v.append(true);
                        }
                        if value_exists {
                            run_start.get_or_insert(compact_idx);
                            compact_idx += 1;
                            cur_offset += 1;
                        } else {
                            child_builder.extend_nulls(1);
                            cur_offset += 1;
                        }
                    } else {
                        if let Some(v) = validity.as_mut() {
                            v.append(d + 1 == self.def_level);
                        }
                    }
                } else if list_exists {
                    if value_exists {
                        run_start.get_or_insert(compact_idx);
                        compact_idx += 1;
                        cur_offset += 1;
                    } else {
                        if let Some(start) = run_start.take() {
                            child_builder.extend(0, start, compact_idx);
                        }
                        child_builder.extend_nulls(1);
                        cur_offset += 1;
                    }
                }
            }

            list_offsets.push(
                OffsetSize::from_usize(cur_offset)
                    .ok_or_else(|| general_err!("offset overflow"))?,
            );

            assert_eq!(
                compact_idx,
                compact_array.len(),
                "not all compact values consumed: used {compact_idx}, have {}",
                compact_array.len()
            );

            // Flush final run
            if let Some(start) = run_start {
                child_builder.extend(0, start, compact_idx);
            }
            child_data = child_builder.freeze();
        }

        if list_offsets.last().map(|o| o.as_usize()) != Some(child_data.len()) {
            return Err(general_err!(
                "Failed to reconstruct list from level data: \
                 expected {} child values but got {}",
                list_offsets.last().map(|o| o.as_usize()).unwrap_or(0),
                child_data.len()
            ));
        }

        let value_offsets = Buffer::from(list_offsets.to_byte_slice());

        let mut data_builder = ArrayData::builder(self.get_data_type().clone())
            .len(list_offsets.len() - 1)
            .add_buffer(value_offsets)
            .add_child_data(child_data);

        if let Some(builder) = validity {
            assert_eq!(builder.len(), list_offsets.len() - 1);
            data_builder = data_builder.null_bit_buffer(Some(builder.into()));
        }

        let list_data = unsafe { data_builder.build_unchecked() };
        Ok(Arc::new(GenericListArray::<OffsetSize>::from(list_data)))
    }
}

/// Implementation of ListArrayReader. Nested lists and lists of structs are not yet supported.
impl<OffsetSize: OffsetSizeTrait> ArrayReader for ListArrayReader<OffsetSize> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns data type.
    /// This must be a List.
    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        let size = self.item_reader.read_records(batch_size)?;
        Ok(size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        if self.unpadded_child {
            self.consume_batch_unpadded()
        } else {
            self.consume_batch_padded()
        }
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        self.item_reader.skip_records(num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.item_reader.get_def_levels()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.item_reader.get_rep_levels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::array_reader::ArrayReaderBuilder;
    use crate::arrow::array_reader::list_array::ListArrayReader;
    use crate::arrow::array_reader::test_util::InMemoryArrayReader;
    use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
    use crate::arrow::schema::parquet_to_arrow_schema_and_fields;
    use crate::arrow::{ArrowWriter, ProjectionMask, parquet_to_arrow_schema};
    use crate::file::properties::WriterProperties;
    use crate::file::reader::{FileReader, SerializedFileReader};
    use crate::schema::parser::parse_message_type;
    use crate::schema::types::SchemaDescriptor;
    use arrow::datatypes::{Field, Int32Type as ArrowInt32, Int32Type};
    use arrow_array::{Array, PrimitiveArray};
    use arrow_data::ArrayDataBuilder;
    use arrow_schema::Fields;
    use std::sync::Arc;

    fn list_type<OffsetSize: OffsetSizeTrait>(
        data_type: ArrowType,
        item_nullable: bool,
    ) -> ArrowType {
        let field = Arc::new(Field::new_list_field(data_type, item_nullable));
        GenericListArray::<OffsetSize>::DATA_TYPE_CONSTRUCTOR(field)
    }

    fn downcast<OffsetSize: OffsetSizeTrait>(array: &ArrayRef) -> &'_ GenericListArray<OffsetSize> {
        array
            .as_any()
            .downcast_ref::<GenericListArray<OffsetSize>>()
            .unwrap()
    }

    fn to_offsets<OffsetSize: OffsetSizeTrait>(values: Vec<usize>) -> Buffer {
        Buffer::from_iter(
            values
                .into_iter()
                .map(|x| OffsetSize::from_usize(x).unwrap()),
        )
    }

    fn test_nested_list<OffsetSize: OffsetSizeTrait>() {
        // 3 lists, with first and third nullable
        // [
        //     [
        //         [[1, null], null, [4], []],
        //         [],
        //         [[7]],
        //         [[]],
        //         [[1, 2, 3], [4, null, 6], null]
        //     ],
        //     null,
        //     [],
        //     [[[11]]]
        // ]

        let l3_item_type = ArrowType::Int32;
        let l3_type = list_type::<OffsetSize>(l3_item_type, true);

        let l2_item_type = l3_type.clone();
        let l2_type = list_type::<OffsetSize>(l2_item_type, true);

        let l1_item_type = l2_type.clone();
        let l1_type = list_type::<OffsetSize>(l1_item_type, false);

        let leaf = PrimitiveArray::<Int32Type>::from_iter(vec![
            Some(1),
            None,
            Some(4),
            Some(7),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(6),
            Some(11),
        ]);

        // [[1, null], null, [4], [], [7], [], [1, 2, 3], [4, null, 6], null, [11]]
        let offsets = to_offsets::<OffsetSize>(vec![0, 2, 2, 3, 3, 4, 4, 7, 10, 10, 11]);
        let l3 = ArrayDataBuilder::new(l3_type.clone())
            .len(10)
            .add_buffer(offsets)
            .add_child_data(leaf.into_data())
            .null_bit_buffer(Some(Buffer::from([0b11111101, 0b00000010])))
            .build()
            .unwrap();

        // [[[1, null], null, [4], []], [], [[7]], [[]], [[1, 2, 3], [4, null, 6], null], [[11]]]
        let offsets = to_offsets::<OffsetSize>(vec![0, 4, 4, 5, 6, 9, 10]);
        let l2 = ArrayDataBuilder::new(l2_type.clone())
            .len(6)
            .add_buffer(offsets)
            .add_child_data(l3)
            .build()
            .unwrap();

        let offsets = to_offsets::<OffsetSize>(vec![0, 5, 5, 5, 6]);
        let l1 = ArrayDataBuilder::new(l1_type.clone())
            .len(4)
            .add_buffer(offsets)
            .add_child_data(l2)
            .null_bit_buffer(Some(Buffer::from([0b00001101])))
            .build()
            .unwrap();

        let expected = GenericListArray::<OffsetSize>::from(l1);

        let values = Arc::new(PrimitiveArray::<Int32Type>::from(vec![
            Some(1),
            None,
            None,
            Some(4),
            None,
            None,
            Some(7),
            None,
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(6),
            None,
            None,
            None,
            Some(11),
        ]));

        let item_array_reader = InMemoryArrayReader::new(
            ArrowType::Int32,
            values,
            Some(vec![6, 5, 3, 6, 4, 2, 6, 4, 6, 6, 6, 6, 5, 6, 3, 0, 1, 6]),
            Some(vec![0, 3, 2, 2, 2, 1, 1, 1, 1, 3, 3, 2, 3, 3, 2, 0, 0, 0]),
        );

        let l3 =
            ListArrayReader::<OffsetSize>::new(Box::new(item_array_reader), l3_type, 5, 3, true);

        let l2 = ListArrayReader::<OffsetSize>::new(Box::new(l3), l2_type, 3, 2, false);

        let mut l1 = ListArrayReader::<OffsetSize>::new(Box::new(l2), l1_type, 2, 1, true);

        let expected_1 = expected.slice(0, 2);
        let expected_2 = expected.slice(2, 2);

        let actual = l1.next_batch(2).unwrap();
        assert_eq!(actual.as_ref(), &expected_1);

        let actual = l1.next_batch(1024).unwrap();
        assert_eq!(actual.as_ref(), &expected_2);
    }

    fn test_required_list<OffsetSize: OffsetSizeTrait>() {
        // [[1, null, 2], [], [3, 4], [], [], [null, 1]]
        let expected =
            GenericListArray::<OffsetSize>::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(1), None, Some(2)]),
                Some(vec![]),
                Some(vec![Some(3), Some(4)]),
                Some(vec![]),
                Some(vec![]),
                Some(vec![None, Some(1)]),
            ]);

        let array = Arc::new(PrimitiveArray::<ArrowInt32>::from(vec![
            Some(1),
            None,
            Some(2),
            None,
            Some(3),
            Some(4),
            None,
            None,
            None,
            Some(1),
        ]));

        let item_array_reader = InMemoryArrayReader::new(
            ArrowType::Int32,
            array,
            Some(vec![2, 1, 2, 0, 2, 2, 0, 0, 1, 2]),
            Some(vec![0, 1, 1, 0, 0, 1, 0, 0, 0, 1]),
        );

        let mut list_array_reader = ListArrayReader::<OffsetSize>::new(
            Box::new(item_array_reader),
            list_type::<OffsetSize>(ArrowType::Int32, true),
            1,
            1,
            false,
        );

        let actual = list_array_reader.next_batch(1024).unwrap();
        let actual = downcast::<OffsetSize>(&actual);

        assert_eq!(&expected, actual)
    }

    fn test_nullable_list<OffsetSize: OffsetSizeTrait>() {
        // [[1, null, 2], null, [], [3, 4], [], [], null, [], [null, 1]]
        let expected =
            GenericListArray::<OffsetSize>::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(1), None, Some(2)]),
                None,
                Some(vec![]),
                Some(vec![Some(3), Some(4)]),
                Some(vec![]),
                Some(vec![]),
                None,
                Some(vec![]),
                Some(vec![None, Some(1)]),
            ]);

        let array = Arc::new(PrimitiveArray::<ArrowInt32>::from(vec![
            Some(1),
            None,
            Some(2),
            None,
            None,
            Some(3),
            Some(4),
            None,
            None,
            None,
            None,
            None,
            Some(1),
        ]));

        let item_array_reader = InMemoryArrayReader::new(
            ArrowType::Int32,
            array,
            Some(vec![3, 2, 3, 0, 1, 3, 3, 1, 1, 0, 1, 2, 3]),
            Some(vec![0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
        );

        let mut list_array_reader = ListArrayReader::<OffsetSize>::new(
            Box::new(item_array_reader),
            list_type::<OffsetSize>(ArrowType::Int32, true),
            2,
            1,
            true,
        );

        let actual = list_array_reader.next_batch(1024).unwrap();
        let actual = downcast::<OffsetSize>(&actual);

        assert_eq!(&expected, actual)
    }

    fn test_list_array<OffsetSize: OffsetSizeTrait>() {
        test_nullable_list::<OffsetSize>();
        test_required_list::<OffsetSize>();
        test_nested_list::<OffsetSize>();
    }

    #[test]
    fn test_list_array_reader() {
        test_list_array::<i32>();
    }

    #[test]
    fn test_large_list_array_reader() {
        test_list_array::<i64>()
    }

    #[test]
    fn test_nested_lists() {
        // Construct column schema
        let message_type = "
        message table {
            REPEATED group table_info {
                REQUIRED BYTE_ARRAY name;
                REPEATED group cols {
                    REQUIRED BYTE_ARRAY name;
                    REQUIRED INT32 type;
                    OPTIONAL INT32 length;
                }
                REPEATED group tags {
                    REQUIRED BYTE_ARRAY name;
                    REQUIRED INT32 type;
                    OPTIONAL INT32 length;
                }
            }
        }
        ";

        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();

        let arrow_schema = parquet_to_arrow_schema(schema.as_ref(), None).unwrap();

        let file = tempfile::tempfile().unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(200)
            .build();

        let writer = ArrowWriter::try_new(
            file.try_clone().unwrap(),
            Arc::new(arrow_schema),
            Some(props),
        )
        .unwrap();
        writer.close().unwrap();

        let file_reader: Arc<dyn FileReader> = Arc::new(SerializedFileReader::new(file).unwrap());

        let file_metadata = file_reader.metadata().file_metadata();
        let schema = file_metadata.schema_descr();
        let mask = ProjectionMask::leaves(schema, vec![0]);
        let (_, fields) = parquet_to_arrow_schema_and_fields(
            schema,
            ProjectionMask::all(),
            file_metadata.key_value_metadata(),
            &[],
        )
        .unwrap();

        let metrics = ArrowReaderMetrics::disabled();
        let mut array_reader = ArrayReaderBuilder::new(&file_reader, &metrics)
            .build_array_reader(fields.as_ref(), &mask)
            .unwrap();

        let batch = array_reader.next_batch(100).unwrap();
        assert_eq!(batch.data_type(), array_reader.get_data_type());
        assert_eq!(
            batch.data_type(),
            &ArrowType::Struct(Fields::from(vec![Field::new(
                "table_info",
                ArrowType::List(Arc::new(Field::new(
                    "table_info",
                    ArrowType::Struct(vec![Field::new("name", ArrowType::Binary, false)].into()),
                    false
                ))),
                false
            )]))
        );
        assert_eq!(batch.len(), 0);
    }
}
