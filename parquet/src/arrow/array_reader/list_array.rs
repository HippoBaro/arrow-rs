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
use crate::column::reader::run_level_buffer::{AlignedRunIter, levels_to_runs};
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
    /// Whether the child reader produces compact (non-padded) values.
    /// Set once at construction time based on set_skip_padding() support.
    compact_child: bool,
    /// When true, produce compact output: only non-null list entries,
    /// no validity bitmap, and populate record_def_runs as side output.
    skip_padding: bool,
    /// Record-level def runs populated during compact consume_batch.
    /// Each entry is (def_value, count) at the list level — non-null
    /// lists have def >= def_level, null lists have def < def_level.
    record_def_runs: Vec<(i16, u32)>,
    _marker: PhantomData<OffsetSize>,
}

impl<OffsetSize: OffsetSizeTrait> ListArrayReader<OffsetSize> {
    /// Construct list array reader.
    pub fn new(
        mut item_reader: Box<dyn ArrayReader>,
        data_type: ArrowType,
        def_level: i16,
        rep_level: i16,
        nullable: bool,
    ) -> Self {
        // Try to enable compact mode on the child reader. If supported,
        // the child produces only non-null values (no padding), which
        // saves O(rows) memory allocation for sparse columns.
        let compact_child = item_reader.set_skip_padding(true);

        Self {
            item_reader,
            data_type,
            def_level,
            rep_level,
            nullable,
            compact_child,
            skip_padding: false,
            record_def_runs: Vec::new(),
            _marker: PhantomData,
        }
    }
}

/// Implementation of ListArrayReader.
impl<OffsetSize: OffsetSizeTrait> ArrayReader for ListArrayReader<OffsetSize> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        let size = self.item_reader.read_records(batch_size)?;
        Ok(size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        // Clear record_def_runs early, BEFORE any early-return paths,
        // to prevent stale data from a previous batch leaking through.
        if self.skip_padding {
            self.record_def_runs.clear();
        }

        let child_array = self.item_reader.consume_batch()?;

        // Get def and rep level runs, falling back to flat conversion
        let def_level_runs_owned;
        let def_runs = if let Some(runs) = self.item_reader.get_def_level_runs() {
            runs
        } else {
            let flat = self
                .item_reader
                .get_def_levels()
                .ok_or_else(|| general_err!("item_reader def levels are None."))?;
            def_level_runs_owned = levels_to_runs(flat);
            &def_level_runs_owned
        };

        let rep_level_runs_owned;
        let rep_runs = if let Some(runs) = self.item_reader.get_rep_level_runs() {
            runs
        } else {
            let flat = self
                .item_reader
                .get_rep_levels()
                .ok_or_else(|| general_err!("item_reader rep levels are None."))?;
            rep_level_runs_owned = levels_to_runs(flat);
            &rep_level_runs_owned
        };

        if rep_runs.is_empty() {
            return Ok(new_empty_array(&self.data_type));
        }

        if rep_runs[0].0 != 0 {
            return Err(general_err!("first repetition level of batch must be 0"));
        }

        // Max definition level from runs — O(runs) not O(rows).
        let max_def_level = def_runs
            .iter()
            .map(|(v, _)| *v)
            .max()
            .unwrap_or(self.def_level);

        // Whether items within non-null lists can themselves be null.
        let items_nullable = max_def_level > self.def_level;

        // Estimate list count from rep runs for bitmap pre-sizing
        let est_lists: usize = rep_runs
            .iter()
            .filter(|(v, _)| *v == 0)
            .map(|(_, c)| *c as usize)
            .sum();

        let mut list_offsets: Vec<OffsetSize> = Vec::with_capacity(est_lists + 1);
        // In compact mode, all output entries are non-null so no validity bitmap.
        let mut validity = if self.skip_padding {
            None
        } else {
            self.nullable
                .then(|| BooleanBufferBuilder::new(est_lists))
        };

        let child_data;

        if !items_nullable {
            // Items within non-null lists cannot be null.
            // With a compact child, the child array maps 1:1 to values
            // where def >= max_def_level. With a padded child, we need
            // MutableArrayData to filter out padding entries.
            //
            // The AlignedRunIter processes both cases in O(runs).
            let mut cur_offset: usize = 0;

            if self.compact_child {
                // Compact child: child array has only real values.
                for (d, r, count) in AlignedRunIter::new(def_runs, &rep_runs) {
                    let count = count as usize;
                    if r < self.rep_level {
                        if d >= self.def_level {
                            if self.skip_padding {
                                self.record_def_runs.push((self.def_level, count as u32));
                            }
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
                        } else if self.skip_padding {
                            self.record_def_runs.push((d, count as u32));
                        } else {
                            let offset = OffsetSize::from_usize(cur_offset)
                                .ok_or_else(|| general_err!("offset overflow"))?;
                            list_offsets.extend(std::iter::repeat_n(offset, count));
                            if let Some(v) = validity.as_mut() {
                                v.append_n(count, d + 1 == self.def_level);
                            }
                        }
                    } else if d >= self.def_level {
                        cur_offset += count;
                    }
                }

                list_offsets.push(
                    OffsetSize::from_usize(cur_offset)
                        .ok_or_else(|| general_err!("offset overflow"))?,
                );
                child_data = child_array.to_data();
            } else {
                // Padded child: child array has one entry per level.
                // Need MutableArrayData to filter out null/empty entries.
                let mut filter_start = None;
                let mut skipped = 0;

                let data = child_array.to_data();
                let mut child_data_builder =
                    MutableArrayData::new(vec![&data], false, child_array.len());

                for (d, r, count) in AlignedRunIter::new(def_runs, &rep_runs) {
                    let count = count as usize;
                    if r > self.rep_level {
                        // Inner repetition — handled by child
                    } else if r == self.rep_level {
                        cur_offset += count;
                    } else {
                        // Record boundaries
                        if d >= self.def_level {
                            if self.skip_padding {
                                self.record_def_runs.push((self.def_level, count as u32));
                            }
                            filter_start.get_or_insert(cur_offset + skipped);
                            for _ in 0..count {
                                list_offsets.push(
                                    OffsetSize::from_usize(cur_offset).unwrap(),
                                );
                                cur_offset += 1;
                            }
                            if let Some(v) = validity.as_mut() {
                                v.append_n(count, true);
                            }
                        } else if self.skip_padding {
                            if let Some(start) = filter_start.take() {
                                child_data_builder
                                    .extend(0, start, cur_offset + skipped);
                            }
                            skipped += count;
                            self.record_def_runs.push((d, count as u32));
                        } else {
                            if let Some(start) = filter_start.take() {
                                child_data_builder
                                    .extend(0, start, cur_offset + skipped);
                            }
                            let offset =
                                OffsetSize::from_usize(cur_offset).unwrap();
                            list_offsets
                                .extend(std::iter::repeat_n(offset, count));
                            if let Some(v) = validity.as_mut() {
                                v.append_n(count, d + 1 == self.def_level);
                            }
                            skipped += count;
                        }
                    }
                }

                list_offsets.push(OffsetSize::from_usize(cur_offset).unwrap());

                child_data = if skipped == 0 {
                    child_array.to_data()
                } else {
                    if let Some(start) = filter_start.take() {
                        child_data_builder
                            .extend(0, start, cur_offset + skipped);
                    }
                    child_data_builder.freeze()
                };
            }
        } else if self.compact_child {
            // Items nullable, compact child: child array has only values
            // where def >= max_def_level. Need to re-insert null items.
            let mut cur_offset: usize = 0;
            let mut compact_idx: usize = 0;
            let mut def_cursor =
                crate::column::reader::run_level_buffer::RunCursor::new(def_runs);
            let mut rep_cursor =
                crate::column::reader::run_level_buffer::RunCursor::new(&rep_runs);
            let compact_data = child_array.to_data();
            let mut child_builder =
                MutableArrayData::new(vec![&compact_data], true, child_array.len());
            let mut run_start: Option<usize> = None;
            let total_levels: usize =
                rep_runs.iter().map(|(_, c)| *c as usize).sum();

            for _ in 0..total_levels {
                let r = rep_cursor.next();
                let d = def_cursor.next();
                let is_list_boundary = r < self.rep_level;
                let list_exists = d >= self.def_level;
                let value_exists = d >= max_def_level;

                if is_list_boundary {
                    if list_exists {
                        if let Some(start) = run_start.take() {
                            child_builder.extend(0, start, compact_idx);
                        }
                        list_offsets.push(
                            OffsetSize::from_usize(cur_offset)
                                .ok_or_else(|| general_err!("offset overflow"))?,
                        );
                        if self.skip_padding {
                            self.record_def_runs.push((self.def_level, 1));
                        }
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
                    } else if self.skip_padding {
                        self.record_def_runs.push((d, 1));
                    } else {
                        if let Some(start) = run_start.take() {
                            child_builder.extend(0, start, compact_idx);
                        }
                        list_offsets.push(
                            OffsetSize::from_usize(cur_offset)
                                .ok_or_else(|| general_err!("offset overflow"))?,
                        );
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
            if let Some(start) = run_start {
                child_builder.extend(0, start, compact_idx);
            }
            child_data = child_builder.freeze();
        } else {
            // Items nullable, padded child: child array has one entry per
            // level. Filter out null/empty list entries and re-insert null
            // items within non-null lists.
            let mut cur_offset: usize = 0;
            let mut def_cursor =
                crate::column::reader::run_level_buffer::RunCursor::new(def_runs);
            let mut rep_cursor =
                crate::column::reader::run_level_buffer::RunCursor::new(&rep_runs);
            let data = child_array.to_data();
            let mut child_data_builder =
                MutableArrayData::new(vec![&data], false, child_array.len());
            let mut filter_start: Option<usize> = None;
            let mut skipped: usize = 0;
            let total_levels: usize =
                rep_runs.iter().map(|(_, c)| *c as usize).sum();

            for _ in 0..total_levels {
                let r = rep_cursor.next();
                let d = def_cursor.next();

                if r > self.rep_level {
                    if d < self.def_level {
                        return Err(general_err!(
                            "Encountered repetition level too large for definition level"
                        ));
                    }
                } else if r == self.rep_level {
                    cur_offset += 1;
                } else {
                    // Record boundary
                    if d >= self.def_level {
                        list_offsets.push(OffsetSize::from_usize(cur_offset).unwrap());
                        filter_start.get_or_insert(cur_offset + skipped);
                        cur_offset += 1;
                        if self.skip_padding {
                            self.record_def_runs.push((self.def_level, 1));
                        }
                        if let Some(v) = validity.as_mut() {
                            v.append(true);
                        }
                    } else if self.skip_padding {
                        if let Some(start) = filter_start.take() {
                            child_data_builder
                                .extend(0, start, cur_offset + skipped);
                        }
                        skipped += 1;
                        self.record_def_runs.push((d, 1));
                    } else {
                        list_offsets.push(OffsetSize::from_usize(cur_offset).unwrap());
                        if let Some(start) = filter_start.take() {
                            child_data_builder
                                .extend(0, start, cur_offset + skipped);
                        }
                        if let Some(v) = validity.as_mut() {
                            v.append(d + 1 == self.def_level);
                        }
                        skipped += 1;
                    }
                }
            }

            list_offsets.push(OffsetSize::from_usize(cur_offset).unwrap());
            child_data = if skipped == 0 {
                child_array.to_data()
            } else {
                if let Some(start) = filter_start.take() {
                    child_data_builder.extend(0, start, cur_offset + skipped);
                }
                child_data_builder.freeze()
            };
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

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        self.item_reader.skip_records(num_records)
    }

    fn set_compact_record_output(&mut self, compact: bool) -> bool {
        self.skip_padding = compact;
        true
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.item_reader.get_def_levels()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.item_reader.get_rep_levels()
    }

    fn get_def_level_runs(&self) -> Option<&[(i16, u32)]> {
        if self.skip_padding {
            Some(&self.record_def_runs)
        } else {
            None
        }
    }

    fn peek_def_level_runs(&self) -> Option<&[(i16, u32)]> {
        self.item_reader.peek_def_level_runs()
    }

    fn discard_batch(&mut self) -> Result<usize> {
        self.item_reader.discard_batch()
    }
}
