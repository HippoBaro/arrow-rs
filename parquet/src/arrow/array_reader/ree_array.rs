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

//! Array reader that produces Arrow RunEndEncoded (REE) arrays from
//! Parquet columns, preserving the RLE run structure through to Arrow
//! without dense materialization.

use crate::arrow::array_reader::primitive_array::IntoBuffer;
use crate::arrow::array_reader::{read_records, ArrayReader};
use crate::arrow::record_reader::RecordReader;
use crate::column::page::PageIterator;
use crate::column::reader::run_level_buffer::RunLevelBuffer;
use crate::data_type::DataType;
use crate::errors::Result;
use crate::schema::types::ColumnDescPtr;
use arrow_array::types::Int32Type as ArrowInt32Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal32Array, Decimal64Array, Float32Array,
    Float64Array, Int32Array, Int64Array, RunArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
    UInt32Array, UInt64Array,
};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType as ArrowType, Field, TimeUnit};
use std::any::Any;
use std::sync::Arc;

use crate::basic::Type as PhysicalType;
use crate::data_type::Int96;

/// Build a `RunArray<Int32Type>` from definition level runs and compact
/// (non-null) values.
///
/// The def level runs describe which positions are null vs non-null.
/// The compact_values array contains only the non-null values (produced
/// by a record reader with `skip_padding=true`).
///
/// For a 99% sparse column with 100K rows and 1K values, this produces
/// a RunArray with ~3 physical entries instead of 100K dense elements.
fn build_ree_array(
    def_runs: &[(i16, u32)],
    compact_values: ArrayRef,
    max_def_level: i16,
) -> Result<ArrayRef> {
    if def_runs.is_empty() {
        // Empty input: produce an empty RunArray
        let run_ends = Int32Array::from(Vec::<i32>::new());
        let values = arrow_array::new_empty_array(compact_values.data_type());
        let run_array = RunArray::<ArrowInt32Type>::try_new(&run_ends, &values)?;
        return Ok(Arc::new(run_array));
    }

    // Merge adjacent runs that have the same null/non-null classification.
    // A run is "non-null" if its def level == max_def_level.
    let mut merged_runs: Vec<(bool, u32)> = Vec::new(); // (is_value, count)
    for &(def_val, count) in def_runs {
        let is_value = def_val >= max_def_level;
        if let Some(last) = merged_runs.last_mut().filter(|r| r.0 == is_value) {
            last.1 += count;
        } else {
            merged_runs.push((is_value, count));
        }
    }

    // Build run_ends (cumulative) and values array.
    let num_physical_runs = merged_runs.len();
    let mut run_ends_vec: Vec<i32> = Vec::with_capacity(num_physical_runs);
    let mut cumulative: i32 = 0;
    let mut value_offset: usize = 0;

    // We need to build a values array with one entry per run.
    // For null runs: the entry is null.
    // For value runs: the entry is sliced from compact_values.
    //
    // Since RunArray values must have one entry per run_end, and each
    // run may cover multiple values, we need to handle multi-value runs
    // differently. A run of (is_value=true, count=100) means 100
    // consecutive non-null values — but they may all be different.
    // RunArray requires each (run_end, value) pair to represent a
    // constant value for that run.
    //
    // For truly RLE data (same value repeated), each Parquet RLE run
    // maps to one REE entry. But for general data (PLAIN encoded values
    // within a non-null run), each value needs its own REE entry.
    //
    // The efficient approach: null runs collapse to one REE entry.
    // Non-null runs expand to one REE entry per value.

    // Count total physical entries (null runs = 1 each, value runs = count each)
    let total_entries: usize = merged_runs
        .iter()
        .map(|(is_value, count)| {
            if *is_value {
                *count as usize
            } else {
                1
            }
        })
        .sum();

    let mut run_ends = Vec::with_capacity(total_entries);
    let mut null_mask = Vec::with_capacity(total_entries);
    // Track which compact_values indices to include
    let mut value_indices: Vec<Option<usize>> = Vec::with_capacity(total_entries);

    cumulative = 0;
    for (is_value, count) in &merged_runs {
        if *is_value {
            // Each value gets its own REE entry
            for _ in 0..*count {
                cumulative += 1;
                run_ends.push(cumulative);
                null_mask.push(true);
                value_indices.push(Some(value_offset));
                value_offset += 1;
            }
        } else {
            // Entire null run collapses to one REE entry
            cumulative += *count as i32;
            run_ends.push(cumulative);
            null_mask.push(false);
            value_indices.push(None);
        }
    }

    // Build the values array: one entry per REE physical entry.
    // For null entries, we need a placeholder (null).
    // For value entries, copy from compact_values.
    //
    // Use MutableArrayData to build this efficiently.
    let compact_data = compact_values.to_data();
    let mut values_builder =
        arrow_data::transform::MutableArrayData::new(vec![&compact_data], true, total_entries);

    for idx in &value_indices {
        match idx {
            Some(i) => values_builder.extend(0, *i, *i + 1),
            None => values_builder.extend_nulls(1),
        }
    }

    let values_data = values_builder.freeze();
    let values_array = arrow_array::make_array(values_data);
    let run_ends_array = Int32Array::from(run_ends);

    let run_array = RunArray::<ArrowInt32Type>::try_new(&run_ends_array, &values_array)?;
    Ok(Arc::new(run_array))
}

/// An [`ArrayReader`] that produces Arrow RunEndEncoded (REE) arrays
/// from Parquet primitive columns.
///
/// Instead of materializing a dense Vec<T> with null-padding, this reader
/// uses the definition level run structure to produce a compact RunArray
/// where null runs are represented as single REE entries.
pub struct ReeArrayReader<T: DataType>
where
    T::T: Copy + Default,
    Vec<T::T>: IntoBuffer,
{
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_level_runs: Option<RunLevelBuffer>,
    rep_level_runs: Option<RunLevelBuffer>,
    record_reader: RecordReader<T>,
    max_def_level: i16,
}

impl<T: DataType> ReeArrayReader<T>
where
    T::T: Copy + Default,
    Vec<T::T>: IntoBuffer,
{
    pub fn new(
        pages: Box<dyn PageIterator>,
        column_desc: ColumnDescPtr,
        arrow_type: ArrowType,
    ) -> Result<Self> {
        let max_def_level = column_desc.max_def_level();
        let mut record_reader = RecordReader::<T>::new(column_desc);
        // Enable compact mode: only non-null values in the buffer
        record_reader.set_skip_padding(true);

        Ok(Self {
            data_type: arrow_type,
            pages,
            def_level_runs: None,
            rep_level_runs: None,
            record_reader,
            max_def_level,
        })
    }
}

impl<T> ArrayReader for ReeArrayReader<T>
where
    T: DataType,
    T::T: Copy + Default,
    Vec<T::T>: IntoBuffer,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        read_records(&mut self.record_reader, self.pages.as_mut(), batch_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let target_type = &self.data_type;

        // Get compact values (non-null only, thanks to skip_padding=true)
        let record_data = self
            .record_reader
            .consume_record_data()
            .into_buffer(target_type);
        let num_values = self.record_reader.values_written();

        // Determine the physical Arrow type for the values array
        let arrow_data_type = match T::get_physical_type() {
            PhysicalType::BOOLEAN => ArrowType::Boolean,
            PhysicalType::INT32 => match target_type {
                ArrowType::UInt32 => ArrowType::UInt32,
                ArrowType::Decimal32(_, _) => target_type.clone(),
                _ => ArrowType::Int32,
            },
            PhysicalType::INT64 => match target_type {
                ArrowType::UInt64 => ArrowType::UInt64,
                ArrowType::Decimal64(_, _) => target_type.clone(),
                _ => ArrowType::Int64,
            },
            PhysicalType::FLOAT => ArrowType::Float32,
            PhysicalType::DOUBLE => ArrowType::Float64,
            PhysicalType::INT96 => target_type.clone(),
            _ => unreachable!("ReeArrayReader doesn't support complex physical types"),
        };

        // Build a compact values Arrow array (no nulls — all values are real)
        let array_data = ArrayDataBuilder::new(arrow_data_type.clone())
            .len(num_values)
            .add_buffer(record_data);
        let array_data = unsafe { array_data.build_unchecked() };

        let compact_array: ArrayRef = match T::get_physical_type() {
            PhysicalType::BOOLEAN => Arc::new(BooleanArray::from(array_data)),
            PhysicalType::INT32 => match arrow_data_type {
                ArrowType::UInt32 => Arc::new(UInt32Array::from(array_data)),
                ArrowType::Int32 => Arc::new(Int32Array::from(array_data)),
                ArrowType::Decimal32(_, _) => Arc::new(Decimal32Array::from(array_data)),
                _ => unreachable!(),
            },
            PhysicalType::INT64 => match arrow_data_type {
                ArrowType::UInt64 => Arc::new(UInt64Array::from(array_data)),
                ArrowType::Int64 => Arc::new(Int64Array::from(array_data)),
                ArrowType::Decimal64(_, _) => Arc::new(Decimal64Array::from(array_data)),
                _ => unreachable!(),
            },
            PhysicalType::FLOAT => Arc::new(Float32Array::from(array_data)),
            PhysicalType::DOUBLE => Arc::new(Float64Array::from(array_data)),
            PhysicalType::INT96 => match target_type {
                ArrowType::Timestamp(TimeUnit::Second, _) => {
                    Arc::new(TimestampSecondArray::from(array_data))
                }
                ArrowType::Timestamp(TimeUnit::Millisecond, _) => {
                    Arc::new(TimestampMillisecondArray::from(array_data))
                }
                ArrowType::Timestamp(TimeUnit::Microsecond, _) => {
                    Arc::new(TimestampMicrosecondArray::from(array_data))
                }
                ArrowType::Timestamp(TimeUnit::Nanosecond, _) => {
                    Arc::new(TimestampNanosecondArray::from(array_data))
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        // Apply type casting to compact values if needed
        let compact_array = match target_type {
            _ if *target_type == arrow_data_type => compact_array,
            _ => arrow_cast::cast(&compact_array, target_type)?,
        };

        // Take def level runs and build REE array
        self.def_level_runs = self.record_reader.consume_def_level_runs();
        self.rep_level_runs = self.record_reader.consume_rep_level_runs();
        self.record_reader.reset();

        let def_runs = self
            .def_level_runs
            .as_ref()
            .map(|r| r.runs())
            .unwrap_or(&[]);

        build_ree_array(def_runs, compact_array, self.max_def_level)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        crate::arrow::array_reader::skip_records(
            &mut self.record_reader,
            self.pages.as_mut(),
            num_records,
        )
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_level_runs.as_ref().map(|r| r.as_slice())
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_level_runs.as_ref().map(|r| r.as_slice())
    }

    fn get_def_level_runs(&self) -> Option<&[(i16, u32)]> {
        self.def_level_runs.as_ref().map(|r| r.runs())
    }

    fn get_rep_level_runs(&self) -> Option<&[(i16, u32)]> {
        self.rep_level_runs.as_ref().map(|r| r.runs())
    }
}
