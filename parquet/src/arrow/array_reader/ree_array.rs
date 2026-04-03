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
    /// The RunEndEncoded data type (returned by get_data_type)
    ree_data_type: ArrowType,
    /// The inner value type (used for building the compact values array)
    inner_data_type: ArrowType,
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
        record_reader.set_skip_padding(true);

        // Wrap the inner type in RunEndEncoded
        let ree_data_type = ArrowType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ArrowType::Int32, false)),
            Arc::new(Field::new("values", arrow_type.clone(), true)),
        );

        Ok(Self {
            ree_data_type,
            inner_data_type: arrow_type,
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
        &self.ree_data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        read_records(&mut self.record_reader, self.pages.as_mut(), batch_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let target_type = &self.inner_data_type;

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

/// Convert a dense array (which may contain nulls) into a RunEndEncoded
/// array by collapsing consecutive null entries into single runs.
/// Non-null entries get one REE entry each.
///
/// For an all-null array of 100K entries, this produces a RunArray with
/// 1 physical entry (~9 bytes) instead of 100K dense entries.
pub(crate) fn dense_to_ree(dense: ArrayRef) -> Result<ArrayRef> {
    let len = dense.len();
    if len == 0 {
        let run_ends = Int32Array::from(Vec::<i32>::new());
        let values = arrow_array::new_empty_array(dense.data_type());
        let run_array = RunArray::<ArrowInt32Type>::try_new(&run_ends, &values)?;
        return Ok(Arc::new(run_array));
    }

    // Scan the validity bitmap to identify null and non-null runs.
    let mut run_ends: Vec<i32> = Vec::new();
    let mut value_indices: Vec<Option<usize>> = Vec::new();

    let mut i = 0;
    while i < len {
        if dense.is_null(i) {
            // Start of a null run — find how far it extends
            let start = i;
            while i < len && dense.is_null(i) {
                i += 1;
            }
            // One REE entry for the entire null run
            run_ends.push(i as i32);
            value_indices.push(None);
        } else {
            // Single non-null entry
            run_ends.push((i + 1) as i32);
            value_indices.push(Some(i));
            i += 1;
        }
    }

    // Build the values array using MutableArrayData
    let dense_data = dense.to_data();
    let mut values_builder =
        arrow_data::transform::MutableArrayData::new(vec![&dense_data], true, value_indices.len());

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

/// A generic [`ArrayReader`] wrapper that converts any inner reader's
/// dense output into RunEndEncoded (REE) arrays.
///
/// Delegates all operations to the inner reader, then wraps the output
/// of `consume_batch` in a `RunArray` by collapsing null runs.
///
/// Works for any column type: List, ByteArray, Primitive, Struct, etc.
pub struct ReeWrappingReader {
    inner: Box<dyn ArrayReader>,
    ree_data_type: ArrowType,
}

impl ReeWrappingReader {
    pub fn new(inner: Box<dyn ArrayReader>) -> Self {
        let inner_type = inner.get_data_type().clone();
        let ree_data_type = ArrowType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ArrowType::Int32, false)),
            Arc::new(Field::new("values", inner_type, true)),
        );
        Self {
            inner,
            ree_data_type,
        }
    }
}

impl ReeWrappingReader {
    /// Check if the buffered batch is entirely null by inspecting the
    /// inner reader's def level runs. Returns Some(num_rows) if all-null.
    fn check_all_null(&self) -> Option<usize> {
        let runs = self.inner.peek_def_level_runs()?;
        // All null if every run has def_level == 0
        if runs.iter().all(|(def, _)| *def == 0) {
            let total: usize = runs.iter().map(|(_, c)| *c as usize).sum();
            if total > 0 {
                return Some(total);
            }
        }
        None
    }
}

impl ArrayReader for ReeWrappingReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.ree_data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        self.inner.read_records(batch_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        // Short-circuit: if the entire batch is null, produce a compact
        // REE null array directly WITHOUT materializing the dense array.
        // This avoids the O(rows) allocation that List/ByteArray columns
        // incur even when all-null.
        if let Some(num_rows) = self.check_all_null() {
            // Discard the buffered data without materializing
            self.inner.discard_batch()?;

            // Produce a single-run REE null array (~9 bytes)
            let run_ends = Int32Array::from(vec![num_rows as i32]);
            let inner_type = match &self.ree_data_type {
                ArrowType::RunEndEncoded(_, v) => v.data_type(),
                _ => unreachable!(),
            };
            let null_value = arrow_array::new_null_array(inner_type, 1);
            let run_array =
                RunArray::<ArrowInt32Type>::try_new(&run_ends, &null_value)?;
            return Ok(Arc::new(run_array));
        }

        let dense = self.inner.consume_batch()?;
        dense_to_ree(dense)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        self.inner.skip_records(num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.inner.get_def_levels()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.inner.get_rep_levels()
    }

    fn get_def_level_runs(&self) -> Option<&[(i16, u32)]> {
        self.inner.get_def_level_runs()
    }

    fn get_rep_level_runs(&self) -> Option<&[(i16, u32)]> {
        self.inner.get_rep_level_runs()
    }
}
