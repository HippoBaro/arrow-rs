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

//! Utilities for building Arrow RunEndEncoded (REE) arrays from
//! Parquet definition level runs and compact (non-null-only) values.

use crate::errors::Result;
use arrow_array::types::Int32Type as ArrowInt32Type;
use arrow_array::{ArrayRef, Int32Array, RunArray};
use arrow_schema::DataType as ArrowType;
use std::sync::Arc;

/// Build a `RunArray<Int32Type>` from definition level runs and compact
/// (non-null) values.
///
/// The def level runs describe which positions are null vs non-null.
/// The compact_values array contains only the non-null values (produced
/// by a record reader with `skip_padding=true`).
///
/// When `value_runs` is provided (from dictionary-encoded columns), consecutive
/// non-null values with the same dictionary index are collapsed into single REE
/// entries instead of expanding each value individually.
///
/// For a 99% sparse column with 100K rows and 1K values, this produces
/// a RunArray with ~3 physical entries instead of 100K dense elements.
pub(crate) fn build_ree_array(
    def_runs: &[(i16, u32)],
    compact_values: ArrayRef,
    max_def_level: i16,
    value_runs: Option<&[(u32, u32)]>,
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

    // Build run_ends and values array in a single pass, avoiding an
    // intermediate index vector. Values are built directly using
    // MutableArrayData with batched extend calls.
    let mut run_ends: Vec<i32> = Vec::new();
    let mut cumulative: i32 = 0;
    let mut value_offset: usize = 0;

    let compact_data = compact_values.to_data();
    let mut values_builder = arrow_data::transform::MutableArrayData::new(
        vec![&compact_data],
        true,
        merged_runs.len(), // lower-bound estimate
    );

    let mut vr_iter = value_runs.map(ValueRunIterator::new);

    for &(is_value, count) in &merged_runs {
        if is_value {
            if let Some(ref mut vr) = vr_iter {
                // With value runs: collapse identical consecutive values.
                // Each value run produces one REE entry (one extend call).
                let mut remaining = count;
                while remaining > 0 {
                    let (_, run_count) = vr.consume(remaining);
                    cumulative += run_count as i32;
                    run_ends.push(cumulative);
                    values_builder.extend(0, value_offset, value_offset + 1);
                    value_offset += run_count as usize;
                    remaining -= run_count;
                }
            } else {
                // No value runs: each value gets its own REE entry.
                // Batch the entire consecutive range in ONE extend call.
                let n = count as usize;
                for i in 0..n {
                    cumulative += 1;
                    run_ends.push(cumulative);
                }
                values_builder.extend(0, value_offset, value_offset + n);
                value_offset += n;
            }
        } else {
            // Entire null run collapses to one REE entry
            cumulative += count as i32;
            run_ends.push(cumulative);
            values_builder.extend_nulls(1);
        }
    }

    let values_data = values_builder.freeze();
    let values_array = arrow_array::make_array(values_data);
    let run_ends_array = Int32Array::from(run_ends);

    let run_array = RunArray::<ArrowInt32Type>::try_new(&run_ends_array, &values_array)?;
    Ok(Arc::new(run_array))
}

/// Build a `RunArray<Int32Type>` directly from dictionary values and value runs,
/// without requiring an expanded flat values array.
///
/// `dict_values` is the Arrow array of unique dictionary entries (N unique strings).
/// `value_runs` contains `(dict_index, count)` pairs referencing positions in `dict_values`.
/// `def_runs` describes null/non-null regions at the record level.
pub(crate) fn build_ree_array_from_dict(
    def_runs: &[(i16, u32)],
    dict_values: ArrayRef,
    max_def_level: i16,
    value_runs: &[(u32, u32)],
) -> Result<ArrayRef> {
    if def_runs.is_empty() {
        let run_ends = Int32Array::from(Vec::<i32>::new());
        let values = arrow_array::new_empty_array(dict_values.data_type());
        let run_array = RunArray::<ArrowInt32Type>::try_new(&run_ends, &values)?;
        return Ok(Arc::new(run_array));
    }

    // Merge adjacent def runs with the same null/non-null classification.
    let mut merged_runs: Vec<(bool, u32)> = Vec::new();
    for &(def_val, count) in def_runs {
        let is_value = def_val >= max_def_level;
        if let Some(last) = merged_runs.last_mut().filter(|r| r.0 == is_value) {
            last.1 += count;
        } else {
            merged_runs.push((is_value, count));
        }
    }

    let mut run_ends: Vec<i32> = Vec::new();
    let mut cumulative: i32 = 0;

    let dict_data = dict_values.to_data();
    let mut values_builder = arrow_data::transform::MutableArrayData::new(
        vec![&dict_data],
        true,
        merged_runs.len(),
    );

    let mut vr_iter = ValueRunIterator::new(value_runs);

    for &(is_value, count) in &merged_runs {
        if is_value {
            // Each value run produces one REE entry, indexing into the
            // dictionary by dict_index instead of into a flat expanded array.
            let mut remaining = count;
            while remaining > 0 {
                let (dict_index, run_count) = vr_iter.consume(remaining);
                cumulative += run_count as i32;
                run_ends.push(cumulative);
                let idx = dict_index as usize;
                values_builder.extend(0, idx, idx + 1);
                remaining -= run_count;
            }
        } else {
            // Entire null run collapses to one REE entry.
            cumulative += count as i32;
            run_ends.push(cumulative);
            values_builder.extend_nulls(1);
        }
    }

    let values_data = values_builder.freeze();
    let values_array = arrow_array::make_array(values_data);
    let run_ends_array = Int32Array::from(run_ends);

    let run_array = RunArray::<ArrowInt32Type>::try_new(&run_ends_array, &values_array)?;
    Ok(Arc::new(run_array))
}

/// Iterator that consumes value runs `(dict_index, count)` and supports
/// partial consumption of individual entries.
struct ValueRunIterator<'a> {
    runs: &'a [(u32, u32)],
    pos: usize,
    /// Remaining count in the current run entry
    current_remaining: u32,
}

impl<'a> ValueRunIterator<'a> {
    fn new(runs: &'a [(u32, u32)]) -> Self {
        let current_remaining = runs.first().map_or(0, |r| r.1);
        Self {
            runs,
            pos: 0,
            current_remaining,
        }
    }

    /// Consume up to `max` values from the current run, returning
    /// `(dict_index, count_consumed)`. All consumed values share the
    /// same dictionary index.
    fn consume(&mut self, max: u32) -> (u32, u32) {
        let (idx, _) = self.runs[self.pos];
        let take = max.min(self.current_remaining);
        self.current_remaining -= take;
        if self.current_remaining == 0 {
            self.pos += 1;
            if self.pos < self.runs.len() {
                self.current_remaining = self.runs[self.pos].1;
            }
        }
        (idx, take)
    }
}

/// Extract `(key, count)` runs from a `DictionaryArray` by scanning its keys.
/// Returns `None` if the array is not a dictionary type.
/// The keys array is compact (no nulls expected since it comes from skip_padding mode).
fn extract_key_runs_from_dict_array(array: &ArrayRef) -> Option<Vec<(u32, u32)>> {
    use arrow_array::cast::AsArray;

    macro_rules! extract_runs {
        ($keys:expr) => {{
            let mut runs = Vec::new();
            for key in $keys.values().iter() {
                let k = *key as u32;
                if let Some(last) = runs.last_mut().filter(|(idx, _): &&mut (u32, u32)| *idx == k) {
                    last.1 += 1;
                } else {
                    runs.push((k, 1u32));
                }
            }
            Some(runs)
        }};
    }

    match array.data_type() {
        ArrowType::Dictionary(key_type, _) => match key_type.as_ref() {
            ArrowType::Int8 => extract_runs!(array.as_dictionary::<arrow_array::types::Int8Type>().keys()),
            ArrowType::Int16 => extract_runs!(array.as_dictionary::<arrow_array::types::Int16Type>().keys()),
            ArrowType::Int32 => extract_runs!(array.as_dictionary::<arrow_array::types::Int32Type>().keys()),
            ArrowType::Int64 => extract_runs!(array.as_dictionary::<arrow_array::types::Int64Type>().keys()),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_array::Int32Array;

    #[test]
    fn test_build_ree_array_with_value_runs() {
        // All non-null: def_runs = [(1, 9)] (all max_def_level=1)
        // Values: [1, 1, 1, 1, 1, 2, 2, 2, 3]
        // Value runs: [(0, 5), (1, 3), (2, 1)] — dict indices
        let def_runs: &[(i16, u32)] = &[(1, 9)];
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 1, 1, 1, 1, 2, 2, 2, 3]));
        let value_runs: &[(u32, u32)] = &[(0, 5), (1, 3), (2, 1)];

        let result = build_ree_array(def_runs, values, 1, Some(value_runs)).unwrap();
        let run_array = result
            .as_any()
            .downcast_ref::<RunArray<ArrowInt32Type>>()
            .unwrap();

        assert_eq!(run_array.len(), 9);
        assert_eq!(
            run_array.run_ends().values(),
            &[5, 8, 9],
            "run_ends mismatch: got {:?}",
            run_array.run_ends().values()
        );

        let vals = run_array.values().as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(vals.value(0), 1);
        assert_eq!(vals.value(1), 2);
        assert_eq!(vals.value(2), 3);
    }

    #[test]
    fn test_build_ree_array_with_value_runs_and_nulls() {
        // [1, 1, NULL, NULL, 2, 2, 2, NULL, 3]
        // def_runs: [(1, 2), (0, 2), (1, 3), (0, 1), (1, 1)]
        // compact values (non-null only): [1, 1, 2, 2, 2, 3]
        // value runs (covering compact values): [(0, 2), (1, 3), (2, 1)]
        let def_runs: &[(i16, u32)] = &[(1, 2), (0, 2), (1, 3), (0, 1), (1, 1)];
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2, 3]));
        let value_runs: &[(u32, u32)] = &[(0, 2), (1, 3), (2, 1)];

        let result = build_ree_array(def_runs, values, 1, Some(value_runs)).unwrap();
        let run_array = result
            .as_any()
            .downcast_ref::<RunArray<ArrowInt32Type>>()
            .unwrap();

        assert_eq!(run_array.len(), 9);
        // Expected: [1x2, NULLx2, 2x3, NULLx1, 3x1] = 5 runs
        assert_eq!(
            run_array.run_ends().values(),
            &[2, 4, 7, 8, 9],
            "Got {} runs: {:?}",
            run_array.run_ends().len(),
            run_array.run_ends().values()
        );
    }

    #[test]
    fn test_build_ree_array_without_value_runs() {
        // Same data but no value runs → one entry per non-null value
        let def_runs: &[(i16, u32)] = &[(1, 3)];
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));

        let result = build_ree_array(def_runs, values, 1, None).unwrap();
        let run_array = result
            .as_any()
            .downcast_ref::<RunArray<ArrowInt32Type>>()
            .unwrap();

        assert_eq!(run_array.len(), 3);
        assert_eq!(run_array.run_ends().len(), 3); // No collapsing
        assert_eq!(run_array.run_ends().values(), &[1, 2, 3]);
    }
}
