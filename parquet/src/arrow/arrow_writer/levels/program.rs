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

use super::{LevelContext, LevelData};
use crate::column::value::{SelectionRange, ValueSelectionRef};
use crate::column::writer::LevelDataRef;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::NullBuffer;

pub(super) const LEVEL_RUN_PROBE_SIZE: usize = 128;
pub(super) const MIN_AVERAGE_LEVEL_RUN_LENGTH: usize = 8;
use std::ops::Range;
use std::sync::OnceLock;

/// Owned value positions for a leaf column.
///
/// [`LeafPlanBuilder`] owns this while levels are built. Writers borrow it
/// through [`LeafBatch`] as a [`ValueSelectionRef`], allowing coordinated
/// windows without slicing or rebasing the Arrow array.
#[derive(Debug, Clone)]
pub(crate) enum ValueSelection {
    /// No leaf values are written.
    Empty,
    /// A contiguous range of positions in the leaf array.
    Dense { offset: usize, len: usize },
    /// Ordered contiguous ranges retained while their metadata is smaller
    /// than one explicit index per selected value.
    Ranges {
        ranges: Vec<SelectionRange>,
        len: usize,
    },
    /// Bitmap-backed positions materialized only when a writer borrows them.
    DeferredSparse {
        nulls: NullBuffer,
        offset: usize,
        len: usize,
        indices: OnceLock<Vec<usize>>,
    },
    /// Explicit positions in the leaf array.
    Sparse(Vec<usize>),
}

impl ValueSelection {
    pub(crate) fn as_ref(&self) -> ValueSelectionRef<'_> {
        match self {
            Self::Empty => ValueSelectionRef::Empty,
            Self::Dense { offset, len } => ValueSelectionRef::Dense {
                offset: *offset,
                len: *len,
            },
            Self::Ranges { ranges, len } => ValueSelectionRef::Ranges(
                crate::column::value::RangesSelectionRef::new(ranges, *len),
            ),
            Self::DeferredSparse {
                nulls,
                offset,
                len,
                indices,
            } => ValueSelectionRef::Sparse(
                indices.get_or_init(|| Self::collect_sparse(nulls, *offset, *len)),
            ),
            Self::Sparse(indices) => ValueSelectionRef::Sparse(indices),
        }
    }

    pub(super) fn append_range(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let range_len = range.end - range.start;

        match self {
            Self::Empty => {
                *self = Self::Dense {
                    offset: range.start,
                    len: range_len,
                };
            }
            Self::Dense { offset, len } if *offset + *len == range.start => {
                *len += range_len;
            }
            Self::Dense { offset, len } => {
                let (offset, len) = (*offset, *len);
                if range.start < offset + len {
                    let mut indices: Vec<_> = (offset..offset + len).collect();
                    indices.extend(range);
                    *self = Self::Sparse(indices);
                    return;
                }
                let ranges = vec![
                    SelectionRange::new(offset..offset + len, len),
                    SelectionRange::new(range.clone(), len + range_len),
                ];
                let total_len = len + range_len;
                if Self::ranges_are_expensive(ranges.len(), total_len) {
                    let mut indices: Vec<_> = (offset..offset + len).collect();
                    indices.extend(range);
                    *self = Self::Sparse(indices);
                } else {
                    *self = Self::Ranges {
                        ranges,
                        len: total_len,
                    };
                }
            }
            Self::Ranges { ranges, len } => {
                let last_index = ranges.len() - 1;
                let selected_start = last_index
                    .checked_sub(1)
                    .map_or(0, |previous| ranges[previous].selected_end);
                let last_end = ranges[last_index].source_range(selected_start).end;
                if last_end == range.start {
                    let last = ranges.last_mut().unwrap();
                    last.selected_end += range_len;
                } else if range.start < last_end {
                    let mut indices = Vec::with_capacity(*len + range_len);
                    indices.extend(ranges.iter().enumerate().flat_map(|(idx, range)| {
                        let start = idx
                            .checked_sub(1)
                            .map_or(0, |previous| ranges[previous].selected_end);
                        range.source_range(start)
                    }));
                    indices.extend(range);
                    *self = Self::Sparse(indices);
                    return;
                } else {
                    ranges.push(SelectionRange::new(range, *len + range_len));
                }
                *len += range_len;
                if Self::ranges_are_expensive(ranges.len(), *len) {
                    let indices = ranges
                        .iter()
                        .enumerate()
                        .flat_map(|(idx, range)| {
                            let start = idx
                                .checked_sub(1)
                                .map_or(0, |previous| ranges[previous].selected_end);
                            range.source_range(start)
                        })
                        .collect();
                    *self = Self::Sparse(indices);
                }
            }
            Self::DeferredSparse {
                nulls,
                offset,
                len,
                indices,
            } => {
                let mut materialized = indices
                    .take()
                    .unwrap_or_else(|| Self::collect_sparse(nulls, *offset, *len));
                materialized.extend(range);
                *self = Self::Sparse(materialized);
            }
            Self::Sparse(indices) => indices.extend(range),
        }
    }

    /// Append explicit non-null positions, reserving for the known number of
    /// values.
    pub(super) fn extend_sparse_indices<I>(&mut self, value_count: usize, iter: I)
    where
        I: IntoIterator<Item = usize>,
    {
        let iter = iter.into_iter();
        let mut materialized = match self {
            Self::Empty => Vec::with_capacity(value_count),
            Self::Dense { offset, len } => {
                let mut indices = Vec::with_capacity(*len + value_count);
                indices.extend(*offset..*offset + *len);
                indices
            }
            Self::Ranges { ranges, len, .. } => {
                let mut indices = Vec::with_capacity(*len + value_count);
                let mut selected_start = 0;
                for range in ranges {
                    indices.extend(range.source_range(selected_start));
                    selected_start = range.selected_end;
                }
                indices
            }
            Self::DeferredSparse {
                nulls,
                offset,
                len,
                indices,
            } => {
                let mut indices = indices
                    .take()
                    .unwrap_or_else(|| Self::collect_sparse(nulls, *offset, *len));
                indices.reserve(value_count);
                indices
            }
            Self::Sparse(indices) => {
                indices.reserve(value_count);
                indices.extend(iter);
                return;
            }
        };
        materialized.extend(iter);
        if !materialized.is_empty() {
            *self = Self::Sparse(materialized);
        }
    }

    fn append_deferred_sparse(&mut self, nulls: NullBuffer, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        if matches!(self, Self::Empty) {
            *self = Self::DeferredSparse {
                nulls,
                offset,
                len,
                indices: OnceLock::new(),
            };
        } else {
            self.extend_sparse_indices(len, nulls.valid_indices().map(|index| index + offset));
        }
    }

    fn collect_sparse(nulls: &NullBuffer, offset: usize, len: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(len);
        indices.extend(nulls.valid_indices().map(|index| index + offset));
        debug_assert_eq!(indices.len(), len);
        indices
    }

    fn ranges_are_expensive(num_ranges: usize, num_values: usize) -> bool {
        num_ranges.saturating_mul(std::mem::size_of::<SelectionRange>())
            >= num_values.saturating_mul(std::mem::size_of::<usize>())
    }

    /// Releases geometric spare capacity so a completed plan retains the
    /// metadata savings of this representation.
    fn finish(&mut self) {
        if let Self::Ranges { ranges, .. } = self {
            ranges.shrink_to_fit();
        }
    }
}

/// One borrowed batch presented to the column writer.
#[derive(Clone, Copy)]
pub(crate) struct LeafBatch<'a> {
    array: &'a (dyn Array + 'static),
    def_levels: LevelDataRef<'a>,
    rep_levels: LevelDataRef<'a>,
    values: ValueSelectionRef<'a>,
}

impl<'a> LeafBatch<'a> {
    pub(crate) fn new(
        array: &'a (dyn Array + 'static),
        def_levels: LevelDataRef<'a>,
        rep_levels: LevelDataRef<'a>,
        values: ValueSelectionRef<'a>,
    ) -> Self {
        Self {
            array,
            def_levels,
            rep_levels,
            values,
        }
    }

    pub(crate) fn array(&self) -> &'a (dyn Array + 'static) {
        self.array
    }

    pub(crate) fn def_level_data(&self) -> LevelDataRef<'a> {
        self.def_levels
    }

    pub(crate) fn rep_level_data(&self) -> LevelDataRef<'a> {
        self.rep_levels
    }

    pub(crate) fn value_selection(&self) -> ValueSelectionRef<'a> {
        self.values
    }

    /// Slice only the streams and selection. Value indices remain absolute to
    /// the original Arrow array.
    pub(crate) fn slice(self, window: LeafBatchSlice) -> Self {
        Self {
            array: self.array,
            def_levels: self
                .def_levels
                .slice(window.level_offset, window.num_levels),
            rep_levels: self
                .rep_levels
                .slice(window.level_offset, window.num_levels),
            values: self.values.slice(window.value_offset, window.num_values),
        }
    }
}

/// A coordinated window into a leaf batch's level and selected-value streams.
///
/// Level and value offsets are separate because null slots occupy the former
/// but not the latter. Value indices themselves remain absolute to the batch's
/// Arrow array.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LeafBatchSlice {
    pub(crate) level_offset: usize,
    pub(crate) num_levels: usize,
    pub(crate) value_offset: usize,
    pub(crate) num_values: usize,
}

/// Owned level streams and value selection for one write batch.
#[derive(Debug, Clone)]
pub(crate) struct BatchPlan {
    /// Definition levels (present if `max_def_level != 0`).
    pub(super) def_levels: LevelData,
    /// Repetition levels (present if `max_rep_level != 0`).
    pub(super) rep_levels: LevelData,
    /// Value positions in the leaf array to write.
    pub(super) values: ValueSelection,
}

impl BatchPlan {
    /// A fresh, empty triple for a leaf with the given level bounds.
    pub(super) fn empty(max_def_level: i16, max_rep_level: i16) -> Self {
        Self {
            def_levels: LevelData::new(max_def_level != 0),
            rep_levels: LevelData::new(max_rep_level != 0),
            values: ValueSelection::Empty,
        }
    }

    fn finish(&mut self) {
        self.def_levels.finish();
        self.rep_levels.finish();
        self.values.finish();
    }

    /// Validates the three coordinated streams at the construction boundary,
    /// where the builder knows the leaf's level bounds. The column writer relies
    /// on these relationships for unchecked, count-based slicing of both levels
    /// and values.
    #[cfg(debug_assertions)]
    fn assert_valid(&self, max_def_level: i16, max_rep_level: i16) {
        let has_def_levels = !matches!(self.def_levels, LevelData::Absent);
        let has_rep_levels = !matches!(self.rep_levels, LevelData::Absent);
        assert_eq!(
            has_def_levels,
            max_def_level != 0,
            "definition-level representation disagrees with max definition level"
        );
        assert_eq!(
            has_rep_levels,
            max_rep_level != 0,
            "repetition-level representation disagrees with max repetition level"
        );

        let def_len = self.def_levels.len();
        let rep_len = self.rep_levels.len();
        if has_def_levels && has_rep_levels {
            assert_eq!(
                def_len, rep_len,
                "definition and repetition streams must describe the same leaf slots"
            );
        }

        let num_levels = if has_def_levels {
            def_len
        } else if has_rep_levels {
            rep_len
        } else {
            self.values.as_ref().len()
        };
        let expected_values = self
            .def_levels
            .as_ref()
            .value_count(num_levels, max_def_level);
        assert_eq!(
            self.values.as_ref().len(),
            expected_values,
            "selected value count must equal the max-definition occurrences"
        );
    }

    /// Borrow this triple as a write view over `array`.
    pub(crate) fn view<'a>(&'a self, array: &'a (dyn Array + 'static)) -> LeafBatch<'a> {
        LeafBatch {
            array,
            def_levels: self.def_levels.as_ref(),
            rep_levels: self.rep_levels.as_ref(),
            values: self.values.as_ref(),
        }
    }
}

/// Finalized write plan for one primitive Parquet leaf column.
#[derive(Debug, Clone)]
pub(crate) struct LeafPlan {
    pub(super) array: ArrayRef,
    pub(super) batch: BatchPlan,
}

impl LeafPlan {
    pub(crate) fn view(&self) -> LeafBatch<'_> {
        self.batch.view(self.array.as_ref())
    }
}

/// Mutable construction state for one dense leaf batch.
#[derive(Debug, Clone)]
pub(super) struct LeafPlanBuilder {
    /// The appendable `(def, rep, values)` batch.
    pub(super) tail: BatchPlan,

    /// The maximum definition level for this leaf column
    pub(super) max_def_level: i16,

    /// The maximum repetition for this leaf column
    pub(super) max_rep_level: i16,

    /// The arrow array
    pub(super) array: ArrayRef,
}

impl LeafPlanBuilder {
    pub(super) fn new(ctx: LevelContext, is_nullable: bool, array: ArrayRef) -> Self {
        let max_rep_level = ctx.rep_level;
        let max_def_level = ctx.def_level + is_nullable as i16;

        Self {
            tail: BatchPlan::empty(max_def_level, max_rep_level),
            max_def_level,
            max_rep_level,
            array,
        }
    }

    #[inline(always)]
    pub(super) fn finish(mut self) -> LeafPlan {
        self.tail.finish();
        let Self {
            tail,
            array,
            max_def_level,
            max_rep_level,
        } = self;
        #[cfg(not(debug_assertions))]
        let _ = (max_def_level, max_rep_level);
        #[cfg(debug_assertions)]
        tail.assert_valid(max_def_level, max_rep_level);
        LeafPlan { array, batch: tail }
    }

    /// Bulk-emit `count` uniform def/rep levels.
    pub(super) fn extend_uniform_levels(&mut self, def_val: i16, rep_val: i16, count: usize) {
        self.tail.def_levels.append_run(def_val, count);
        self.tail.rep_levels.append_run(rep_val, count);
    }

    pub(super) fn append_value_range(&mut self, range: Range<usize>) {
        self.tail.values.append_range(range);
    }

    pub(super) fn append_deferred_sparse_values(
        &mut self,
        nulls: NullBuffer,
        offset: usize,
        len: usize,
    ) {
        self.tail.values.append_deferred_sparse(nulls, offset, len);
    }

    pub(super) fn append_def_level_run(&mut self, value: i16, count: usize) {
        self.tail.def_levels.append_run(value, count);
    }

    pub(super) fn append_rep_level_run(&mut self, value: i16, count: usize) {
        self.tail.rep_levels.append_run(value, count);
    }

    pub(super) fn extend_def_levels<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = i16>,
    {
        self.tail.def_levels.extend_from_iter(iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::value::ValueSelectionRef;
    use crate::column::writer::LevelDataRef;
    use arrow_array::{ArrayRef, Int32Array};
    use std::sync::Arc;

    fn only_batch_view(plan: &LeafPlan) -> LeafBatch<'_> {
        plan.view()
    }

    fn batch_plan(
        array: ArrayRef,
        max_def_level: i16,
        max_rep_level: i16,
        def_levels: LevelData,
        rep_levels: LevelData,
        values: ValueSelection,
    ) -> LeafPlan {
        LeafPlanBuilder {
            tail: BatchPlan {
                def_levels,
                rep_levels,
                values,
            },
            max_def_level,
            max_rep_level,
            array,
        }
        .finish()
    }

    fn assert_batch_slice(
        plan: &LeafPlan,
        window: LeafBatchSlice,
        def_levels: LevelDataRef<'_>,
        rep_levels: LevelDataRef<'_>,
        values: ValueSelectionRef<'_>,
    ) {
        let view = only_batch_view(plan).slice(window);
        assert_eq!(view.def_level_data(), def_levels);
        assert_eq!(view.rep_level_data(), rep_levels);
        assert_eq!(view.value_selection(), values);
        assert_eq!(view.array().len(), plan.array.len());
    }

    fn materialize_levels(levels: LevelDataRef<'_>) -> Vec<i16> {
        (0..levels.len())
            .map(|idx| levels.value_at(idx).expect("level in bounds"))
            .collect()
    }

    fn collect_selection(selection: ValueSelectionRef<'_>) -> Vec<usize> {
        let mut values = Vec::with_capacity(selection.len());
        selection
            .try_for_each(|idx| -> Result<(), ()> {
                values.push(idx);
                Ok(())
            })
            .unwrap();
        values
    }

    fn assert_levels(data: &LevelData, expected: &[i16]) {
        assert_eq!(materialize_levels(data.as_ref()), expected);
    }

    fn assert_selection(selection: &ValueSelection, expected: impl IntoIterator<Item = usize>) {
        assert_eq!(
            collect_selection(selection.as_ref()),
            expected.into_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn leaf_batch_slice_flat() {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6]));
        let levels = batch_plan(
            array,
            0,
            0,
            LevelData::Absent,
            LevelData::Absent,
            ValueSelection::Dense { offset: 0, len: 6 },
        );
        assert_batch_slice(
            &levels,
            LeafBatchSlice {
                level_offset: 0,
                num_levels: 0,
                value_offset: 2,
                num_values: 3,
            },
            LevelDataRef::Absent,
            LevelDataRef::Absent,
            ValueSelectionRef::Dense { offset: 2, len: 3 },
        );

        let array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6]));
        let levels = batch_plan(
            array,
            1,
            0,
            LevelData::Uniform { value: 1, count: 6 },
            LevelData::Absent,
            ValueSelection::Dense { offset: 0, len: 6 },
        );
        assert_batch_slice(
            &levels,
            LeafBatchSlice {
                level_offset: 2,
                num_levels: 3,
                value_offset: 2,
                num_values: 3,
            },
            LevelDataRef::Uniform { value: 1, count: 3 },
            LevelDataRef::Absent,
            ValueSelectionRef::Dense { offset: 2, len: 3 },
        );

        let array: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
            Some(6),
        ]));
        let levels = batch_plan(
            array,
            1,
            0,
            LevelData::Materialized(vec![1, 0, 1, 0, 1, 1]),
            LevelData::Absent,
            ValueSelection::Sparse(vec![0, 2, 4, 5]),
        );
        assert_batch_slice(
            &levels,
            LeafBatchSlice {
                level_offset: 1,
                num_levels: 3,
                value_offset: 1,
                num_values: 1,
            },
            LevelDataRef::Materialized(&[0, 1, 0]),
            LevelDataRef::Absent,
            ValueSelectionRef::Sparse(&[2]),
        );
    }

    #[test]
    fn leaf_batch_slice_nested_with_nulls() {
        // Regression test for https://github.com/apache/arrow-rs/issues/9637
        // Null list entries own non-empty child ranges, leaving gaps in the
        // selected leaf positions: 0→3 (skip 1,2) and 3→8 (skip 5,6,7).
        let array: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1), // 0: row 0
            None,    // 1: gap (null list row 1)
            None,    // 2: gap (null list row 1)
            Some(2), // 3: row 2
            None,    // 4: row 2, null element
            None,    // 5: gap (null list row 3)
            None,    // 6: gap (null list row 3)
            None,    // 7: gap (null list row 3)
            Some(4), // 8: row 4
            Some(5), // 9: row 4
        ]));
        let levels = batch_plan(
            array,
            3,
            1,
            LevelData::Materialized(vec![3, 0, 3, 2, 0, 3, 3]),
            LevelData::Materialized(vec![0, 0, 0, 1, 0, 0, 1]),
            ValueSelection::Sparse(vec![0, 3, 8, 9]),
        );

        for (chunk, defs, reps, values) in [
            (
                LeafBatchSlice {
                    level_offset: 0,
                    num_levels: 2,
                    value_offset: 0,
                    num_values: 1,
                },
                &[3, 0][..],
                &[0, 0][..],
                &[0][..],
            ),
            (
                LeafBatchSlice {
                    level_offset: 2,
                    num_levels: 3,
                    value_offset: 1,
                    num_values: 1,
                },
                &[3, 2, 0],
                &[0, 1, 0],
                &[3],
            ),
            (
                LeafBatchSlice {
                    level_offset: 5,
                    num_levels: 2,
                    value_offset: 2,
                    num_values: 2,
                },
                &[3, 3],
                &[0, 1],
                &[8, 9],
            ),
        ] {
            assert_batch_slice(
                &levels,
                chunk,
                LevelDataRef::Materialized(defs),
                LevelDataRef::Materialized(reps),
                ValueSelectionRef::Sparse(values),
            );
        }
    }

    #[test]
    fn leaf_batch_slice_all_null() {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, None, Some(4)]));
        let levels = batch_plan(
            array,
            1,
            0,
            LevelData::Materialized(vec![1, 0, 0, 1]),
            LevelData::Absent,
            ValueSelection::Sparse(vec![0, 3]),
        );
        assert_batch_slice(
            &levels,
            LeafBatchSlice {
                level_offset: 1,
                num_levels: 2,
                value_offset: 1,
                num_values: 0,
            },
            LevelDataRef::Materialized(&[0, 0]),
            LevelDataRef::Absent,
            ValueSelectionRef::Sparse(&[]),
        );
    }

    #[test]
    fn level_data_retains_and_coalesces_runs() {
        let mut uniform = LevelData::Uniform { value: 1, count: 3 };
        uniform.append_run(1, 0);
        assert!(matches!(uniform, LevelData::Uniform { value: 1, count: 3 }));
        let mut materialized = LevelData::Materialized(vec![1, 2]);
        materialized.append_run(3, 0);
        assert_levels(&materialized, &[1, 2]);

        let mut data = LevelData::new(true);
        data.append_run(2, 4);
        assert!(matches!(data, LevelData::Uniform { value: 2, count: 4 }));

        data.append_run(2, 3);
        assert!(matches!(data, LevelData::Uniform { value: 2, count: 7 }));

        data.append_run(4, 5);
        let LevelData::Runs(runs) = &data else {
            panic!("expected runs, got {data:?}");
        };
        assert_eq!(runs.ends(), &[7, 12]);
        assert_eq!(runs.values(), &[2, 4]);

        // The adjacent equal run extends the cumulative end in place.
        data.append_run(4, 2);
        data.append_run(1, 1);
        let LevelData::Runs(runs) = &data else {
            panic!("expected runs, got {data:?}");
        };
        assert_eq!(runs.ends(), &[7, 14, 15]);
        assert_eq!(runs.values(), &[2, 4, 1]);
        assert_levels(
            &data,
            &vec![2; 7]
                .into_iter()
                .chain(vec![4; 7])
                .chain([1])
                .collect::<Vec<_>>(),
        );

        let expected: Vec<_> = std::iter::repeat_n(1, 64)
            .chain(std::iter::repeat_n(0, 32))
            .chain(std::iter::repeat_n(2, 64))
            .collect();
        let mut data = LevelData::new(true);
        data.append_run(1, 64);
        data.append_run(0, 32);
        data.append_run(2, 64);

        let LevelData::Runs(runs) = &data else {
            panic!("expected runs, got {data:?}");
        };
        assert_eq!(runs.ends(), &[64, 96, 160]);
        assert_eq!(runs.values(), &[1, 0, 2]);
        assert_levels(&data, &expected);
    }

    #[test]
    fn level_data_runs_adaptively_materialize_high_entropy() {
        let high_entropy: Vec<_> = (0..256).map(|idx| (idx % 2) as i16).collect();
        let mut data = LevelData::new(true);
        for &value in &high_entropy {
            data.append_run(value, 1);
        }
        assert_levels(&data, &high_entropy);
        assert!(matches!(data, LevelData::Materialized(_)));

        // Average run length exactly eight remains compact; only an average
        // strictly below eight triggers the one-way fallback.
        let eight_wide: Vec<_> = (0..32)
            .flat_map(|run| std::iter::repeat_n((run % 2) as i16, 8))
            .collect();
        let mut compact = LevelData::new(true);
        for run in 0..32 {
            compact.append_run((run % 2) as i16, 8);
        }
        assert!(matches!(compact, LevelData::Runs(_)));
        assert_levels(&compact, &eight_wide);

        let mut extended = LevelData::new(true);
        extended.extend_from_iter(high_entropy.iter().copied());
        assert!(matches!(extended, LevelData::Materialized(_)));

        // Once materialized, subsequent long runs don't reconstruct metadata.
        data.append_run(3, 64);
        assert!(matches!(data, LevelData::Materialized(_)));
    }

    #[test]
    fn level_data_runs_materialize_mut() {
        let mut data = LevelData::new(true);
        data.append_run(1, 3);
        data.append_run(2, 2);
        assert!(matches!(data, LevelData::Runs(_)));

        data.materialize_mut().unwrap().push(3);
        assert_levels(&data, &[1, 1, 1, 2, 2, 3]);
        let mut absent = LevelData::Absent;
        assert!(absent.materialize_mut().is_none());
    }

    #[test]
    fn value_selection_ranges_slice_and_traverse_in_value_space() {
        let mut dense = ValueSelection::Dense { offset: 0, len: 3 };
        dense.append_range(0..0);
        assert!(matches!(dense, ValueSelection::Dense { offset: 0, len: 3 }));

        let mut selection = ValueSelection::Empty;
        selection.append_range(10..20);
        selection.append_range(30..40);
        // Adjacent source spans coalesce without adding metadata.
        selection.append_range(40..50);

        let ValueSelection::Ranges { ranges, len, .. } = &selection else {
            panic!("expected compact ranges, got {selection:?}");
        };
        assert_eq!(*len, 30);
        assert_eq!(ranges.len(), 2);
        assert_eq!(
            std::mem::size_of::<SelectionRange>(),
            2 * std::mem::size_of::<usize>()
        );
        assert_eq!(
            collect_selection(selection.as_ref()),
            (10..20).chain(30..50).collect::<Vec<_>>()
        );
        let sliced = selection.as_ref().slice(5, 20);
        assert_eq!(sliced.len(), 20);
        assert_eq!(sliced.index_at(0), 15);
        assert_eq!(sliced.index_at(4), 19);
        assert_eq!(sliced.index_at(5), 30);
        assert_eq!(sliced.index_at(19), 44);
        assert_eq!(
            sliced.cursor().collect::<Vec<_>>(),
            (15..20).chain(30..45).collect::<Vec<_>>()
        );

        let mut spans = Vec::new();
        let ValueSelectionRef::Ranges(ranges) = sliced else {
            panic!("expected sliced ranges")
        };
        ranges
            .try_for_each_range(|start, len| -> Result<(), ()> {
                spans.push((start, len));
                Ok(())
            })
            .unwrap();
        assert_eq!(spans, vec![(15, 5), (30, 15)]);

        let at_end = selection.as_ref().slice(30, 0);
        assert_eq!(at_end.cursor().len(), 0);
        assert_eq!(at_end.cursor().next(), None);

        selection.extend_sparse_indices(2, [60, 62]);
        assert!(matches!(selection, ValueSelection::Sparse(_)));
        assert_eq!(
            collect_selection(selection.as_ref()),
            (10..20).chain(30..50).chain([60, 62]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn value_selection_ranges_fall_back_for_entropy_or_non_monotonicity() {
        for ranges in [
            vec![0..4, 5..6, 7..8, 9..10, 11..12, 13..14],
            vec![100..200, 0..100],
            vec![10..30, 20..40],
        ] {
            let expected = ranges.iter().flat_map(Clone::clone).collect::<Vec<_>>();
            let mut selection = ValueSelection::Empty;
            for range in ranges {
                selection.append_range(range);
            }
            assert!(matches!(selection, ValueSelection::Sparse(_)));
            assert_selection(&selection, expected);
        }
    }

    #[test]
    fn deferred_sparse_materializes_once_when_extended() {
        let mut selection = ValueSelection::Empty;
        selection.append_deferred_sparse(NullBuffer::from(vec![true, false, true]), 10, 2);
        selection.append_deferred_sparse(NullBuffer::from(vec![false, true, true]), 20, 2);
        assert_selection(&selection, [10, 12, 21, 22]);

        let mut initialized = ValueSelection::Empty;
        initialized.append_deferred_sparse(NullBuffer::from(vec![true, false, true]), 30, 2);
        assert_selection(&initialized, [30, 32]);
        initialized.extend_sparse_indices(1, [40]);
        initialized.append_range(50..52);
        assert!(matches!(initialized, ValueSelection::Sparse(_)));
        assert_selection(&initialized, [30, 32, 40, 50, 51]);
    }
}
