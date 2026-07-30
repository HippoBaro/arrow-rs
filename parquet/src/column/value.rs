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

//! Selection, storage routing, and the statically dispatched handoff of native
//! physical values to column encoders.
//!
//! [`ValueSelectionRef`] describes selected logical value positions after level
//! planning, [`PhysicalIndexPlan`] adds optional dictionary routing and grouped
//! terminal indices, and [`Sink`] hands each native batch descriptor to its
//! encoder. Descriptors borrow either source payloads or bounded producer-owned
//! scratch; the protocol requires no canonical scalar representation or
//! heap-owned transfer object.

use std::{mem::MaybeUninit, slice};

#[cfg(feature = "arrow")]
use std::ops::Range;

#[cfg(feature = "arrow")]
use arrow_buffer::ArrowNativeType;

use crate::errors::Result;

/// Run boundaries of a run-end-encoded array, type-erased over the run-end
/// index width. `run_ends[j]` is the logical position one past the end of
/// physical run `j`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "arrow"), allow(dead_code))]
pub(crate) enum RunEnds<'a> {
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl RunEnds<'_> {
    /// Physical run index containing absolute logical position `pos` — the
    /// first run whose end is strictly past `pos`. Equivalent to
    /// `RunEndBuffer::get_physical_index`.
    #[cfg(any(feature = "arrow", test))]
    #[inline(always)]
    pub(crate) fn run_of(self, pos: usize) -> usize {
        match self {
            Self::I16(ends) => ends.partition_point(|&end| (end as usize) <= pos),
            Self::I32(ends) => ends.partition_point(|&end| (end as usize) <= pos),
            Self::I64(ends) => ends.partition_point(|&end| (end as usize) <= pos),
        }
    }

    /// Logical end (one past the last row) of physical run `run`.
    #[cfg(any(feature = "arrow", test))]
    #[inline(always)]
    pub(crate) fn end_of(self, run: usize) -> usize {
        match self {
            Self::I16(ends) => ends[run] as usize,
            Self::I32(ends) => ends[run] as usize,
            Self::I64(ends) => ends[run] as usize,
        }
    }
}

/// Borrowed view of a selected set of values.
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectionRange {
    /// First source position. The source length is the difference between this
    /// range's cumulative end and the preceding range's cumulative end. This
    /// keeps range metadata to the same two words as `Range<usize>` while
    /// retaining logarithmic value-space slicing.
    pub(crate) source_start: usize,
    /// Exclusive end in the concatenated selected-value stream.
    pub(crate) selected_end: usize,
}

#[cfg(feature = "arrow")]
impl SelectionRange {
    pub(crate) fn new(source: Range<usize>, selected_end: usize) -> Self {
        Self {
            source_start: source.start,
            selected_end,
        }
    }

    #[inline]
    pub(crate) fn source_range(&self, selected_start: usize) -> Range<usize> {
        self.source_start..self.source_start + (self.selected_end - selected_start)
    }
}

#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RangesSelectionRef<'a> {
    ranges: &'a [SelectionRange],
    /// Number of values skipped from the concatenated range stream.
    offset: usize,
    len: usize,
}

#[cfg(feature = "arrow")]
impl<'a> RangesSelectionRef<'a> {
    pub(crate) fn new(ranges: &'a [SelectionRange], len: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            let mut selected_start = 0;
            let mut previous_source_end = None;
            for range in ranges {
                debug_assert!(range.selected_end > selected_start, "empty selection range");
                let source_end = range
                    .source_start
                    .checked_add(range.selected_end - selected_start)
                    .expect("selection range source end overflow");
                if let Some(previous_end) = previous_source_end {
                    debug_assert!(
                        range.source_start > previous_end,
                        "selection ranges must be ordered, disjoint, and coalesced"
                    );
                }
                selected_start = range.selected_end;
                previous_source_end = Some(source_end);
            }
            debug_assert_eq!(selected_start, len);
            debug_assert_eq!(ranges.is_empty(), len == 0);
        }
        Self {
            ranges,
            offset: 0,
            len,
        }
    }

    pub(crate) fn slice(self, offset: usize, len: usize) -> Self {
        debug_assert!(offset <= self.len && len <= self.len - offset);
        Self {
            offset: self.offset + offset,
            len,
            ..self
        }
    }

    #[inline(always)]
    fn range_index(self, selected: usize) -> usize {
        self.ranges
            .partition_point(|range| range.selected_end <= selected)
    }

    #[inline(always)]
    fn selected_start(self, range_index: usize) -> usize {
        range_index
            .checked_sub(1)
            .map_or(0, |previous| self.ranges[previous].selected_end)
    }

    fn single_range(self) -> Option<Range<usize>> {
        if self.len == 0 {
            return None;
        }
        let first_selected = self.offset;
        let last_selected = self.offset + self.len - 1;
        let first_range = self.range_index(first_selected);
        let last_range = self.range_index(last_selected);
        if first_range != last_range {
            return None;
        }
        let selected_start = self.selected_start(first_range);
        let start = self.ranges[first_range].source_start + first_selected - selected_start;
        Some(start..start + self.len)
    }

    #[inline]
    pub(crate) fn index_at(self, index: usize) -> usize {
        debug_assert!(index < self.len);
        let selected = self.offset + index;
        let range_index = self.range_index(selected);
        let selected_start = self.selected_start(range_index);
        self.ranges[range_index].source_start + selected - selected_start
    }

    #[inline]
    pub(crate) fn try_for_each_range<E>(
        self,
        mut f: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut selected = self.offset;
        let mut remaining = self.len;
        let mut range_index = self.range_index(selected);
        while remaining != 0 {
            let range = &self.ranges[range_index];
            let selected_start = self.selected_start(range_index);
            let skip = selected - selected_start;
            let start = range.source_start + skip;
            let len = (range.selected_end - selected).min(remaining);
            f(start, len)?;
            remaining -= len;
            selected += len;
            range_index += 1;
        }
        debug_assert_eq!(remaining, 0, "range selection exceeded stored ranges");
        Ok(())
    }
}

/// Borrowed output-order groups of equal physical indices. `ends[i]` is the
/// cumulative logical end of `indices[i]`, keeping repeated values O(groups).
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupedSelectionRef<'a> {
    indices: &'a [usize],
    ends: &'a [usize],
    offset: usize,
    len: usize,
}

#[cfg(feature = "arrow")]
impl<'a> GroupedSelectionRef<'a> {
    pub(crate) fn new(indices: &'a [usize], ends: &'a [usize]) -> Self {
        debug_assert_eq!(indices.len(), ends.len());
        debug_assert!(ends.first().is_none_or(|&end| end != 0));
        debug_assert!(ends.windows(2).all(|ends| ends[0] < ends[1]));
        debug_assert!(indices.windows(2).all(|indices| indices[0] != indices[1]));
        Self {
            indices,
            ends,
            offset: 0,
            len: ends.last().copied().unwrap_or(0),
        }
    }

    fn slice(self, offset: usize, len: usize) -> Self {
        debug_assert!(offset <= self.len && len <= self.len - offset);
        Self {
            offset: self.offset + offset,
            len,
            ..self
        }
    }

    #[inline(always)]
    fn group(self, position: usize) -> usize {
        self.ends.partition_point(|&end| end <= position)
    }

    #[inline(always)]
    fn index_at(self, index: usize) -> usize {
        debug_assert!(index < self.len);
        self.indices[self.group(self.offset + index)]
    }

    #[inline]
    fn try_for_each_group<E>(
        self,
        mut f: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let end = self.offset + self.len;
        let mut position = self.offset;
        let mut group = self.group(position);
        while position != end {
            let group_end = self.ends[group].min(end);
            f(self.indices[group], group_end - position)?;
            position = group_end;
            group += 1;
        }
        Ok(())
    }
}

/// Borrowed view of a selected set of values.
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueSelectionRef<'a> {
    Empty,
    Dense {
        offset: usize,
        len: usize,
    },
    /// Ordered contiguous source ranges. The view may begin/end inside the
    /// first/last stored range after value-space slicing.
    Ranges(RangesSelectionRef<'a>),
    Sparse(&'a [usize]),
    Grouped(GroupedSelectionRef<'a>),
}

#[cfg(feature = "arrow")]
impl<'a> ValueSelectionRef<'a> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Dense { len, .. } => len,
            Self::Ranges(ranges) => ranges.len,
            Self::Sparse(indices) => indices.len(),
            Self::Grouped(grouped) => grouped.len,
        }
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
                len: selection_len,
            } => {
                // The count-based guard in `write_mini_batch` is the only other
                // bounds protection on this path; catch a miscounted window
                // here to prevent an out-of-bounds value read.
                debug_assert!(offset + len <= selection_len);
                Self::Dense {
                    offset: base + offset,
                    len,
                }
            }
            Self::Ranges(ranges) => Self::Ranges(ranges.slice(offset, len)),
            Self::Sparse(indices) => Self::Sparse(&indices[offset..offset + len]),
            Self::Grouped(grouped) => Self::Grouped(grouped.slice(offset, len)),
        }
    }

    pub(crate) fn cursor(self) -> ValueSelectionCursor<'a> {
        match self {
            Self::Empty => ValueSelectionCursor::Empty,
            Self::Dense { offset, len } => ValueSelectionCursor::Dense(offset..offset + len),
            Self::Ranges(ranges) => {
                ValueSelectionCursor::Ranges(RangesSelectionCursor::new(ranges))
            }
            Self::Sparse(indices) => ValueSelectionCursor::Sparse(indices.iter()),
            Self::Grouped(grouped) => ValueSelectionCursor::Grouped {
                indices: grouped.indices,
                ends: grouped.ends,
                group: grouped.group(grouped.offset),
                position: grouped.offset,
                remaining: grouped.len,
            },
        }
    }

    #[inline(always)]
    pub(crate) fn index_at(self, idx: usize) -> usize {
        debug_assert!(idx < self.len());
        match self {
            Self::Empty => unreachable!("empty value selection has no values"),
            Self::Dense { offset, .. } => offset + idx,
            Self::Ranges(ranges) => ranges.index_at(idx),
            Self::Sparse(indices) => indices[idx],
            Self::Grouped(grouped) => grouped.index_at(idx),
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
            Self::Ranges(ranges) => ranges.try_for_each_range(|start, len| {
                for idx in start..start + len {
                    f(idx)?;
                }
                Ok(())
            }),
            Self::Sparse(indices) => {
                for &idx in indices {
                    f(idx)?;
                }
                Ok(())
            }
            Self::Grouped(grouped) => grouped.try_for_each_group(|index, count| {
                for _ in 0..count {
                    f(index)?;
                }
                Ok(())
            }),
        }
    }
}

/// Exact-size sequential traversal of a value selection. This is deliberately
/// separate from the encoder-facing `ValueCursor`: it yields source positions
/// while CDC zips positions with definition and repetition levels.
#[cfg(feature = "arrow")]
pub(crate) enum ValueSelectionCursor<'a> {
    Empty,
    Dense(std::ops::Range<usize>),
    Ranges(RangesSelectionCursor<'a>),
    Sparse(slice::Iter<'a, usize>),
    Grouped {
        indices: &'a [usize],
        ends: &'a [usize],
        group: usize,
        position: usize,
        remaining: usize,
    },
}

#[cfg(feature = "arrow")]
impl Iterator for ValueSelectionCursor<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Dense(range) => range.next(),
            Self::Ranges(ranges) => ranges.next(),
            Self::Sparse(indices) => indices.next().copied(),
            Self::Grouped {
                indices,
                ends,
                group,
                position,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                while *position == ends[*group] {
                    *group += 1;
                }
                let value = indices[*group];
                *position += 1;
                *remaining -= 1;
                Some(value)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

#[cfg(feature = "arrow")]
impl ExactSizeIterator for ValueSelectionCursor<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Dense(range) => range.len(),
            Self::Ranges(ranges) => ranges.len(),
            Self::Sparse(indices) => indices.len(),
            Self::Grouped { remaining, .. } => *remaining,
        }
    }
}

#[cfg(feature = "arrow")]
pub(crate) struct RangesSelectionCursor<'a> {
    ranges: &'a [SelectionRange],
    range_index: usize,
    position: usize,
    range_end: usize,
    remaining: usize,
}

#[cfg(feature = "arrow")]
impl<'a> RangesSelectionCursor<'a> {
    pub(crate) fn new(selection: RangesSelectionRef<'a>) -> Self {
        let range_index = selection.range_index(selection.offset);
        let selected_start = selection.selected_start(range_index);
        let skip = selection.offset - selected_start;
        let mut position = 0;
        let mut range_end = 0;
        if selection.len != 0 {
            let range = &selection.ranges[range_index];
            position = range.source_start + skip;
            range_end = range.source_start + (range.selected_end - selected_start);
        }
        Self {
            ranges: selection.ranges,
            range_index,
            position,
            range_end,
            remaining: selection.len,
        }
    }
}

#[cfg(feature = "arrow")]
impl Iterator for RangesSelectionCursor<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while self.position == self.range_end {
            self.range_index += 1;
            let range = &self.ranges[self.range_index];
            let selected_start = self.ranges[self.range_index - 1].selected_end;
            self.position = range.source_start;
            self.range_end = range.source_start + (range.selected_end - selected_start);
        }
        let value = self.position;
        self.position += 1;
        self.remaining -= 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(feature = "arrow")]
impl ExactSizeIterator for RangesSelectionCursor<'_> {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Type-erased Arrow dictionary keys. Keeping the eight legal key widths in a
/// borrowed enum lets index-map composition remain statically dispatched and
/// allocation free.
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum DictionaryKeys<'a> {
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

#[cfg(feature = "arrow")]
impl<'a> DictionaryKeys<'a> {
    fn contiguous_range(self, selection: ValueSelectionRef<'_>) -> Option<Range<usize>> {
        match self {
            Self::I8(keys) => contiguous_key_range(keys, selection),
            Self::I16(keys) => contiguous_key_range(keys, selection),
            Self::I32(keys) => contiguous_key_range(keys, selection),
            Self::I64(keys) => contiguous_key_range(keys, selection),
            Self::U8(keys) => contiguous_key_range(keys, selection),
            Self::U16(keys) => contiguous_key_range(keys, selection),
            Self::U32(keys) => contiguous_key_range(keys, selection),
            Self::U64(keys) => contiguous_key_range(keys, selection),
        }
    }

    /// Visit selected dictionary values with the key width dispatched once,
    /// outside the value loop.
    #[inline]
    fn try_for_each<E>(
        self,
        selection: ValueSelectionRef<'a>,
        mut f: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        self.try_for_each_group(selection, |index, count| {
            for _ in 0..count {
                f(index)?;
            }
            Ok(())
        })
    }

    /// Visit selected dictionary run groups with the key width dispatched
    /// once. Non-grouped selections emit one group per value.
    #[inline]
    fn try_for_each_group<E>(
        self,
        selection: ValueSelectionRef<'a>,
        mut f: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        macro_rules! visit {
            ($keys:expr) => {
                match selection {
                    ValueSelectionRef::Grouped(grouped) => {
                        grouped.try_for_each_group(|row, count| f($keys[row].as_usize(), count))
                    }
                    _ => selection.try_for_each(|row| f($keys[row].as_usize(), 1)),
                }
            };
        }
        match self {
            Self::I8(keys) => visit!(keys),
            Self::I16(keys) => visit!(keys),
            Self::I32(keys) => visit!(keys),
            Self::I64(keys) => visit!(keys),
            Self::U8(keys) => visit!(keys),
            Self::U16(keys) => visit!(keys),
            Self::U32(keys) => visit!(keys),
            Self::U64(keys) => visit!(keys),
        }
    }
}

/// Return the mapped range when selected dictionary keys are consecutive.
/// First/last/middle probes reject cyclic and low-cardinality maps before the
/// full scan used by the high-cardinality identity-like case.
#[cfg(feature = "arrow")]
fn contiguous_key_range<K: ArrowNativeType>(
    keys: &[K],
    selection: ValueSelectionRef<'_>,
) -> Option<Range<usize>> {
    let len = selection.len();
    if len == 0 {
        return None;
    }
    let first = keys[selection.index_at(0)].as_usize();
    let end = first.checked_add(len)?;
    let expected_last = end - 1;
    if keys[selection.index_at(len - 1)].as_usize() != expected_last {
        return None;
    }
    let middle = len / 2;
    if keys[selection.index_at(middle)].as_usize() != first + middle {
        return None;
    }

    let mut expected = first;
    selection
        .try_for_each(|row| -> Result<(), ()> {
            if keys[row].as_usize() != expected {
                return Err(());
            }
            expected += 1;
            Ok(())
        })
        .ok()?;
    debug_assert_eq!(expected, end);
    Some(first..end)
}

/// A compact output-order span of physical value indices.
///
/// `Range` is a sequence of consecutive indices. `Repeat` is one repeated
/// physical index. The span count always contributes to the plan's logical
/// length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "arrow")]
pub(crate) enum IndexSpan {
    Range { start: usize, len: usize },
    Repeat { index: usize, count: usize },
}

#[cfg(feature = "arrow")]
impl IndexSpan {
    #[inline]
    fn from_group(index: usize, count: usize) -> Self {
        debug_assert_ne!(count, 0);
        match count {
            1 => Self::Range {
                start: index,
                len: 1,
            },
            _ => Self::Repeat { index, count },
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Range { len, .. } => len,
            Self::Repeat { count, .. } => count,
        }
    }
}

/// Allocation-free composition of a value selection with an optional physical
/// dictionary mapping. A grouped selection preserves repeated source indices;
/// dictionary mapping resolves each source group once.
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PhysicalIndexPlan<'a> {
    selection: ValueSelectionRef<'a>,
    dictionary: Option<DictionaryKeys<'a>>,
}

#[cfg(feature = "arrow")]
impl<'a> PhysicalIndexPlan<'a> {
    pub(crate) fn identity(selection: ValueSelectionRef<'a>) -> Self {
        Self {
            selection,
            dictionary: None,
        }
    }

    pub(crate) fn dictionary(selection: ValueSelectionRef<'a>, keys: DictionaryKeys<'a>) -> Self {
        Self {
            selection,
            dictionary: Some(keys),
        }
    }

    pub(crate) fn slice(self, offset: usize, len: usize) -> Self {
        Self {
            selection: self.selection.slice(offset, len),
            ..self
        }
    }

    pub(crate) fn len(self) -> usize {
        self.selection.len()
    }

    /// Return the source-coordinate selection when no dictionary mapping is
    /// present. Grouped identity selections remain directly observable.
    pub(crate) fn unmapped_selection(self) -> Option<ValueSelectionRef<'a>> {
        self.dictionary.is_none().then_some(self.selection)
    }

    pub(crate) fn is_grouped(self) -> bool {
        matches!(self.selection, ValueSelectionRef::Grouped(_))
    }

    pub(crate) fn has_dictionary_mapping(self) -> bool {
        self.dictionary.is_some()
    }

    /// Return the directly borrowable physical range, including a dictionary
    /// selection whose keys are consecutive. Grouped plans retain their
    /// explicit run boundaries and are never flattened here.
    pub(crate) fn direct_physical_range(self) -> Option<Range<usize>> {
        if self.is_grouped() {
            return None;
        }
        match self.dictionary {
            None => match self.selection {
                ValueSelectionRef::Dense { offset, len } => Some(offset..offset + len),
                ValueSelectionRef::Ranges(ranges) => ranges.single_range(),
                _ => None,
            },
            Some(keys) => keys.contiguous_range(self.selection),
        }
    }

    /// Visit producer-provided ranges and groups in output order.
    pub(crate) fn try_for_each_span<E>(
        self,
        mut f: impl FnMut(IndexSpan) -> Result<(), E>,
    ) -> Result<(), E> {
        match self.dictionary {
            None => match self.selection {
                ValueSelectionRef::Empty => Ok(()),
                ValueSelectionRef::Dense { offset, len } => {
                    if len != 0 {
                        f(IndexSpan::Range { start: offset, len })?;
                    }
                    Ok(())
                }
                ValueSelectionRef::Ranges(ranges) => {
                    ranges.try_for_each_range(|start, len| f(IndexSpan::Range { start, len }))
                }
                ValueSelectionRef::Sparse(indices) => indices.iter().try_for_each(|&index| {
                    f(IndexSpan::Range {
                        start: index,
                        len: 1,
                    })
                }),
                ValueSelectionRef::Grouped(grouped) => grouped
                    .try_for_each_group(|index, count| f(IndexSpan::from_group(index, count))),
            },
            Some(keys) => keys.try_for_each_group(self.selection, |index, count| {
                f(IndexSpan::from_group(index, count))
            }),
        }
    }

    pub(crate) fn try_for_each_index<E>(
        self,
        f: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        match self.dictionary {
            None => self.selection.try_for_each(f),
            Some(keys) => keys.try_for_each(self.selection, f),
        }
    }

    pub(crate) fn try_for_each_index_group<E>(
        self,
        mut f: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        match self.dictionary {
            Some(keys) => keys.try_for_each_group(self.selection, f),
            None => match self.selection {
                ValueSelectionRef::Grouped(grouped) => grouped.try_for_each_group(f),
                selection => selection.try_for_each(|index| f(index, 1)),
            },
        }
    }
}

/// The single batch handoff from a producer to a column encoder.
///
/// Implementations receive a directly consumable native descriptor: a borrowed
/// dense batch, a packed-Boolean selection, a native byte span, or a bounded
/// materialized batch. Statistics, bloom filters, dictionary interning, and
/// fallback encoding remain private consumer policy; the batch contract imposes
/// neither a scalar representation nor an accumulator type.
///
/// Most implementations make `commit` the out-of-line boundary. Byte arrays
/// inline this small dispatcher and keep the selected encoder-target helper out
/// of line, preserving one effective call per batch.
pub(crate) trait Sink<B> {
    fn commit(&mut self, batch: B) -> Result<()>;
}

/// Optional selection-resolved traversal for addressable producers.
///
/// Inputs already consumable as one native batch may bypass the cursor. Numeric
/// casts, non-contiguous byte values, and computed FLBA use dedicated bounded
/// packers. Bulk eligibility remains explicit at the producer rather than being
/// a second cursor mode.
pub(crate) trait ValueCursor<T: Copy>: Copy {
    /// Exact number of logical values [`Self::try_for_each`] emits.
    fn len(self) -> usize;

    /// Emit selected values in output order.
    fn try_for_each<E>(self, f: impl FnMut(T) -> Result<(), E>) -> Result<(), E>;

    /// Emit source-provided run groups. The default emits one group per item.
    #[inline]
    fn for_each_run_group<E>(self, mut f: impl FnMut(T, usize) -> Result<(), E>) -> Result<(), E> {
        self.try_for_each(|value| f(value, 1))
    }
}

#[cfg(feature = "arrow")]
impl ValueCursor<usize> for PhysicalIndexPlan<'_> {
    fn len(self) -> usize {
        PhysicalIndexPlan::len(self)
    }

    fn try_for_each<E>(self, f: impl FnMut(usize) -> Result<(), E>) -> Result<(), E> {
        self.try_for_each_index(f)
    }

    fn for_each_run_group<E>(self, f: impl FnMut(usize, usize) -> Result<(), E>) -> Result<(), E> {
        self.try_for_each_index_group(f)
    }
}

/// A total index-to-value projection that preserves the source cursor's run
/// groups.
#[cfg(feature = "arrow")]
#[derive(Clone, Copy)]
pub(crate) struct MappedValueCursor<C, F> {
    indices: C,
    map: F,
}

#[cfg(feature = "arrow")]
pub(crate) fn map_values<C, F>(indices: C, map: F) -> MappedValueCursor<C, F> {
    MappedValueCursor { indices, map }
}

#[cfg(feature = "arrow")]
impl<T, C, F> ValueCursor<T> for MappedValueCursor<C, F>
where
    T: Copy,
    C: ValueCursor<usize>,
    F: Fn(usize) -> T + Copy,
{
    fn len(self) -> usize {
        self.indices.len()
    }

    fn try_for_each<E>(self, mut f: impl FnMut(T) -> Result<(), E>) -> Result<(), E> {
        self.indices.try_for_each(|index| f((self.map)(index)))
    }

    fn for_each_run_group<E>(self, mut f: impl FnMut(T, usize) -> Result<(), E>) -> Result<(), E> {
        self.indices
            .for_each_run_group(|index, count| f((self.map)(index), count))
    }
}

/// View the initialized prefix of a `MaybeUninit` slice as initialized values.
///
/// # Safety
///
/// `len` must not exceed `values.len()`, and every element in `values[..len]`
/// must have been initialized as a valid `T`.
#[inline(always)]
pub(crate) unsafe fn assume_init_prefix<T>(values: &[MaybeUninit<T>], len: usize) -> &[T] {
    debug_assert!(len <= values.len());
    // SAFETY: guaranteed by the caller. `MaybeUninit<T>` has the same layout
    // and alignment as `T`.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<T>(), len) }
}

/// Gather selected values into bounded `N`-element stack batches.
#[inline(always)]
pub(crate) fn gather_tiled<const N: usize, T, C, Flush>(values: C, mut flush: Flush) -> Result<()>
where
    T: Copy,
    C: ValueCursor<T>,
    Flush: FnMut(&[T]) -> Result<()>,
{
    let mut batch = [MaybeUninit::<T>::uninit(); N];
    let mut filled = 0;
    values.try_for_each(
        #[inline(always)]
        |value| -> Result<()> {
            batch[filled].write(value);
            filled += 1;
            if filled == N {
                // SAFETY: this loop initialized every slot before `filled`
                // reached `N`.
                flush(unsafe { assume_init_prefix(&batch, filled) })?;
                filled = 0;
            }
            Ok(())
        },
    )?;
    if filled != 0 {
        // SAFETY: this loop initializes slots sequentially and `filled` is the
        // length of that initialized prefix.
        flush(unsafe { assume_init_prefix(&batch, filled) })?;
    }
    Ok(())
}

/// Gather run-collapsed values into bounded `(value, count)` stack batches.
/// Handoffs are proportional to selected run groups; reordered sparse
/// selections may emit more than one group for a physical run.
#[inline(always)]
pub(crate) fn gather_run_groups_tiled<const N: usize, T, Flush>(
    values: impl ValueCursor<T>,
    mut flush: Flush,
) -> Result<()>
where
    T: Copy,
    Flush: FnMut(&[T], &[usize]) -> Result<()>,
{
    let mut value_batch = [MaybeUninit::<T>::uninit(); N];
    let mut count_batch = [MaybeUninit::<usize>::uninit(); N];
    let mut filled = 0;
    values.for_each_run_group(
        #[inline(always)]
        |value, count| -> Result<()> {
            value_batch[filled].write(value);
            count_batch[filled].write(count);
            filled += 1;
            if filled == N {
                // SAFETY: both arrays are initialized at the current slot before
                // `filled` advances, so their initialized prefixes match.
                flush(
                    unsafe { assume_init_prefix(&value_batch, filled) },
                    unsafe { assume_init_prefix(&count_batch, filled) },
                )?;
                filled = 0;
            }
            Ok(())
        },
    )?;
    if filled != 0 {
        // SAFETY: both arrays are initialized at the current slot before
        // `filled` advances, so their initialized prefixes match.
        flush(
            unsafe { assume_init_prefix(&value_batch, filled) },
            unsafe { assume_init_prefix(&count_batch, filled) },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_end_widths_resolve_identically() {
        fn describe(run_ends: RunEnds<'_>) -> (Vec<usize>, Vec<usize>) {
            (
                (0..9).map(|position| run_ends.run_of(position)).collect(),
                (0..3).map(|run| run_ends.end_of(run)).collect(),
            )
        }

        let i16_ends = [2i16, 5, 9];
        let i32_ends = [2i32, 5, 9];
        let i64_ends = [2i64, 5, 9];
        let expected = describe(RunEnds::I16(&i16_ends));
        assert_eq!(describe(RunEnds::I32(&i32_ends)), expected);
        assert_eq!(describe(RunEnds::I64(&i64_ends)), expected);
        assert_eq!(expected.0, [0, 0, 1, 1, 1, 2, 2, 2, 2]);
        assert_eq!(expected.1, [2, 5, 9]);
    }

    #[cfg(feature = "arrow")]
    fn physical_spans(plan: PhysicalIndexPlan<'_>) -> Vec<IndexSpan> {
        let mut spans = Vec::new();
        plan.try_for_each_span(|span| -> Result<(), ()> {
            spans.push(span);
            Ok(())
        })
        .unwrap();
        spans
    }

    #[cfg(feature = "arrow")]
    fn expand_spans(spans: &[IndexSpan]) -> Vec<usize> {
        let mut values = Vec::new();
        for &span in spans {
            match span {
                IndexSpan::Range { start, len } => {
                    values.extend(start..start + len);
                }
                IndexSpan::Repeat { index, count } => {
                    values.extend(std::iter::repeat_n(index, count));
                }
            }
        }
        values
    }

    #[cfg(feature = "arrow")]
    fn physical_indices(plan: PhysicalIndexPlan<'_>) -> Vec<usize> {
        let mut values = Vec::new();
        plan.try_for_each_index(|index| -> Result<(), ()> {
            values.push(index);
            Ok(())
        })
        .unwrap();
        values
    }

    #[cfg(feature = "arrow")]
    fn range(start: usize, len: usize) -> IndexSpan {
        IndexSpan::Range { start, len }
    }

    #[cfg(feature = "arrow")]
    fn repeat(index: usize, count: usize) -> IndexSpan {
        IndexSpan::Repeat { index, count }
    }

    #[cfg(feature = "arrow")]
    fn assert_span_plan(plan: PhysicalIndexPlan<'_>, expected: &[usize]) {
        let spans = physical_spans(plan);
        assert_eq!(
            spans.iter().map(|span| span.len()).sum::<usize>(),
            plan.len()
        );
        assert_eq!(expand_spans(&spans), expected);
        assert_eq!(physical_indices(plan), expected);

        let mut grouped = Vec::new();
        plan.try_for_each_index_group(|index, count| -> Result<(), ()> {
            grouped.push((index, count));
            Ok(())
        })
        .unwrap();
        assert_eq!(
            grouped
                .iter()
                .flat_map(|&(index, count)| std::iter::repeat_n(index, count))
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn physical_identity_dense_ranges_sparse_and_slices() {
        let dense = PhysicalIndexPlan::identity(ValueSelectionRef::Dense { offset: 10, len: 8 });
        assert_eq!(
            dense.unmapped_selection(),
            Some(ValueSelectionRef::Dense { offset: 10, len: 8 })
        );
        assert_eq!(dense.direct_physical_range(), Some(10..18));
        assert_eq!(physical_spans(dense.slice(2, 4)), [range(12, 4)]);

        let stored_ranges = [SelectionRange::new(2..5, 3), SelectionRange::new(8..12, 7)];
        let ranges = PhysicalIndexPlan::identity(ValueSelectionRef::Ranges(
            RangesSelectionRef::new(&stored_ranges, 7),
        ));
        assert_eq!(ranges.direct_physical_range(), None);
        assert_eq!(ranges.slice(1, 2).direct_physical_range(), Some(3..5));
        assert_eq!(
            physical_spans(ranges.slice(2, 4)),
            [range(4, 1), range(8, 3)]
        );
        assert_span_plan(ranges, &[2, 3, 4, 8, 9, 10, 11]);

        let sparse_values = [3, 4, 4, 5, 9, 9, 8, 1, 2];
        let sparse = PhysicalIndexPlan::identity(ValueSelectionRef::Sparse(&sparse_values));
        assert_span_plan(sparse, &sparse_values);
        assert_span_plan(sparse.slice(1, 7), &sparse_values[1..8]);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn physical_grouped_identity_preserves_compact_repeats() {
        let indices = [7, 2, 3];
        let ends = [3, 4, 6];
        let selection = ValueSelectionRef::Grouped(GroupedSelectionRef::new(&indices, &ends));
        let grouped = PhysicalIndexPlan::identity(selection);

        assert_eq!(grouped.unmapped_selection(), Some(selection));
        assert!(grouped.is_grouped());
        assert_eq!(
            physical_spans(grouped),
            [repeat(7, 3), range(2, 1), repeat(3, 2),]
        );
        assert_eq!(
            physical_spans(grouped.slice(1, 4)),
            [repeat(7, 2), range(2, 1), range(3, 1)]
        );

        let expanded = [7, 7, 7, 2, 3, 3];
        for offset in 0..=expanded.len() {
            for len in 0..=expanded.len() - offset {
                let sliced = selection.slice(offset, len);
                let expected = &expanded[offset..offset + len];
                assert_eq!(sliced.len(), len);
                assert_eq!(sliced.cursor().collect::<Vec<_>>(), expected);
                let mut visited = Vec::new();
                sliced
                    .try_for_each(|index| -> Result<(), ()> {
                        visited.push(index);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(visited, expected);
                assert_span_plan(PhysicalIndexPlan::identity(sliced), expected);
                for (position, &expected) in expected.iter().enumerate() {
                    assert_eq!(sliced.index_at(position), expected);
                }
            }
        }
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn physical_sparse_spans_match_exhaustive_flat_oracle() {
        // Every reordered/duplicated sequence over 0..3 up to length six.
        for len in 0..=6 {
            for encoded in 0..3usize.pow(len as u32) {
                let mut value = encoded;
                let mut indices = Vec::with_capacity(len);
                for _ in 0..len {
                    indices.push(value % 3);
                    value /= 3;
                }
                let plan = PhysicalIndexPlan::identity(ValueSelectionRef::Sparse(&indices));
                assert_span_plan(plan, &indices);
            }
        }
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn physical_dictionary_widths_preserve_mapped_order() {
        macro_rules! assert_widths {
            ($($width:ty => $variant:ident),+ $(,)?) => {
                $({
                    let keys: [$width; 3] = [2, 0, 1];
                    let plan = PhysicalIndexPlan::dictionary(
                        ValueSelectionRef::Dense { offset: 0, len: 3 },
                        DictionaryKeys::$variant(&keys),
                    );
                    assert_eq!(plan.direct_physical_range(), None);
                    assert_span_plan(plan, &[2, 0, 1]);
                })+
            };
        }
        assert_widths!(
            i8 => I8,
            i16 => I16,
            i32 => I32,
            i64 => I64,
            u8 => U8,
            u16 => U16,
            u32 => U32,
            u64 => U64,
        );

        let keys = [4u8, 5, 6, 6, 7];
        let plan = PhysicalIndexPlan::dictionary(
            ValueSelectionRef::Dense {
                offset: 0,
                len: keys.len(),
            },
            DictionaryKeys::U8(&keys),
        );
        assert_eq!(plan.unmapped_selection(), None);
        assert!(plan.has_dictionary_mapping());
        assert!(!plan.is_grouped());
        assert_span_plan(plan, &[4, 5, 6, 6, 7]);
        let present = plan.slice(0, 4);
        assert_eq!(present.direct_physical_range(), None);
        let mut flat = Vec::new();
        present
            .try_for_each_index(|index| -> Result<(), ()> {
                flat.push(index);
                Ok(())
            })
            .unwrap();
        assert_eq!(flat, [4, 5, 6, 6]);

        let keys = [99_i32, 4, 5, 99, 6];
        let selected = [1_usize, 2, 4];
        let contiguous = PhysicalIndexPlan::dictionary(
            ValueSelectionRef::Sparse(&selected),
            DictionaryKeys::I32(&keys),
        );
        assert_eq!(contiguous.direct_physical_range(), Some(4..7));

        let ranges = [SelectionRange::new(1..3, 2), SelectionRange::new(4..5, 3)];
        let mapped = PhysicalIndexPlan::dictionary(
            ValueSelectionRef::Ranges(RangesSelectionRef::new(&ranges, 3)),
            DictionaryKeys::I32(&keys),
        );
        assert_span_plan(mapped, &[4, 5, 6]);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn value_selection_cursor_matches_callback_traversal() {
        let grouped_indices = [4, 1, 4, 9];
        let grouped_ends = [2, 3, 6, 7];
        for selection in [
            ValueSelectionRef::Empty,
            ValueSelectionRef::Dense { offset: 3, len: 5 },
            ValueSelectionRef::Sparse(&[4, 1, 4, 9]),
            ValueSelectionRef::Grouped(GroupedSelectionRef::new(&grouped_indices, &grouped_ends)),
        ] {
            let mut expected = Vec::new();
            selection
                .try_for_each(|idx| -> Result<(), ()> {
                    expected.push(idx);
                    Ok(())
                })
                .unwrap();
            let cursor = selection.cursor();
            assert_eq!(cursor.len(), expected.len());
            assert_eq!(cursor.collect::<Vec<_>>(), expected);
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct PanicDefault(u32);

    impl Default for PanicDefault {
        fn default() -> Self {
            panic!("gathering must not initialize unused batch entries")
        }
    }

    #[derive(Clone, Copy)]
    struct SliceCursor<'a, T>(&'a [T]);

    impl<T: Copy> ValueCursor<T> for SliceCursor<'_, T> {
        fn len(self) -> usize {
            self.0.len()
        }

        fn try_for_each<E>(self, mut f: impl FnMut(T) -> Result<(), E>) -> Result<(), E> {
            for &value in self.0 {
                f(value)?;
            }
            Ok(())
        }
    }

    fn assert_tiled_batches<const N: usize, T>(batches: &[Vec<T>], expected: &[T])
    where
        T: Copy + std::fmt::Debug + PartialEq,
    {
        let mut actual = Vec::new();
        let mut lengths = Vec::new();
        for batch in batches {
            actual.extend_from_slice(batch);
            lengths.push(batch.len());
        }
        assert_eq!(actual, expected);
        assert_eq!(
            lengths,
            (0..expected.len() / N)
                .map(|_| N)
                .chain((expected.len() % N != 0).then_some(expected.len() % N))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn gather_tiled_initializes_only_emitted_values() {
        for len in [0usize, 3, 4, 5, 8, 9] {
            let input: Vec<_> = (0..len).map(|value| PanicDefault(value as u32)).collect();
            let mut batches = Vec::new();
            gather_tiled::<4, _, _, _>(SliceCursor(&input), |batch| {
                batches.push(batch.to_vec());
                Ok(())
            })
            .unwrap();

            assert_tiled_batches::<4, _>(&batches, &input);
        }
    }

    #[derive(Clone, Copy)]
    struct RunCursor<'a> {
        values: &'a [&'a [u8]],
        counts: &'a [usize],
    }

    impl<'a> ValueCursor<&'a [u8]> for RunCursor<'a> {
        fn len(self) -> usize {
            self.counts.iter().sum()
        }

        fn try_for_each<E>(self, mut f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
            for (&value, &count) in self.values.iter().zip(self.counts) {
                for _ in 0..count {
                    f(value)?;
                }
            }
            Ok(())
        }

        fn for_each_run_group<E>(
            self,
            mut f: impl FnMut(&'a [u8], usize) -> Result<(), E>,
        ) -> Result<(), E> {
            for (&value, &count) in self.values.iter().zip(self.counts) {
                f(value, count)?;
            }
            Ok(())
        }
    }

    #[test]
    fn gather_run_groups_tiled_initializes_matching_prefixes() {
        for len in [0usize, 3, 4, 5, 8, 9] {
            let storage: Vec<_> = (0..len).map(|value| vec![value as u8]).collect();
            let values: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
            let counts: Vec<_> = (1..=values.len()).collect();
            let mut value_batches = Vec::new();
            let mut count_batches = Vec::new();

            gather_run_groups_tiled::<4, _, _>(
                RunCursor {
                    values: &values,
                    counts: &counts,
                },
                |batch_values, batch_counts| {
                    assert_eq!(batch_values.len(), batch_counts.len());
                    value_batches.push(batch_values.to_vec());
                    count_batches.push(batch_counts.to_vec());
                    Ok(())
                },
            )
            .unwrap();

            assert_tiled_batches::<4, _>(&value_batches, &values);
            assert_tiled_batches::<4, _>(&count_batches, &counts);
        }
    }
}
