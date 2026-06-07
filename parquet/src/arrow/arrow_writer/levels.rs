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

//! Parquet definition and repetition levels
//!
//! Contains the algorithm for computing definition and repetition levels.
//! The algorithm works by tracking the slots of an array that should
//! ultimately be populated when writing to Parquet.
//! Parquet achieves nesting through definition levels and repetition levels \[1\].
//! Definition levels specify how many optional fields in the part for the column
//! are defined.
//! Repetition levels specify at what repeated field (list) in the path a column
//! is defined.
//!
//! In a nested data structure such as `a.b.c`, one can see levels as defining
//! whether a record is defined at `a`, `a.b`, or `a.b.c`.
//! Optional fields are nullable fields, thus if all 3 fields
//! are nullable, the maximum definition could be = 3 if there are no lists.
//!
//! The algorithm in this module computes the necessary information to enable
//! the writer to keep track of which columns are at which levels, and to extract
//! the correct values at the correct slots from Arrow arrays.
//!
//! It works by walking a record batch's arrays, keeping track of what values
//! are non-null, their positions and computing what their levels are.
//!
//! \[1\] [parquet-format#nested-encoding](https://github.com/apache/parquet-format#nested-encoding)

use crate::column::writer::{LevelDataRef, RunLevelsRef};
use crate::errors::{ParquetError, Result};
use crate::file::properties::DEFAULT_WRITE_BATCH_SIZE;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, OffsetSizeTrait};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use std::ops::Range;
use std::sync::Arc;

pub(crate) mod cursor;
mod program;

#[cfg(test)]
use program::ValueSelection;
use program::{LEVEL_RUN_PROBE_SIZE, LeafPlanBuilder, MIN_AVERAGE_LEVEL_RUN_LENGTH};
pub(crate) use program::{LeafBatch, LeafBatchSlice, LeafPlan};

/// Performs a depth-first scan of the children of `array`, constructing a
/// finalized [`LeafPlan`] for each leaf column encountered.
pub(crate) fn calculate_array_levels(array: &ArrayRef, field: &Field) -> Result<Vec<LeafPlan>> {
    let mut builder = LevelInfoBuilder::try_new(field, LevelContext::default(), array)?;
    builder.write(0..array.len())?;
    Ok(builder.into_leaf_plans())
}

/// Returns true if the DataType can be represented as a primitive parquet column,
/// i.e. a leaf array with no children
pub(super) fn is_leaf(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Null
                | DataType::Boolean
                | DataType::Utf8
                | DataType::Utf8View
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

#[derive(Clone, Copy)]
struct FieldContract<'a> {
    data_type: &'a DataType,
    nullable: bool,
    name: &'a str,
}

/// Erase schema-only dictionary and REE wrappers. REE value-field
/// nullability belongs to the logical node it exposes.
fn normalized(field: &Field) -> FieldContract<'_> {
    let (data_type, nullable) = logical_type(field.data_type());
    FieldContract {
        data_type,
        nullable: field.is_nullable() || nullable,
        name: field.name(),
    }
}

fn logical_type(mut data_type: &DataType) -> (&DataType, bool) {
    let mut nullable = false;
    loop {
        match data_type {
            DataType::Dictionary(_, value) => data_type = value,
            DataType::RunEndEncoded(_, value) => {
                nullable |= value.is_nullable();
                data_type = value.data_type();
            }
            _ => return (data_type, nullable),
        }
    }
}

fn leaf_types_compatible(expected: &DataType, actual: &DataType) -> bool {
    is_leaf(expected)
        && is_leaf(actual)
        && (expected.equals_datatype(actual)
            || matches!(
                (expected, actual),
                (
                    DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8,
                    DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
                ) | (
                    DataType::Binary | DataType::BinaryView | DataType::LargeBinary,
                    DataType::Binary | DataType::BinaryView | DataType::LargeBinary
                )
            ))
}

/// The definition and repetition level of an array within a potentially nested hierarchy
#[derive(Debug, Default, Clone, Copy)]
struct LevelContext {
    /// The current repetition level
    rep_level: i16,
    /// The current definition level
    def_level: i16,
}

/// Null handling for one logical node.
///
/// All-valid buffers are discarded. Optional nodes expose their semantic
/// nulls to level construction, while required nodes validate only ranges
/// reached through valid ancestors and otherwise behave as all-valid.
#[derive(Debug)]
enum NodeNulls {
    None,
    Optional(NullBuffer),
    Required(NullBuffer, Box<str>),
}

impl NodeNulls {
    fn new(field: FieldContract<'_>, nulls: Option<NullBuffer>) -> Self {
        let Some(nulls) = nulls.filter(|nulls| nulls.null_count() != 0) else {
            return Self::None;
        };
        if field.nullable {
            Self::Optional(nulls)
        } else {
            Self::Required(nulls, field.name.into())
        }
    }

    fn for_range(&self, mut range: Range<usize>) -> Result<Option<&NullBuffer>> {
        match self {
            Self::None => Ok(None),
            Self::Optional(nulls) => Ok(Some(nulls)),
            Self::Required(nulls, field) => match range.find(|&index| nulls.is_null(index)) {
                Some(index) => Err(required_null(field, index)),
                None => Ok(None),
            },
        }
    }
}

fn required_null(field: &str, index: usize) -> ParquetError {
    ParquetError::ArrowError(format!(
        "Found null at index {index} for required field '{field}'"
    ))
}

/// Traversal state for a potentially nested [`Field`].
#[derive(Debug)]
struct LevelInfoBuilder {
    node: LevelInfoNode,
    nulls: NodeNulls,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum LevelInfoNode {
    /// A primitive, leaf array
    Primitive(LeafPlanBuilder),
    /// A list array
    List(ListChildBuilder, LevelContext, OffsetBuffer<i32>),
    /// A large list array
    LargeList(ListChildBuilder, LevelContext, OffsetBuffer<i64>),
    /// A fixed size list array
    FixedSizeList(Box<LevelInfoBuilder>, LevelContext, usize),
    /// A list view array
    ListView(
        Box<LevelInfoBuilder>, // Child Values
        LevelContext,          // Context
        ScalarBuffer<i32>,     // Offsets
        ScalarBuffer<i32>,     // Sizes
    ),
    /// A large list view array
    LargeListView(
        Box<LevelInfoBuilder>, // Child Values
        LevelContext,          // Context
        ScalarBuffer<i64>,     // Offsets
        ScalarBuffer<i64>,     // Sizes
    ),
    /// A struct array
    Struct(Vec<LevelInfoBuilder>, LevelContext),
}

/// A list child and whether every child element emits exactly one level slot.
#[derive(Debug)]
struct ListChildBuilder(Box<LevelInfoBuilder>, bool);

/// Minimum sub-range length before the bulk-fill fast path in `write_leaf`
/// becomes profitable for null-heavy leaf columns. Below this, per-call
/// slice and popcount overhead outweighs the bulk-fill savings for
/// list/struct paths that call `write_leaf` with tiny ranges.
const BULK_FILL_MIN_LEN: usize = 64;
/// Bound speculative run retention to one default writer mini-batch. Larger
/// streams materialize so compact metadata cannot widen their write batches.
const COMPACT_LEVEL_PROBE_MAX_LEN: usize = DEFAULT_WRITE_BATCH_SIZE;

/// Classification of one list row.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RowKind {
    Null,
    Empty,
    NonEmpty,
}

#[inline(always)]
fn scan_list_row_spans(
    range: Range<usize>,
    mut classify: impl FnMut(usize) -> RowKind,
    emit: &mut impl FnMut(RowKind, Range<usize>) -> Result<()>,
) -> Result<()> {
    if range.is_empty() {
        return Ok(());
    }
    let mut kind = classify(range.start);
    let mut start = range.start;
    for row in range.start + 1..range.end {
        let next = classify(row);
        if next != kind {
            emit(kind, start..row)?;
            kind = next;
            start = row;
        }
    }
    emit(kind, start..range.end)
}

#[inline(always)]
fn for_each_list_row_span<O: OffsetSizeTrait>(
    offsets: &[O],
    nulls: Option<&NullBuffer>,
    range: Range<usize>,
    mut emit: impl FnMut(RowKind, Range<usize>) -> Result<()>,
) -> Result<()> {
    match nulls {
        Some(nulls) => scan_list_row_spans(
            range,
            |row| {
                if nulls.is_null(row) {
                    RowKind::Null
                } else if offsets[row] == offsets[row + 1] {
                    RowKind::Empty
                } else {
                    RowKind::NonEmpty
                }
            },
            &mut emit,
        ),
        None => scan_list_row_spans(
            range,
            |row| {
                if offsets[row] == offsets[row + 1] {
                    RowKind::Empty
                } else {
                    RowKind::NonEmpty
                }
            },
            &mut emit,
        ),
    }
}

impl LevelInfoBuilder {
    /// Finalize leaves directly while flattening the traversal tree, avoiding
    /// an intermediate vector of builders at the module boundary.
    #[inline(always)]
    fn into_leaf_plans(self) -> Vec<LeafPlan> {
        match self.node {
            LevelInfoNode::Primitive(builder) => vec![builder.finish()],
            LevelInfoNode::List(child, _, _) | LevelInfoNode::LargeList(child, _, _) => {
                child.0.into_leaf_plans()
            }
            LevelInfoNode::FixedSizeList(child, _, _)
            | LevelInfoNode::ListView(child, _, _, _)
            | LevelInfoNode::LargeListView(child, _, _, _) => child.into_leaf_plans(),
            LevelInfoNode::Struct(children, _) => children
                .into_iter()
                .flat_map(LevelInfoBuilder::into_leaf_plans)
                .collect(),
        }
    }

    /// Create a new [`LevelInfoBuilder`] for the given [`Field`] and parent [`LevelContext`]
    fn try_new(field: &Field, parent_ctx: LevelContext, array: &ArrayRef) -> Result<Self> {
        if !Self::logical_types_compatible(field.data_type(), array.data_type()) {
            return Err(arrow_err!(format!(
                "Incompatible type. Field '{}' has type {}, array has type {}",
                field.name(),
                field.data_type(),
                array.data_type(),
            )));
        }

        let field = normalized(field);
        let is_nullable = field.nullable;
        let nulls = NodeNulls::new(field, array.logical_nulls());

        match array.data_type() {
            d if is_leaf(d) || matches!(d, DataType::Dictionary(_, value) if is_leaf(value)) => {
                let levels = LeafPlanBuilder::new(parent_ctx, is_nullable, array.clone());
                Ok(Self {
                    node: LevelInfoNode::Primitive(levels),
                    nulls,
                })
            }
            DataType::Struct(_) => {
                let DataType::Struct(children) = field.data_type else {
                    unreachable!("compatible logical struct field changed kind")
                };
                let array = array.as_struct();
                let def_level = parent_ctx.def_level + is_nullable as i16;

                let ctx = LevelContext {
                    def_level,
                    ..parent_ctx
                };

                let children = children
                    .iter()
                    .zip(array.columns())
                    .map(|(f, a)| Self::try_new(f, ctx, a))
                    .collect::<Result<_>>()?;

                Ok(Self {
                    node: LevelInfoNode::Struct(children, ctx),
                    nulls,
                })
            }
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Map(_, _)
            | DataType::FixedSizeList(_, _)
            | DataType::ListView(_)
            | DataType::LargeListView(_) => {
                let def_level = parent_ctx.def_level + 1 + is_nullable as i16;

                let ctx = LevelContext {
                    rep_level: parent_ctx.rep_level + 1,
                    def_level,
                };

                let node = match field.data_type {
                    DataType::List(child) => {
                        let list = array.as_list();
                        let child = Self::try_new_list_child(child, ctx, list.values())?;
                        let offsets = list.offsets().clone();
                        LevelInfoNode::List(child, ctx, offsets)
                    }
                    DataType::LargeList(child) => {
                        let list = array.as_list();
                        let child = Self::try_new_list_child(child, ctx, list.values())?;
                        let offsets = list.offsets().clone();
                        LevelInfoNode::LargeList(child, ctx, offsets)
                    }
                    DataType::Map(child, _) => {
                        let map = array.as_map();
                        let entries = Arc::new(map.entries().clone()) as ArrayRef;
                        let child = Self::try_new_list_child(child, ctx, &entries)?;
                        let offsets = map.offsets().clone();
                        LevelInfoNode::List(child, ctx, offsets)
                    }
                    DataType::FixedSizeList(child, size) => {
                        let list = array.as_fixed_size_list();
                        let child = Self::try_new(child.as_ref(), ctx, list.values())?;
                        LevelInfoNode::FixedSizeList(Box::new(child), ctx, *size as _)
                    }
                    DataType::ListView(child) => {
                        let list = array.as_list_view();
                        let child = Self::try_new(child.as_ref(), ctx, list.values())?;
                        let offsets = list.offsets().clone();
                        let sizes = list.sizes().clone();
                        LevelInfoNode::ListView(Box::new(child), ctx, offsets, sizes)
                    }
                    DataType::LargeListView(child) => {
                        let list = array.as_list_view();
                        let child = Self::try_new(child.as_ref(), ctx, list.values())?;
                        let offsets = list.offsets().clone();
                        let sizes = list.sizes().clone();
                        LevelInfoNode::LargeListView(Box::new(child), ctx, offsets, sizes)
                    }
                    _ => unreachable!(),
                };
                Ok(Self { node, nulls })
            }
            d => Err(nyi_err!("Datatype {} is not yet supported", d)),
        }
    }

    fn try_new_list_child(
        field: &Field,
        ctx: LevelContext,
        values: &ArrayRef,
    ) -> Result<ListChildBuilder> {
        let child = Self::try_new(field, ctx, values)?;
        let flat = child.child_has_no_nested_rep();
        Ok(ListChildBuilder(Box::new(child), flat))
    }

    /// Given an `array`, write the level data for the elements in `range`
    fn write(&mut self, range: Range<usize>) -> Result<()> {
        let nulls = self.nulls.for_range(range.clone())?;
        match &mut self.node {
            LevelInfoNode::Primitive(info) => Self::write_leaf(info, nulls, range),
            LevelInfoNode::List(child, ctx, offsets) => {
                Self::write_list(child, ctx, offsets, nulls, range)
            }
            LevelInfoNode::LargeList(child, ctx, offsets) => {
                Self::write_list(child, ctx, offsets, nulls, range)
            }
            LevelInfoNode::FixedSizeList(child, ctx, size) => {
                Self::write_fixed_size_list(child, ctx, *size, nulls, range)
            }
            LevelInfoNode::ListView(child, ctx, offsets, sizes) => {
                Self::write_list_view(child, ctx, offsets, sizes, nulls, range)
            }
            LevelInfoNode::LargeListView(child, ctx, offsets, sizes) => {
                Self::write_list_view(child, ctx, offsets, sizes, nulls, range)
            }
            LevelInfoNode::Struct(children, ctx) => Self::write_struct(children, ctx, nulls, range),
        }
    }

    /// Returns `true` if the child contains no nested repetition levels, meaning
    /// each child element produces exactly one rep_level entry in the leaf.
    /// This is true for `Primitive` children and `Struct` trees with no list descendants.
    fn child_has_no_nested_rep(&self) -> bool {
        match &self.node {
            LevelInfoNode::Primitive(..) => true,
            LevelInfoNode::Struct(children, _) => {
                children.iter().all(|c| c.child_has_no_nested_rep())
            }
            _ => false,
        }
    }

    /// Write `range` elements from ListArray `array`
    ///
    /// Note: MapArrays are `ListArray<i32>` under the hood and so are dispatched to this method
    fn write_list<O: OffsetSizeTrait>(
        child: &mut ListChildBuilder,
        ctx: &LevelContext,
        offsets: &[O],
        nulls: Option<&NullBuffer>,
        range: Range<usize>,
    ) -> Result<()> {
        if nulls.is_some_and(|nulls| nulls.null_count() == nulls.len()) {
            let count = range.end - range.start;
            child.0.visit_leaves(|leaf| {
                leaf.extend_uniform_levels(ctx.def_level - 2, ctx.rep_level - 1, count);
            });
            return Ok(());
        }

        if child.1 {
            return Self::write_flat_list(&mut child.0, ctx, offsets, nulls, range);
        }

        let child = &mut child.0;

        let offsets = &offsets[range.start..range.end + 1];

        // In a list column, each row falls into one of three categories:
        // - "null": the list slot is absent (!is_valid), encoded at def_level - 2
        // - "empty": the list slot is present but has zero elements
        //   (offsets[i] == offsets[i+1]), encoded at def_level - 1
        // - non-empty: the list slot has child values, which are recursed into
        //
        // Consecutive runs of null or empty rows are batched and written together.
        // Consecutive non-empty valid rows may also be batched as child ranges.
        match nulls {
            Some(nulls) => {
                if nulls.null_count() == 0 {
                    return Self::write_valid_list_rows(child, ctx, offsets);
                }

                let row_count = range.end - range.start;
                let validity = nulls.inner().slice(range.start, row_count);
                let mut cursor = 0;

                for (valid_start, valid_end) in validity.set_slices() {
                    Self::write_list_level_run(child, ctx, ctx.def_level - 2, valid_start - cursor);
                    Self::write_valid_list_rows(child, ctx, &offsets[valid_start..valid_end + 1])?;
                    cursor = valid_end;
                }

                Self::write_list_level_run(child, ctx, ctx.def_level - 2, row_count - cursor);
            }
            None => {
                Self::write_valid_list_rows(child, ctx, offsets)?;
            }
        }
        Ok(())
    }

    /// Flat-list path for children where each element has one level slot.
    ///
    /// The one-to-one child-element/repetition-level mapping lets contiguous
    /// non-empty rows cross in one child batch, after which list starts can be
    /// stamped directly from their offsets.
    fn write_flat_list<O: OffsetSizeTrait>(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        offsets: &[O],
        nulls: Option<&NullBuffer>,
        range: Range<usize>,
    ) -> Result<()> {
        let list_start_rep = ctx.rep_level - 1;
        for_each_list_row_span(offsets, nulls, range, |kind, rows| match kind {
            RowKind::Null | RowKind::Empty => {
                let def = ctx.def_level - if kind == RowKind::Null { 2 } else { 1 };
                let count = rows.end - rows.start;
                child.visit_leaves(|leaf| {
                    leaf.append_rep_level_run(list_start_rep, count);
                    leaf.append_def_level_run(def, count);
                });
                Ok(())
            }
            RowKind::NonEmpty => Self::write_flat_list_span(child, ctx, offsets, rows),
        })
    }

    fn write_flat_list_span<O: OffsetSizeTrait>(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        offsets: &[O],
        rows: Range<usize>,
    ) -> Result<()> {
        let values_start = offsets[rows.start].as_usize();
        let values_end = offsets[rows.end].as_usize();
        child.write(values_start..values_end)?;
        // `child.write` appended one repetition level per child element, so
        // each row's first entry is directly addressable from its list offset.
        child.visit_leaves(|leaf| {
            let rep_levels = leaf.tail.rep_levels.materialize_mut().unwrap();
            let batch_base = rep_levels.len() - (values_end - values_start);
            for row in rows.clone() {
                rep_levels[batch_base + offsets[row].as_usize() - values_start] = ctx.rep_level - 1;
            }
        });
        Ok(())
    }

    /// Write a contiguous run of known-valid ListArray rows.
    ///
    /// Each row is either empty, emitted as a uniform level run, or non-empty,
    /// recursively written as one batched child range.
    fn write_valid_list_rows<O: OffsetSizeTrait>(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        offsets: &[O],
    ) -> Result<()> {
        let mut pending_empties: usize = 0;
        let mut pending_values_start: Option<usize> = None;

        for (idx, w) in offsets.windows(2).enumerate() {
            let start_idx = w[0].as_usize();
            let end_idx = w[1].as_usize();
            if start_idx == end_idx {
                if let Some(start) = pending_values_start.take() {
                    Self::write_non_null_list_run(child, ctx, &offsets[start..idx + 1])?;
                }
                pending_empties += 1;
            } else {
                Self::write_list_level_run(child, ctx, ctx.def_level - 1, pending_empties);
                pending_empties = 0;
                pending_values_start.get_or_insert(idx);
            }
        }

        if let Some(start) = pending_values_start.take() {
            Self::write_non_null_list_run(child, ctx, &offsets[start..offsets.len()])?;
        }
        Self::write_list_level_run(child, ctx, ctx.def_level - 1, pending_empties);
        Ok(())
    }

    #[inline(always)]
    fn write_list_level_run(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        def_level: i16,
        count: usize,
    ) {
        if count > 0 {
            child.visit_leaves(|leaf| {
                leaf.append_rep_level_run(ctx.rep_level - 1, count);
                leaf.append_def_level_run(def_level, count);
            });
        }
    }

    fn write_non_null_list_run<O: OffsetSizeTrait>(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        offsets: &[O],
    ) -> Result<()> {
        debug_assert!(offsets.len() >= 2);
        debug_assert!(
            offsets
                .windows(2)
                .all(|w| w[0].as_usize() < w[1].as_usize())
        );

        let start_idx = offsets[0].as_usize();
        let end_idx = offsets[offsets.len() - 1].as_usize();
        child.write(start_idx..end_idx)?;

        child.visit_leaves(|leaf| {
            let rep_levels = leaf.tail.rep_levels.materialize_mut().unwrap();
            let mut rev = rep_levels.iter_mut().rev();

            for w in offsets.windows(2).rev() {
                let mut remaining = w[1].as_usize() - w[0].as_usize();
                debug_assert!(remaining > 0);

                while remaining > 0 {
                    let next = rev.next().unwrap();
                    if *next > ctx.rep_level {
                        // Nested element - ignore
                        continue;
                    }

                    remaining -= 1;
                    if remaining == 0 {
                        *next = ctx.rep_level - 1;
                    }
                }
            }
        });
        Ok(())
    }

    /// Write `range` elements from ListViewArray `array`
    fn write_list_view<O: OffsetSizeTrait>(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        offsets: &[O],
        sizes: &[O],
        nulls: Option<&NullBuffer>,
        range: Range<usize>,
    ) -> Result<()> {
        let offsets = &offsets[range.start..range.end];
        let sizes = &sizes[range.start..range.end];

        let write_non_null_slice =
            |child: &mut LevelInfoBuilder, start_idx: usize, end_idx: usize| -> Result<()> {
                child.write(start_idx..end_idx)?;
                child.visit_leaves(|leaf| {
                    let rep_levels = leaf.tail.rep_levels.materialize_mut().unwrap();
                    let mut rev = rep_levels.iter_mut().rev();
                    let mut remaining = end_idx - start_idx;

                    loop {
                        let next = rev.next().unwrap();
                        if *next > ctx.rep_level {
                            // Nested element - ignore
                            continue;
                        }

                        remaining -= 1;
                        if remaining == 0 {
                            *next = ctx.rep_level - 1;
                            break;
                        }
                    }
                });
                Ok(())
            };

        match nulls {
            Some(nulls) => {
                let null_offset = range.start;
                // TODO: Faster bitmask iteration (#1757)
                for (idx, (offset, size)) in offsets.iter().zip(sizes.iter()).enumerate() {
                    let is_valid = nulls.is_valid(idx + null_offset);
                    let start_idx = offset.as_usize();
                    let size = size.as_usize();
                    let end_idx = start_idx + size;
                    if !is_valid {
                        Self::write_list_level_run(child, ctx, ctx.def_level - 2, 1)
                    } else if size == 0 {
                        Self::write_list_level_run(child, ctx, ctx.def_level - 1, 1)
                    } else {
                        write_non_null_slice(child, start_idx, end_idx)?
                    }
                }
            }
            None => {
                for (offset, size) in offsets.iter().zip(sizes.iter()) {
                    let start_idx = offset.as_usize();
                    let size = size.as_usize();
                    let end_idx = start_idx + size;
                    if size == 0 {
                        Self::write_list_level_run(child, ctx, ctx.def_level - 1, 1)
                    } else {
                        write_non_null_slice(child, start_idx, end_idx)?
                    }
                }
            }
        }
        Ok(())
    }

    /// Write `range` elements from StructArray `array`
    fn write_struct(
        children: &mut [LevelInfoBuilder],
        ctx: &LevelContext,
        nulls: Option<&NullBuffer>,
        range: Range<usize>,
    ) -> Result<()> {
        let write_null = |children: &mut [LevelInfoBuilder], range: Range<usize>| {
            let len = range.end - range.start;
            for child in children {
                child.visit_leaves(|info| {
                    info.extend_uniform_levels(ctx.def_level - 1, ctx.rep_level, len);
                })
            }
        };

        // Fast path: entire struct array is null; emit bulk null def/rep levels
        if nulls.is_some_and(|nulls| nulls.null_count() == nulls.len()) {
            write_null(children, range);
            return Ok(());
        }

        let write_non_null =
            |children: &mut [LevelInfoBuilder], range: Range<usize>| -> Result<()> {
                for child in children {
                    child.write(range.clone())?;
                }
                Ok(())
            };

        match nulls {
            Some(validity) => {
                let mut last_non_null_idx = None;
                let mut last_null_idx = None;

                // TODO: Faster bitmask iteration (#1757)
                for i in range.clone() {
                    match validity.is_valid(i) {
                        true => {
                            if let Some(last_idx) = last_null_idx.take() {
                                write_null(children, last_idx..i)
                            }
                            last_non_null_idx.get_or_insert(i);
                        }
                        false => {
                            if let Some(last_idx) = last_non_null_idx.take() {
                                write_non_null(children, last_idx..i)?
                            }
                            last_null_idx.get_or_insert(i);
                        }
                    }
                }

                if let Some(last_idx) = last_null_idx.take() {
                    write_null(children, last_idx..range.end)
                }

                if let Some(last_idx) = last_non_null_idx.take() {
                    write_non_null(children, last_idx..range.end)?
                }
            }
            None => write_non_null(children, range)?,
        }
        Ok(())
    }

    /// Write `range` elements from FixedSizeListArray with child data `values` and null bitmap `nulls`.
    fn write_fixed_size_list(
        child: &mut LevelInfoBuilder,
        ctx: &LevelContext,
        fixed_size: usize,
        nulls: Option<&NullBuffer>,
        range: Range<usize>,
    ) -> Result<()> {
        // Fast path: entire fixed-size list array is null
        if nulls.is_some_and(|nulls| nulls.null_count() == nulls.len()) {
            let count = range.end - range.start;
            Self::write_list_level_run(child, ctx, ctx.def_level - 2, count);
            return Ok(());
        }

        let write_non_null =
            |child: &mut LevelInfoBuilder, start_idx: usize, end_idx: usize| -> Result<()> {
                let values_start = start_idx * fixed_size;
                let values_end = end_idx * fixed_size;
                child.write(values_start..values_end)?;

                child.visit_leaves(|leaf| {
                    let rep_levels = leaf.tail.rep_levels.materialize_mut().unwrap();

                    let row_indices = (0..fixed_size)
                        .rev()
                        .cycle()
                        .take(values_end - values_start);

                    // Step backward over the child rep levels and mark the start of each list
                    rep_levels
                        .iter_mut()
                        .rev()
                        // Filter out reps from nested children
                        .filter(|&&mut r| r == ctx.rep_level)
                        .zip(row_indices)
                        .for_each(|(r, idx)| {
                            if idx == 0 {
                                *r = ctx.rep_level - 1;
                            }
                        });
                });
                Ok(())
            };

        let write_rows =
            |child: &mut LevelInfoBuilder, start_idx: usize, end_idx: usize| -> Result<()> {
                if fixed_size > 0 {
                    write_non_null(child, start_idx, end_idx)
                } else {
                    Self::write_list_level_run(child, ctx, ctx.def_level - 1, end_idx - start_idx);
                    Ok(())
                }
            };

        match nulls {
            Some(nulls) => {
                let mut start_idx = None;
                for idx in range.clone() {
                    if nulls.is_valid(idx) {
                        // Start a run of valid rows if not already inside of one
                        start_idx.get_or_insert(idx);
                    } else {
                        // Write out any pending valid rows
                        if let Some(start) = start_idx.take() {
                            write_rows(child, start, idx)?;
                        }
                        // Add null row
                        Self::write_list_level_run(child, ctx, ctx.def_level - 2, 1)
                    }
                }
                // Write out any remaining valid rows
                if let Some(start) = start_idx.take() {
                    write_rows(child, start, range.end)?;
                }
            }
            // If all rows are valid then write the whole array
            None => write_rows(child, range.start, range.end)?,
        }
        Ok(())
    }

    /// Write a primitive array, as defined by [`is_leaf`]
    fn write_leaf(
        info: &mut LeafPlanBuilder,
        nulls: Option<&NullBuffer>,
        range: Range<usize>,
    ) -> Result<()> {
        let len = range.end - range.start;

        // Fast path: entire leaf array is null
        if nulls.is_some_and(|nulls| nulls.null_count() == nulls.len()) {
            info.extend_uniform_levels(info.max_def_level - 1, info.max_rep_level, len);
            return Ok(());
        }

        match nulls {
            Some(nulls) => {
                assert!(range.end <= nulls.len());
                let max_def_level = info.max_def_level;
                let range_nulls = nulls.slice(range.start, len);
                let null_count = range_nulls.null_count();
                if null_count == 0 {
                    info.append_def_level_run(max_def_level, len);
                    info.append_value_range(range.clone());
                    if !matches!(info.tail.rep_levels, LevelData::Absent) {
                        info.append_rep_level_run(info.max_rep_level, len);
                    }
                    return Ok(());
                }
                // For long null-heavy windows, derive both level runs and
                // selected-value ranges from the same valid-span scan. The
                // adaptive level builder retains long runs and falls back to a
                // flat buffer when the bitmap is too fragmented.
                if len >= BULK_FILL_MIN_LEN && nulls.null_count() * 2 >= nulls.len() {
                    let null_def_level = max_def_level - 1;
                    let (def_levels, values) = (&mut info.tail.def_levels, &mut info.tail.values);
                    let mut cursor = 0;
                    for (start, end) in range_nulls.valid_slices() {
                        def_levels.append_run(null_def_level, start - cursor);
                        def_levels.append_run(max_def_level, end - start);
                        values.append_range(range.start + start..range.start + end);
                        cursor = end;
                    }
                    def_levels.append_run(null_def_level, len - cursor);
                } else {
                    if info.tail.def_levels.len().saturating_add(len) <= COMPACT_LEVEL_PROBE_MAX_LEN
                        && Self::levels_have_compact_runs(&range_nulls)
                    {
                        let null_def_level = max_def_level - 1;
                        let mut cursor = 0;
                        for (start, end) in range_nulls.valid_slices() {
                            info.append_def_level_run(null_def_level, start - cursor);
                            info.append_def_level_run(max_def_level, end - start);
                            cursor = end;
                        }
                        info.append_def_level_run(null_def_level, len - cursor);
                    } else {
                        let bits = nulls.inner();
                        info.extend_def_levels(range.clone().map(|i| {
                            // Safety: range.end was asserted to be in bounds earlier
                            let valid = unsafe { bits.value_unchecked(i) };
                            max_def_level - (!valid as i16)
                        }));
                    }
                    info.append_deferred_sparse_values(range_nulls, range.start, len - null_count);
                }
            }
            None => {
                info.append_def_level_run(info.max_def_level, len);
                info.append_value_range(range);
            }
        }

        if !matches!(info.tail.rep_levels, LevelData::Absent) {
            info.append_rep_level_run(info.max_rep_level, len);
        }
        Ok(())
    }

    /// Probe a bounded prefix before choosing run-form definition levels. This
    /// retains long ordinary runs without making fragmented nullable columns
    /// pay the adaptive run-building cost that they immediately discard.
    fn levels_have_compact_runs(nulls: &NullBuffer) -> bool {
        let len = nulls.len().min(LEVEL_RUN_PROBE_SIZE);
        if len < MIN_AVERAGE_LEVEL_RUN_LENGTH {
            return false;
        }
        let mut runs = 1;
        let mut previous = nulls.is_valid(0);
        for index in 1..len {
            let valid = nulls.is_valid(index);
            runs += usize::from(valid != previous);
            if runs * MIN_AVERAGE_LEVEL_RUN_LENGTH > len {
                return false;
            }
            previous = valid;
        }
        true
    }

    /// Visits all children of this node in depth first order
    fn visit_leaves(&mut self, visit: impl Fn(&mut LeafPlanBuilder) + Copy) {
        match &mut self.node {
            LevelInfoNode::Primitive(info) => visit(info),
            LevelInfoNode::List(c, _, _) | LevelInfoNode::LargeList(c, _, _) => {
                c.0.visit_leaves(visit)
            }
            LevelInfoNode::FixedSizeList(c, _, _)
            | LevelInfoNode::ListView(c, _, _, _)
            | LevelInfoNode::LargeListView(c, _, _, _) => c.visit_leaves(visit),
            LevelInfoNode::Struct(children, _) => {
                for c in children {
                    c.visit_leaves(visit)
                }
            }
        }
    }

    /// Determine if non-identical nodes are logically compatible.
    ///
    /// Dictionary and run-end encoded wrappers are transparent. Nested nodes
    /// validate only their own shape here; their children are validated by the
    /// recursive [`Self::try_new`] calls. This accepts, for example,
    /// `Struct<REE<Int32>>` alongside a dense `Struct<Int32>` without walking
    /// the entire hierarchy at every parent node.
    fn logical_types_compatible(a: &DataType, b: &DataType) -> bool {
        let a = logical_type(a).0;
        let b = logical_type(b).0;

        a == b
            || leaf_types_compatible(a, b)
            || match (a, b) {
                // Composite children are checked recursively by `try_new`.
                (DataType::Struct(a), DataType::Struct(b)) => a.len() == b.len(),
                (DataType::List(_), DataType::List(_))
                | (DataType::LargeList(_), DataType::LargeList(_))
                | (DataType::ListView(_), DataType::ListView(_))
                | (DataType::LargeListView(_), DataType::LargeListView(_)) => true,
                (DataType::FixedSizeList(_, a), DataType::FixedSizeList(_, b)) => a == b,
                (DataType::Map(_, a), DataType::Map(_, b)) => a == b,

                // otherwise we have incompatible types
                _ => false,
            }
    }
}

/// One owned definition- or repetition-level stream for a leaf batch.
#[derive(Debug, Clone)]
pub(crate) enum LevelData {
    Absent,
    Materialized(Vec<i16>),
    Uniform {
        value: i16,
        count: usize,
    },
    /// A compact run stream for ordinary Arrow hierarchy levels.
    Runs(LevelRuns),
}

impl LevelData {
    pub(super) fn new(present: bool) -> Self {
        match present {
            true => Self::Materialized(Vec::new()),
            false => Self::Absent,
        }
    }

    /// Number of level entries this stream describes.
    pub(crate) fn len(&self) -> usize {
        self.as_ref().len()
    }

    pub(crate) fn as_ref(&self) -> LevelDataRef<'_> {
        match self {
            Self::Absent => LevelDataRef::Absent,
            Self::Materialized(values) => LevelDataRef::Materialized(values),
            Self::Uniform { value, count } => LevelDataRef::Uniform {
                value: *value,
                count: *count,
            },
            Self::Runs(runs) => LevelDataRef::Runs(RunLevelsRef::from_level_runs(
                runs.ends(),
                runs.values(),
                0,
                runs.len(),
            )),
        }
    }

    pub(super) fn append_run(&mut self, value: i16, count: usize) {
        if count == 0 {
            return;
        }

        match self {
            // No physical level stream exists for this schema. Higher-level
            // traversal may still append implicit levels, so this remains a no-op.
            Self::Absent => {}
            // Start compact: the first appended run can be represented without
            // allocating a level buffer.
            Self::Materialized(values) if values.is_empty() => {
                *self = Self::Uniform { value, count };
            }
            // Already materialized, so preserve the buffer representation and append.
            Self::Materialized(values) => values.extend(std::iter::repeat_n(value, count)),
            // Preserve the compact representation while the appended run has
            // the same value.
            Self::Uniform {
                value: uniform_value,
                count: uniform_count,
            } if *uniform_value == value => {
                *uniform_count += count;
            }
            // A different value breaks uniformity but still admits a compact
            // two-run representation.
            Self::Uniform {
                value: uniform_value,
                count: uniform_count,
            } => {
                let runs = LevelRuns::from_two_runs(*uniform_value, *uniform_count, value, count);
                *self = if runs.should_materialize() {
                    Self::Materialized(runs.into_materialized())
                } else {
                    Self::Runs(runs)
                };
            }
            Self::Runs(runs) => {
                runs.append_run(value, count);
                if runs.should_materialize() {
                    let runs = match std::mem::replace(self, Self::Absent) {
                        Self::Runs(runs) => runs,
                        _ => unreachable!(),
                    };
                    *self = Self::Materialized(runs.into_materialized());
                }
            }
        }
    }

    #[inline(never)]
    pub(super) fn extend_from_iter<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = i16>,
    {
        match self {
            Self::Absent => {}
            Self::Materialized(values) => values.extend(iter),
            _ => self.materialize_mut().unwrap().extend(iter),
        }
    }

    /// Convert a compact level representation into a materialized buffer if
    /// needed, then return the mutable level buffer. Returns `None` when no
    /// physical level stream exists.
    pub(super) fn materialize_mut(&mut self) -> Option<&mut Vec<i16>> {
        let values = match self {
            Self::Absent => return None,
            Self::Materialized(values) => return Some(values),
            Self::Uniform { value, count } => vec![*value; *count],
            Self::Runs(_) => {
                let runs = match std::mem::replace(self, Self::Absent) {
                    Self::Runs(runs) => runs,
                    _ => unreachable!(),
                };
                runs.into_materialized()
            }
        };
        *self = Self::Materialized(values);
        let Self::Materialized(values) = self else {
            unreachable!()
        };
        Some(values)
    }

    /// Releases geometric spare capacity from a completed plan's run buffers.
    fn finish(&mut self) {
        if let Self::Runs(runs) = self {
            runs.ends.shrink_to_fit();
            runs.values.shrink_to_fit();
        }
    }
}

/// Cumulative-end run representation for ordinary definition and repetition
/// levels. These runs are produced directly while
/// shredding the Arrow hierarchy and don't depend on a run-end array.
///
/// The two vectors form a structure-of-arrays representation: entry `i`
/// describes the half-open logical range `ends[i - 1]..ends[i]` (or
/// `0..ends[0]`) with value `values[i]`. The representation is canonical:
/// ends are strictly increasing and adjacent values always differ.
#[derive(Debug, Clone)]
pub(crate) struct LevelRuns {
    ends: Vec<usize>,
    values: Vec<i16>,
}

impl LevelRuns {
    fn from_two_runs(
        first_value: i16,
        first_count: usize,
        second_value: i16,
        second_count: usize,
    ) -> Self {
        debug_assert_ne!(first_value, second_value);
        debug_assert_ne!(first_count, 0);
        debug_assert_ne!(second_count, 0);
        let ends = vec![first_count];
        let values = vec![first_value];
        let mut runs = Self { ends, values };
        runs.append_run(second_value, second_count);
        runs
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.ends.last().copied().unwrap_or(0)
    }

    #[inline]
    pub(crate) fn ends(&self) -> &[usize] {
        &self.ends
    }

    #[inline]
    pub(crate) fn values(&self) -> &[i16] {
        &self.values
    }

    fn append_run(&mut self, value: i16, count: usize) {
        debug_assert_ne!(count, 0);
        let end = self
            .len()
            .checked_add(count)
            .expect("level stream length overflow");
        if self.values.last().copied() == Some(value) {
            *self.ends.last_mut().expect("a value has a matching end") = end;
        } else {
            self.ends.push(end);
            self.values.push(value);
        }
        self.debug_assert_valid();
    }

    /// Once a sufficiently large stream averages fewer than eight logical
    /// entries per run, explicit levels are both smaller and cheaper to walk.
    /// The transition is deliberately one-way: [`LevelData::Materialized`]
    /// never attempts to reconstruct runs.
    #[inline]
    fn should_materialize(&self) -> bool {
        let len = self.len();
        len >= LEVEL_RUN_PROBE_SIZE
            && self
                .values
                .len()
                .saturating_mul(MIN_AVERAGE_LEVEL_RUN_LENGTH)
                > len
    }

    fn into_materialized(self) -> Vec<i16> {
        let mut materialized = Vec::with_capacity(self.len());
        let mut start = 0;
        for (end, value) in self.ends.into_iter().zip(self.values) {
            materialized.extend(std::iter::repeat_n(value, end - start));
            start = end;
        }
        materialized
    }

    #[inline]
    fn debug_assert_valid(&self) {
        debug_assert_eq!(self.ends.len(), self.values.len());
        debug_assert!(!self.ends.is_empty());
        debug_assert!(self.ends[0] > 0);
        debug_assert!(self.ends.windows(2).all(|window| window[0] < window[1]));
        debug_assert!(self.values.windows(2).all(|window| window[0] != window[1]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::builder::*;
    use arrow_array::types::Int32Type;
    use arrow_array::*;
    use arrow_buffer::{Buffer, ToByteSlice};
    use arrow_cast::display::array_value_to_string;
    use arrow_data::{ArrayData, ArrayDataBuilder};
    use arrow_schema::{Fields, Schema};

    fn only_batch(plan: &LeafPlan) -> LeafBatch<'_> {
        plan.view()
    }

    fn assert_level_stream(
        actual: crate::column::writer::LevelDataRef<'_>,
        expected: Option<&[i16]>,
    ) {
        match expected {
            None => assert!(matches!(
                actual,
                crate::column::writer::LevelDataRef::Absent
            )),
            Some(expected) => assert_eq!(actual.cursor().collect::<Vec<_>>(), expected),
        }
    }

    /// Compare the logical contents of an ordinary leaf plan without coupling
    /// behavior tests to its compact level or selection representation.
    fn assert_leaf_plan(
        actual: &LeafPlan,
        expected_array: &dyn Array,
        expected_defs: Option<&[i16]>,
        expected_reps: Option<&[i16]>,
        expected_values: impl IntoIterator<Item = usize>,
    ) {
        assert_eq!(actual.array.as_ref(), expected_array);
        let batch = only_batch(actual);
        assert_level_stream(batch.def_level_data(), expected_defs);
        assert_level_stream(batch.rep_level_data(), expected_reps);
        assert_eq!(
            batch.value_selection().cursor().collect::<Vec<_>>(),
            expected_values.into_iter().collect::<Vec<_>>()
        );
    }

    fn assert_nullable_spans(valid: &[std::ops::Range<usize>], expect_run_levels: bool) {
        let is_valid = |idx| valid.iter().any(|range| range.contains(&idx));
        let array = Arc::new(Int32Array::from(
            (0..80)
                .map(|idx| is_valid(idx).then_some(idx as i32))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let field = Field::new_list_field(DataType::Int32, true);
        let levels = calculate_array_levels(&array, &field).unwrap();
        let batch = &levels[0].batch;
        if expect_run_levels {
            assert!(matches!(batch.values, ValueSelection::Ranges { .. }));
        } else {
            assert!(matches!(
                batch.values,
                ValueSelection::DeferredSparse { .. }
            ));
        }
        assert_eq!(
            batch.values.as_ref().cursor().collect::<Vec<_>>(),
            valid.iter().flat_map(Clone::clone).collect::<Vec<_>>()
        );
        if expect_run_levels {
            assert!(matches!(batch.def_levels, LevelData::Runs(_)));
            assert_eq!(
                batch.def_levels.as_ref().cursor().collect::<Vec<_>>(),
                (0..80)
                    .map(|idx| i16::from(is_valid(idx)))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_calculate_array_levels_twitter_example() {
        // based on the example at https://blog.twitter.com/engineering/en_us/a/2013/dremel-made-simple-with-parquet.html
        // [[a, b, c], [d, e, f, g]], [[h], [i,j]]

        let leaf_type = Field::new_list_field(DataType::Int32, false);
        let inner_type = DataType::List(Arc::new(leaf_type));
        let inner_field = Field::new("l2", inner_type.clone(), false);
        let outer_type = DataType::List(Arc::new(inner_field));
        let outer_field = Field::new("l1", outer_type.clone(), false);

        let primitives = Int32Array::from_iter(0..10);

        // Cannot use from_iter_primitive as always infers nullable
        let offsets = Buffer::from_iter([0_i32, 3, 7, 8, 10]);
        let inner_list = ArrayDataBuilder::new(inner_type)
            .len(4)
            .add_buffer(offsets)
            .add_child_data(primitives.to_data())
            .build()
            .unwrap();

        let offsets = Buffer::from_iter([0_i32, 2, 4]);
        let outer_list = ArrayDataBuilder::new(outer_type)
            .len(2)
            .add_buffer(offsets)
            .add_child_data(inner_list)
            .build()
            .unwrap();
        let outer_list = make_array(outer_list);

        let levels = calculate_array_levels(&outer_list, &outer_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &primitives,
            Some(&[2; 10]),
            Some(&[0, 2, 2, 1, 2, 2, 2, 0, 1, 2]),
            0..10,
        );
    }

    #[test]
    fn test_calculate_one_level_1() {
        // This test calculates the levels for a non-null primitive array
        let array = Arc::new(Int32Array::from_iter(0..10)) as ArrayRef;
        let field = Field::new_list_field(DataType::Int32, false);

        let levels = calculate_array_levels(&array, &field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(&levels[0], array.as_ref(), None, None, 0..10);
    }

    #[test]
    fn test_calculate_one_level_2() {
        // This test calculates the levels for a nullable primitive array
        let array = Arc::new(Int32Array::from_iter([
            Some(0),
            None,
            Some(0),
            Some(0),
            None,
        ])) as ArrayRef;
        let field = Field::new_list_field(DataType::Int32, true);

        let levels = calculate_array_levels(&array, &field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            array.as_ref(),
            Some(&[1, 0, 1, 1, 0]),
            None,
            [0, 2, 3],
        );
    }

    #[test]
    fn test_calculate_one_level_nullable_no_nulls_uses_uniform_dense() {
        let array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let field = Field::new_list_field(DataType::Int32, true);

        let levels = calculate_array_levels(&array, &field).unwrap();
        assert_eq!(levels.len(), 1);

        let batch = &levels[0].batch;
        assert!(matches!(
            batch.def_levels,
            LevelData::Uniform { value: 1, count: 3 }
        ));
        assert!(matches!(
            batch.values,
            ValueSelection::Dense { offset: 0, len: 3 }
        ));
        assert_leaf_plan(&levels[0], array.as_ref(), Some(&[1, 1, 1]), None, 0..3);
    }

    #[test]
    fn nullable_leaf_defers_sparse_indices_or_retains_compact_ranges() {
        // Mostly-valid input defers sparse-index construction until writing;
        // null-heavy input retains compact ranges and run definitions.
        assert_nullable_spans(&[7..39, 47..76], false);
        assert_nullable_spans(&[10..20, 50..60], true);
    }

    #[test]
    fn test_calculate_array_levels_1() {
        let leaf_field = Field::new_list_field(DataType::Int32, false);
        let list_type = DataType::List(Arc::new(leaf_field));

        // if all array values are defined (e.g. batch<list<_>>)
        // [[0], [1], [2], [3], [4]]

        let leaf_array = Int32Array::from_iter(0..5);
        // Cannot use from_iter_primitive as always infers nullable
        let offsets = Buffer::from_iter(0_i32..6);
        let list = ArrayDataBuilder::new(list_type.clone())
            .len(5)
            .add_buffer(offsets)
            .add_child_data(leaf_array.to_data())
            .build()
            .unwrap();
        let list = make_array(list);

        let list_field = Field::new("list", list_type.clone(), false);
        let levels = calculate_array_levels(&list, &list_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(&levels[0], &leaf_array, Some(&[1; 5]), Some(&[0; 5]), 0..5);

        // array: [[0, 0], NULL, [2, 2], [3, 3, 3, 3], [4, 4, 4]]
        // all values are defined as we do not have nulls on the root (batch)
        // repetition:
        //   0: 0, 1
        //   1: 0
        //   2: 0, 1
        //   3: 0, 1, 1, 1
        //   4: 0, 1, 1
        let leaf_array = Int32Array::from_iter([0, 0, 2, 2, 3, 3, 3, 3, 4, 4, 4]);
        let offsets = Buffer::from_iter([0_i32, 2, 2, 4, 8, 11]);
        let list = ArrayDataBuilder::new(list_type.clone())
            .len(5)
            .add_buffer(offsets)
            .add_child_data(leaf_array.to_data())
            .null_bit_buffer(Some(Buffer::from([0b00011101])))
            .build()
            .unwrap();
        let list = make_array(list);

        let list_field = Field::new("list", list_type, true);
        let levels = calculate_array_levels(&list, &list_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &leaf_array,
            Some(&[2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            Some(&[0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]),
            0..11,
        );
    }

    #[test]
    fn test_write_list_interleaved_null_empty() {
        let leaf_field = Field::new_list_field(DataType::Int32, false);
        let list_type = DataType::List(Arc::new(leaf_field));

        let leaf_array = Int32Array::from(vec![1, 2, 3]);
        let offsets = Buffer::from_iter([0_i32, 0, 0, 2, 2, 2, 2, 3, 3]);
        let null_bitmap = Buffer::from([0b11100110_u8]);
        let list = ArrayDataBuilder::new(list_type.clone())
            .len(8)
            .add_buffer(offsets)
            .add_child_data(leaf_array.to_data())
            .null_bit_buffer(Some(null_bitmap))
            .build()
            .unwrap();
        let list = make_array(list);

        let list_field = Field::new("list", list_type, true);
        let levels = calculate_array_levels(&list, &list_field).unwrap();
        assert_eq!(levels.len(), 1);
        let levels = &levels[0];
        let batch = only_batch(levels);

        assert_eq!(
            batch.def_level_data().cursor().collect::<Vec<_>>(),
            [0, 1, 2, 2, 0, 0, 1, 2, 1],
        );
        assert_eq!(
            batch.rep_level_data().cursor().collect::<Vec<_>>(),
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
        );
        let mut indices = Vec::new();
        batch
            .value_selection()
            .try_for_each(|idx| -> std::result::Result<(), ()> {
                indices.push(idx);
                Ok(())
            })
            .unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_write_list_batches_consecutive_nullable_child_values() {
        let leaf_field = Field::new_list_field(DataType::Int32, true);
        let list_type = DataType::List(Arc::new(leaf_field));

        let leaf_array = Int32Array::from(vec![Some(10), None, Some(12), None, Some(14), Some(15)]);
        let offsets = Buffer::from_iter([0_i32, 2, 4, 4, 4, 6]);
        let null_bitmap = Buffer::from([0b00011011_u8]);
        let list = ArrayDataBuilder::new(list_type.clone())
            .len(5)
            .add_buffer(offsets)
            .add_child_data(leaf_array.to_data())
            .null_bit_buffer(Some(null_bitmap))
            .build()
            .unwrap();
        let list = make_array(list);

        let list_field = Field::new("list", list_type, true);
        let levels = calculate_array_levels(&list, &list_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &leaf_array,
            Some(&[3, 2, 3, 2, 0, 1, 3, 3]),
            Some(&[0, 1, 0, 1, 0, 0, 0, 1]),
            [0, 2, 4, 5],
        );
    }

    #[test]
    fn test_calculate_array_levels_2() {
        // If some values are null
        //
        // This emulates an array in the form: <struct<list<?>>
        // with values:
        // - 0: [0, 1], but is null because of the struct
        // - 1: []
        // - 2: [2, 3], but is null because of the struct
        // - 3: [4, 5, 6, 7]
        // - 4: [8, 9, 10]
        //
        // If the first values of a list are null due to a parent, we have to still account for them
        // while indexing, because they would affect the way the child is indexed
        // i.e. in the above example, we have to know that [0, 1] has to be skipped
        let leaf = Int32Array::from_iter(0..11);
        let leaf_field = Field::new("leaf", DataType::Int32, false);

        let list_type = DataType::List(Arc::new(leaf_field));
        let list = ArrayData::builder(list_type.clone())
            .len(5)
            .add_child_data(leaf.to_data())
            .add_buffer(Buffer::from_iter([0_i32, 2, 2, 4, 8, 11]))
            .build()
            .unwrap();

        let list = make_array(list);
        let list_field = Arc::new(Field::new("list", list_type, true));

        let struct_array =
            StructArray::from((vec![(list_field, list)], Buffer::from([0b00011010])));
        let array = Arc::new(struct_array) as ArrayRef;

        let struct_field = Field::new("struct", array.data_type().clone(), true);

        let levels = calculate_array_levels(&array, &struct_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &leaf,
            Some(&[0, 2, 0, 3, 3, 3, 3, 3, 3, 3]),
            Some(&[0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            4..11,
        );

        // nested lists

        // 0: [[100, 101], [102, 103]]
        // 1: []
        // 2: [[104, 105], [106, 107]]
        // 3: [[108, 109], [110, 111], [112, 113], [114, 115]]
        // 4: [[116, 117], [118, 119], [120, 121]]

        let leaf = Int32Array::from_iter(100..122);
        let leaf_field = Field::new("leaf", DataType::Int32, true);

        let l1_type = DataType::List(Arc::new(leaf_field));
        let offsets = Buffer::from_iter([0_i32, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
        let l1 = ArrayData::builder(l1_type.clone())
            .len(11)
            .add_child_data(leaf.to_data())
            .add_buffer(offsets)
            .build()
            .unwrap();

        let l1_field = Field::new("l1", l1_type, true);
        let l2_type = DataType::List(Arc::new(l1_field));
        let l2 = ArrayData::builder(l2_type)
            .len(5)
            .add_child_data(l1)
            .add_buffer(Buffer::from_iter([0, 2, 2, 4, 8, 11]))
            .build()
            .unwrap();

        let l2 = make_array(l2);
        let l2_field = Field::new("l2", l2.data_type().clone(), true);

        let levels = calculate_array_levels(&l2, &l2_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &leaf,
            Some(&[
                5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            ]),
            Some(&[
                0, 2, 1, 2, 0, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2,
            ]),
            0..22,
        );
    }

    #[test]
    fn test_calculate_array_levels_nested_list() {
        let leaf_field = Field::new("leaf", DataType::Int32, false);
        let list_type = DataType::List(Arc::new(leaf_field));

        // if all array values are defined (e.g. batch<list<_>>)
        // The array at this level looks like:
        // 0: [a]
        // 1: [a]
        // 2: [a]
        // 3: [a]

        let leaf = Int32Array::from_iter([0; 4]);
        let list = ArrayData::builder(list_type.clone())
            .len(4)
            .add_buffer(Buffer::from_iter(0_i32..5))
            .add_child_data(leaf.to_data())
            .build()
            .unwrap();
        let list = make_array(list);

        let list_field = Field::new("list", list_type.clone(), false);
        let levels = calculate_array_levels(&list, &list_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(&levels[0], &leaf, Some(&[1; 4]), Some(&[0; 4]), 0..4);

        // 0: null
        // 1: [1, 2, 3]
        // 2: [4, 5]
        // 3: [6, 7]
        let leaf = Int32Array::from_iter(0..8);
        let list = ArrayData::builder(list_type.clone())
            .len(4)
            .add_buffer(Buffer::from_iter([0_i32, 0, 3, 5, 7]))
            .null_bit_buffer(Some(Buffer::from([0b00001110])))
            .add_child_data(leaf.to_data())
            .build()
            .unwrap();
        let list = make_array(list);
        let list_field = Arc::new(Field::new("list", list_type, true));

        let struct_array = StructArray::from(vec![(list_field, list)]);
        let array = Arc::new(struct_array) as ArrayRef;

        let struct_field = Field::new("struct", array.data_type().clone(), true);
        let levels = calculate_array_levels(&array, &struct_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &leaf,
            Some(&[1, 3, 3, 3, 3, 3, 3, 3]),
            Some(&[0, 0, 1, 1, 0, 1, 0, 1]),
            0..7,
        );

        // nested lists
        // In a JSON syntax with the schema: <struct<list<list<primitive>>>>, this translates into:
        // 0: {"struct": null }
        // 1: {"struct": [ [201], [202, 203], [] ]}
        // 2: {"struct": [ [204, 205, 206], [207, 208, 209, 210] ]}
        // 3: {"struct": [ [], [211, 212, 213, 214, 215] ]}

        let leaf = Int32Array::from_iter(201..216);
        let leaf_field = Field::new("leaf", DataType::Int32, false);
        let list_1_type = DataType::List(Arc::new(leaf_field));
        let list_1 = ArrayData::builder(list_1_type.clone())
            .len(7)
            .add_buffer(Buffer::from_iter([0_i32, 1, 3, 3, 6, 10, 10, 15]))
            .add_child_data(leaf.to_data())
            .build()
            .unwrap();

        let list_1_field = Field::new("l1", list_1_type, true);
        let list_2_type = DataType::List(Arc::new(list_1_field));
        let list_2 = ArrayData::builder(list_2_type.clone())
            .len(4)
            .add_buffer(Buffer::from_iter([0_i32, 0, 3, 5, 7]))
            .null_bit_buffer(Some(Buffer::from([0b00001110])))
            .add_child_data(list_1)
            .build()
            .unwrap();

        let list_2 = make_array(list_2);
        let list_2_field = Arc::new(Field::new("list_2", list_2_type, true));

        let struct_array =
            StructArray::from((vec![(list_2_field, list_2)], Buffer::from([0b00001111])));
        let struct_field = Field::new("struct", struct_array.data_type().clone(), true);

        let array = Arc::new(struct_array) as ArrayRef;
        let levels = calculate_array_levels(&array, &struct_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            &leaf,
            Some(&[1, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5]),
            Some(&[0, 0, 1, 2, 1, 0, 2, 2, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2]),
            0..15,
        );
    }

    #[test]
    fn test_calculate_nested_struct_levels() {
        // tests a <struct[a]<struct[b]<int[c]>>
        // array:
        //  - {a: {b: {c: 1}}}
        //  - {a: {b: {c: null}}}
        //  - {a: {b: {c: 3}}}
        //  - {a: {b: null}}
        //  - {a: null}}
        //  - {a: {b: {c: 6}}}

        let c = Int32Array::from_iter([Some(1), None, Some(3), None, Some(5), Some(6)]);
        let leaf = Arc::new(c) as ArrayRef;
        let c_field = Arc::new(Field::new("c", DataType::Int32, true));
        let b = StructArray::from(((vec![(c_field, leaf.clone())]), Buffer::from([0b00110111])));

        let b_field = Arc::new(Field::new("b", b.data_type().clone(), true));
        let a = StructArray::from((
            (vec![(b_field, Arc::new(b) as ArrayRef)]),
            Buffer::from([0b00101111]),
        ));

        let a_field = Field::new("a", a.data_type().clone(), true);
        let a_array = Arc::new(a) as ArrayRef;

        let levels = calculate_array_levels(&a_array, &a_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            leaf.as_ref(),
            Some(&[3, 2, 3, 1, 0, 3]),
            None,
            [0, 2, 5],
        );
    }

    #[test]
    fn list_single_column() {
        // this tests the level generation from the arrow_writer equivalent test

        let a_values = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let a_value_offsets = arrow::buffer::Buffer::from_iter([0_i32, 1, 3, 3, 6, 10]);
        let a_list_type = DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true)));
        let a_list_data = ArrayData::builder(a_list_type.clone())
            .len(5)
            .add_buffer(a_value_offsets)
            .null_bit_buffer(Some(Buffer::from([0b00011011])))
            .add_child_data(a_values.to_data())
            .build()
            .unwrap();

        assert_eq!(a_list_data.null_count(), 1);

        let a = ListArray::from(a_list_data);

        let item_field = Field::new_list_field(a_list_type, true);
        let mut builder = levels(&item_field, a);
        builder.write(2..4).unwrap();
        let levels = builder.into_leaf_plans();

        assert_eq!(levels.len(), 1);

        let list_level = &levels[0];

        assert_leaf_plan(
            list_level,
            &a_values,
            Some(&[0, 3, 3, 3]),
            Some(&[0, 0, 1, 1]),
            [3, 4, 5],
        );
    }

    #[test]
    fn mixed_struct_list() {
        // this tests the level generation from the equivalent arrow_writer_complex test

        // define schema
        let struct_field_d = Arc::new(Field::new("d", DataType::Float64, true));
        let struct_field_f = Arc::new(Field::new("f", DataType::Float32, true));
        let struct_field_g = Arc::new(Field::new(
            "g",
            DataType::List(Arc::new(Field::new("items", DataType::Int16, false))),
            false,
        ));
        let struct_field_e = Arc::new(Field::new(
            "e",
            DataType::Struct(vec![struct_field_f.clone(), struct_field_g.clone()].into()),
            true,
        ));
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
            Field::new(
                "c",
                DataType::Struct(vec![struct_field_d.clone(), struct_field_e.clone()].into()),
                true, // https://github.com/apache/arrow-rs/issues/245
            ),
        ]);

        // create some data
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![Some(1), None, None, Some(4), Some(5)]);
        let d = Float64Array::from(vec![None, None, None, Some(1.0), None]);
        let f = Float32Array::from(vec![Some(0.0), None, Some(333.3), None, Some(5.25)]);

        let g_value = Int16Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Construct a buffer for value offsets, for the nested array:
        //  [[1], [2, 3], null, [4, 5, 6], [7, 8, 9, 10]]
        let g_value_offsets = arrow::buffer::Buffer::from([0, 1, 3, 3, 6, 10].to_byte_slice());

        // Construct a list array from the above two
        let g_list_data = ArrayData::builder(struct_field_g.data_type().clone())
            .len(5)
            .add_buffer(g_value_offsets)
            .add_child_data(g_value.into_data())
            .build()
            .unwrap();
        let g = ListArray::from(g_list_data);

        let e = StructArray::from(vec![
            (struct_field_f, Arc::new(f.clone()) as ArrayRef),
            (struct_field_g, Arc::new(g) as ArrayRef),
        ]);

        let c = StructArray::from(vec![
            (struct_field_d, Arc::new(d.clone()) as ArrayRef),
            (struct_field_e, Arc::new(e) as ArrayRef),
        ]);

        // build a record batch
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(a.clone()), Arc::new(b.clone()), Arc::new(c)],
        )
        .unwrap();

        //////////////////////////////////////////////
        // calculate the list's level
        let mut levels = vec![];
        batch
            .columns()
            .iter()
            .zip(batch.schema().fields())
            .for_each(|(array, field)| {
                let mut array_levels = calculate_array_levels(array, field).unwrap();
                levels.append(&mut array_levels);
            });
        assert_eq!(levels.len(), 5);

        // test "a" levels
        let list_level = &levels[0];

        assert_leaf_plan(list_level, &a, None, None, 0..5);

        // test "b" levels
        let list_level = levels.get(1).unwrap();

        assert_leaf_plan(list_level, &b, Some(&[1, 0, 0, 1, 1]), None, [0, 3, 4]);

        // test "d" levels
        let list_level = levels.get(2).unwrap();

        assert_leaf_plan(list_level, &d, Some(&[1, 1, 1, 2, 1]), None, [3]);

        // test "f" levels
        let list_level = levels.get(3).unwrap();

        assert_leaf_plan(list_level, &f, Some(&[3, 2, 3, 2, 3]), None, [0, 2, 4]);
    }

    #[test]
    fn test_null_vs_nonnull_struct() {
        // define schema
        let offset_field = Arc::new(Field::new("offset", DataType::Int32, true));
        let schema = Schema::new(vec![Field::new(
            "some_nested_object",
            DataType::Struct(vec![offset_field.clone()].into()),
            false,
        )]);

        // create some data
        let offset = Int32Array::from(vec![1, 2, 3, 4, 5]);

        let some_nested_object =
            StructArray::from(vec![(offset_field, Arc::new(offset) as ArrayRef)]);

        // build a record batch
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(some_nested_object)]).unwrap();

        let struct_null_level =
            calculate_array_levels(batch.column(0), batch.schema().field(0)).unwrap();

        // create second batch
        // define schema
        let offset_field = Arc::new(Field::new("offset", DataType::Int32, true));
        let schema = Schema::new(vec![Field::new(
            "some_nested_object",
            DataType::Struct(vec![offset_field.clone()].into()),
            true,
        )]);

        // create some data
        let offset = Int32Array::from(vec![1, 2, 3, 4, 5]);

        let some_nested_object =
            StructArray::from(vec![(offset_field, Arc::new(offset) as ArrayRef)]);

        // build a record batch
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(some_nested_object)]).unwrap();

        let struct_non_null_level =
            calculate_array_levels(batch.column(0), batch.schema().field(0)).unwrap();

        let required_defs = only_batch(&struct_null_level[0])
            .def_level_data()
            .cursor()
            .collect::<Vec<_>>();
        let optional_defs = only_batch(&struct_non_null_level[0])
            .def_level_data()
            .cursor()
            .collect::<Vec<_>>();
        assert_ne!(required_defs, optional_defs);
    }

    #[test]
    fn test_map_array() {
        // Note: we are using the JSON Arrow reader for brevity
        let json_content = r#"
        {"stocks":{"long": "$AAA", "short": "$BBB"}}
        {"stocks":{"long": "$CCC", "short": null}}
        {"stocks":{"hedged": "$YYY", "long": null, "short": "$D"}}
        "#;
        let entries_struct_type = DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let stocks_field = Field::new(
            "stocks",
            DataType::Map(
                Arc::new(Field::new("entries", entries_struct_type, false)),
                false,
            ),
            // not nullable, so the keys have max level = 1
            false,
        );
        let schema = Arc::new(Schema::new(vec![stocks_field]));
        let builder = arrow::json::ReaderBuilder::new(schema).with_batch_size(64);
        let mut reader = builder.build(std::io::Cursor::new(json_content)).unwrap();

        let batch = reader.next().unwrap().unwrap();

        // calculate the map's level
        let mut levels = vec![];
        batch
            .columns()
            .iter()
            .zip(batch.schema().fields())
            .for_each(|(array, field)| {
                let mut array_levels = calculate_array_levels(array, field).unwrap();
                levels.append(&mut array_levels);
            });
        assert_eq!(levels.len(), 2);

        let map = batch.column(0).as_map();
        // test key levels
        let list_level = &levels[0];

        assert_leaf_plan(
            list_level,
            map.keys().as_ref(),
            Some(&[1; 7]),
            Some(&[0, 1, 0, 1, 0, 1, 1]),
            0..7,
        );

        // test values levels
        let list_level = levels.get(1).unwrap();
        assert_leaf_plan(
            list_level,
            map.values().as_ref(),
            Some(&[2, 2, 2, 1, 2, 1, 2]),
            Some(&[0, 1, 0, 1, 0, 1, 1]),
            [0, 1, 2, 4, 6],
        );
    }

    #[test]
    fn test_list_of_struct() {
        // define schema
        let int_field = Field::new("a", DataType::Int32, true);
        let fields = Fields::from([Arc::new(int_field)]);
        let item_field = Field::new_list_field(DataType::Struct(fields.clone()), true);
        let list_field = Field::new("list", DataType::List(Arc::new(item_field)), true);

        let int_builder = Int32Builder::with_capacity(10);
        let struct_builder = StructBuilder::new(fields, vec![Box::new(int_builder)]);
        let mut list_builder = ListBuilder::new(struct_builder);

        // [{a: 1}], [], null, [null, null], [{a: null}], [{a: 2}]
        //
        // [{a: 1}]
        let values = list_builder.values();
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_value(1);
        values.append(true);
        list_builder.append(true);

        // []
        list_builder.append(true);

        // null
        list_builder.append(false);

        // [null, null]
        let values = list_builder.values();
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values.append(false);
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values.append(false);
        list_builder.append(true);

        // [{a: null}]
        let values = list_builder.values();
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values.append(true);
        list_builder.append(true);

        // [{a: 2}]
        let values = list_builder.values();
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_value(2);
        values.append(true);
        list_builder.append(true);

        let array = Arc::new(list_builder.finish());

        let values = array.values().as_struct().column(0).clone();
        let values_len = values.len();
        assert_eq!(values_len, 5);

        let schema = Arc::new(Schema::new(vec![list_field]));

        let rb = RecordBatch::try_new(schema, vec![array]).unwrap();

        let levels = calculate_array_levels(rb.column(0), rb.schema().field(0)).unwrap();
        let list_level = &levels[0];

        assert_leaf_plan(
            list_level,
            values.as_ref(),
            Some(&[4, 1, 0, 2, 2, 3, 4]),
            Some(&[0, 0, 0, 0, 1, 0, 0]),
            [0, 4],
        );
    }

    #[test]
    fn test_struct_mask_list() {
        // Test the null mask of a struct array masking out non-empty slices of a child ListArray
        let inner = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![None]),
            Some(vec![]),
            Some(vec![Some(3), None]), // Masked by struct array
            Some(vec![Some(4), Some(5)]),
            None, // Masked by struct array
            None,
        ]);
        let values = inner.values().clone();

        // This test assumes that nulls don't take up space
        assert_eq!(inner.values().len(), 7);

        let field = Arc::new(Field::new("list", inner.data_type().clone(), true));
        let array = Arc::new(inner) as ArrayRef;
        let nulls = Buffer::from([0b01010111]);
        let struct_a = StructArray::from((vec![(field, array)], nulls));

        let field = Field::new("struct", struct_a.data_type().clone(), true);
        let array = Arc::new(struct_a) as ArrayRef;
        let levels = calculate_array_levels(&array, &field).unwrap();

        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            values.as_ref(),
            Some(&[4, 4, 3, 2, 0, 4, 4, 0, 1]),
            Some(&[0, 1, 0, 0, 0, 0, 1, 0, 0]),
            [0, 1, 5, 6],
        );
    }

    #[test]
    fn test_list_mask_struct() {
        // Test the null mask of a struct array and the null mask of a list array
        // masking out non-null elements of their children

        let a1 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![None]), // Masked by list array
            Some(vec![]),     // Masked by list array
            Some(vec![Some(3), None]),
            Some(vec![Some(4), Some(5), None, Some(6)]), // Masked by struct array
            None,
            None,
        ]);
        let a1_values = a1.values().clone();
        let a1 = Arc::new(a1) as ArrayRef;

        let a2 = Arc::new(Int32Array::from_iter(vec![
            Some(1), // Masked by list array
            Some(2), // Masked by list array
            None,
            Some(4), // Masked by struct array
            Some(5),
            None,
        ])) as ArrayRef;
        let a2_values = a2.clone();

        let field_a1 = Arc::new(Field::new("list", a1.data_type().clone(), true));
        let field_a2 = Arc::new(Field::new("integers", a2.data_type().clone(), true));

        let nulls = Buffer::from([0b00110111]);
        let struct_a = Arc::new(StructArray::from((
            vec![(field_a1, a1), (field_a2, a2)],
            nulls,
        ))) as ArrayRef;

        let offsets = Buffer::from_iter([0_i32, 0, 2, 2, 3, 5, 5]);
        let nulls = Buffer::from([0b00111100]);

        let list_type = DataType::List(Arc::new(Field::new(
            "struct",
            struct_a.data_type().clone(),
            true,
        )));

        let data = ArrayDataBuilder::new(list_type.clone())
            .len(6)
            .null_bit_buffer(Some(nulls))
            .add_buffer(offsets)
            .add_child_data(struct_a.into_data())
            .build()
            .unwrap();

        let list = make_array(data);
        let list_field = Field::new("col", list_type, true);

        let expected = vec![
            r#""#.to_string(),
            r#""#.to_string(),
            r#"[]"#.to_string(),
            r#"[{list: [3, ], integers: }]"#.to_string(),
            r#"[, {list: , integers: 5}]"#.to_string(),
            r#"[]"#.to_string(),
        ];

        let actual: Vec<_> = (0..6)
            .map(|x| array_value_to_string(&list, x).unwrap())
            .collect();
        assert_eq!(actual, expected);

        let levels = calculate_array_levels(&list, &list_field).unwrap();

        assert_eq!(levels.len(), 2);

        assert_leaf_plan(
            &levels[0],
            a1_values.as_ref(),
            Some(&[0, 0, 1, 6, 5, 2, 3, 1]),
            Some(&[0, 0, 0, 0, 2, 0, 1, 0]),
            [1],
        );

        assert_leaf_plan(
            &levels[1],
            a2_values.as_ref(),
            Some(&[0, 0, 1, 3, 2, 4, 1]),
            Some(&[0, 0, 0, 0, 0, 1, 0]),
            [4],
        );
    }

    #[test]
    fn test_fixed_size_list() {
        // [[1, 2], null, null, [7, 8], null]
        let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), 2);
        builder.values().append_slice(&[1, 2]);
        builder.append(true);
        builder.values().append_slice(&[3, 4]);
        builder.append(false);
        builder.values().append_slice(&[5, 6]);
        builder.append(false);
        builder.values().append_slice(&[7, 8]);
        builder.append(true);
        builder.values().append_slice(&[9, 10]);
        builder.append(false);
        let a = builder.finish();
        let values = a.values().clone();

        let item_field = Field::new_list_field(a.data_type().clone(), true);
        let mut builder = levels(&item_field, a);
        builder.write(1..4).unwrap();
        let levels = builder.into_leaf_plans();

        assert_eq!(levels.len(), 1);

        let list_level = &levels[0];

        assert_leaf_plan(
            list_level,
            values.as_ref(),
            Some(&[0, 0, 3, 3]),
            Some(&[0, 0, 0, 1]),
            [6, 7],
        );
    }

    #[test]
    fn test_fixed_size_list_of_struct() {
        // define schema
        let field_a = Field::new("a", DataType::Int32, true);
        let field_b = Field::new("b", DataType::Int64, false);
        let fields = Fields::from([Arc::new(field_a), Arc::new(field_b)]);
        let item_field = Field::new_list_field(DataType::Struct(fields.clone()), true);
        let list_field = Field::new(
            "list",
            DataType::FixedSizeList(Arc::new(item_field), 2),
            true,
        );

        let builder_a = Int32Builder::with_capacity(10);
        let builder_b = Int64Builder::with_capacity(10);
        let struct_builder =
            StructBuilder::new(fields, vec![Box::new(builder_a), Box::new(builder_b)]);
        let mut list_builder = FixedSizeListBuilder::new(struct_builder, 2);

        // [
        //   [{a: 1, b: 2}, null],
        //   null,
        //   [null, null],
        //   [{a: null, b: 3}, {a: 2, b: 4}]
        // ]

        // [{a: 1, b: 2}, null]
        let values = list_builder.values();
        // {a: 1, b: 2}
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_value(1);
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(2);
        values.append(true);
        // null
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(0);
        values.append(false);
        list_builder.append(true);

        // null
        let values = list_builder.values();
        // null
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(0);
        values.append(false);
        // null
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(0);
        values.append(false);
        list_builder.append(false);

        // [null, null]
        let values = list_builder.values();
        // null
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(0);
        values.append(false);
        // null
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(0);
        values.append(false);
        list_builder.append(true);

        // [{a: null, b: 3}, {a: 2, b: 4}]
        let values = list_builder.values();
        // {a: null, b: 3}
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_null();
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(3);
        values.append(true);
        // {a: 2, b: 4}
        values
            .field_builder::<Int32Builder>(0)
            .unwrap()
            .append_value(2);
        values
            .field_builder::<Int64Builder>(1)
            .unwrap()
            .append_value(4);
        values.append(true);
        list_builder.append(true);

        let array = Arc::new(list_builder.finish());

        assert_eq!(array.values().len(), 8);
        assert_eq!(array.len(), 4);

        let struct_values = array.values().as_struct();
        let values_a = struct_values.column(0).clone();
        let values_b = struct_values.column(1).clone();

        let schema = Arc::new(Schema::new(vec![list_field]));
        let rb = RecordBatch::try_new(schema, vec![array]).unwrap();

        let levels = calculate_array_levels(rb.column(0), rb.schema().field(0)).unwrap();
        let a_levels = &levels[0];
        let b_levels = &levels[1];

        // [[{a: 1}, null], null, [null, null], [{a: null}, {a: 2}]]
        assert_leaf_plan(
            a_levels,
            values_a.as_ref(),
            Some(&[4, 2, 0, 2, 2, 3, 4]),
            Some(&[0, 1, 0, 0, 1, 0, 1]),
            [0, 7],
        );
        // [[{b: 2}, null], null, [null, null], [{b: 3}, {b: 4}]]
        assert_leaf_plan(
            b_levels,
            values_b.as_ref(),
            Some(&[3, 2, 0, 2, 2, 3, 3]),
            Some(&[0, 1, 0, 0, 1, 0, 1]),
            [0, 6, 7],
        );
    }

    #[test]
    fn test_fixed_size_list_empty() {
        let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), 0);
        builder.append(true);
        builder.append(false);
        builder.append(true);
        let array = builder.finish();
        let values = array.values().clone();

        let item_field = Field::new_list_field(array.data_type().clone(), true);
        let mut builder = levels(&item_field, array);
        builder.write(0..3).unwrap();
        let levels = builder.into_leaf_plans();

        assert_eq!(levels.len(), 1);

        let list_level = &levels[0];

        assert_leaf_plan(
            list_level,
            values.as_ref(),
            Some(&[1, 0, 1]),
            Some(&[0, 0, 0]),
            [],
        );
    }

    #[test]
    fn test_fixed_size_list_of_var_lists() {
        // [[[1, null, 3], null], [[4], []], [[5, 6], [null, null]], null]
        let mut builder = FixedSizeListBuilder::new(ListBuilder::new(Int32Builder::new()), 2);
        builder.values().append_value([Some(1), None, Some(3)]);
        builder.values().append_null();
        builder.append(true);
        builder.values().append_value([Some(4)]);
        builder.values().append_value([]);
        builder.append(true);
        builder.values().append_value([Some(5), Some(6)]);
        builder.values().append_value([None, None]);
        builder.append(true);
        builder.values().append_null();
        builder.values().append_null();
        builder.append(false);
        let a = builder.finish();
        let values = a.values().as_list::<i32>().values().clone();

        let item_field = Field::new_list_field(a.data_type().clone(), true);
        let mut builder = levels(&item_field, a);
        builder.write(0..4).unwrap();
        let levels = builder.into_leaf_plans();

        assert_leaf_plan(
            &levels[0],
            values.as_ref(),
            Some(&[5, 4, 5, 2, 5, 3, 5, 5, 4, 4, 0]),
            Some(&[0, 2, 2, 1, 0, 1, 0, 2, 1, 2, 0]),
            [0, 2, 3, 4, 5],
        );
    }

    #[test]
    fn test_null_dictionary_values() {
        let values = Int32Array::new(
            vec![1, 2, 3, 4].into(),
            Some(NullBuffer::from(vec![true, false, true, true])),
        );
        let keys = Int32Array::new(
            vec![1, 54, 2, 0].into(),
            Some(NullBuffer::from(vec![true, false, true, true])),
        );
        // [NULL, NULL, 3, 0]
        let dict = DictionaryArray::new(keys, Arc::new(values));

        let item_field = Field::new_list_field(dict.data_type().clone(), true);

        let mut builder = levels(&item_field, dict.clone());
        builder.write(0..4).unwrap();
        let levels = builder.into_leaf_plans();

        assert_leaf_plan(&levels[0], &dict, Some(&[0, 0, 1, 1]), None, [2, 3]);
    }

    #[test]
    fn required_dictionary_rejects_logical_null_value() {
        let values = Int32Array::new(vec![1, 2].into(), Some(NullBuffer::from(vec![true, false])));
        let keys = Int32Array::from(vec![1, 0]);
        let dict = Arc::new(DictionaryArray::new(keys, Arc::new(values))) as ArrayRef;
        let field = Field::new("item", dict.data_type().clone(), false);

        let err = calculate_array_levels(&dict, &field).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Arrow: Found null at index 0 for required field 'item'"
        );
    }

    #[test]
    fn mismatched_types() {
        let array = Arc::new(Int32Array::from_iter(0..10)) as ArrayRef;
        let field = Field::new_list_field(DataType::Float64, false);

        let err = LevelInfoBuilder::try_new(&field, Default::default(), &array)
            .unwrap_err()
            .to_string();

        assert_eq!(
            err,
            "Arrow: Incompatible type. Field 'item' has type Float64, array has type Int32",
        );
    }

    fn levels<T: Array + 'static>(field: &Field, array: T) -> LevelInfoBuilder {
        let v = Arc::new(array) as ArrayRef;
        LevelInfoBuilder::try_new(field, Default::default(), &v).unwrap()
    }

    #[test]
    fn test_all_null_list() {
        // A list where every slot is null — hits the all-null fast path in write_list.
        let leaf_field = Field::new_list_field(DataType::Int32, false);
        let list_type = DataType::List(Arc::new(leaf_field));

        let leaf_array = Int32Array::from(Vec::<i32>::new());
        let offsets = Buffer::from_iter([0_i32, 0, 0, 0]);
        let null_bitmap = Buffer::from([0b00000000_u8]); // all null
        let list = ArrayDataBuilder::new(list_type.clone())
            .len(3)
            .add_buffer(offsets)
            .add_child_data(leaf_array.to_data())
            .null_bit_buffer(Some(null_bitmap))
            .build()
            .unwrap();
        let list = make_array(list);

        let list_field = Field::new("list", list_type, true);
        let levels = calculate_array_levels(&list, &list_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(&levels[0], &leaf_array, Some(&[0; 3]), Some(&[0; 3]), []);
    }

    #[test]
    fn test_all_null_list_nullable_item() {
        // List<Int32> where every list slot is null.
        // Schema: list (nullable) -> item (int32, nullable)
        // Data: [null, null, null, null]
        //
        // Expected: max_def=3, max_rep=1, def/rep levels all 0.
        let item_field = Arc::new(Field::new_list_field(DataType::Int32, true));
        let list = ListArray::new_null(item_field, 4);
        let values = list.values().clone();
        let field = Field::new("list", list.data_type().clone(), true);
        let array = Arc::new(list) as ArrayRef;

        let levels = calculate_array_levels(&array, &field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            values.as_ref(),
            Some(&[0; 4]),
            Some(&[0; 4]),
            [],
        );
    }

    #[test]
    fn test_all_null_fixed_size_list_nullable_item() {
        // FixedSizeList<Int32; 2> where every list slot is null.
        // Schema: list (nullable) -> item (int32, nullable)
        // Data: [null, null, null]
        //
        // Expected: max_def=3, max_rep=1, def/rep levels all 0.
        let item_field = Arc::new(Field::new_list_field(DataType::Int32, true));
        let list = FixedSizeListArray::new_null(item_field, 2, 3);
        let values = list.values().clone();
        let field = Field::new("list", list.data_type().clone(), true);
        let array = Arc::new(list) as ArrayRef;

        let levels = calculate_array_levels(&array, &field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            values.as_ref(),
            Some(&[0; 3]),
            Some(&[0; 3]),
            [],
        );
    }

    #[test]
    fn test_all_null_struct() {
        // Struct<Int32> where every struct slot is null.
        // Schema: a (struct, nullable) -> c (int32, nullable)
        // Data: [null, null, null, null]
        //
        // Expected: max_def=2, def_levels all 0 (struct is null → child never reached),
        // leaf values are empty.
        let c = Int32Array::from(vec![None::<i32>; 4]);
        let leaf = Arc::new(c) as ArrayRef;
        let c_field = Arc::new(Field::new("c", DataType::Int32, true));
        let a = StructArray::from((vec![(c_field, leaf.clone())], Buffer::from([0b00000000])));
        let a_field = Field::new("a", a.data_type().clone(), true);
        let a_array = Arc::new(a) as ArrayRef;

        let levels = calculate_array_levels(&a_array, &a_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(&levels[0], leaf.as_ref(), Some(&[0; 4]), None, []);
    }

    #[test]
    fn test_all_null_fixed_size_list() {
        // A fixed-size list where every slot is null. Hits the all-null fast path
        // in write_fixed_size_list.
        let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), 2);
        builder.values().append_slice(&[0, 0]);
        builder.append(false);
        builder.values().append_slice(&[0, 0]);
        builder.append(false);
        builder.values().append_slice(&[0, 0]);
        builder.append(false);
        let a = builder.finish();
        let values = a.values().clone();

        let item_field = Field::new_list_field(a.data_type().clone(), true);
        let levels = calculate_array_levels(&(Arc::new(a) as ArrayRef), &item_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(
            &levels[0],
            values.as_ref(),
            Some(&[0; 3]),
            Some(&[0; 3]),
            [],
        );
    }

    #[test]
    fn test_non_nullable_field_with_nulls_in_array() {
        let array = Arc::new(Int32Array::from_iter([Some(1), None, Some(3)])) as ArrayRef;
        let field = Field::new("item", DataType::Int32, false);

        let err = calculate_array_levels(&array, &field).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Arrow: Found null at index 1 for required field 'item'"
        );
    }

    #[test]
    fn required_struct_child_rejects_reachable_null() {
        let batch_fields = Fields::from(vec![Field::new("child", DataType::Int32, true)]);
        let child = Arc::new(Int32Array::from(vec![None::<i32>, None])) as ArrayRef;
        let array = Arc::new(StructArray::new(
            batch_fields,
            vec![child],
            Some(NullBuffer::from(vec![false, true])),
        )) as ArrayRef;

        let writer_fields = Fields::from(vec![Field::new("child", DataType::Int32, false)]);
        let field = Field::new("parent", DataType::Struct(writer_fields), true);

        let err = calculate_array_levels(&array, &field).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Arrow: Found null at index 1 for required field 'child'"
        );
    }

    #[test]
    fn required_struct_child_accepts_null_masked_by_parent() {
        let batch_fields = Fields::from(vec![Field::new("child", DataType::Int32, true)]);
        let child = Arc::new(Int32Array::from(vec![None::<i32>])) as ArrayRef;
        let array = Arc::new(StructArray::new(
            batch_fields,
            vec![child.clone()],
            Some(NullBuffer::from(vec![false])),
        )) as ArrayRef;

        let writer_fields = Fields::from(vec![Field::new("child", DataType::Int32, false)]);
        let field = Field::new("parent", DataType::Struct(writer_fields), true);

        let levels = calculate_array_levels(&array, &field).unwrap();
        assert_leaf_plan(&levels[0], child.as_ref(), Some(&[0]), None, []);
    }

    #[test]
    fn required_struct_rejects_null_before_visiting_children() {
        let fields = Fields::from(vec![Field::new("child", DataType::Int32, true)]);
        let child = Arc::new(Int32Array::from(vec![None::<i32>])) as ArrayRef;
        let array = Arc::new(StructArray::new(
            fields,
            vec![child],
            Some(NullBuffer::from(vec![false])),
        )) as ArrayRef;
        let field = Field::new("parent", array.data_type().clone(), false);

        let err = calculate_array_levels(&array, &field).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Arrow: Found null at index 0 for required field 'parent'"
        );
    }

    #[test]
    fn test_all_null_nested_struct() {
        // Struct<Struct<Int32>> where the outer struct is entirely null.
        // Schema: a (struct, nullable) -> b (struct, nullable) -> c (int32, nullable)
        // Data: [null, null, null]
        //
        // Expected: max_def=3, def_levels all 0.
        let c = Int32Array::from(vec![None::<i32>; 3]);
        let leaf = Arc::new(c) as ArrayRef;
        let c_field = Arc::new(Field::new("c", DataType::Int32, true));
        let b = StructArray::from((vec![(c_field, leaf.clone())], Buffer::from([0b00000000])));
        let b_field = Arc::new(Field::new("b", b.data_type().clone(), true));
        let a = StructArray::from((
            vec![(b_field, Arc::new(b) as ArrayRef)],
            Buffer::from([0b00000000]),
        ));
        let a_field = Field::new("a", a.data_type().clone(), true);
        let a_array = Arc::new(a) as ArrayRef;

        let levels = calculate_array_levels(&a_array, &a_field).unwrap();
        assert_eq!(levels.len(), 1);

        assert_leaf_plan(&levels[0], leaf.as_ref(), Some(&[0; 3]), None, []);
    }

    #[test]
    fn test_all_null_struct_multiple_children() {
        // Struct with two leaf children, entirely null.
        // Schema: a (struct, nullable) -> { c1 (int32, nullable), c2 (int32, nullable) }
        // Data: [null, null]
        //
        // Both leaf columns should get uniform def_levels=0.
        let c1 = Arc::new(Int32Array::from(vec![None::<i32>; 2])) as ArrayRef;
        let c2 = Arc::new(Int32Array::from(vec![None::<i32>; 2])) as ArrayRef;
        let c1_field = Arc::new(Field::new("c1", DataType::Int32, true));
        let c2_field = Arc::new(Field::new("c2", DataType::Int32, true));
        let a = StructArray::from((
            vec![(c1_field, c1.clone()), (c2_field, c2.clone())],
            Buffer::from([0b00000000]),
        ));
        let a_field = Field::new("a", a.data_type().clone(), true);
        let a_array = Arc::new(a) as ArrayRef;

        let levels = calculate_array_levels(&a_array, &a_field).unwrap();
        assert_eq!(levels.len(), 2);

        for (i, leaf) in [c1, c2].into_iter().enumerate() {
            assert_leaf_plan(&levels[i], leaf.as_ref(), Some(&[0; 2]), None, []);
        }
    }

    #[test]
    fn unsupported_nested_dictionary_returns_nyi_before_physical_dispatch() {
        let inner_dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![0, 1]),
                Arc::new(Int32Array::from(vec![11, 13])),
            )
            .unwrap(),
        );
        let dictionary_of_dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(Int32Array::from(vec![0, 1]), inner_dictionary)
                .unwrap(),
        );

        let field = Field::new("c", dictionary_of_dictionary.data_type().clone(), true);
        match calculate_array_levels(&dictionary_of_dictionary, &field).unwrap_err() {
            ParquetError::NYI(message) => assert!(message.contains("not yet supported")),
            other => panic!("expected nested dictionary NYI error, got {other}"),
        }
    }
}
