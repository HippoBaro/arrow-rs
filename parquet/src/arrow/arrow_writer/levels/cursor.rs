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

//! A bounded, borrowed cursor over the canonical Parquet leaf stream.
//!
//! This module deliberately does not build a reusable program for an Arrow
//! hierarchy.  The Arrow tree is the program: a cursor follows one primitive
//! leaf through it, resolving run ends and dictionary keys as it goes.  Only a
//! record-aligned tile of levels and terminal value indices is retained.

use super::{FieldContract, is_leaf, leaf_types_compatible, normalized, program::LeafBatch};
use crate::column::value::{GroupedSelectionRef, ValueSelectionRef};
use crate::column::writer::LevelDataRef;
use crate::errors::{ParquetError, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_buffer::{ArrowNativeType, NullBuffer};
use arrow_schema::{DataType, Field};

/// Returns whether an actual batch layout needs recursive cursor lowering.
///
/// This intentionally examines the batch type, rather than the writer schema:
/// a writer schema containing REE may receive a dense compatible batch, which
/// remains on the dense hot path.
pub(crate) fn needs_cursor(data_type: &DataType) -> bool {
    match data_type {
        DataType::RunEndEncoded(_, _) => true,
        DataType::Struct(fields) => fields.iter().any(|f| needs_cursor(f.data_type())),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => needs_cursor(field.data_type()),
        DataType::Dictionary(_, value) => !is_leaf(value) || needs_cursor(value),
        _ => false,
    }
}

/// A plan for one primitive Parquet leaf of a top-level Arrow array.
///
/// The plan owns the input objects needed to make a cursor independent of the
/// record batch. `path` contains only branching struct-child ordinals; lists,
/// dictionaries, and REE wrappers have a single logical child.
#[derive(Debug, Clone)]
pub(crate) struct CursorLeafPlan {
    root: ArrayRef,
    field: Field,
    path: Box<[usize]>,
    terminal: ArrayRef,
    max_def_level: i16,
    max_rep_level: i16,
}

impl CursorLeafPlan {
    pub(crate) fn cursor(&self, target_slots: usize, target_rows: usize) -> LeafCursor<'_> {
        LeafCursor {
            plan: self,
            next_row: 0,
            target_slots: target_slots.max(1),
            target_rows: target_rows.max(1),
            tile: LeafTile::new(self.max_def_level, self.max_rep_level),
            probe: RepeatProbe::new(self.root.as_ref(), &self.path),
        }
    }
}

/// The loop-invariant part of bounding a run of identical top-level records.
///
/// Descending a struct child preserves the row index space, so the structural
/// facts of a node under a chain of structs still bound top-level rows. The
/// chain is resolved once per cursor: probing it per row costs two virtual
/// calls that are pure waste on shapes whose bound can never fire.
#[derive(Debug)]
struct RepeatProbe<'a> {
    /// Validity of the nullable structs above `node`, outermost first.
    parents: Vec<&'a NullBuffer>,
    /// The node whose own structure can bound a run of identical records.
    node: Option<&'a dyn Array>,
}

impl<'a> RepeatProbe<'a> {
    fn new(root: &'a dyn Array, path: &[usize]) -> Self {
        let mut parents = Vec::new();
        let mut current = root;
        let mut path_pos = 0;
        let node = loop {
            match current.data_type() {
                DataType::RunEndEncoded(_, _) | DataType::List(_) | DataType::LargeList(_) => {
                    break Some(current);
                }
                DataType::Struct(_) => {
                    let Some(&ordinal) = path.get(path_pos) else {
                        break None;
                    };
                    if let Some(nulls) = current.nulls() {
                        parents.push(nulls);
                    }
                    current = current.as_struct().column(ordinal).as_ref();
                    path_pos += 1;
                }
                _ => break None,
            }
        };
        Self { parents, node }
    }

    /// Return the bounded end of a run of identical top-level leaf records.
    fn repeat_end(&self, row: usize, limit: usize) -> Result<Option<usize>> {
        if self.parents.is_empty() {
            return match self.node {
                Some(node) => node_repeat_end(node, row, limit),
                None => Ok(None),
            };
        }
        if row + 1 >= limit {
            return Ok(None);
        }

        // Outermost first: below the first null ancestor nothing is emitted, so
        // deeper validity is not consulted. A validity change at `row + 1`
        // cannot yield a copy, so bail before paying for any run scan.
        let mut null_depth = None;
        for (depth, nulls) in self.parents.iter().enumerate() {
            let valid = nulls.is_valid(row);
            if valid != nulls.is_valid(row + 1) {
                return Ok(None);
            }
            if !valid {
                null_depth = Some(depth);
                break;
            }
        }

        // Each bound is scanned innermost first so the outer set-bit scans are
        // already clamped, keeping the bit tests linear in the rows skipped.
        let mut end = match null_depth {
            // A null struct emits one bare definition level and descends
            // nowhere, so its whole clear-bit run is one repeated record.
            Some(depth) => bit_run_end(self.parents[depth], row, limit, false),
            None => match self.node {
                Some(node) => match node_repeat_end(node, row, limit)? {
                    Some(end) => end,
                    None => return Ok(None),
                },
                None => return Ok(None),
            },
        };
        for nulls in &self.parents[..null_depth.unwrap_or(self.parents.len())] {
            end = bit_run_end(nulls, row, end, true);
        }
        Ok(Some(end))
    }
}

/// End of the run of `valid` bits starting at `row`, bounded by `limit`.
fn bit_run_end(nulls: &NullBuffer, row: usize, limit: usize, valid: bool) -> usize {
    let mut end = row + 1;
    while end < limit && nulls.is_valid(end) == valid {
        end += 1;
    }
    end
}

/// Collect one cursor plan per primitive leaf, in Parquet depth-first order.
///
/// This validates structural compatibility but deliberately does not scan for
/// nulls. Reachability depends on list offsets, run slices, and nullable
/// ancestors, so required-null validation happens while the cursor walks the
/// logical records.
pub(crate) fn collect_cursor_leaves(root: &ArrayRef, field: &Field) -> Result<Vec<CursorLeafPlan>> {
    let mut leaves = Vec::new();
    let mut path = Vec::new();
    collect_node(
        root,
        normalized(field),
        LevelContext::default(),
        &mut path,
        root,
        field,
        &mut leaves,
    )?;
    Ok(leaves)
}

/// One reusable, record-aligned portion of a leaf stream.
///
/// `target_slots` is a soft bound: a tile is stopped only between top-level
/// records, and therefore one record may take it beyond the target. Allocation
/// capacity is retained when the cursor advances to the next tile.
#[derive(Debug)]
pub(crate) struct LeafTile {
    slots: usize,
    def_levels: LevelBuffer,
    rep_levels: LevelBuffer,
    value_indices: Vec<usize>,
    value_ends: Vec<usize>,
    // Last physical run at each REE depth; retained across tile clears.
    ree_runs: Vec<usize>,
    ree_depth: usize,
}

impl LeafTile {
    fn new(max_def_level: i16, max_rep_level: i16) -> Self {
        Self {
            slots: 0,
            def_levels: LevelBuffer::new(max_def_level != 0),
            rep_levels: LevelBuffer::new(max_rep_level != 0),
            value_indices: Vec::new(),
            value_ends: Vec::new(),
            ree_runs: Vec::new(),
            ree_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.slots = 0;
        self.def_levels.clear();
        self.rep_levels.clear();
        self.value_indices.clear();
        self.value_ends.clear();
    }

    fn push_level(&mut self, def: i16, rep: i16) {
        self.slots += 1;
        self.def_levels.push(def);
        self.rep_levels.push(rep);
    }

    fn push_value(&mut self, def: i16, rep: i16, index: usize) {
        self.push_level(def, rep);
        self.push_group(index, 1);
    }

    fn push_group(&mut self, index: usize, len: usize) {
        let end = self.value_ends.last().copied().unwrap_or(0) + len;
        if self.value_indices.last() == Some(&index) {
            *self.value_ends.last_mut().unwrap() = end;
        } else {
            self.value_indices.push(index);
            self.value_ends.push(end);
        }
    }

    /// Repeat the record appended after `(slot_start, value_start)` without
    /// walking its Arrow hierarchy again.
    fn repeat_record(&mut self, slot_start: usize, value_start: usize, copies: usize) {
        if copies == 0 {
            return;
        }

        let record_slots = self.slots - slot_start;
        self.def_levels
            .repeat_range(slot_start, record_slots, copies);
        self.rep_levels
            .repeat_range(slot_start, record_slots, copies);
        self.slots += record_slots * copies;

        let value_end = self.value_ends.last().copied().unwrap_or(0);
        if value_start == value_end {
            return;
        }
        let first_group = self.value_ends.partition_point(|&end| end <= value_start);
        let group_end = self.value_indices.len();
        if first_group + 1 == group_end {
            self.push_group(
                self.value_indices[first_group],
                (value_end - value_start) * copies,
            );
            return;
        }

        // The source prefix remains immutable while a multi-group pattern is
        // appended; duplicated record boundaries are then coalesced.
        for _ in 0..copies {
            for group in first_group..group_end {
                let start = if group == 0 {
                    0
                } else {
                    self.value_ends[group - 1]
                };
                let len = self.value_ends[group].min(value_end) - start.max(value_start);
                if len != 0 {
                    let end = self.value_ends.last().copied().unwrap_or(0) + len;
                    self.value_indices.push(self.value_indices[group]);
                    self.value_ends.push(end);
                }
            }
        }
        if self.value_indices[first_group] == self.value_indices[group_end - 1] {
            self.coalesce_groups();
        }
    }

    fn coalesce_groups(&mut self) {
        let mut write = 0;
        let mut source_start = 0;
        for read in 0..self.value_indices.len() {
            let source_end = self.value_ends[read];
            let len = source_end - source_start;
            source_start = source_end;
            if write != 0 && self.value_indices[write - 1] == self.value_indices[read] {
                self.value_ends[write - 1] += len;
            } else {
                self.value_indices[write] = self.value_indices[read];
                let end = write
                    .checked_sub(1)
                    .map_or(0, |previous| self.value_ends[previous])
                    + len;
                self.value_ends[write] = end;
                write += 1;
            }
        }
        self.value_indices.truncate(write);
        self.value_ends.truncate(write);
    }

    pub(crate) fn batch<'a>(&'a self, plan: &'a CursorLeafPlan) -> LeafBatch<'a> {
        let values = if self.value_indices.is_empty() {
            ValueSelectionRef::Empty
        } else if self.value_indices.len() == self.value_ends.last().copied().unwrap_or(0) {
            ValueSelectionRef::Sparse(&self.value_indices)
        } else {
            ValueSelectionRef::Grouped(GroupedSelectionRef::new(
                &self.value_indices,
                &self.value_ends,
            ))
        };
        LeafBatch::new(
            plan.terminal.as_ref(),
            self.def_levels.as_ref(),
            self.rep_levels.as_ref(),
            values,
        )
    }
}

/// A pull cursor whose returned tile is reused by the next call.
pub(crate) struct LeafCursor<'a> {
    plan: &'a CursorLeafPlan,
    next_row: usize,
    target_slots: usize,
    target_rows: usize,
    tile: LeafTile,
    probe: RepeatProbe<'a>,
}

impl<'a> LeafCursor<'a> {
    pub(crate) fn next_tile(&mut self) -> Result<Option<&LeafTile>> {
        if self.next_row == self.plan.root.len() {
            return Ok(None);
        }

        let first_row = self.next_row;
        let rows_to_boundary = self.target_rows - first_row % self.target_rows;
        self.tile.clear();
        let contract = normalized(&self.plan.field);

        while self.next_row < self.plan.root.len()
            && self.tile.slots < self.target_slots
            && self.next_row - first_row < rows_to_boundary
        {
            let repeat_limit = self
                .next_row
                .saturating_add(self.target_slots.saturating_add(1))
                .min(first_row + rows_to_boundary)
                .min(self.plan.root.len());
            let run_end = self.probe.repeat_end(self.next_row, repeat_limit)?;
            let slot_start = self.tile.slots;
            let value_start = self.tile.value_ends.last().copied().unwrap_or(0);
            visit_node(
                self.plan.root.as_ref(),
                self.next_row,
                contract,
                LevelContext::default(),
                0,
                &self.plan.path,
                0,
                &mut self.tile,
            )?;
            self.next_row += 1;

            if let Some(run_end) = run_end {
                let record_slots = self.tile.slots - slot_start;
                let slot_rows = self
                    .target_slots
                    .saturating_sub(self.tile.slots)
                    .div_ceil(record_slots);
                let copies = (run_end - self.next_row)
                    .min(first_row + rows_to_boundary - self.next_row)
                    .min(slot_rows);
                self.tile.repeat_record(slot_start, value_start, copies);
                self.next_row += copies;
            }
        }

        debug_assert!(self.tile.slots != 0);
        debug_assert_eq!(
            self.tile.def_levels.len(),
            usize::from(self.plan.max_def_level != 0) * self.tile.slots
        );
        debug_assert_eq!(
            self.tile.rep_levels.len(),
            usize::from(self.plan.max_rep_level != 0) * self.tile.slots
        );
        Ok(Some(&self.tile))
    }
}

/// Return the bounded end of a run of identical records emitted by `array`.
///
/// A REE supplies this directly. A regular list can supply the same fact when
/// equal-width rows fall wholly within one run of its REE child: each row then
/// invokes the same physical child block the same number of times.
fn node_repeat_end(array: &dyn Array, row: usize, limit: usize) -> Result<Option<usize>> {
    match array.data_type() {
        DataType::RunEndEncoded(_, _) => {
            let (run_ends, base, _) = super::super::run_ends_of(array)?;
            let run = run_ends.run_of(base + row);
            Ok(Some(
                (run_ends.end_of(run) - base).min(array.len()).min(limit),
            ))
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            list_repeat_end(
                array,
                row,
                limit,
                list.value_offsets(),
                list.values().as_ref(),
            )
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            list_repeat_end(
                array,
                row,
                limit,
                list.value_offsets(),
                list.values().as_ref(),
            )
        }
        _ => Ok(None),
    }
}

fn list_repeat_end<O: ArrowNativeType>(
    list: &dyn Array,
    row: usize,
    limit: usize,
    offsets: &[O],
    child: &dyn Array,
) -> Result<Option<usize>> {
    if list.is_null(row) || !matches!(child.data_type(), DataType::RunEndEncoded(_, _)) {
        return Ok(None);
    }

    let start = offsets[row].as_usize();
    let end = offsets[row + 1].as_usize();
    if start == end {
        return Ok(None);
    }

    // Resolving the child run costs a search over the run ends, which only pays
    // off if the very next row can join the repeat. `limit <= root.len()` keeps
    // the `row + 2` offset and the validity probe in bounds.
    let width = end - start;
    if row + 1 >= limit || list.is_null(row + 1) || offsets[row + 2].as_usize() - end != width {
        return Ok(None);
    }

    let (run_ends, base, _) = super::super::run_ends_of(child)?;
    let run = run_ends.run_of(base + start);
    let run_end = (run_ends.end_of(run) - base).min(child.len());
    if end > run_end {
        return Ok(None);
    }

    let mut row_end = row + 1;
    while row_end < limit && !list.is_null(row_end) {
        let next_start = offsets[row_end].as_usize();
        let next_end = offsets[row_end + 1].as_usize();
        if next_end - next_start != width || next_end > run_end {
            break;
        }
        row_end += 1;
    }
    Ok(Some(row_end))
}

#[derive(Debug, Default, Clone, Copy)]
struct LevelContext {
    def: i16,
    rep: i16,
}

#[allow(clippy::too_many_arguments)]
fn collect_node(
    array: &ArrayRef,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    path: &mut Vec<usize>,
    root: &ArrayRef,
    root_field: &Field,
    leaves: &mut Vec<CursorLeafPlan>,
) -> Result<()> {
    match array.data_type() {
        DataType::RunEndEncoded(_, _) => {
            let (_, _, values) = super::super::run_ends_of(array.as_ref())?;
            return collect_node(values, contract, ctx, path, root, root_field, leaves);
        }
        DataType::Dictionary(_, value) if !is_leaf(value) => {
            return collect_node(
                array.as_any_dictionary().values(),
                contract,
                ctx,
                path,
                root,
                root_field,
                leaves,
            );
        }
        _ => {}
    }

    match array.data_type() {
        actual
            if is_leaf(actual)
                || matches!(actual, DataType::Dictionary(_, value) if is_leaf(value)) =>
        {
            let logical = match actual {
                DataType::Dictionary(_, value) => value.as_ref(),
                _ => actual,
            };
            if !leaf_types_compatible(contract.data_type, logical) {
                return Err(incompatible(contract, logical));
            }
            leaves.push(CursorLeafPlan {
                root: root.clone(),
                field: root_field.clone(),
                path: path.clone().into_boxed_slice(),
                terminal: array.clone(),
                max_def_level: ctx.def + contract.nullable as i16,
                max_rep_level: ctx.rep,
            });
            Ok(())
        }
        DataType::Struct(actual_fields) => {
            let DataType::Struct(expected_fields) = contract.data_type else {
                return Err(incompatible(contract, array.data_type()));
            };
            if actual_fields.len() != expected_fields.len() {
                return Err(arrow_err!(
                    "Incompatible struct field '{}': expected {} children, got {}",
                    contract.name,
                    expected_fields.len(),
                    actual_fields.len()
                ));
            }
            let structure = array.as_struct();
            let child_ctx = LevelContext {
                def: ctx.def + contract.nullable as i16,
                ..ctx
            };
            for (child_ordinal, (child, child_field)) in
                structure.columns().iter().zip(expected_fields).enumerate()
            {
                path.push(child_ordinal);
                collect_node(
                    child,
                    normalized(child_field),
                    child_ctx,
                    path,
                    root,
                    root_field,
                    leaves,
                )?;
                path.pop();
            }
            Ok(())
        }
        actual if actual.is_list() || matches!(actual, DataType::Map(_, _)) => {
            let (child, child_field) = match (actual, contract.data_type) {
                (DataType::List(_), DataType::List(field)) => {
                    (array.as_list::<i32>().values().clone(), field.as_ref())
                }
                (DataType::LargeList(_), DataType::LargeList(field)) => {
                    (array.as_list::<i64>().values().clone(), field.as_ref())
                }
                (DataType::FixedSizeList(_, actual), DataType::FixedSizeList(field, expected))
                    if actual == expected =>
                {
                    (array.as_fixed_size_list().values().clone(), field.as_ref())
                }
                (DataType::ListView(_), DataType::ListView(field)) => {
                    (array.as_list_view::<i32>().values().clone(), field.as_ref())
                }
                (DataType::LargeListView(_), DataType::LargeListView(field)) => {
                    (array.as_list_view::<i64>().values().clone(), field.as_ref())
                }
                (DataType::Map(_, actual), DataType::Map(field, expected))
                    if actual == expected =>
                {
                    (
                        std::sync::Arc::new(array.as_map().entries().clone()) as ArrayRef,
                        field.as_ref(),
                    )
                }
                _ => return Err(incompatible(contract, actual)),
            };
            collect_list_child(
                &child,
                child_field,
                contract,
                ctx,
                path,
                root,
                root_field,
                leaves,
            )
        }
        actual => Err(nyi_err!(format!(
            "Datatype {actual} is not supported by recursive leaf cursor"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_list_child(
    child: &ArrayRef,
    child_field: &Field,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    path: &mut Vec<usize>,
    root: &ArrayRef,
    root_field: &Field,
    leaves: &mut Vec<CursorLeafPlan>,
) -> Result<()> {
    collect_node(
        child,
        normalized(child_field),
        LevelContext {
            def: ctx.def + contract.nullable as i16 + 1,
            rep: ctx.rep + 1,
        },
        path,
        root,
        root_field,
        leaves,
    )
}

#[allow(clippy::too_many_arguments)]
fn visit_node(
    array: &dyn Array,
    index: usize,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    path: &[usize],
    path_pos: usize,
    out: &mut LeafTile,
) -> Result<()> {
    match array.data_type() {
        DataType::RunEndEncoded(_, _) => {
            let (run_ends, base, values) = super::super::run_ends_of(array)?;
            let position = base + index;
            let depth = out.ree_depth;
            let physical = match out.ree_runs.get(depth).copied() {
                Some(mut run) if run == 0 || position >= run_ends.end_of(run.saturating_sub(1)) => {
                    while run_ends.end_of(run) <= position {
                        run += 1;
                    }
                    run
                }
                _ => run_ends.run_of(position),
            };
            if let Some(run) = out.ree_runs.get_mut(depth) {
                *run = physical;
            } else {
                debug_assert_eq!(out.ree_runs.len(), depth);
                out.ree_runs.push(physical);
            }
            out.ree_depth += 1;
            let result = visit_node(
                values.as_ref(),
                physical,
                contract,
                ctx,
                rep,
                path,
                path_pos,
                out,
            );
            out.ree_depth -= 1;
            return result;
        }
        DataType::Dictionary(_, value) if is_leaf(value) => {
            let dictionary = array.as_any_dictionary();
            if dictionary.keys().is_null(index) {
                return emit_null(contract, ctx, rep, index, out);
            }
            let key = dictionary_key(dictionary.keys(), index)?;
            if matches!(value.as_ref(), DataType::Null) || dictionary.values().is_null(key) {
                return emit_null(contract, ctx, rep, index, out);
            }
            out.push_value(ctx.def + contract.nullable as i16, rep, index);
            return Ok(());
        }
        DataType::Dictionary(_, _) => {
            let dictionary = array.as_any_dictionary();
            if dictionary.keys().is_null(index) {
                return emit_null(contract, ctx, rep, index, out);
            }
            let key = dictionary_key(dictionary.keys(), index)?;
            return visit_node(
                dictionary.values().as_ref(),
                key,
                contract,
                ctx,
                rep,
                path,
                path_pos,
                out,
            );
        }
        _ => {}
    }

    match array.data_type() {
        DataType::Null => emit_null(contract, ctx, rep, index, out),
        actual if is_leaf(actual) => {
            if array.is_null(index) {
                emit_null(contract, ctx, rep, index, out)
            } else {
                out.push_value(ctx.def + contract.nullable as i16, rep, index);
                Ok(())
            }
        }
        DataType::Struct(_) => {
            if array.is_null(index) {
                return emit_null(contract, ctx, rep, index, out);
            }
            let DataType::Struct(fields) = contract.data_type else {
                return Err(incompatible(contract, array.data_type()));
            };
            let child_ordinal = path[path_pos];
            visit_node(
                array.as_struct().column(child_ordinal).as_ref(),
                index,
                normalized(&fields[child_ordinal]),
                LevelContext {
                    def: ctx.def + contract.nullable as i16,
                    ..ctx
                },
                rep,
                path,
                path_pos + 1,
                out,
            )
        }
        actual if actual.is_list() || matches!(actual, DataType::Map(_, _)) => {
            let (start, end, values, child): (usize, usize, &dyn Array, &Field) =
                match (actual, contract.data_type) {
                    (DataType::List(_), DataType::List(field)) => {
                        let list = array.as_list::<i32>();
                        let offsets = list.value_offsets();
                        (
                            offsets[index].as_usize(),
                            offsets[index + 1].as_usize(),
                            list.values().as_ref(),
                            field,
                        )
                    }
                    (DataType::LargeList(_), DataType::LargeList(field)) => {
                        let list = array.as_list::<i64>();
                        let offsets = list.value_offsets();
                        (
                            offsets[index].as_usize(),
                            offsets[index + 1].as_usize(),
                            list.values().as_ref(),
                            field,
                        )
                    }
                    (
                        DataType::FixedSizeList(_, actual),
                        DataType::FixedSizeList(field, expected),
                    ) if actual == expected => {
                        let list = array.as_fixed_size_list();
                        let start = list.value_offset(index) as usize;
                        (
                            start,
                            start + list.value_length() as usize,
                            list.values().as_ref(),
                            field,
                        )
                    }
                    (DataType::ListView(_), DataType::ListView(field)) => {
                        let list = array.as_list_view::<i32>();
                        let start = list.value_offset(index).as_usize();
                        (
                            start,
                            start + list.value_size(index).as_usize(),
                            list.values().as_ref(),
                            field,
                        )
                    }
                    (DataType::LargeListView(_), DataType::LargeListView(field)) => {
                        let list = array.as_list_view::<i64>();
                        let start = list.value_offset(index).as_usize();
                        (
                            start,
                            start + list.value_size(index).as_usize(),
                            list.values().as_ref(),
                            field,
                        )
                    }
                    (DataType::Map(_, actual), DataType::Map(field, expected))
                        if actual == expected =>
                    {
                        let map = array.as_map();
                        let offsets = map.value_offsets();
                        (
                            offsets[index] as usize,
                            offsets[index + 1] as usize,
                            map.entries(),
                            field,
                        )
                    }
                    _ => return Err(incompatible(contract, actual)),
                };
            visit_list(
                array, index, start, end, values, child, contract, ctx, rep, path, path_pos, out,
            )
        }
        actual => Err(nyi_err!(format!(
            "Datatype {actual} is not supported by recursive leaf cursor"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn visit_list(
    list: &dyn Array,
    row: usize,
    start: usize,
    end: usize,
    child: &dyn Array,
    child_field: &Field,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    path: &[usize],
    path_pos: usize,
    out: &mut LeafTile,
) -> Result<()> {
    if list.is_null(row) {
        return emit_null(contract, ctx, rep, row, out);
    }

    let list_def = ctx.def + contract.nullable as i16;
    if start == end {
        out.push_level(list_def, rep);
        return Ok(());
    }

    let child_ctx = LevelContext {
        def: list_def + 1,
        rep: ctx.rep + 1,
    };
    let child_contract = normalized(child_field);
    // A run-encoded child invokes the same physical block for every element of
    // a run, so walk it once per run and replay the emitted record. Elements are
    // then all emitted as interior repetitions and the record's first repetition
    // level is fixed once at the end, so the replayed prefix is verbatim.
    let child_runs = match child.data_type() {
        DataType::RunEndEncoded(_, _) => Some(super::super::run_ends_of(child)?),
        _ => None,
    };
    let depth = out.ree_depth;
    let row_slot = out.slots;
    // Single-element rows can never replay, so they keep emitting the record's
    // repetition level directly and leave a uniform buffer uniform.
    let patched = child_runs.is_some() && end - start > 1;
    let mut child_rep = if patched { child_ctx.rep } else { rep };
    let mut child_index = start;
    while child_index < end {
        let slot_start = out.slots;
        let value_start = out.value_ends.last().copied().unwrap_or(0);
        visit_node(
            child,
            child_index,
            child_contract,
            child_ctx,
            child_rep,
            path,
            path_pos,
            out,
        )?;
        child_rep = child_ctx.rep;
        child_index += 1;
        if let Some((run_ends, base, _)) = child_runs {
            let run_end = run_ends.end_of(out.ree_runs[depth]).saturating_sub(base);
            let copies = run_end.min(end) - child_index;
            out.repeat_record(slot_start, value_start, copies);
            child_index += copies;
        }
    }
    if patched {
        out.rep_levels.set(row_slot, rep);
    }
    Ok(())
}

fn emit_null(
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    index: usize,
    out: &mut LeafTile,
) -> Result<()> {
    if !contract.nullable {
        return Err(super::required_null(contract.name, index));
    }
    out.push_level(ctx.def, rep);
    Ok(())
}

fn dictionary_key(keys: &dyn Array, index: usize) -> Result<usize> {
    macro_rules! key {
        ($ty:ty) => {
            keys.as_primitive::<$ty>().value(index) as usize
        };
    }
    Ok(match keys.data_type() {
        DataType::Int8 => key!(Int8Type),
        DataType::Int16 => key!(Int16Type),
        DataType::Int32 => key!(Int32Type),
        DataType::Int64 => key!(Int64Type),
        DataType::UInt8 => key!(UInt8Type),
        DataType::UInt16 => key!(UInt16Type),
        DataType::UInt32 => key!(UInt32Type),
        DataType::UInt64 => key!(UInt64Type),
        other => {
            return Err(nyi_err!(format!("Unsupported dictionary key type {other}")));
        }
    })
}

fn incompatible(contract: FieldContract<'_>, actual: &DataType) -> ParquetError {
    ParquetError::ArrowError(format!(
        "Incompatible type. Field '{}' has type {}, array has type {}",
        contract.name, contract.data_type, actual
    ))
}

#[derive(Debug)]
struct LevelBuffer {
    enabled: bool,
    uniform: Option<i16>,
    len: usize,
    values: Vec<i16>,
}

impl LevelBuffer {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            uniform: None,
            len: 0,
            values: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.uniform = None;
        self.len = 0;
        self.values.clear();
    }

    fn push(&mut self, value: i16) {
        if !self.enabled {
            return;
        }
        match self.uniform {
            None if self.values.is_empty() && self.len == 0 => self.uniform = Some(value),
            Some(uniform) if uniform == value => {}
            Some(uniform) => {
                self.values.resize(self.len, uniform);
                self.values.push(value);
                self.uniform = None;
            }
            None => self.values.push(value),
        }
        self.len += 1;
    }

    /// Overwrite an already-pushed level, materialising the uniform
    /// representation only when the value actually differs.
    fn set(&mut self, index: usize, value: i16) {
        if !self.enabled {
            return;
        }
        match self.uniform {
            Some(uniform) if uniform == value => {}
            Some(uniform) => {
                self.values.resize(self.len, uniform);
                self.uniform = None;
                self.values[index] = value;
            }
            None => self.values[index] = value,
        }
    }

    fn repeat_range(&mut self, start: usize, len: usize, copies: usize) {
        if !self.enabled || copies == 0 {
            return;
        }
        if self.uniform.is_some() {
            self.len += len * copies;
            return;
        }
        let end = start + len;
        self.values.reserve(len * copies);
        for _ in 0..copies {
            self.values.extend_from_within(start..end);
        }
        self.len += len * copies;
    }

    fn len(&self) -> usize {
        if self.enabled { self.len } else { 0 }
    }

    fn as_ref(&self) -> LevelDataRef<'_> {
        if !self.enabled {
            LevelDataRef::Absent
        } else if let Some(value) = self.uniform {
            LevelDataRef::Uniform {
                value,
                count: self.len,
            }
        } else {
            LevelDataRef::Materialized(&self.values)
        }
    }
}
