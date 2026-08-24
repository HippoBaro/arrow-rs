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

//! A borrowed cursor over the canonical Parquet leaf stream.
//!
//! A cursor follows one primitive leaf through the Arrow tree. Direct paths are
//! traversed in ranges. Paths through run ends or non-leaf dictionary values
//! use indexed traversal, retaining only a record-aligned tile of levels and
//! terminal value indices.

use super::{
    FieldContract, LevelContext, LevelData, is_leaf, leaf_types_compatible, normalized,
    plan::{LeafBatch, ValueSelection},
};
use crate::column::value_selection::{GroupedSelectionRef, ValueSelectionRef};
use crate::column::writer::LevelDataRef;
use crate::errors::{ParquetError, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, GenericListArray, GenericListViewArray, OffsetSizeTrait};
use arrow_buffer::{ArrowNativeType, NullBuffer};
use arrow_schema::{DataType, Field};
use std::ops::Range;

/// A plan for one primitive Parquet leaf of a top-level Arrow array.
///
/// The plan owns the input objects needed to make a cursor independent of the
/// record batch. `path` classifies each node and contains the selected child
/// ordinal for branching structs.
#[derive(Debug, Clone)]
pub(crate) struct CursorLeafPlan {
    root: ArrayRef,
    field: Field,
    path: Box<[PathNode]>,
    terminal: ArrayRef,
    max_def_level: i16,
    max_rep_level: i16,
    indexed_traversal: bool,
}

impl CursorLeafPlan {
    pub(crate) fn cursor(&self, target_slots: usize, target_rows: usize) -> LeafCursor<'_> {
        LeafCursor {
            plan: self,
            next_row: 0,
            target_slots: target_slots.max(1),
            target_rows: target_rows.max(1),
            tile: LeafTile::new(
                self.max_def_level,
                self.max_rep_level,
                self.indexed_traversal,
            ),
            probe: RepeatProbe::new(self.root.as_ref(), &self.path),
        }
    }
}

/// The loop-invariant part of bounding a run of identical top-level records.
///
/// Descending a struct child preserves the row index space, so the structural
/// facts of a node under a chain of structs still bound top-level rows. The
/// chain is resolved when the cursor is created.
#[derive(Debug)]
struct RepeatProbe<'a> {
    /// Validity of the nullable structs above `node`, outermost first.
    parents: Vec<&'a NullBuffer>,
    /// The node whose own structure can bound a run of identical records.
    node: Option<&'a dyn Array>,
}

impl<'a> RepeatProbe<'a> {
    fn new(root: &'a dyn Array, path: &[PathNode]) -> Self {
        let mut parents = Vec::new();
        let mut current = root;
        let mut node = None;
        for &PathNode { kind, struct_child } in path {
            match kind {
                NodeKind::RunEndEncoded | NodeKind::List(ListKind::List | ListKind::LargeList) => {
                    node = Some(current);
                    break;
                }
                NodeKind::Struct => {
                    if let Some(nulls) = current.nulls() {
                        parents.push(nulls);
                    }
                    current = current.as_struct().column(struct_child).as_ref();
                }
                _ => break,
            }
        }
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

        // The node or null-ancestor run supplies the initial bound. Each
        // enclosing parent validity run then clamps it further.
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

#[derive(Debug, Clone, Copy)]
enum ListKind {
    List,
    LargeList,
    FixedSizeList,
    ListView,
    LargeListView,
    Map,
}

impl ListKind {
    fn try_new(actual: &DataType, contract: FieldContract<'_>) -> Result<Self> {
        Ok(match (actual, contract.data_type) {
            (DataType::List(_), DataType::List(_)) => Self::List,
            (DataType::LargeList(_), DataType::LargeList(_)) => Self::LargeList,
            (DataType::FixedSizeList(_, a), DataType::FixedSizeList(_, e)) if a == e => {
                Self::FixedSizeList
            }
            (DataType::ListView(_), DataType::ListView(_)) => Self::ListView,
            (DataType::LargeListView(_), DataType::LargeListView(_)) => Self::LargeListView,
            (DataType::Map(_, a), DataType::Map(_, e)) if a == e => Self::Map,
            _ => return Err(incompatible(contract, actual)),
        })
    }

    fn field<'a>(self, contract: FieldContract<'a>) -> &'a Field {
        match (self, contract.data_type) {
            (Self::List, DataType::List(field))
            | (Self::LargeList, DataType::LargeList(field))
            | (Self::FixedSizeList, DataType::FixedSizeList(field, _))
            | (Self::ListView, DataType::ListView(field))
            | (Self::LargeListView, DataType::LargeListView(field))
            | (Self::Map, DataType::Map(field, _)) => field,
            _ => unreachable!("list kind was validated during planning"),
        }
    }

    fn child_owned(self, array: &dyn Array) -> ArrayRef {
        match self {
            Self::List => array.as_list::<i32>().values().clone(),
            Self::LargeList => array.as_list::<i64>().values().clone(),
            Self::FixedSizeList => array.as_fixed_size_list().values().clone(),
            Self::ListView => array.as_list_view::<i32>().values().clone(),
            Self::LargeListView => array.as_list_view::<i64>().values().clone(),
            Self::Map => std::sync::Arc::new(array.as_map().entries().clone()),
        }
    }

    #[inline]
    fn row(self, array: &dyn Array, row: usize) -> (usize, usize, &dyn Array) {
        match self {
            Self::List => list_row(array.as_list::<i32>(), row),
            Self::LargeList => list_row(array.as_list::<i64>(), row),
            Self::FixedSizeList => {
                let array = array.as_fixed_size_list();
                let start = array.value_offset(row) as usize;
                (
                    start,
                    start + array.value_length() as usize,
                    array.values().as_ref(),
                )
            }
            Self::ListView => list_view_row(array.as_list_view::<i32>(), row),
            Self::LargeListView => list_view_row(array.as_list_view::<i64>(), row),
            Self::Map => {
                let array = array.as_map();
                let offsets = array.value_offsets();
                (
                    offsets[row] as usize,
                    offsets[row + 1] as usize,
                    array.entries(),
                )
            }
        }
    }
}

#[inline]
fn list_row<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    row: usize,
) -> (usize, usize, &dyn Array) {
    let offsets = array.value_offsets();
    (
        offsets[row].as_usize(),
        offsets[row + 1].as_usize(),
        array.values().as_ref(),
    )
}

#[inline]
fn list_view_row<O: OffsetSizeTrait>(
    array: &GenericListViewArray<O>,
    row: usize,
) -> (usize, usize, &dyn Array) {
    let start = array.value_offset(row).as_usize();
    (
        start,
        start + array.value_size(row).as_usize(),
        array.values().as_ref(),
    )
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
    RunEndEncoded,
    Dictionary,
    Null,
    Leaf,
    DictionaryLeaf,
    Struct,
    List(ListKind),
}

#[derive(Debug, Clone, Copy)]
struct PathNode {
    kind: NodeKind,
    struct_child: usize,
}

fn classify_node(array: &dyn Array, contract: FieldContract<'_>) -> Result<NodeKind> {
    Ok(match array.data_type() {
        DataType::RunEndEncoded(_, _) => NodeKind::RunEndEncoded,
        DataType::Dictionary(_, value) if is_leaf(value) => {
            if !leaf_types_compatible(contract.data_type, value) {
                return Err(incompatible(contract, value));
            }
            NodeKind::DictionaryLeaf
        }
        DataType::Dictionary(_, _) => NodeKind::Dictionary,
        actual if is_leaf(actual) => {
            if !leaf_types_compatible(contract.data_type, actual) {
                return Err(incompatible(contract, actual));
            }
            if matches!(actual, DataType::Null) {
                NodeKind::Null
            } else {
                NodeKind::Leaf
            }
        }
        DataType::Struct(actual) => {
            let DataType::Struct(expected) = contract.data_type else {
                return Err(incompatible(contract, array.data_type()));
            };
            if actual.len() != expected.len() {
                return Err(arrow_err!(
                    "Incompatible struct field '{}': expected {} children, got {}",
                    contract.name,
                    expected.len(),
                    actual.len()
                ));
            }
            NodeKind::Struct
        }
        actual if actual.is_list() || matches!(actual, DataType::Map(_, _)) => {
            NodeKind::List(ListKind::try_new(actual, contract)?)
        }
        actual => {
            return Err(nyi_err!(format!(
                "Datatype {actual} is not supported by recursive leaf cursor"
            )));
        }
    })
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
        false,
        &mut leaves,
    )?;
    Ok(leaves)
}

/// Compact level storage for indexed traversal, which appends one slot at a time.
#[derive(Debug)]
struct ScalarLevels {
    enabled: bool,
    uniform: Option<i16>,
    len: usize,
    values: Vec<i16>,
}

impl ScalarLevels {
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

    #[inline]
    fn push(&mut self, value: i16) {
        if !self.enabled {
            return;
        }
        match self.uniform {
            None if self.len == 0 => self.uniform = Some(value),
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

    fn set(&mut self, index: usize, value: i16) {
        if !self.enabled || self.uniform == Some(value) {
            return;
        }
        if let Some(uniform) = self.uniform.take() {
            self.values.resize(self.len, uniform);
        }
        self.values[index] = value;
    }

    #[inline]
    fn repeat_range(&mut self, start: usize, len: usize, copies: usize) {
        if !self.enabled || copies == 0 {
            return;
        }
        if self.uniform.is_none() {
            let end = start + len;
            self.values.reserve(len * copies);
            for _ in 0..copies {
                self.values.extend_from_within(start..end);
            }
        }
        self.len += len * copies;
    }

    fn len(&self) -> usize {
        usize::from(self.enabled) * self.len
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

/// One reusable portion of a leaf stream. Indexed traversal accumulates
/// terminal indices, while direct traversal can emit contiguous ranges.
/// Repeated paths are bounded on record boundaries, and the cursor reuses the
/// tile between calls.
#[derive(Debug)]
pub(crate) struct LeafTile {
    slots: usize,
    def_levels: ScalarLevels,
    rep_levels: ScalarLevels,
    direct: Option<Box<DirectTile>>,
    indexed_traversal: bool,
    value_indices: Vec<usize>,
    value_ends: Vec<usize>,
    // Last physical run at each REE depth; retained across tile clears.
    ree_runs: Vec<usize>,
    ree_depth: usize,
}

#[derive(Debug)]
struct DirectTile {
    def_levels: LevelData,
    rep_levels: LevelData,
    values: ValueSelection,
}

impl LeafTile {
    fn new(max_def_level: i16, max_rep_level: i16, indexed_traversal: bool) -> Self {
        Self {
            slots: 0,
            def_levels: ScalarLevels::new(indexed_traversal && max_def_level != 0),
            rep_levels: ScalarLevels::new(indexed_traversal && max_rep_level != 0),
            direct: (!indexed_traversal).then(|| {
                Box::new(DirectTile {
                    def_levels: LevelData::new(max_def_level != 0),
                    rep_levels: LevelData::new(max_rep_level != 0),
                    values: ValueSelection::Empty,
                })
            }),
            indexed_traversal,
            value_indices: Vec::new(),
            value_ends: Vec::new(),
            ree_runs: Vec::new(),
            ree_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.slots = 0;
        if self.indexed_traversal {
            self.def_levels.clear();
            self.rep_levels.clear();
        } else {
            let direct = self.direct.as_mut().unwrap();
            direct.def_levels.clear();
            direct.rep_levels.clear();
            direct.values.clear();
        }
        self.value_indices.clear();
        self.value_ends.clear();
    }

    fn push_level(&mut self, def: i16, rep: i16) {
        debug_assert!(self.indexed_traversal);
        self.slots += 1;
        self.def_levels.push(def);
        self.rep_levels.push(rep);
    }

    fn push_level_run(&mut self, def: i16, rep: i16, count: usize) {
        debug_assert!(!self.indexed_traversal);
        self.slots += count;
        let direct = self.direct.as_mut().unwrap();
        direct.def_levels.append_run(def, count);
        direct.rep_levels.append_run(rep, count);
    }

    fn push_value_range(&mut self, def: i16, rep: i16, range: std::ops::Range<usize>) {
        debug_assert!(!self.indexed_traversal);
        let len = range.len();
        self.push_level_run(def, rep, len);
        self.direct.as_mut().unwrap().values.append_range(range);
    }

    fn push_value(&mut self, def: i16, rep: i16, index: usize) {
        debug_assert!(self.indexed_traversal);
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

    /// Repeat the leaf output appended since the slot and value checkpoints
    /// without walking its Arrow hierarchy again.
    fn repeat_since(&mut self, slot_checkpoint: usize, value_checkpoint: usize, copies: usize) {
        if copies == 0 {
            return;
        }

        let appended_slots = self.slots - slot_checkpoint;
        self.def_levels
            .repeat_range(slot_checkpoint, appended_slots, copies);
        self.rep_levels
            .repeat_range(slot_checkpoint, appended_slots, copies);
        self.slots += appended_slots * copies;

        debug_assert!(self.indexed_traversal);
        let value_end = self.value_ends.last().copied().unwrap_or(0);
        if value_checkpoint == value_end {
            return;
        }
        let first_group = self
            .value_ends
            .partition_point(|&end| end <= value_checkpoint);
        let group_end = self.value_indices.len();
        if first_group + 1 == group_end {
            self.push_group(
                self.value_indices[first_group],
                (value_end - value_checkpoint) * copies,
            );
            return;
        }

        // The source prefix remains immutable while a multi-group pattern is
        // appended; adjacent duplicate groups are then coalesced.
        for _ in 0..copies {
            for group in first_group..group_end {
                let start = if group == 0 {
                    0
                } else {
                    self.value_ends[group - 1]
                };
                let len = self.value_ends[group].min(value_end) - start.max(value_checkpoint);
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
        let (def_levels, rep_levels) = if self.indexed_traversal {
            (self.def_levels.as_ref(), self.rep_levels.as_ref())
        } else {
            let direct = self.direct.as_ref().unwrap();
            (direct.def_levels.as_ref(), direct.rep_levels.as_ref())
        };
        let values = if !self.indexed_traversal {
            self.direct.as_ref().unwrap().values.as_ref()
        } else if self.value_indices.is_empty() {
            ValueSelectionRef::Empty
        } else if self.value_indices.len() == self.value_ends.last().copied().unwrap_or(0) {
            ValueSelectionRef::Sparse(&self.value_indices)
        } else {
            ValueSelectionRef::Grouped(GroupedSelectionRef::new(
                &self.value_indices,
                &self.value_ends,
            ))
        };
        LeafBatch::new(plan.terminal.as_ref(), def_levels, rep_levels, values)
    }

    fn def_levels_len(&self) -> usize {
        if self.indexed_traversal {
            self.def_levels.len()
        } else {
            self.direct.as_ref().unwrap().def_levels.len()
        }
    }

    fn rep_levels_len(&self) -> usize {
        if self.indexed_traversal {
            self.rep_levels.len()
        } else {
            self.direct.as_ref().unwrap().rep_levels.len()
        }
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

        if !self.plan.indexed_traversal {
            let end = if self.plan.max_rep_level == 0 {
                self.plan.root.len()
            } else {
                first_row
                    .saturating_add(rows_to_boundary)
                    .min(self.plan.root.len())
            };
            visit_range(
                self.plan.root.as_ref(),
                first_row..end,
                contract,
                LevelContext::default(),
                0,
                &self.plan.path,
                &mut self.tile,
            )?;
            self.next_row = end;
            debug_assert!(self.tile.slots != 0);
            return Ok(Some(&self.tile));
        }

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
            let slot_checkpoint = self.tile.slots;
            let value_checkpoint = self.tile.value_ends.last().copied().unwrap_or(0);
            visit_node(
                self.plan.root.as_ref(),
                self.next_row,
                contract,
                LevelContext::default(),
                0,
                &self.plan.path,
                &mut self.tile,
            )?;
            self.next_row += 1;

            if let Some(run_end) = run_end {
                let appended_slots = self.tile.slots - slot_checkpoint;
                let rows_within_slot_limit = self
                    .target_slots
                    .saturating_sub(self.tile.slots)
                    .div_ceil(appended_slots);
                let copies = (run_end - self.next_row)
                    .min(first_row + rows_to_boundary - self.next_row)
                    .min(rows_within_slot_limit);
                self.tile
                    .repeat_since(slot_checkpoint, value_checkpoint, copies);
                self.next_row += copies;
            }
        }

        debug_assert!(self.tile.slots != 0);
        debug_assert_eq!(
            self.tile.def_levels_len(),
            usize::from(self.plan.max_def_level != 0) * self.tile.slots
        );
        debug_assert_eq!(
            self.tile.rep_levels_len(),
            usize::from(self.plan.max_rep_level != 0) * self.tile.slots
        );
        Ok(Some(&self.tile))
    }
}

fn visit_range(
    array: &dyn Array,
    range: Range<usize>,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    path: &[PathNode],
    out: &mut LeafTile,
) -> Result<()> {
    debug_assert!(!out.indexed_traversal);
    if range.is_empty() {
        return Ok(());
    }

    let (&PathNode { kind, struct_child }, child_path) = path.split_first().unwrap();
    match kind {
        NodeKind::Null | NodeKind::Leaf | NodeKind::DictionaryLeaf => {
            visit_leaf_range(array, range, contract, ctx, rep, out)
        }
        NodeKind::Struct => {
            let DataType::Struct(fields) = contract.data_type else {
                unreachable!("struct contract was validated during planning")
            };
            let child = array.as_struct().column(struct_child).as_ref();
            let child_contract = normalized(&fields[struct_child]);
            let child_ctx = LevelContext {
                def_level: ctx.def_level + contract.nullable as i16,
                ..ctx
            };
            let Some(nulls) = array
                .logical_nulls()
                .filter(|nulls| nulls.null_count() != 0)
            else {
                return visit_range(
                    child,
                    range,
                    child_contract,
                    child_ctx,
                    rep,
                    child_path,
                    out,
                );
            };

            if !contract.nullable {
                if let Some(index) = range.clone().find(|&index| nulls.is_null(index)) {
                    return Err(super::required_null(contract.name, index));
                }
                return visit_range(
                    child,
                    range,
                    child_contract,
                    child_ctx,
                    rep,
                    child_path,
                    out,
                );
            }

            let range_nulls = nulls.slice(range.start, range.len());
            if range_nulls.null_count() == range.len() {
                out.push_level_run(ctx.def_level, rep, range.len());
                return Ok(());
            }
            let mut position = 0;
            for (start, end) in range_nulls.valid_slices() {
                out.push_level_run(ctx.def_level, rep, start - position);
                visit_range(
                    child,
                    range.start + start..range.start + end,
                    child_contract,
                    child_ctx,
                    rep,
                    child_path,
                    out,
                )?;
                position = end;
            }
            out.push_level_run(ctx.def_level, rep, range.len() - position);
            Ok(())
        }
        NodeKind::List(kind) => {
            visit_list_range(kind, array, range, contract, ctx, rep, child_path, out)
        }
        NodeKind::RunEndEncoded | NodeKind::Dictionary => {
            unreachable!("indexed node reached direct leaf traversal")
        }
    }
}

fn visit_leaf_range(
    array: &dyn Array,
    range: Range<usize>,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    out: &mut LeafTile,
) -> Result<()> {
    let len = range.len();
    let def = ctx.def_level + contract.nullable as i16;
    let Some(nulls) = array
        .logical_nulls()
        .filter(|nulls| nulls.null_count() != 0)
    else {
        out.push_value_range(def, rep, range);
        return Ok(());
    };

    if !contract.nullable {
        if let Some(index) = range.clone().find(|&index| nulls.is_null(index)) {
            return Err(super::required_null(contract.name, index));
        }
        out.push_value_range(def, rep, range);
        return Ok(());
    }

    let range_nulls = nulls.slice(range.start, len);
    let null_count = range_nulls.null_count();
    if null_count == 0 {
        out.push_value_range(def, rep, range);
        return Ok(());
    }
    if null_count == len {
        out.push_level_run(ctx.def_level, rep, len);
        return Ok(());
    }

    out.slots += len;
    let direct = out.direct.as_mut().unwrap();
    direct.rep_levels.append_run(rep, len);
    if len >= super::BULK_FILL_MIN_LEN && nulls.null_count() * 2 >= nulls.len() {
        let mut position = 0;
        for (start, end) in range_nulls.valid_slices() {
            direct
                .def_levels
                .append_run(ctx.def_level, start - position);
            direct.def_levels.append_run(def, end - start);
            direct
                .values
                .append_range(range.start + start..range.start + end);
            position = end;
        }
        direct.def_levels.append_run(ctx.def_level, len - position);
    } else {
        if direct.def_levels.len().saturating_add(len) <= super::COMPACT_LEVEL_PROBE_MAX_LEN
            && levels_have_compact_runs(&range_nulls)
        {
            let mut position = 0;
            for (start, end) in range_nulls.valid_slices() {
                direct
                    .def_levels
                    .append_run(ctx.def_level, start - position);
                direct.def_levels.append_run(def, end - start);
                position = end;
            }
            direct.def_levels.append_run(ctx.def_level, len - position);
        } else {
            let bits = nulls.inner();
            direct
                .def_levels
                .extend_from_iter(range.clone().map(|index| {
                    // SAFETY: `range` is a valid slice of `array` and therefore of
                    // its logical null buffer.
                    let valid = unsafe { bits.value_unchecked(index) };
                    def - (!valid as i16)
                }));
        }
        direct
            .values
            .append_sparse(range_nulls, range.start, len - null_count);
    }
    Ok(())
}

fn levels_have_compact_runs(nulls: &NullBuffer) -> bool {
    let len = nulls.len().min(super::plan::LEVEL_RUN_PROBE_SIZE);
    if len < super::plan::MIN_AVERAGE_LEVEL_RUN_LENGTH {
        return false;
    }
    let mut runs = 1;
    let mut previous = nulls.is_valid(0);
    for index in 1..len {
        let valid = nulls.is_valid(index);
        runs += usize::from(valid != previous);
        if runs * super::plan::MIN_AVERAGE_LEVEL_RUN_LENGTH > len {
            return false;
        }
        previous = valid;
    }
    true
}

#[expect(clippy::too_many_arguments)]
fn visit_list_range(
    kind: ListKind,
    array: &dyn Array,
    range: Range<usize>,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    path: &[PathNode],
    out: &mut LeafTile,
) -> Result<()> {
    let child_field = kind.field(contract);
    macro_rules! visit {
        ($child:expr, $bounds:expr) => {
            visit_list_rows(
                array,
                range,
                $child,
                child_field,
                contract,
                ctx,
                rep,
                path,
                out,
                $bounds,
            )
        };
    }
    match kind {
        ListKind::List => {
            let list = array.as_list::<i32>();
            let offsets = list.value_offsets();
            visit!(list.values().as_ref(), |row| (
                offsets[row].as_usize(),
                offsets[row + 1].as_usize()
            ))
        }
        ListKind::LargeList => {
            let list = array.as_list::<i64>();
            let offsets = list.value_offsets();
            visit!(list.values().as_ref(), |row| (
                offsets[row].as_usize(),
                offsets[row + 1].as_usize()
            ))
        }
        ListKind::FixedSizeList => {
            let list = array.as_fixed_size_list();
            let width = list.value_length() as usize;
            visit!(list.values().as_ref(), |row| {
                let start = list.value_offset(row) as usize;
                (start, start + width)
            })
        }
        ListKind::ListView => {
            let list = array.as_list_view::<i32>();
            visit!(list.values().as_ref(), |row| {
                let start = list.value_offset(row).as_usize();
                (start, start + list.value_size(row).as_usize())
            })
        }
        ListKind::LargeListView => {
            let list = array.as_list_view::<i64>();
            visit!(list.values().as_ref(), |row| {
                let start = list.value_offset(row).as_usize();
                (start, start + list.value_size(row).as_usize())
            })
        }
        ListKind::Map => {
            let map = array.as_map();
            let offsets = map.value_offsets();
            visit_list_rows(
                array,
                range,
                map.entries(),
                child_field,
                contract,
                ctx,
                rep,
                path,
                out,
                |row| (offsets[row] as usize, offsets[row + 1] as usize),
            )
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn visit_list_rows(
    list: &dyn Array,
    range: Range<usize>,
    child: &dyn Array,
    child_field: &Field,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    path: &[PathNode],
    out: &mut LeafTile,
    bounds: impl Fn(usize) -> (usize, usize) + Copy,
) -> Result<()> {
    let list_def = ctx.def_level + contract.nullable as i16;
    let child_ctx = LevelContext {
        def_level: list_def + 1,
        rep_level: ctx.rep_level + 1,
    };
    let child_contract = normalized(child_field);
    let flat_child = path_has_no_list(path);
    let mut pending_rows: Option<Range<usize>> = None;
    let mut pending_end = 0;

    let mut flush = |rows: Range<usize>, out: &mut LeafTile| -> Result<()> {
        let values_start = bounds(rows.start).0;
        let values_end = bounds(rows.end - 1).1;
        let slot_start = out.slots;
        visit_range(
            child,
            values_start..values_end,
            child_contract,
            child_ctx,
            child_ctx.rep_level,
            path,
            out,
        )?;
        patch_list_starts(
            out,
            slot_start,
            values_start,
            rows,
            child_ctx.rep_level,
            rep,
            flat_child,
            bounds,
        );
        Ok(())
    };

    let nulls = list.logical_nulls().filter(|nulls| nulls.null_count() != 0);
    if let Some(nulls) = nulls.as_ref() {
        if !contract.nullable {
            if let Some(index) = range.clone().find(|&index| nulls.is_null(index)) {
                return Err(super::required_null(contract.name, index));
            }
            visit_valid_list_rows(
                range,
                list_def,
                rep,
                out,
                bounds,
                &mut pending_rows,
                &mut pending_end,
                &mut flush,
            )?;
        } else {
            let range_nulls = nulls.slice(range.start, range.len());
            let mut position = 0;
            for (start, end) in range_nulls.valid_slices() {
                if let Some(rows) = pending_rows.take() {
                    flush(rows, out)?;
                }
                out.push_level_run(ctx.def_level, rep, start - position);
                visit_valid_list_rows(
                    range.start + start..range.start + end,
                    list_def,
                    rep,
                    out,
                    bounds,
                    &mut pending_rows,
                    &mut pending_end,
                    &mut flush,
                )?;
                position = end;
            }
            if let Some(rows) = pending_rows.take() {
                flush(rows, out)?;
            }
            out.push_level_run(ctx.def_level, rep, range.len() - position);
        }
    } else {
        visit_valid_list_rows(
            range,
            list_def,
            rep,
            out,
            bounds,
            &mut pending_rows,
            &mut pending_end,
            &mut flush,
        )?;
    }
    if let Some(rows) = pending_rows {
        flush(rows, out)?;
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn visit_valid_list_rows(
    range: Range<usize>,
    list_def: i16,
    rep: i16,
    out: &mut LeafTile,
    bounds: impl Fn(usize) -> (usize, usize) + Copy,
    pending_rows: &mut Option<Range<usize>>,
    pending_end: &mut usize,
    flush: &mut impl FnMut(Range<usize>, &mut LeafTile) -> Result<()>,
) -> Result<()> {
    for row in range {
        let (start, end) = bounds(row);
        if start == end {
            if let Some(rows) = pending_rows.take() {
                flush(rows, out)?;
            }
            out.push_level_run(list_def, rep, 1);
        } else if pending_rows.is_some() && *pending_end == start {
            pending_rows.as_mut().unwrap().end = row + 1;
            *pending_end = end;
        } else {
            if let Some(rows) = pending_rows.take() {
                flush(rows, out)?;
            }
            *pending_rows = Some(row..row + 1);
            *pending_end = end;
        }
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn patch_list_starts(
    out: &mut LeafTile,
    slot_start: usize,
    values_start: usize,
    rows: Range<usize>,
    child_rep: i16,
    rep: i16,
    flat_child: bool,
    bounds: impl Fn(usize) -> (usize, usize),
) {
    if rep == child_rep {
        return;
    }
    let levels = out
        .direct
        .as_mut()
        .unwrap()
        .rep_levels
        .materialize_mut()
        .unwrap();
    if flat_child {
        for row in rows {
            levels[slot_start + bounds(row).0 - values_start] = rep;
        }
        return;
    }
    let mut slot = levels.len();
    for row in rows.rev() {
        let (start, end) = bounds(row);
        let mut remaining = end - start;
        while remaining != 0 {
            slot -= 1;
            if levels[slot] <= child_rep {
                remaining -= 1;
                if remaining == 0 {
                    levels[slot] = rep;
                }
            }
        }
    }
    debug_assert_eq!(slot, slot_start);
}

fn path_has_no_list(path: &[PathNode]) -> bool {
    path.iter().all(|node| {
        matches!(
            node.kind,
            NodeKind::Struct | NodeKind::Null | NodeKind::Leaf | NodeKind::DictionaryLeaf
        )
    })
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

    // Require a second candidate row before resolving the child run.
    // `limit <= list.len()` keeps the `row + 2` offset and validity probe in
    // bounds.
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

#[expect(clippy::too_many_arguments)]
fn collect_node(
    array: &ArrayRef,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    path: &mut Vec<PathNode>,
    root: &ArrayRef,
    root_field: &Field,
    indexed_traversal: bool,
    leaves: &mut Vec<CursorLeafPlan>,
) -> Result<()> {
    let kind = classify_node(array.as_ref(), contract)?;
    path.push(PathNode {
        kind,
        struct_child: 0,
    });
    let result = (|| match kind {
        NodeKind::RunEndEncoded => {
            let (_, _, values) = super::super::run_ends_of(array.as_ref())?;
            collect_node(values, contract, ctx, path, root, root_field, true, leaves)
        }
        NodeKind::Dictionary => collect_node(
            array.as_any_dictionary().values(),
            contract,
            ctx,
            path,
            root,
            root_field,
            true,
            leaves,
        ),
        NodeKind::Null | NodeKind::Leaf | NodeKind::DictionaryLeaf => {
            leaves.push(CursorLeafPlan {
                root: root.clone(),
                field: root_field.clone(),
                path: path.clone().into_boxed_slice(),
                terminal: array.clone(),
                max_def_level: ctx.def_level + contract.nullable as i16,
                max_rep_level: ctx.rep_level,
                indexed_traversal,
            });
            Ok(())
        }
        NodeKind::Struct => {
            let DataType::Struct(fields) = contract.data_type else {
                unreachable!("struct contract was validated during planning")
            };
            let structure = array.as_struct();
            let child_ctx = LevelContext {
                def_level: ctx.def_level + contract.nullable as i16,
                ..ctx
            };
            for (child_ordinal, (child, child_field)) in
                structure.columns().iter().zip(fields).enumerate()
            {
                path.last_mut().unwrap().struct_child = child_ordinal;
                collect_node(
                    child,
                    normalized(child_field),
                    child_ctx,
                    path,
                    root,
                    root_field,
                    indexed_traversal,
                    leaves,
                )?;
            }
            Ok(())
        }
        NodeKind::List(kind) => {
            let child = kind.child_owned(array.as_ref());
            collect_list_child(
                &child,
                kind.field(contract),
                contract,
                ctx,
                path,
                root,
                root_field,
                indexed_traversal,
                leaves,
            )
        }
    })();
    path.pop();
    result
}

#[expect(clippy::too_many_arguments)]
fn collect_list_child(
    child: &ArrayRef,
    child_field: &Field,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    path: &mut Vec<PathNode>,
    root: &ArrayRef,
    root_field: &Field,
    indexed_traversal: bool,
    leaves: &mut Vec<CursorLeafPlan>,
) -> Result<()> {
    collect_node(
        child,
        normalized(child_field),
        LevelContext {
            def_level: ctx.def_level + contract.nullable as i16 + 1,
            rep_level: ctx.rep_level + 1,
        },
        path,
        root,
        root_field,
        indexed_traversal,
        leaves,
    )
}

fn visit_node(
    array: &dyn Array,
    index: usize,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    rep: i16,
    path: &[PathNode],
    out: &mut LeafTile,
) -> Result<()> {
    let (&PathNode { kind, struct_child }, child_path) = path.split_first().unwrap();
    match kind {
        NodeKind::RunEndEncoded => {
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
                child_path,
                out,
            );
            out.ree_depth -= 1;
            result
        }
        NodeKind::DictionaryLeaf => {
            let dictionary = array.as_any_dictionary();
            if dictionary.keys().is_null(index) {
                return emit_null(contract, ctx, rep, index, out);
            }
            let key = dictionary_key(dictionary.keys(), index)?;
            if matches!(dictionary.values().data_type(), DataType::Null)
                || dictionary.values().is_null(key)
            {
                return emit_null(contract, ctx, rep, index, out);
            }
            out.push_value(ctx.def_level + contract.nullable as i16, rep, index);
            Ok(())
        }
        NodeKind::Dictionary => {
            let dictionary = array.as_any_dictionary();
            if dictionary.keys().is_null(index) {
                return emit_null(contract, ctx, rep, index, out);
            }
            let key = dictionary_key(dictionary.keys(), index)?;
            visit_node(
                dictionary.values().as_ref(),
                key,
                contract,
                ctx,
                rep,
                child_path,
                out,
            )
        }
        NodeKind::Null => emit_null(contract, ctx, rep, index, out),
        NodeKind::Leaf => {
            if array.is_null(index) {
                emit_null(contract, ctx, rep, index, out)
            } else {
                out.push_value(ctx.def_level + contract.nullable as i16, rep, index);
                Ok(())
            }
        }
        NodeKind::Struct => {
            if array.is_null(index) {
                return emit_null(contract, ctx, rep, index, out);
            }
            let DataType::Struct(fields) = contract.data_type else {
                unreachable!("struct contract was validated during planning")
            };
            visit_node(
                array.as_struct().column(struct_child).as_ref(),
                index,
                normalized(&fields[struct_child]),
                LevelContext {
                    def_level: ctx.def_level + contract.nullable as i16,
                    ..ctx
                },
                rep,
                child_path,
                out,
            )
        }
        NodeKind::List(kind) => {
            let (start, end, child) = kind.row(array, index);
            visit_list(
                array,
                index,
                start,
                end,
                child,
                kind.field(contract),
                contract,
                ctx,
                rep,
                child_path,
                out,
            )
        }
    }
}

#[expect(clippy::too_many_arguments)]
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
    path: &[PathNode],
    out: &mut LeafTile,
) -> Result<()> {
    if list.is_null(row) {
        return emit_null(contract, ctx, rep, row, out);
    }

    let list_def = ctx.def_level + contract.nullable as i16;
    if start == end {
        out.push_level(list_def, rep);
        return Ok(());
    }

    let child_ctx = LevelContext {
        def_level: list_def + 1,
        rep_level: ctx.rep_level + 1,
    };
    let child_contract = normalized(child_field);
    // A run-encoded child invokes the same physical block for every element of
    // a run, so walk it once per run and repeat the emitted leaf segment. The
    // elements are emitted as interior repetitions, and the list row's first
    // repetition level is fixed once at the end.
    let child_runs = match child.data_type() {
        DataType::RunEndEncoded(_, _) => Some(super::super::run_ends_of(child)?),
        _ => None,
    };
    let depth = out.ree_depth;
    let row_slot = out.slots;
    // Single-element rows can never repeat, so they keep emitting the row's
    // repetition level directly and leave a uniform buffer uniform.
    let patch_first_rep_level = child_runs.is_some() && end - start > 1;
    let mut child_rep = if patch_first_rep_level {
        child_ctx.rep_level
    } else {
        rep
    };
    let mut child_index = start;
    while child_index < end {
        let slot_checkpoint = out.slots;
        let value_checkpoint = out.value_ends.last().copied().unwrap_or(0);
        visit_node(
            child,
            child_index,
            child_contract,
            child_ctx,
            child_rep,
            path,
            out,
        )?;
        child_rep = child_ctx.rep_level;
        child_index += 1;
        if let Some((run_ends, base, _)) = child_runs {
            let run_end = run_ends.end_of(out.ree_runs[depth]).saturating_sub(base);
            let copies = run_end.min(end) - child_index;
            out.repeat_since(slot_checkpoint, value_checkpoint, copies);
            child_index += copies;
        }
    }
    if patch_first_rep_level {
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
    out.push_level(ctx.def_level, rep);
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;

    #[test]
    fn repeated_multi_group_patterns_coalesce_boundaries() {
        let mut tile = LeafTile::new(0, 0, true);
        tile.push_group(1, 2);
        tile.push_group(2, 1);
        tile.push_group(1, 2);

        tile.repeat_since(0, 0, 2);

        assert_eq!(tile.value_indices, [1, 2, 1, 2, 1, 2, 1]);
        assert_eq!(tile.value_ends, [2, 3, 7, 8, 12, 13, 15]);
    }

    #[test]
    fn scalar_levels_guard_and_materialization_paths() {
        let mut disabled = ScalarLevels::new(false);
        disabled.set(0, 1);
        disabled.repeat_range(0, 0, 1);
        assert_eq!(disabled.as_ref(), LevelDataRef::Absent);

        let mut levels = ScalarLevels::new(true);
        levels.push(1);
        levels.push(1);
        levels.set(0, 1);
        levels.repeat_range(0, 2, 0);
        assert_eq!(
            levels.as_ref(),
            LevelDataRef::Uniform { value: 1, count: 2 }
        );

        levels.set(1, 2);
        levels.repeat_range(0, 2, 2);
        assert_eq!(
            levels.as_ref(),
            LevelDataRef::Materialized(&[1, 2, 1, 2, 1, 2])
        );
    }

    #[test]
    fn direct_leaf_empty_required_and_bulk_null_paths() {
        let field = Field::new("a", DataType::Int32, false);
        let array = Int32Array::from(vec![Some(1), None]);
        let mut tile = LeafTile::new(0, 0, false);
        let path = [PathNode {
            kind: NodeKind::Leaf,
            struct_child: 0,
        }];
        visit_range(
            &array,
            0..0,
            normalized(&field),
            LevelContext::default(),
            0,
            &path,
            &mut tile,
        )
        .unwrap();
        assert_eq!(tile.slots, 0);
        assert!(
            visit_range(
                &array,
                0..2,
                normalized(&field),
                LevelContext::default(),
                0,
                &path,
                &mut tile,
            )
            .is_err()
        );

        let field = Field::new("a", DataType::Int32, true);
        let array = Int32Array::from(
            (0..80)
                .map(|index| (index % 4 == 0).then_some(index))
                .collect::<Vec<_>>(),
        );
        let mut tile = LeafTile::new(1, 0, false);
        visit_range(
            &array,
            0..array.len(),
            normalized(&field),
            LevelContext::default(),
            0,
            &path,
            &mut tile,
        )
        .unwrap();
        assert_eq!(tile.slots, 80);
        assert_eq!(tile.direct.unwrap().values.as_ref().len(), 20);

        let primitive = Int32Array::from(vec![1, 2]);
        assert_eq!(
            RepeatProbe::new(&primitive, &[]).repeat_end(0, 2).unwrap(),
            None
        );
    }
}
