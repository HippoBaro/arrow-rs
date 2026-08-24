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
use crate::errors::{ParquetError, Result};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
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
}

impl CursorLeafPlan {
    pub(crate) fn cursor(&self, target_rows: usize) -> LeafCursor<'_> {
        LeafCursor {
            plan: self,
            next_row: 0,
            target_rows: target_rows.max(1),
            tile: LeafTile::new(self.max_def_level, self.max_rep_level),
        }
    }
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
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
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
        DataType::Dictionary(_, value) if is_leaf(value) => {
            if !leaf_types_compatible(contract.data_type, value) {
                return Err(incompatible(contract, value));
            }
            NodeKind::DictionaryLeaf
        }
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
        &mut leaves,
    )?;
    Ok(leaves)
}

/// One reusable portion of a leaf stream. Indexed traversal accumulates
/// terminal indices, while direct traversal can emit contiguous ranges.
/// Repeated paths are bounded on record boundaries, and the cursor reuses the
/// tile between calls.
#[derive(Debug)]
pub(crate) struct LeafTile {
    slots: usize,
    direct: DirectTile,
}

#[derive(Debug)]
struct DirectTile {
    def_levels: LevelData,
    rep_levels: LevelData,
    values: ValueSelection,
}

impl LeafTile {
    fn new(max_def_level: i16, max_rep_level: i16) -> Self {
        Self {
            slots: 0,
            direct: DirectTile {
                def_levels: LevelData::new(max_def_level != 0),
                rep_levels: LevelData::new(max_rep_level != 0),
                values: ValueSelection::Empty,
            },
        }
    }

    fn clear(&mut self) {
        self.slots = 0;
        self.direct.def_levels.clear();
        self.direct.rep_levels.clear();
        self.direct.values.clear();
    }

    fn push_level_run(&mut self, def: i16, rep: i16, count: usize) {
        self.slots += count;
        self.direct.def_levels.append_run(def, count);
        self.direct.rep_levels.append_run(rep, count);
    }

    fn push_value_range(&mut self, def: i16, rep: i16, range: std::ops::Range<usize>) {
        let len = range.len();
        self.push_level_run(def, rep, len);
        self.direct.values.append_range(range);
    }

    pub(crate) fn batch<'a>(&'a self, plan: &'a CursorLeafPlan) -> LeafBatch<'a> {
        LeafBatch::new(
            plan.terminal.as_ref(),
            self.direct.def_levels.as_ref(),
            self.direct.rep_levels.as_ref(),
            self.direct.values.as_ref(),
        )
    }

    fn def_levels_len(&self) -> usize {
        self.direct.def_levels.len()
    }

    fn rep_levels_len(&self) -> usize {
        self.direct.rep_levels.len()
    }
}

/// A pull cursor whose returned tile is reused by the next call.
pub(crate) struct LeafCursor<'a> {
    plan: &'a CursorLeafPlan,
    next_row: usize,
    target_rows: usize,
    tile: LeafTile,
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
    let direct = &mut out.direct;
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
    let levels = out.direct.rep_levels.materialize_mut().unwrap();
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

fn collect_node(
    array: &ArrayRef,
    contract: FieldContract<'_>,
    ctx: LevelContext,
    path: &mut Vec<PathNode>,
    root: &ArrayRef,
    root_field: &Field,
    leaves: &mut Vec<CursorLeafPlan>,
) -> Result<()> {
    let kind = classify_node(array.as_ref(), contract)?;
    path.push(PathNode {
        kind,
        struct_child: 0,
    });
    let result = (|| match kind {
        NodeKind::Null | NodeKind::Leaf | NodeKind::DictionaryLeaf => {
            leaves.push(CursorLeafPlan {
                root: root.clone(),
                field: root_field.clone(),
                path: path.clone().into_boxed_slice(),
                terminal: array.clone(),
                max_def_level: ctx.def_level + contract.nullable as i16,
                max_rep_level: ctx.rep_level,
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
        leaves,
    )
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
    fn direct_leaf_empty_required_and_bulk_null_paths() {
        let field = Field::new("a", DataType::Int32, false);
        let array = Int32Array::from(vec![Some(1), None]);
        let mut tile = LeafTile::new(0, 0);
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
        let mut tile = LeafTile::new(1, 0);
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
        assert_eq!(tile.direct.values.as_ref().len(), 20);
    }
}
