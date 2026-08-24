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
use arrow_buffer::{ArrowNativeType, NullBuffer, OffsetBuffer, ScalarBuffer};
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
    direct_path: Option<Box<[DirectNode]>>,
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

/// A direct-traversal node bound to its Arrow buffers and schema contract.
///
/// Nested list runs are often only a handful of values long. Keeping these
/// loop-invariant objects in the plan avoids rediscovering them for every run.
#[derive(Debug, Clone)]
struct DirectNode {
    kind: DirectNodeKind,
    struct_child: usize,
    nullable: bool,
    name: Box<str>,
    nulls: Option<NullBuffer>,
    flat_child: bool,
}

#[derive(Debug, Clone)]
enum DirectNodeKind {
    Leaf,
    Struct,
    List(OffsetBuffer<i32>),
    LargeList(OffsetBuffer<i64>),
    FixedSizeList(usize),
    ListView {
        offsets: ScalarBuffer<i32>,
        sizes: ScalarBuffer<i32>,
    },
    LargeListView {
        offsets: ScalarBuffer<i64>,
        sizes: ScalarBuffer<i64>,
    },
    Map(OffsetBuffer<i32>),
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

fn bind_direct_path(
    root: &ArrayRef,
    field: &Field,
    path: &[PathNode],
) -> Result<Box<[DirectNode]>> {
    let mut bound = Vec::with_capacity(path.len());
    let mut array = root.clone();
    let mut contract = normalized(field);

    for (depth, node) in path.iter().enumerate() {
        let nulls = array
            .logical_nulls()
            .filter(|nulls| nulls.null_count() != 0);
        let flat_child = path[depth + 1..].iter().all(|node| {
            matches!(
                node.kind,
                NodeKind::Struct | NodeKind::Null | NodeKind::Leaf | NodeKind::DictionaryLeaf
            )
        });

        let (kind, child) = match node.kind {
            NodeKind::Null | NodeKind::Leaf | NodeKind::DictionaryLeaf => {
                (DirectNodeKind::Leaf, None)
            }
            NodeKind::Struct => {
                let DataType::Struct(fields) = contract.data_type else {
                    unreachable!("struct contract was validated during planning")
                };
                let child = array.as_struct().column(node.struct_child).clone();
                let child_contract = normalized(&fields[node.struct_child]);
                (DirectNodeKind::Struct, Some((child, child_contract)))
            }
            NodeKind::List(kind) => {
                let direct_kind = match kind {
                    ListKind::List => {
                        DirectNodeKind::List(array.as_list::<i32>().offsets().clone())
                    }
                    ListKind::LargeList => {
                        DirectNodeKind::LargeList(array.as_list::<i64>().offsets().clone())
                    }
                    ListKind::FixedSizeList => DirectNodeKind::FixedSizeList(
                        array.as_fixed_size_list().value_length() as usize,
                    ),
                    ListKind::ListView => {
                        let list = array.as_list_view::<i32>();
                        DirectNodeKind::ListView {
                            offsets: list.offsets().clone(),
                            sizes: list.sizes().clone(),
                        }
                    }
                    ListKind::LargeListView => {
                        let list = array.as_list_view::<i64>();
                        DirectNodeKind::LargeListView {
                            offsets: list.offsets().clone(),
                            sizes: list.sizes().clone(),
                        }
                    }
                    ListKind::Map => DirectNodeKind::Map(array.as_map().offsets().clone()),
                };
                let child = kind.child_owned(array.as_ref());
                let child_contract = normalized(kind.field(contract));
                (direct_kind, Some((child, child_contract)))
            }
        };

        bound.push(DirectNode {
            kind,
            struct_child: node.struct_child,
            nullable: contract.nullable,
            name: contract.name.into(),
            nulls,
            flat_child,
        });
        if let Some((child, child_contract)) = child {
            array = child;
            contract = child_contract;
        }
    }
    Ok(bound.into_boxed_slice())
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
        self.direct.rep_levels.append_dense_run(rep, count);
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
        let end = if self.plan.max_rep_level == 0 {
            self.plan.root.len()
        } else {
            first_row
                .saturating_add(rows_to_boundary)
                .min(self.plan.root.len())
        };
        visit_direct_range(
            first_row..end,
            LevelContext::default(),
            0,
            self.plan.direct_path.as_deref().unwrap(),
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

/// A row-aligned cursor over all direct leaves of one top-level Arrow field.
///
/// The individual leaf plans remain usable by the public parallel-column API;
/// the row-group writer uses this cursor to share structural traversal while it
/// has all descendant column writers together.
pub(crate) struct DirectGroupCursor<'a> {
    plans: &'a [&'a CursorLeafPlan],
    next_row: usize,
    target_rows: usize,
    tiles: Vec<LeafTile>,
}

impl<'a> DirectGroupCursor<'a> {
    pub(crate) fn try_new(plans: &'a [&'a CursorLeafPlan], target_rows: usize) -> Option<Self> {
        let first = *plans.first()?;
        if plans.len() == 1
            || first.direct_path.is_none()
            || plans
                .iter()
                .any(|plan| plan.direct_path.is_none() || plan.root.len() != first.root.len())
            || !has_shared_repeated_node(plans, 0)
        {
            return None;
        }
        let tiles = plans
            .iter()
            .map(|plan| LeafTile::new(plan.max_def_level, plan.max_rep_level))
            .collect();
        Some(Self {
            plans,
            next_row: 0,
            target_rows: target_rows.max(1),
            tiles,
        })
    }

    pub(crate) fn next_tiles(&mut self) -> Result<Option<&[LeafTile]>> {
        let len = self.plans[0].root.len();
        if self.next_row == len {
            return Ok(None);
        }
        for tile in &mut self.tiles {
            tile.clear();
        }
        let first_row = self.next_row;
        let rows_to_boundary = self.target_rows - first_row % self.target_rows;
        let end = first_row.saturating_add(rows_to_boundary).min(len);
        visit_direct_group_range(
            self.plans,
            0,
            first_row..end,
            LevelContext::default(),
            0,
            &mut self.tiles,
        )?;
        self.next_row = end;

        for (plan, tile) in self.plans.iter().zip(&self.tiles) {
            debug_assert_ne!(tile.slots, 0);
            debug_assert_eq!(
                tile.def_levels_len(),
                usize::from(plan.max_def_level != 0) * tile.slots
            );
            debug_assert_eq!(
                tile.rep_levels_len(),
                usize::from(plan.max_rep_level != 0) * tile.slots
            );
        }
        Ok(Some(&self.tiles))
    }
}

fn has_shared_repeated_node(plans: &[&CursorLeafPlan], depth: usize) -> bool {
    if plans.len() < 2 {
        return false;
    }
    let node = &plans[0].direct_path.as_deref().unwrap()[depth];
    match &node.kind {
        DirectNodeKind::List(_)
        | DirectNodeKind::LargeList(_)
        | DirectNodeKind::FixedSizeList(_)
        | DirectNodeKind::ListView { .. }
        | DirectNodeKind::LargeListView { .. }
        | DirectNodeKind::Map(_) => true,
        DirectNodeKind::Struct => {
            let mut start = 0;
            while start != plans.len() {
                let child = plans[start].direct_path.as_deref().unwrap()[depth].struct_child;
                let mut end = start + 1;
                while end != plans.len()
                    && plans[end].direct_path.as_deref().unwrap()[depth].struct_child == child
                {
                    end += 1;
                }
                if has_shared_repeated_node(&plans[start..end], depth + 1) {
                    return true;
                }
                start = end;
            }
            false
        }
        DirectNodeKind::Leaf => false,
    }
}

fn visit_direct_group_range(
    plans: &[&CursorLeafPlan],
    depth: usize,
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    out: &mut [LeafTile],
) -> Result<()> {
    if range.is_empty() {
        return Ok(());
    }
    debug_assert_eq!(plans.len(), out.len());
    let node = &plans[0].direct_path.as_deref().unwrap()[depth];
    match &node.kind {
        DirectNodeKind::Leaf => {
            debug_assert_eq!(plans.len(), 1);
            visit_direct_leaf(node, range, ctx, rep, &mut out[0])
        }
        DirectNodeKind::Struct => visit_direct_group_struct(plans, depth, range, ctx, rep, out),
        DirectNodeKind::List(offsets) => {
            visit_direct_group_list_rows(plans, depth, range, ctx, rep, out, |row| {
                (offsets[row].as_usize(), offsets[row + 1].as_usize())
            })
        }
        DirectNodeKind::LargeList(offsets) => {
            visit_direct_group_list_rows(plans, depth, range, ctx, rep, out, |row| {
                (offsets[row].as_usize(), offsets[row + 1].as_usize())
            })
        }
        DirectNodeKind::FixedSizeList(width) => {
            visit_direct_group_list_rows(plans, depth, range, ctx, rep, out, |row| {
                let start = row * width;
                (start, start + width)
            })
        }
        DirectNodeKind::ListView { offsets, sizes } => {
            visit_direct_group_list_rows(plans, depth, range, ctx, rep, out, |row| {
                let start = offsets[row].as_usize();
                (start, start + sizes[row].as_usize())
            })
        }
        DirectNodeKind::LargeListView { offsets, sizes } => {
            visit_direct_group_list_rows(plans, depth, range, ctx, rep, out, |row| {
                let start = offsets[row].as_usize();
                (start, start + sizes[row].as_usize())
            })
        }
        DirectNodeKind::Map(offsets) => {
            visit_direct_group_list_rows(plans, depth, range, ctx, rep, out, |row| {
                (offsets[row].as_usize(), offsets[row + 1].as_usize())
            })
        }
    }
}

fn visit_direct_group_struct(
    plans: &[&CursorLeafPlan],
    depth: usize,
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    out: &mut [LeafTile],
) -> Result<()> {
    let node = &plans[0].direct_path.as_deref().unwrap()[depth];
    let child_ctx = LevelContext {
        def_level: ctx.def_level + node.nullable as i16,
        ..ctx
    };
    scan_nullable_runs(node, range, |valid, range| {
        if !valid {
            for out in &mut *out {
                out.push_level_run(ctx.def_level, rep, range.len());
            }
            return Ok(());
        }

        let mut start = 0;
        while start != plans.len() {
            let child = plans[start].direct_path.as_deref().unwrap()[depth].struct_child;
            let mut end = start + 1;
            while end != plans.len()
                && plans[end].direct_path.as_deref().unwrap()[depth].struct_child == child
            {
                end += 1;
            }
            visit_direct_group_range(
                &plans[start..end],
                depth + 1,
                range.clone(),
                child_ctx,
                rep,
                &mut out[start..end],
            )?;
            start = end;
        }
        Ok(())
    })
}

fn visit_direct_group_list_rows(
    plans: &[&CursorLeafPlan],
    depth: usize,
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    out: &mut [LeafTile],
    bounds: impl Fn(usize) -> (usize, usize) + Copy,
) -> Result<()> {
    let node = &plans[0].direct_path.as_deref().unwrap()[depth];
    let list_def = ctx.def_level + node.nullable as i16;
    let child_ctx = LevelContext {
        def_level: list_def + 1,
        rep_level: ctx.rep_level + 1,
    };
    let mut flush = |kind: ListSlotKind, rows: Range<usize>| -> Result<()> {
        match kind {
            ListSlotKind::Null | ListSlotKind::Empty => {
                let def = if kind == ListSlotKind::Null {
                    ctx.def_level
                } else {
                    list_def
                };
                for out in &mut *out {
                    out.push_level_run(def, rep, rows.len());
                }
            }
            ListSlotKind::NonEmpty => {
                let values_start = bounds(rows.start).0;
                let values_end = bounds(rows.end - 1).1;
                visit_direct_group_range(
                    plans,
                    depth + 1,
                    values_start..values_end,
                    child_ctx,
                    child_ctx.rep_level,
                    out,
                )?;
                for (plan, out) in plans.iter().zip(&mut *out) {
                    let list = &plan.direct_path.as_deref().unwrap()[depth];
                    patch_list_starts(
                        out,
                        values_start,
                        rows.clone(),
                        child_ctx.rep_level,
                        rep,
                        list.flat_child,
                        bounds,
                    );
                }
            }
        }
        Ok(())
    };
    scan_list_slots(node, range, bounds, &mut flush)
}

fn visit_direct_range(
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    path: &[DirectNode],
    out: &mut LeafTile,
) -> Result<()> {
    if range.is_empty() {
        return Ok(());
    }

    let (node, child_path) = path.split_first().unwrap();
    match &node.kind {
        DirectNodeKind::Leaf => visit_direct_leaf(node, range, ctx, rep, out),
        DirectNodeKind::Struct => visit_direct_struct(node, range, ctx, rep, child_path, out),
        DirectNodeKind::List(offsets) => {
            visit_direct_list_rows(node, range, ctx, rep, child_path, out, |row| {
                (offsets[row].as_usize(), offsets[row + 1].as_usize())
            })
        }
        DirectNodeKind::LargeList(offsets) => {
            visit_direct_list_rows(node, range, ctx, rep, child_path, out, |row| {
                (offsets[row].as_usize(), offsets[row + 1].as_usize())
            })
        }
        DirectNodeKind::FixedSizeList(width) => {
            visit_direct_list_rows(node, range, ctx, rep, child_path, out, |row| {
                let start = row * width;
                (start, start + width)
            })
        }
        DirectNodeKind::ListView { offsets, sizes } => {
            visit_direct_list_rows(node, range, ctx, rep, child_path, out, |row| {
                let start = offsets[row].as_usize();
                (start, start + sizes[row].as_usize())
            })
        }
        DirectNodeKind::LargeListView { offsets, sizes } => {
            visit_direct_list_rows(node, range, ctx, rep, child_path, out, |row| {
                let start = offsets[row].as_usize();
                (start, start + sizes[row].as_usize())
            })
        }
        DirectNodeKind::Map(offsets) => {
            visit_direct_list_rows(node, range, ctx, rep, child_path, out, |row| {
                (offsets[row].as_usize(), offsets[row + 1].as_usize())
            })
        }
    }
}

fn visit_direct_struct(
    node: &DirectNode,
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    child_path: &[DirectNode],
    out: &mut LeafTile,
) -> Result<()> {
    let child_ctx = LevelContext {
        def_level: ctx.def_level + node.nullable as i16,
        ..ctx
    };
    scan_nullable_runs(node, range, |valid, range| {
        if valid {
            visit_direct_range(range, child_ctx, rep, child_path, out)
        } else {
            out.push_level_run(ctx.def_level, rep, range.len());
            Ok(())
        }
    })
}

fn visit_direct_leaf(
    node: &DirectNode,
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    out: &mut LeafTile,
) -> Result<()> {
    let len = range.len();
    let def = ctx.def_level + node.nullable as i16;
    let Some(nulls) = node.nulls.as_ref() else {
        out.push_value_range(def, rep, range);
        return Ok(());
    };

    if !node.nullable {
        if let Some(index) = range.clone().find(|&index| nulls.is_null(index)) {
            return Err(super::required_null(&node.name, index));
        }
        out.push_value_range(def, rep, range);
        return Ok(());
    }

    if nulls.null_count() == nulls.len() {
        out.push_level_run(ctx.def_level, rep, len);
        return Ok(());
    }

    // Avoid constructing a sliced NullBuffer for the tiny ranges produced by
    // nested lists. NullBuffer::slice recomputes a popcount; reading the source
    // bitmap directly matches the old builder's proven short-range path.
    if len < super::BULK_FILL_MIN_LEN {
        out.slots += len;
        let direct = &mut out.direct;
        direct.rep_levels.append_dense_run(rep, len);
        let bits = nulls.inner();
        direct
            .def_levels
            .extend_from_iter(range.clone().map(|index| {
                // SAFETY: `range` is a valid range of this bound leaf.
                let valid = unsafe { bits.value_unchecked(index) };
                def - (!valid as i16)
            }));
        direct.values.append_sparse_range(nulls, range);
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
    direct.rep_levels.append_dense_run(rep, len);
    // Majority-null is not on its own a reason to build runs: what matters is
    // whether the runs are long enough to pay for their two-word headers. Dense
    // but *scattered* nulls produce short runs that `LevelData::append_run`
    // accumulates and then discards when it materializes, which costs more than
    // filling the level buffer from the bitmap in one pass.
    if len >= super::BULK_FILL_MIN_LEN
        && nulls.null_count() * 2 >= nulls.len()
        && levels_have_compact_runs(&range_nulls)
    {
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

fn visit_direct_list_rows(
    node: &DirectNode,
    range: Range<usize>,
    ctx: LevelContext,
    rep: i16,
    child_path: &[DirectNode],
    out: &mut LeafTile,
    bounds: impl Fn(usize) -> (usize, usize) + Copy,
) -> Result<()> {
    let list_def = ctx.def_level + node.nullable as i16;
    let child_ctx = LevelContext {
        def_level: list_def + 1,
        rep_level: ctx.rep_level + 1,
    };

    let mut flush = |kind: ListSlotKind, rows: Range<usize>| -> Result<()> {
        match kind {
            ListSlotKind::Null => out.push_level_run(ctx.def_level, rep, rows.len()),
            ListSlotKind::Empty => out.push_level_run(list_def, rep, rows.len()),
            ListSlotKind::NonEmpty => {
                let values_start = bounds(rows.start).0;
                let values_end = bounds(rows.end - 1).1;
                visit_direct_range(
                    values_start..values_end,
                    child_ctx,
                    child_ctx.rep_level,
                    child_path,
                    out,
                )?;
                patch_list_starts(
                    out,
                    values_start,
                    rows,
                    child_ctx.rep_level,
                    rep,
                    node.flat_child,
                    bounds,
                );
            }
        }
        Ok(())
    };
    scan_list_slots(node, range, bounds, &mut flush)
}

#[derive(Clone, Copy, PartialEq)]
enum ListSlotKind {
    Null,
    Empty,
    NonEmpty,
}

fn use_sparse_null_runs(nulls: &NullBuffer, len: usize) -> bool {
    len >= super::BULK_FILL_MIN_LEN && nulls.null_count() * 2 >= nulls.len()
}

fn scan_nullable_runs(
    node: &DirectNode,
    range: Range<usize>,
    mut visit: impl FnMut(bool, Range<usize>) -> Result<()>,
) -> Result<()> {
    let Some(nulls) = node.nulls.as_ref() else {
        return visit(true, range);
    };
    if !node.nullable {
        if let Some(index) = range.clone().find(|&index| nulls.is_null(index)) {
            return Err(super::required_null(&node.name, index));
        }
        return visit(true, range);
    }
    if nulls.null_count() == nulls.len() {
        return visit(false, range);
    }
    if use_sparse_null_runs(nulls, range.len()) {
        let range_nulls = nulls.slice(range.start, range.len());
        let mut position = 0;
        for (start, end) in range_nulls.valid_slices() {
            if position != start {
                visit(false, range.start + position..range.start + start)?;
            }
            visit(true, range.start + start..range.start + end)?;
            position = end;
        }
        if position != range.len() {
            visit(false, range.start + position..range.end)?;
        }
        return Ok(());
    }

    let mut run_start = range.start;
    let mut run_valid = nulls.is_valid(run_start);
    for index in run_start + 1..range.end {
        let valid = nulls.is_valid(index);
        if valid != run_valid {
            visit(run_valid, run_start..index)?;
            run_start = index;
            run_valid = valid;
        }
    }
    visit(run_valid, run_start..range.end)
}

fn scan_list_slots(
    node: &DirectNode,
    range: Range<usize>,
    bounds: impl Fn(usize) -> (usize, usize) + Copy,
    mut flush: impl FnMut(ListSlotKind, Range<usize>) -> Result<()>,
) -> Result<()> {
    if node
        .nulls
        .as_ref()
        .is_some_and(|nulls| nulls.null_count() == nulls.len())
    {
        if !node.nullable {
            return Err(super::required_null(&node.name, range.start));
        }
        return flush(ListSlotKind::Null, range);
    }

    if let Some(nulls) = node.nulls.as_ref()
        && node.nullable
        && use_sparse_null_runs(nulls, range.len())
    {
        let range_nulls = nulls.slice(range.start, range.len());
        let mut position = 0;
        for (start, end) in range_nulls.valid_slices() {
            if position != start {
                flush(
                    ListSlotKind::Null,
                    range.start + position..range.start + start,
                )?;
            }
            scan_list_rows(
                range.start + start..range.start + end,
                bounds,
                |_, start, end| {
                    Ok(if start == end {
                        ListSlotKind::Empty
                    } else {
                        ListSlotKind::NonEmpty
                    })
                },
                &mut flush,
            )?;
            position = end;
        }
        if position != range.len() {
            flush(ListSlotKind::Null, range.start + position..range.end)?;
        }
        return Ok(());
    }

    match node.nulls.as_ref() {
        Some(nulls) => scan_list_rows(
            range,
            bounds,
            |row, start, end| {
                if nulls.is_null(row) {
                    if node.nullable {
                        Ok(ListSlotKind::Null)
                    } else {
                        Err(super::required_null(&node.name, row))
                    }
                } else if start == end {
                    Ok(ListSlotKind::Empty)
                } else {
                    Ok(ListSlotKind::NonEmpty)
                }
            },
            flush,
        ),
        None => scan_list_rows(
            range,
            bounds,
            |_, start, end| {
                Ok(if start == end {
                    ListSlotKind::Empty
                } else {
                    ListSlotKind::NonEmpty
                })
            },
            flush,
        ),
    }
}

fn scan_list_rows(
    range: Range<usize>,
    bounds: impl Fn(usize) -> (usize, usize),
    classify: impl Fn(usize, usize, usize) -> Result<ListSlotKind>,
    mut flush: impl FnMut(ListSlotKind, Range<usize>) -> Result<()>,
) -> Result<()> {
    let mut run_start = range.start;
    let (first_start, mut previous_end) = bounds(run_start);
    let mut run_kind = classify(run_start, first_start, previous_end)?;
    for row in run_start + 1..range.end {
        let (start, end) = bounds(row);
        let kind = classify(row, start, end)?;
        let contiguous = kind != ListSlotKind::NonEmpty || previous_end == start;
        if kind != run_kind || !contiguous {
            flush(run_kind, run_start..row)?;
            run_start = row;
            run_kind = kind;
        }
        previous_end = end;
    }
    flush(run_kind, run_start..range.end)
}

fn patch_list_starts(
    out: &mut LeafTile,
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
        let values_end = bounds(rows.end - 1).1;
        let slot_start = levels.len() - (values_end - values_start);
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
            let direct_path = Some(bind_direct_path(root, root_field, path)?);
            leaves.push(CursorLeafPlan {
                root: root.clone(),
                direct_path,
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
    use std::sync::Arc;

    #[test]
    fn direct_leaf_empty_required_and_bulk_null_paths() {
        let field = Field::new("a", DataType::Int32, false);
        let array: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let mut tile = LeafTile::new(0, 0);
        let path = [PathNode {
            kind: NodeKind::Leaf,
            struct_child: 0,
        }];
        let direct = bind_direct_path(&array, &field, &path).unwrap();
        visit_direct_range(0..0, LevelContext::default(), 0, &direct, &mut tile).unwrap();
        assert_eq!(tile.slots, 0);
        assert!(visit_direct_range(0..2, LevelContext::default(), 0, &direct, &mut tile,).is_err());

        let field = Field::new("a", DataType::Int32, true);
        let array: ArrayRef = Arc::new(Int32Array::from(
            (0..80)
                .map(|index| (index % 4 == 0).then_some(index))
                .collect::<Vec<_>>(),
        ));
        let direct = bind_direct_path(&array, &field, &path).unwrap();
        let mut tile = LeafTile::new(1, 0);
        visit_direct_range(
            0..array.len(),
            LevelContext::default(),
            0,
            &direct,
            &mut tile,
        )
        .unwrap();
        assert_eq!(tile.slots, 80);
        assert_eq!(tile.direct.values.as_ref().len(), 20);
    }
}
