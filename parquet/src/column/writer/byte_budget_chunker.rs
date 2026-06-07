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

//! See [`ByteBudgetChunker`] for byte-budget-aware mini-batch sizing.

use crate::basic::Type;
use crate::column::writer::encoder::ColumnValueEncoder;
use crate::column::writer::{ByteBudgetTarget, ColumnValueSource, LevelDataRef};
use crate::file::properties::WriterProperties;
use crate::schema::types::ColumnDescriptor;

/// Static per-value byte width of a fixed-width physical type.
#[derive(Clone, Copy)]
struct FixedWidth {
    bytes_per_value: usize,
}

/// Picks byte-budget-aware mini-batch sizes for one column.
///
/// Given a level window and its values, picks the largest prefix that fits the
/// *remaining* data/dictionary page byte budget. All batches use this policy.
/// Fixed-width candidates that fit are rejected/accepted in O(1); only
/// potentially overflowing variable-width input inspects physical value sizes.
pub(crate) struct ByteBudgetChunker {
    /// Configured data page byte limit for the column.
    page_byte_limit: usize,
    /// Max definition level of the column; a level equal to this marks a
    /// present (non-null) leaf value. Used to count values per chunk.
    max_def_level: i16,
    /// Per-value byte width when the physical type is fixed-width; `None` for
    /// BYTE_ARRAY, whose sizing must inspect the values.
    fixed_width: Option<FixedWidth>,
    /// Configured dictionary page byte limit for the column.
    dict_page_byte_limit: usize,
}

impl ByteBudgetChunker {
    #[inline]
    pub(crate) fn new(descr: &ColumnDescriptor, props: &WriterProperties) -> Self {
        let page_byte_limit = props.column_data_page_size_limit(descr.path());
        let dict_page_byte_limit = props.column_dictionary_page_size_limit(descr.path());
        let fixed = |bytes_per_value: usize| Some(FixedWidth { bytes_per_value });
        let fixed_width = match descr.physical_type() {
            Type::BOOLEAN => fixed(1),
            Type::INT32 | Type::FLOAT => fixed(std::mem::size_of::<i32>()),
            Type::INT64 | Type::DOUBLE => fixed(std::mem::size_of::<i64>()),
            Type::INT96 => fixed(12),
            Type::FIXED_LEN_BYTE_ARRAY => fixed(descr.type_length().max(0) as usize),
            Type::BYTE_ARRAY => None,
        };
        Self {
            page_byte_limit,
            max_def_level: descr.max_def_level(),
            fixed_width,
            dict_page_byte_limit,
        }
    }

    /// Decide how many levels at the start of a chunk belong in one
    /// mini-batch, so the mini-batch cannot overflow whichever page is
    /// currently accumulating value bytes: the data page when plain-encoding,
    /// or the *dictionary* page while dictionary-encoding. A returned value
    /// smaller than `chunk_size` triggers granular sub-batching in
    /// `write_batch_internal`.
    ///
    /// While dictionary-encoding, the data page holds only small RLE indices,
    /// but the dictionary page accumulates the distinct values themselves —
    /// so it is the dictionary page's remaining budget that must bound the
    /// mini-batch. The per-mini-batch dictionary spill check would otherwise
    /// let one mini-batch of large values balloon the dictionary page.
    ///
    /// Returns zero when the first value does not fit a nonempty data page; the
    /// caller flushes and retries. An empty page always accepts at least one
    /// value so an oversize singleton cannot stall progress.
    ///
    /// `#[inline]`: this is a tiny per-chunk dispatcher; the actual byte
    /// inspection lives in the out-of-line `byte_budget_sub_batch_size`.
    #[inline]
    pub(crate) fn pick_sub_batch_size<E, S>(
        &self,
        encoder: &E,
        source: S,
        chunk_def: LevelDataRef<'_>,
        values_offset: usize,
        chunk_size: usize,
        data_page_is_empty: bool,
    ) -> usize
    where
        E: ColumnValueEncoder,
        S: ColumnValueSource<E>,
    {
        if chunk_size == 0 {
            return chunk_size;
        }
        let (budget, target) = if encoder.has_dictionary() {
            // Bound the mini-batch by the dictionary page's *remaining*
            // budget (it accumulates across mini-batches until it spills).
            match encoder.estimated_dict_page_size() {
                Some(used) => (
                    self.dict_page_byte_limit.saturating_sub(used),
                    ByteBudgetTarget::DictionaryPage,
                ),
                None => return chunk_size,
            }
        } else {
            (
                self.page_byte_limit
                    .saturating_sub(encoder.estimated_data_page_size()),
                ByteBudgetTarget::DataPage,
            )
        };
        // A fixed-width upper bound avoids scanning nullable definition levels
        // whenever the entire candidate fits in the *remaining* budget.
        if self
            .fixed_width
            .is_some_and(|width| width.bytes_per_value.saturating_mul(chunk_size) <= budget)
        {
            return chunk_size;
        }
        self.byte_budget_sub_batch_size::<E, S>(
            source,
            chunk_def,
            values_offset,
            chunk_size,
            (budget, target),
            data_page_is_empty,
        )
    }

    /// Inspect value sizes to decide how many of the chunk's values fit in
    /// `budget` bytes (the data page or dictionary page remaining budget).
    ///
    /// `#[inline(never)]` keeps this slow path out of the hot
    /// `write_batch_internal` loop; numeric and bool columns never reach it.
    #[inline(never)]
    fn byte_budget_sub_batch_size<E, S>(
        &self,
        source: S,
        chunk_def: LevelDataRef<'_>,
        values_offset: usize,
        chunk_size: usize,
        (budget, target): (usize, ByteBudgetTarget),
        data_page_is_empty: bool,
    ) -> usize
    where
        E: ColumnValueEncoder,
        S: ColumnValueSource<E>,
    {
        // How many of this chunk's levels carry an actual value. For a
        // non-nullable, unrepeated column every level is a value, so
        // `value_count` is O(1) (`Absent`/`Uniform` def levels); only
        // nullable or nested columns pay the O(chunk_size) def-level scan.
        let vals_in_chunk = chunk_def.value_count(chunk_size, self.max_def_level);
        if vals_in_chunk == 0 {
            return chunk_size;
        }
        // Restrict the source to this chunk's values
        // `[values_offset, values_offset + vals_in_chunk)`, clamped to the
        // values that remain, then ask how many of them fit in one page byte
        // budget. The source carries its own dense/sparse selection, so this
        // mirrors how `write_mini_batch` picks `write_gather` vs `write`.
        let remaining = source.len().saturating_sub(values_offset);
        let take = vals_in_chunk.min(remaining);
        let fit = self
            .fixed_width
            .map(|w| {
                let count = budget / w.bytes_per_value.max(1);
                count.min(take)
            })
            .or_else(|| {
                source
                    .slice(values_offset, take)
                    .count_variable_width_within_byte_budget(budget, target)
            });
        match fit {
            None => chunk_size,
            Some(values_per_subbatch) if values_per_subbatch >= vals_in_chunk => chunk_size,
            // Zero is an intentional retry signal for a nonempty data page:
            // flush it and size the same logical window against a fresh page.
            Some(0) if target == ByteBudgetTarget::DataPage && !data_page_is_empty => 0,
            Some(values_per_subbatch) => chunk_def.level_count_for_value_count(
                chunk_size,
                self.max_def_level,
                values_per_subbatch.max(1),
            ),
        }
    }
}
