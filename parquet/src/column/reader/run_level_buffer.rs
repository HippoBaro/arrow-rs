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

//! A buffer for Parquet definition/repetition levels that preserves RLE run
//! structure.

use std::cell::OnceCell;

/// A level buffer that stores `(value, count)` runs as the primary
/// representation, with lazy materialization to `Vec<i16>` for consumers
/// that need per-element access via `as_slice()`.
#[derive(Debug)]
pub struct RunLevelBuffer {
    runs: Vec<(i16, u32)>,
    total_len: usize,
    /// Lazily materialized flat buffer. Uses OnceCell so `as_slice()` can
    /// work with `&self` (required by the ArrayReader trait which returns
    /// `Option<&[i16]>` from `get_def_levels(&self)`).
    materialized: OnceCell<Vec<i16>>,
}

impl Default for RunLevelBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RunLevelBuffer {
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            total_len: 0,
            materialized: OnceCell::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    #[inline]
    pub fn iter_runs(&self) -> impl Iterator<Item = (i16, u32)> + '_ {
        self.runs.iter().copied()
    }

    #[inline]
    pub fn runs(&self) -> &[(i16, u32)] {
        &self.runs
    }

    /// Returns a mutable reference to the underlying runs vector.
    /// Invalidates the materialized cache.
    #[inline]
    pub fn runs_mut(&mut self) -> &mut Vec<(i16, u32)> {
        self.materialized = OnceCell::new();
        &mut self.runs
    }

    #[inline]
    pub fn set_total_len(&mut self, len: usize) {
        self.total_len = len;
        self.materialized = OnceCell::new();
    }

    /// Append runs, merging adjacent same-value runs.
    pub fn append_runs(&mut self, new_runs: &[(i16, u32)]) {
        self.materialized = OnceCell::new();
        for &(value, count) in new_runs {
            if let Some(last) = self.runs.last_mut().filter(|r| r.0 == value) {
                last.1 += count;
            } else {
                self.runs.push((value, count));
            }
        }
    }

    /// Lazily materialize and return a flat `&[i16]` slice.
    /// Works with `&self` via OnceCell interior mutability.
    pub fn as_slice(&self) -> &[i16] {
        self.materialized.get_or_init(|| {
            let mut flat = Vec::with_capacity(self.total_len);
            for &(value, count) in &self.runs {
                flat.extend(std::iter::repeat_n(value, count as usize));
            }
            flat
        })
    }

    /// Take the materialized flat buffer, or materialize and return it.
    /// After this call the buffer is empty.
    pub fn take_flat(&mut self) -> Vec<i16> {
        let flat = if let Some(m) = self.materialized.take() {
            m
        } else {
            let mut flat = Vec::with_capacity(self.total_len);
            for &(value, count) in &self.runs {
                flat.extend(std::iter::repeat_n(value, count as usize));
            }
            flat
        };
        self.runs.clear();
        self.total_len = 0;
        flat
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.runs.clear();
        self.total_len = 0;
        self.materialized = OnceCell::new();
    }
}

/// A cursor for iterating through run-length encoded levels one element
/// at a time.
pub struct RunCursor<'a> {
    runs: &'a [(i16, u32)],
    run_idx: usize,
    offset_in_run: u32,
}

impl<'a> RunCursor<'a> {
    pub fn new(runs: &'a [(i16, u32)]) -> Self {
        Self {
            runs,
            run_idx: 0,
            offset_in_run: 0,
        }
    }

    #[inline]
    pub fn next(&mut self) -> i16 {
        let (value, count) = self.runs[self.run_idx];
        self.offset_in_run += 1;
        if self.offset_in_run >= count {
            self.run_idx += 1;
            self.offset_in_run = 0;
        }
        value
    }
}

/// Convert a flat `&[i16]` slice to run-length encoded `(value, count)` pairs.
pub fn levels_to_runs(levels: &[i16]) -> Vec<(i16, u32)> {
    let mut runs = Vec::new();
    for &v in levels {
        if let Some(last) = runs.last_mut().filter(|r: &&mut (i16, u32)| r.0 == v) {
            last.1 += 1;
        } else {
            runs.push((v, 1));
        }
    }
    runs
}

/// An iterator that aligns two run-length encoded streams, yielding
/// `(def_value, rep_value, count)` triples where both the def and rep
/// values are constant for `count` consecutive entries.
///
/// When a def run of `(0, 50000)` overlaps a rep run of `(0, 50000)`,
/// this yields a single `(0, 0, 50000)` triple — enabling O(1) batch
/// processing of 50K null records.
pub struct AlignedRunIter<'a> {
    def_runs: &'a [(i16, u32)],
    rep_runs: &'a [(i16, u32)],
    def_idx: usize,
    rep_idx: usize,
    def_remaining: u32,
    rep_remaining: u32,
}

impl<'a> AlignedRunIter<'a> {
    pub fn new(def_runs: &'a [(i16, u32)], rep_runs: &'a [(i16, u32)]) -> Self {
        let def_remaining = def_runs.first().map(|r| r.1).unwrap_or(0);
        let rep_remaining = rep_runs.first().map(|r| r.1).unwrap_or(0);
        Self {
            def_runs,
            rep_runs,
            def_idx: 0,
            rep_idx: 0,
            def_remaining,
            rep_remaining,
        }
    }
}

impl Iterator for AlignedRunIter<'_> {
    /// (def_value, rep_value, count)
    type Item = (i16, i16, u32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.def_remaining == 0 || self.rep_remaining == 0 {
            return None;
        }

        let def_val = self.def_runs[self.def_idx].0;
        let rep_val = self.rep_runs[self.rep_idx].0;
        let count = self.def_remaining.min(self.rep_remaining);

        self.def_remaining -= count;
        self.rep_remaining -= count;

        if self.def_remaining == 0 {
            self.def_idx += 1;
            if self.def_idx < self.def_runs.len() {
                self.def_remaining = self.def_runs[self.def_idx].1;
            }
        }
        if self.rep_remaining == 0 {
            self.rep_idx += 1;
            if self.rep_idx < self.rep_runs.len() {
                self.rep_remaining = self.rep_runs[self.rep_idx].1;
            }
        }

        Some((def_val, rep_val, count))
    }
}
