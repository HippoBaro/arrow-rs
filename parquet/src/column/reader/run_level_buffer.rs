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
//!
//! Instead of materializing every level as an individual `i16`, this stores
//! `(value, count)` runs. Consumers that can operate on runs (bitmap builders,
//! list offset builders) get O(runs) performance instead of O(rows). Consumers
//! that need per-element access can call [`RunLevelBuffer::as_slice`] which
//! lazily materializes the flat `Vec<i16>` on first access.

/// A level buffer that stores `(value, count)` runs as the primary
/// representation, with lazy materialization to `Vec<i16>` for consumers
/// that need per-element access.
#[derive(Debug)]
pub struct RunLevelBuffer {
    /// Runs as (value, count) pairs — the primary representation
    runs: Vec<(i16, u32)>,
    /// Total number of level entries across all runs
    total_len: usize,
    /// Lazily materialized flat buffer
    materialized: Option<Vec<i16>>,
}

impl Default for RunLevelBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RunLevelBuffer {
    /// Create an empty `RunLevelBuffer`.
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            total_len: 0,
            materialized: None,
        }
    }

    /// Returns the total number of level entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Returns an iterator over `(value, count)` runs.
    #[inline]
    pub fn iter_runs(&self) -> impl Iterator<Item = (i16, u32)> + '_ {
        self.runs.iter().copied()
    }

    /// Returns the runs as a slice of `(value, count)` pairs.
    #[inline]
    pub fn runs(&self) -> &[(i16, u32)] {
        &self.runs
    }

    /// Returns a mutable reference to the underlying runs vector, for
    /// direct population by decoders.
    #[inline]
    pub fn runs_mut(&mut self) -> &mut Vec<(i16, u32)> {
        self.materialized = None; // invalidate cache
        &mut self.runs
    }

    /// Set the total length (must match the sum of all run counts).
    /// Called after the decoder has populated the runs.
    #[inline]
    pub fn set_total_len(&mut self, len: usize) {
        self.total_len = len;
        self.materialized = None; // invalidate cache
    }

    /// Append runs from another source, merging with the last existing run
    /// if it has the same value.
    pub fn append_runs(&mut self, new_runs: &[(i16, u32)]) {
        self.materialized = None; // invalidate cache
        for &(value, count) in new_runs {
            if let Some(last) = self.runs.last_mut().filter(|r| r.0 == value) {
                last.1 += count;
            } else {
                self.runs.push((value, count));
            }
        }
    }

    /// Lazily materialize and return a flat `&[i16]` slice.
    ///
    /// The first call allocates and fills the buffer; subsequent calls
    /// return the cached result. The cache is invalidated when runs are
    /// modified.
    pub fn as_slice(&mut self) -> &[i16] {
        if self.materialized.is_none() {
            let mut flat = Vec::with_capacity(self.total_len);
            for &(value, count) in &self.runs {
                flat.extend(std::iter::repeat_n(value, count as usize));
            }
            self.materialized = Some(flat);
        }
        self.materialized.as_ref().unwrap()
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
        self.materialized = None;
    }
}
