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

//! Criterion benchmark measuring the *memory footprint* of `ArrowWriter` when
//! encoding an Arrow `RecordBatch` to Parquet.
//!
//! This is the write-path counterpart of `arrow_reader_peak_memory.rs`: it
//! reuses the exact same thread-local tracking allocator and the two custom
//! criterion measurements ([`PeakMemory`] = live high-water mark,
//! [`AllocatedBytes`] = cumulative bytes requested). The point is to quantify
//! how much transient/peak memory the writer needs over and above the input
//! batch, so the effect of feeding Arrow sources straight to the column
//! encoders (instead of materializing dense intermediate representations) is
//! directly visible.
//!
//! How the measurement isolates *writer* memory:
//! * The `RecordBatch` and the `ArrowWriter` are built in the `iter_batched`
//!   **setup** closure, which runs *outside* the measured window. Their live
//!   allocations therefore land in the per-iteration baseline (subtracted out
//!   of `PeakMemory`) and are never counted by `AllocatedBytes`.
//! * Only `writer.write(&batch)` followed by `writer.close()` runs inside the
//!   measured window — exactly the encoding work, mirroring the reader bench's
//!   `make_reader` (setup) / `drain_reader` (measured) split.
//! * The output sink is [`std::io::Empty`], which discards serialized bytes, so
//!   the figure reflects the writer's internal working set (column encoders,
//!   dictionary interners, level/encoding buffers, retained page data up to the
//!   row-group flush) rather than the size of the produced file.
//! * `BatchSize::PerIteration` is mandatory: it forces criterion to bracket a
//!   single `routine()` call between `Measurement::start()`/`end()`, so each
//!   sample re-anchors the peak and isolates one `write`+`close`.
//!
//! Constraints:
//! * The tracking allocator is **thread-local**, so this only measures the
//!   synchronous, single-threaded `ArrowWriter`. A parallel/async writer would
//!   allocate on other threads and be silently undercounted.
//! * Run-end-encoded (REE) batches can only be *written* on builds that support
//!   them; on older builds `write` returns an error. The REE group therefore
//!   tolerates write errors (so the file still runs everywhere) — only treat
//!   its numbers as meaningful on builds where REE writing is supported.

use std::alloc::Layout;
use std::cell::Cell;
use std::io::Empty;
use std::sync::Arc;

use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use arrow::util::data_gen::create_random_batch;
use arrow_array::{
    ArrayRef, Decimal128Array, DictionaryArray, FixedSizeBinaryArray, Int32Array, Int64Array,
    RunArray, StringArray,
};
use arrow_buffer::NullBuffer;
use criterion::measurement::{Measurement, ValueFormatter};
use criterion::{BatchSize, BenchmarkGroup, Criterion, criterion_group, criterion_main};
use parquet::arrow::ArrowWriter;
use parquet::errors::Result;
use parquet::file::properties::{WriterProperties, WriterVersion};

// ---------------------------------------------------------------------------
// Thread-local tracking allocator (verbatim from arrow_reader_peak_memory.rs)
// ---------------------------------------------------------------------------

thread_local! {
    static LIVE_BYTES: Cell<usize> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct TrackingAllocator {
    inner: std::alloc::System,
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator {
    inner: std::alloc::System,
};

fn add_live_bytes(size: usize) {
    LIVE_BYTES.with(|live| {
        let new = live.get().saturating_add(size);
        live.set(new);
        PEAK_BYTES.with(|peak| {
            if new > peak.get() {
                peak.set(new);
            }
        });
    });
}

fn subtract_live_bytes(size: usize) {
    LIVE_BYTES.with(|live| {
        live.set(live.get().saturating_sub(size));
    });
}

fn add_allocated_bytes(size: usize) {
    ALLOCATED_BYTES.with(|allocated| {
        allocated.set(allocated.get().saturating_add(size));
    });
}

#[allow(unsafe_code)]
unsafe impl std::alloc::GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            add_live_bytes(layout.size());
            add_allocated_bytes(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        subtract_live_bytes(layout.size());
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            add_allocated_bytes(new_size);
            if new_size > old_size {
                add_live_bytes(new_size - old_size);
            } else {
                subtract_live_bytes(old_size - new_size);
            }
        }
        new_ptr
    }
}

fn reset_peak() {
    PEAK_BYTES.with(|peak| {
        LIVE_BYTES.with(|live| {
            peak.set(live.get());
        });
    });
}

fn peak_bytes() -> usize {
    PEAK_BYTES.with(|peak| peak.get())
}

fn live_bytes() -> usize {
    LIVE_BYTES.with(|live| live.get())
}

fn reset_allocated() {
    ALLOCATED_BYTES.with(|allocated| allocated.set(0));
}

fn allocated_bytes() -> usize {
    ALLOCATED_BYTES.with(|allocated| allocated.get())
}

// ---------------------------------------------------------------------------
// Criterion custom measurements (verbatim from arrow_reader_peak_memory.rs)
// ---------------------------------------------------------------------------

struct BytesFormatter;

const BYTE_UNITS: &[(u32, &str)] = &[
    (60, "EiB"),
    (50, "PiB"),
    (40, "TiB"),
    (30, "GiB"),
    (20, "MiB"),
    (10, "KiB"),
    (0, "B"),
];

fn bytes_per_unit(exponent: u32) -> f64 {
    (1_u64 << exponent) as f64
}

fn scale_bytes(typical: f64, values: &mut [f64]) -> &'static str {
    for &(exponent, unit) in BYTE_UNITS {
        let scale = bytes_per_unit(exponent);
        if typical >= scale {
            for v in values.iter_mut() {
                *v /= scale;
            }
            return unit;
        }
    }
    unreachable!("BYTE_UNITS contains B")
}

impl ValueFormatter for BytesFormatter {
    fn scale_values(&self, typical: f64, values: &mut [f64]) -> &'static str {
        scale_bytes(typical, values)
    }

    fn scale_throughputs(
        &self,
        typical: f64,
        _throughput: &criterion::Throughput,
        values: &mut [f64],
    ) -> &'static str {
        scale_bytes(typical, values)
    }

    fn scale_for_machines(&self, values: &mut [f64]) -> &'static str {
        // Machine-readable: always bytes
        let _ = values;
        "B"
    }
}

struct PeakMemory;

impl Measurement for PeakMemory {
    type Intermediate = usize;
    type Value = usize;

    fn start(&self) -> Self::Intermediate {
        reset_peak();
        live_bytes()
    }

    fn end(&self, baseline: Self::Intermediate) -> Self::Value {
        peak_bytes().saturating_sub(baseline)
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        *v1 + *v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &BytesFormatter
    }
}

struct AllocatedBytes;

impl Measurement for AllocatedBytes {
    type Intermediate = ();
    type Value = usize;

    fn start(&self) -> Self::Intermediate {
        reset_allocated();
    }

    fn end(&self, _baseline: Self::Intermediate) -> Self::Value {
        allocated_bytes()
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        *v1 + *v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &BytesFormatter
    }
}

// ---------------------------------------------------------------------------
// Test data generation (subset adapted from arrow_writer.rs, using only APIs
// that exist on both the pre- and post-native-write-path code so the same file
// can be checked out and run on either branch for an A/B comparison).
// ---------------------------------------------------------------------------

const BATCH_SIZE: usize = 1024 * 1024;

fn create_primitive_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Int32, true),
        Field::new("_2", DataType::Int64, true),
        Field::new("_3", DataType::UInt32, true),
        Field::new("_4", DataType::UInt64, true),
        Field::new("_5", DataType::Float32, true),
        Field::new("_6", DataType::Float64, true),
        Field::new("_7", DataType::Date64, true),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn create_primitive_bench_batch_non_null(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Int32, false),
        Field::new("_2", DataType::Int64, false),
        Field::new("_3", DataType::UInt32, false),
        Field::new("_4", DataType::UInt64, false),
        Field::new("_5", DataType::Float32, false),
        Field::new("_6", DataType::Float64, false),
        Field::new("_7", DataType::Date64, false),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn create_bool_bench_batch_non_null(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![Field::new("_1", DataType::Boolean, false)];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn create_string_bench_batch_non_null(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Utf8, false),
        Field::new("_2", DataType::LargeUtf8, false),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

/// High-null byte-array columns (Utf8 / LargeUtf8 / Utf8View / BinaryView).
fn create_byte_array_sparse_bench_batch(size: usize, null_density: f32) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Utf8, true),
        Field::new("_2", DataType::LargeUtf8, true),
        Field::new("_3", DataType::Utf8View, true),
        Field::new("_4", DataType::BinaryView, true),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        0.5,
    )?)
}

/// Many short, fixed-width strings — exercises the BYTE_ARRAY dense hot path.
fn create_short_string_bench_batch(size: usize) -> Result<RecordBatch> {
    let array = Arc::new(StringArray::from_iter_values(
        (0..size).map(|i| format!("{i:08}")),
    )) as _;
    Ok(RecordBatch::try_from_iter([("col", array)])?)
}

/// FixedSizeBinary column → FIXED_LEN_BYTE_ARRAY (raw FLBA path).
fn create_fixed_size_binary_bench_batch(
    size: usize,
    byte_width: i32,
    null_density: f32,
) -> Result<RecordBatch> {
    let width = byte_width as usize;
    let nulls = nulls_for_density(size, null_density);
    let array = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
        (0..size).map(|i| {
            nulls.as_ref().is_none_or(|n| n.is_valid(i)).then(|| {
                let mut v = vec![0u8; width];
                let b = (i as u64).to_le_bytes();
                let n = width.min(8);
                v[..n].copy_from_slice(&b[..n]);
                v
            })
        }),
        byte_width,
    )?;
    Ok(RecordBatch::try_from_iter([(
        "col",
        Arc::new(array) as ArrayRef,
    )])?)
}

/// Decimal128(38, 10) column → FIXED_LEN_BYTE_ARRAY(16) (converted FLBA path).
fn create_decimal128_bench_batch(size: usize, null_density: f32) -> Result<RecordBatch> {
    let values: Vec<i128> = (0..size).map(|i| i as i128).collect();
    let nulls = nulls_for_density(size, null_density);
    let array = Decimal128Array::new(values.into(), nulls).with_precision_and_scale(38, 10)?;
    Ok(RecordBatch::try_from_iter([(
        "col",
        Arc::new(array) as ArrayRef,
    )])?)
}

fn create_int64_dictionary_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    Ok(create_random_batch(
        dictionary_schema(DataType::Int64),
        size,
        null_density,
        true_density,
    )?)
}

fn create_int64_dictionary_bench_batch_1pct_cardinality(
    size: usize,
    null_density: f32,
) -> Result<RecordBatch> {
    let cardinality = dictionary_cardinality_1pct(size);
    let schema = dictionary_schema(DataType::Int64);
    let keys = dictionary_keys(size, cardinality, null_density);
    let values = Int64Array::from_iter_values((0..cardinality).map(|i| i as i64));
    let array = DictionaryArray::<Int32Type>::new(keys, Arc::new(values));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(array) as ArrayRef],
    )?)
}

fn create_string_dictionary_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![Field::new(
        "_1",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        true,
    )];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn create_string_dictionary_bench_batch_1pct_cardinality(
    size: usize,
    null_density: f32,
) -> Result<RecordBatch> {
    let cardinality = dictionary_cardinality_1pct(size);
    let schema = dictionary_schema(DataType::Utf8);
    let keys = dictionary_keys(size, cardinality, null_density);
    let values = StringArray::from_iter_values((0..cardinality).map(|i| format!("value_{i:08}")));
    let array = DictionaryArray::<Int32Type>::new(keys, Arc::new(values));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(array) as ArrayRef],
    )?)
}

/// A single run-end-encoded column with long runs of few distinct values —
/// REE's intended regime. `make_values` produces the per-run value array.
fn create_controlled_ree_batch(
    run_ends_type: DataType,
    size: usize,
    run_len: usize,
    make_values: impl Fn(usize) -> ArrayRef,
) -> RecordBatch {
    let num_runs = size.div_ceil(run_len);
    let mut acc = 0usize;
    let run_ends: Vec<i64> = (0..num_runs)
        .map(|_| {
            acc = (acc + run_len).min(size);
            acc as i64
        })
        .collect();
    let values = make_values(num_runs);
    let run_array: ArrayRef = match run_ends_type {
        DataType::Int32 => {
            let ends = Int32Array::from_iter_values(run_ends.iter().map(|&v| v as i32));
            Arc::new(RunArray::<Int32Type>::try_new(&ends, &values).unwrap())
        }
        DataType::Int64 => {
            let ends = Int64Array::from_iter_values(run_ends.iter().copied());
            Arc::new(RunArray::<Int64Type>::try_new(&ends, &values).unwrap())
        }
        other => panic!("unsupported REE run-ends type for bench: {other:?}"),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_1",
        run_array.data_type().clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![run_array]).unwrap()
}

fn dictionary_schema(value_type: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "_1",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(value_type)),
        true,
    )]))
}

fn dictionary_cardinality_1pct(size: usize) -> usize {
    (size / 100).max(1)
}

fn dictionary_keys(size: usize, cardinality: usize, null_density: f32) -> Int32Array {
    let keys = (0..size)
        .map(|i| (i % cardinality) as i32)
        .collect::<Vec<_>>();
    Int32Array::new(keys.into(), nulls_for_density(size, null_density))
}

fn nulls_for_density(size: usize, null_density: f32) -> Option<NullBuffer> {
    if null_density == 0. {
        return None;
    }
    let null_threshold = (null_density.clamp(0., 1.) * 10_000.) as usize;
    Some(NullBuffer::from(
        (0..size)
            .map(|i| (i * 9973) % 10_000 >= null_threshold)
            .collect::<Vec<_>>(),
    ))
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

/// Writer property variants to measure. Kept small: `default` (dictionary
/// enabled, parquet 1.0 encoders) and `parquet_2` (v2 / DELTA encoders, which
/// is where the converted-FLBA / decimal materialization lived).
fn writer_props() -> Vec<(&'static str, WriterProperties)> {
    vec![
        ("default", WriterProperties::builder().build()),
        (
            "parquet_2",
            WriterProperties::builder()
                .set_writer_version(WriterVersion::PARQUET_2_0)
                .build(),
        ),
    ]
}

/// Measure one `(batch, props)` cell. The `ArrowWriter` and the props clone are
/// built in the unmeasured `iter_batched` setup; only `write` + `close` (into a
/// discarding [`Empty`] sink) are inside the measured window — mirroring the
/// reader bench's `make_reader` (setup) / `drain_reader` (measured) split.
fn bench_cell<M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    prop_name: &str,
    batch: &RecordBatch,
    props: &WriterProperties,
) {
    group.bench_function(prop_name, |b| {
        b.iter_batched(
            || {
                ArrowWriter::try_new(Empty::default(), batch.schema(), Some(props.clone()))
                    .expect("writer construction")
            },
            |mut writer| {
                writer.write(batch).expect("write");
                writer.close().expect("close");
            },
            BatchSize::PerIteration,
        );
    });
}

/// Whether this parquet build can write RunEndEncoded columns at all. Pre-REE
/// builds error during schema conversion inside `ArrowWriter::try_new`, so probe
/// once and skip the REE groups entirely if unsupported (keeps the bench
/// runnable, and thus A/B-comparable, on both old and new code).
fn ree_write_supported() -> bool {
    let probe = create_controlled_ree_batch(DataType::Int32, 16, 4, |n| {
        Arc::new(Int32Array::from_iter_values((0..n).map(|i| i as i32)))
    });
    (|| -> Result<()> {
        let mut writer = ArrowWriter::try_new(Empty::default(), probe.schema(), None)?;
        writer.write(&probe)?;
        writer.close().map(|_metadata| ())
    })()
    .is_ok()
}

/// All batches that build *and* write on any branch (the A/B-comparable core).
fn ab_safe_batches() -> Vec<(&'static str, RecordBatch)> {
    vec![
        // Numeric primitives — the canonical dense-`Vec<T>` materialization.
        (
            "primitive_non_null",
            create_primitive_bench_batch_non_null(BATCH_SIZE, 0.0, 0.75).unwrap(),
        ),
        (
            "primitive_nullable",
            create_primitive_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap(),
        ),
        (
            "primitive_sparse_99pct_null",
            create_primitive_bench_batch(BATCH_SIZE, 0.99, 0.75).unwrap(),
        ),
        // Boolean — packed-bit path.
        (
            "bool_non_null",
            create_bool_bench_batch_non_null(BATCH_SIZE, 0.25, 0.75).unwrap(),
        ),
        // Byte arrays — the dense-`Vec<ByteArray>` materialization.
        (
            "string_non_null",
            create_string_bench_batch_non_null(BATCH_SIZE, 0.0, 0.75).unwrap(),
        ),
        (
            "short_string_non_null",
            create_short_string_bench_batch(BATCH_SIZE).unwrap(),
        ),
        (
            "byte_array_sparse_99pct_null",
            create_byte_array_sparse_bench_batch(BATCH_SIZE, 0.99).unwrap(),
        ),
        // Fixed-length byte arrays — raw + converted (decimal) FLBA paths.
        (
            "fixed_size_binary_non_null",
            create_fixed_size_binary_bench_batch(BATCH_SIZE, 16, 0.0).unwrap(),
        ),
        (
            "fixed_size_binary_sparse",
            create_fixed_size_binary_bench_batch(BATCH_SIZE, 16, 0.25).unwrap(),
        ),
        (
            "decimal128_non_null",
            create_decimal128_bench_batch(BATCH_SIZE, 0.0).unwrap(),
        ),
        (
            "decimal128_sparse",
            create_decimal128_bench_batch(BATCH_SIZE, 0.25).unwrap(),
        ),
        // Dictionaries — native dictionary adoption path.
        (
            "int64_dictionary",
            create_int64_dictionary_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap(),
        ),
        (
            "int64_dictionary_1pct_cardinality",
            create_int64_dictionary_bench_batch_1pct_cardinality(BATCH_SIZE, 0.25).unwrap(),
        ),
        (
            "string_dictionary",
            create_string_dictionary_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap(),
        ),
        (
            "string_dictionary_1pct_cardinality",
            create_string_dictionary_bench_batch_1pct_cardinality(BATCH_SIZE, 0.25).unwrap(),
        ),
    ]
}

/// Run-end-encoded batches (long runs, few distinct values). Only writable on
/// builds with REE write support; measured numbers are meaningless elsewhere.
fn ree_batches() -> Vec<(&'static str, RecordBatch)> {
    vec![
        (
            "string_ree_low_cardinality",
            create_controlled_ree_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
                Arc::new(StringArray::from_iter_values(
                    (0..n).map(|i| format!("category_{:02}", i % 16)),
                ))
            }),
        ),
        (
            "int64_ree_low_cardinality",
            create_controlled_ree_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
                Arc::new(Int64Array::from_iter_values(
                    (0..n).map(|i| (i % 16) as i64),
                ))
            }),
        ),
        (
            "fixed_size_binary_ree_low_cardinality",
            create_controlled_ree_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
                Arc::new(
                    FixedSizeBinaryArray::try_from_iter((0..n).map(|i| {
                        let mut b = [0u8; 16];
                        b[0] = (i % 16) as u8;
                        b
                    }))
                    .unwrap(),
                )
            }),
        ),
    ]
}

fn add_benches<M: Measurement>(c: &mut Criterion<M>, measurement_name: &str) {
    let props = writer_props();

    for (batch_name, batch) in ab_safe_batches() {
        let mut group = c.benchmark_group(format!("arrow_writer_{measurement_name}/{batch_name}"));
        for (prop_name, prop) in &props {
            bench_cell(&mut group, prop_name, &batch, prop);
        }
        group.finish();
    }

    // REE: default props only, and only on builds that can write REE columns
    // (probed above). On pre-REE builds these groups are skipped entirely.
    if ree_write_supported() {
        let (default_name, default_prop) = &props[0];
        for (batch_name, batch) in ree_batches() {
            let mut group =
                c.benchmark_group(format!("arrow_writer_{measurement_name}/{batch_name}"));
            bench_cell(&mut group, default_name, &batch, default_prop);
            group.finish();
        }
    }
}

fn add_peak_memory_benches(c: &mut Criterion<PeakMemory>) {
    add_benches(c, "peak_memory");
}

fn add_allocated_bytes_benches(c: &mut Criterion<AllocatedBytes>) {
    add_benches(c, "allocated_bytes");
}

fn peak_memory_criterion() -> Criterion<PeakMemory> {
    Criterion::default()
        .with_measurement(PeakMemory)
        .sample_size(10)
}

fn allocated_bytes_criterion() -> Criterion<AllocatedBytes> {
    Criterion::default()
        .with_measurement(AllocatedBytes)
        .sample_size(10)
}

criterion_group! {
    name = peak_memory;
    config = peak_memory_criterion();
    targets = add_peak_memory_benches
}

criterion_group! {
    name = cumulative_allocated_bytes;
    config = allocated_bytes_criterion();
    targets = add_allocated_bytes_benches
}
criterion_main!(peak_memory, cumulative_allocated_bytes);
