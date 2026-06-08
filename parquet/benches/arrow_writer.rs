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

#[macro_use]
extern crate criterion;

use criterion::{Bencher, Criterion, Throughput};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_writer::compute_leaves;
use parquet::basic::{Compression, ZstdLevel};

extern crate arrow;
extern crate parquet;

use std::cell::OnceCell;
use std::hint::black_box;
use std::io::Empty;
use std::sync::Arc;

use arrow::datatypes::*;
use arrow::util::bench_util::{create_f16_array, create_f32_array, create_f64_array};
use arrow::{record_batch::RecordBatch, util::data_gen::*};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, DictionaryArray, FixedSizeBinaryArray,
    Float64Array, Int8Array, Int32Array, Int64Array, ListArray, RecordBatchOptions, RunArray,
    StringArray, StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use parquet::errors::Result;
use parquet::file::properties::{CdcOptions, WriterProperties, WriterVersion};

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

fn create_primitive_dictionary_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![Field::new(
        "_1",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int32)),
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

fn create_primitive_dictionary_bench_batch_1pct_cardinality(
    size: usize,
    null_density: f32,
) -> Result<RecordBatch> {
    let cardinality = dictionary_cardinality_1pct(size);
    let schema = dictionary_schema(DataType::Int32);
    let keys = dictionary_keys(size, cardinality, null_density);
    let values = Int32Array::from_iter_values((0..cardinality).map(|i| i as i32));
    let array = DictionaryArray::<Int32Type>::new(keys, Arc::new(values));

    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(array) as ArrayRef],
    )?)
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

fn create_float64_dictionary_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    Ok(create_random_batch(
        dictionary_schema(DataType::Float64),
        size,
        null_density,
        true_density,
    )?)
}

fn create_float64_dictionary_bench_batch_1pct_cardinality(
    size: usize,
    null_density: f32,
) -> Result<RecordBatch> {
    let cardinality = dictionary_cardinality_1pct(size);
    let schema = dictionary_schema(DataType::Float64);
    let keys = dictionary_keys(size, cardinality, null_density);
    let values = Float64Array::from_iter_values((0..cardinality).map(|i| i as f64));
    let array = DictionaryArray::<Int32Type>::new(keys, Arc::new(values));

    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(array) as ArrayRef],
    )?)
}

fn create_string_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Utf8, true),
        Field::new("_2", DataType::LargeUtf8, true),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

/// 1 M short, fixed-width 8-byte strings. Exercises the BYTE_ARRAY hot path
/// for the case where individual values are small enough that the byte-budget
/// based sub-batch sizing in `write_batch_internal` should always resolve to
/// the full chunk (no granular splitting, no regression vs. current behavior).
fn create_short_string_bench_batch(size: usize) -> Result<RecordBatch> {
    let array = Arc::new(StringArray::from_iter_values(
        (0..size).map(|i| format!("{i:08}")),
    )) as _;
    Ok(RecordBatch::try_from_iter([("col", array)])?)
}

/// `size` rows of `value_size`-byte strings. Exercises the BYTE_ARRAY path
/// where individual values are large enough that batching the default
/// `write_batch_size` of them would blow the page byte limit by orders of
/// magnitude — the case the page-size fix targets.
fn create_large_string_bench_batch(size: usize, value_size: usize) -> Result<RecordBatch> {
    let value = "x".repeat(value_size);
    let array = Arc::new(StringArray::from_iter_values(
        (0..size).map(|_| value.as_str()),
    )) as _;
    Ok(RecordBatch::try_from_iter([("col", array)])?)
}

fn create_string_and_binary_view_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Utf8View, true),
        Field::new("_2", DataType::BinaryView, true),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
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

/// Run-end-encoded column whose run VALUES are a low-cardinality
/// `Dictionary<Int32, Utf8>` (run length 8 -> ~size/8 runs, 16 distinct dict
/// entries). Exercises the REE-of-dictionary path: native adoption vs the dense
/// `take` expansion it replaced.
fn create_ree_of_dict_bench_batch(size: usize) -> RecordBatch {
    let run_len = 8usize;
    let num_runs = size / run_len;
    let run_ends = Int32Array::from(
        (1..=num_runs)
            .map(|i| (i * run_len) as i32)
            .collect::<Vec<_>>(),
    );
    let dict_values = StringArray::from((0..16).map(|i| format!("val_{i:04}")).collect::<Vec<_>>());
    let keys = Int32Array::from((0..num_runs).map(|i| (i % 16) as i32).collect::<Vec<_>>());
    let dict = DictionaryArray::<Int32Type>::new(keys, Arc::new(dict_values));
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &dict).unwrap();
    let field = Field::new("_1", ree.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![Arc::new(ree)]).unwrap()
}

/// `RunArray<Int32, Dictionary<Int32, Int64>>` — a run-end column whose values
/// are a *sized* numeric dictionary. Exercises the shallow-decode path (O(runs)
/// dictionary decode onto the native flat numeric leaf) versus the O(rows) dense
/// expand it replaces.
fn create_ree_of_numeric_dict_bench_batch(size: usize) -> RecordBatch {
    let run_len = 8usize;
    let num_runs = size / run_len;
    let run_ends = Int32Array::from(
        (1..=num_runs)
            .map(|i| (i * run_len) as i32)
            .collect::<Vec<_>>(),
    );
    let dict_values = Int64Array::from((0..16).map(|i| i as i64 * 1_000_000).collect::<Vec<_>>());
    let keys = Int32Array::from((0..num_runs).map(|i| (i % 16) as i32).collect::<Vec<_>>());
    let dict = DictionaryArray::<Int32Type>::new(keys, Arc::new(dict_values));
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &dict).unwrap();
    let field = Field::new("_1", ree.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![Arc::new(ree)]).unwrap()
}

/// `RunArray<Int32, Struct<{tag: Dictionary<Int32,Utf8>, val: Int64}>>` — a
/// run-encoded record with a rep-free dictionary field. Exercises the fan-out
/// shallow-decode path (decode the dictionary child at run granularity + shred
/// a flat leaf) versus the O(rows) dense expand it replaces.
fn create_ree_struct_of_dict_bench_batch(size: usize) -> RecordBatch {
    let run_len = 8usize;
    let num_runs = size / run_len;
    let run_ends = Int32Array::from(
        (1..=num_runs)
            .map(|i| (i * run_len) as i32)
            .collect::<Vec<_>>(),
    );
    let dict_values = StringArray::from((0..16).map(|i| format!("tag_{i:04}")).collect::<Vec<_>>());
    let keys = Int32Array::from((0..num_runs).map(|i| (i % 16) as i32).collect::<Vec<_>>());
    let tag = DictionaryArray::<Int32Type>::new(keys, Arc::new(dict_values));
    let val = Int64Array::from((0..num_runs).map(|i| i as i64).collect::<Vec<_>>());
    let fields = Fields::from(vec![
        Field::new("tag", tag.data_type().clone(), true),
        Field::new("val", DataType::Int64, true),
    ]);
    let values = StructArray::new(fields, vec![Arc::new(tag), Arc::new(val)], None);
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();
    let field = Field::new("_1", ree.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![Arc::new(ree)]).unwrap()
}

fn create_ree_bench_batch(
    value_dt: DataType,
    size: usize,
    null_pct: Option<u8>,
    true_density: f32,
) -> Result<RecordBatch> {
    const DEFAULT_NULL_PCT: u8 = 10;
    let null_density = null_pct.unwrap_or(DEFAULT_NULL_PCT) as f32 / 100.0;
    let fields = vec![Field::new(
        "_1",
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(Field::new("values", value_dt, true)),
        ),
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

/// A single run-end-encoded column with configurable run length and run-end
/// index width. `make_values` produces one value per run, allowing benchmarks
/// to vary value type and distinct-value cardinality.
fn create_run_end_encoded_bench_batch(
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

/// Struct-of-leaves run values for the non-leaf REE benches: 16 distinct
/// (int64, utf8) pairs cycled over `n` runs, one in eight struct-null.
fn make_ree_struct_values(n: usize) -> ArrayRef {
    let a = Int64Array::from_iter_values((0..n).map(|i| (i % 16) as i64));
    let b = StringArray::from_iter_values((0..n).map(|i| format!("category_{:02}", i % 16)));
    let fields = Fields::from(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Utf8, true),
    ]);
    let nulls = NullBuffer::from_iter((0..n).map(|i| i % 8 != 7));
    Arc::new(StructArray::new(
        fields,
        vec![Arc::new(a), Arc::new(b)],
        Some(nulls),
    ))
}

/// String-list run values: each run value is a list of 4 strings drawn from
/// 16 distinct lists.
fn make_ree_string_list_values(n: usize) -> ArrayRef {
    let values = StringArray::from_iter_values(
        (0..n * 4).map(|i| format!("category_{:02}_{}", (i / 4) % 16, i % 4)),
    );
    let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(4usize, n));
    let field = Arc::new(Field::new_list_field(DataType::Utf8, true));
    Arc::new(ListArray::new(field, offsets, Arc::new(values), None))
}

/// A batch of `size` lists of 4 elements over a run-end-encoded child with
/// 256-row runs (16 distinct values, one run in eight null). Without list
/// nulls this takes the fused list-of-runs route. `null_every` creates fine
/// null/valid spans for exercising the profitability gate.
fn create_list_of_int32_ree_batch(size: usize, null_every: Option<usize>) -> RecordBatch {
    let child_len = size * 4;
    let num_runs = child_len.div_ceil(256);
    let run_ends = Int32Array::from_iter_values(
        (0..num_runs).map(|i| (((i + 1) * 256).min(child_len)) as i32),
    );
    let values =
        Int32Array::from_iter((0..num_runs).map(|i| (i % 8 != 7).then_some((i % 16) as i32)));
    let ree: ArrayRef = Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap());
    let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(4usize, size));
    let field = Arc::new(Field::new_list_field(ree.data_type().clone(), true));
    let nulls = null_every.map(|n| NullBuffer::from_iter((0..size).map(|i| i % n != n - 1)));
    let list: ArrayRef = Arc::new(ListArray::new(field, offsets, ree, nulls));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_1",
        list.data_type().clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![list]).unwrap()
}

/// A list-of-runs batch whose rows alternate between empty and four values.
/// Every non-empty row is isolated, presenting the profitability gate with the
/// maximum row-kind transition rate without involving list nullability.
fn create_list_of_int32_ree_alternating_empty_batch(size: usize) -> RecordBatch {
    let child_len = size / 2 * 4;
    let num_runs = child_len.div_ceil(256);
    let run_ends = Int32Array::from_iter_values(
        (0..num_runs).map(|i| (((i + 1) * 256).min(child_len)) as i32),
    );
    let values =
        Int32Array::from_iter((0..num_runs).map(|i| (i % 8 != 7).then_some((i % 16) as i32)));
    let ree: ArrayRef = Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap());
    let offsets = OffsetBuffer::from_lengths((0..size).map(|i| if i % 2 == 0 { 0 } else { 4 }));
    let field = Arc::new(Field::new_list_field(ree.data_type().clone(), true));
    let list: ArrayRef = Arc::new(ListArray::new(field, offsets, ree, None));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_1",
        list.data_type().clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![list]).unwrap()
}

/// A uniform list-of-runs column under a nullable struct. Parent validity can
/// divide the child into multiple write ranges even though the list itself has
/// no null or empty rows.
fn create_struct_of_list_of_int32_ree_batch(size: usize, nulls: NullBuffer) -> RecordBatch {
    let list_batch = create_list_of_int32_ree_batch(size, None);
    let list = list_batch.column(0).clone();
    let fields = Fields::from(vec![Field::new("items", list.data_type().clone(), true)]);
    let structs: ArrayRef = Arc::new(StructArray::new(fields, vec![list], Some(nulls)));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_1",
        structs.data_type().clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![structs]).unwrap()
}

/// A batch of `size` outer lists of 2 elements over a run-end-encoded child
/// (128-element runs) whose run values are inner lists of 4 ints — block
/// replay under the outer list with first-slot repetition overrides.
fn create_list_of_list_int32_ree_batch(size: usize) -> RecordBatch {
    let child_len = size * 2;
    let num_runs = child_len.div_ceil(128);
    let run_ends = Int32Array::from_iter_values(
        (0..num_runs).map(|i| (((i + 1) * 128).min(child_len)) as i32),
    );
    let inner_values =
        Int32Array::from_iter_values((0..num_runs * 4).map(|i| ((i / 4) % 16 * 4 + i % 4) as i32));
    let inner_offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(4usize, num_runs));
    let inner_field = Arc::new(Field::new_list_field(DataType::Int32, true));
    let inner = ListArray::new(inner_field, inner_offsets, Arc::new(inner_values), None);
    let ree: ArrayRef = Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &inner).unwrap());
    let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(2usize, size));
    let field = Arc::new(Field::new_list_field(ree.data_type().clone(), true));
    let list: ArrayRef = Arc::new(ListArray::new(field, offsets, ree, None));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_1",
        list.data_type().clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![list]).unwrap()
}

/// `RunEndEncoded<List<RunEndEncoded<List<Int32>>>>` with 256-row outer runs,
/// two inner entries per physical outer value, and 128-element inner runs.
fn create_ree_list_ree_list_batch(size: usize) -> RecordBatch {
    let outer_runs = size.div_ceil(256);
    let inner_len = outer_runs * 2;
    let inner_runs = inner_len.div_ceil(128);

    let values = Int32Array::from_iter_values(
        (0..inner_runs * 4).map(|i| ((i / 4) % 16 * 4 + i % 4) as i32),
    );
    let value_offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(4usize, inner_runs));
    let value_field = Arc::new(Field::new_list_field(DataType::Int32, true));
    let value_lists = ListArray::new(value_field, value_offsets, Arc::new(values), None);

    let inner_ends = Int32Array::from_iter_values(
        (0..inner_runs).map(|i| (((i + 1) * 128).min(inner_len)) as i32),
    );
    let inner: ArrayRef =
        Arc::new(RunArray::<Int32Type>::try_new(&inner_ends, &value_lists).unwrap());
    let outer_value_offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(2usize, outer_runs));
    let outer_value_field = Arc::new(Field::new_list_field(inner.data_type().clone(), true));
    let outer_values: ArrayRef = Arc::new(ListArray::new(
        outer_value_field,
        outer_value_offsets,
        inner,
        None,
    ));

    let outer_ends =
        Int32Array::from_iter_values((0..outer_runs).map(|i| (((i + 1) * 256).min(size)) as i32));
    let array: ArrayRef =
        Arc::new(RunArray::<Int32Type>::try_new(&outer_ends, &outer_values).unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new(
        "_1",
        array.data_type().clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

/// List run values for the non-leaf REE benches: each run value is a list of
/// 8 ints drawn from 16 distinct lists.
fn make_ree_list_values(n: usize) -> ArrayRef {
    let values =
        Int32Array::from_iter_values((0..n * 8).map(|i| ((i / 8) % 16 * 8 + i % 8) as i32));
    let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(8usize, n));
    let field = Arc::new(Field::new_list_field(DataType::Int32, true));
    Arc::new(ListArray::new(field, offsets, Arc::new(values), None))
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

fn create_bool_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![Field::new("_1", DataType::Boolean, true)];
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

fn create_float_bench_batch_with_nans(size: usize, nan_density: f32) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("_1", DataType::Float16, false),
        Field::new("_2", DataType::Float32, false),
        Field::new("_3", DataType::Float64, false),
    ];
    let schema = Schema::new(fields);
    let columns: Vec<arrow_array::ArrayRef> = vec![
        Arc::new(create_f16_array(size, nan_density)),
        Arc::new(create_f32_array(size, nan_density)),
        Arc::new(create_f64_array(size, nan_density)),
    ];
    Ok(RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &RecordBatchOptions::new().with_match_field_names(false),
    )?)
}

fn create_list_primitive_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new(
            "_1",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            true,
        ),
        Field::new(
            "_2",
            DataType::List(Arc::new(Field::new_list_field(DataType::Boolean, true))),
            true,
        ),
        Field::new(
            "_3",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Utf8, true))),
            true,
        ),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn create_list_primitive_bench_batch_non_null(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new(
            "_1",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, false))),
            false,
        ),
        Field::new(
            "_2",
            DataType::List(Arc::new(Field::new_list_field(DataType::Boolean, false))),
            false,
        ),
        Field::new(
            "_3",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Utf8, false))),
            false,
        ),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn create_struct_bench_batch(size: usize, null_density: f32) -> Result<RecordBatch> {
    let fields = vec![Field::new(
        "_1",
        DataType::Struct(Fields::from(vec![
            Field::new("_1", DataType::Int32, false),
            Field::new("_2", DataType::Int64, false),
            Field::new("_3", DataType::Float32, false),
        ])),
        true,
    )];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        0.75,
    )?)
}

fn create_nested_list_bench_batch(size: usize, null_density: f32) -> Result<RecordBatch> {
    // List<List<Int32>> — exercises the nested repetition (non-batched) path
    let fields = vec![Field::new(
        "_1",
        DataType::List(Arc::new(Field::new_list_field(
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            true,
        ))),
        true,
    )];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        0.75,
    )?)
}

fn create_list_struct_with_list_batch(size: usize, null_density: f32) -> Result<RecordBatch> {
    // List<Struct<a:Int32, b:Float32, c:List<Int32>>>
    // The struct child contains a nested list, so child_has_no_nested_rep() = false.
    // This exercises the per-slot (non-batched) write path in level computation.
    let fields = vec![Field::new(
        "_1",
        DataType::List(Arc::new(Field::new_list_field(
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Float32, true),
                Field::new(
                    "c",
                    DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                    true,
                ),
            ])),
            true,
        ))),
        true,
    )];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        0.75,
    )?)
}

fn _create_nested_bench_batch(
    size: usize,
    null_density: f32,
    true_density: f32,
) -> Result<RecordBatch> {
    let fields = vec![
        Field::new(
            "_1",
            DataType::Struct(Fields::from(vec![
                Field::new("_1", DataType::Int8, true),
                Field::new(
                    "_2",
                    DataType::Struct(Fields::from(vec![
                        Field::new("_1", DataType::Int8, true),
                        Field::new(
                            "_1",
                            DataType::Struct(Fields::from(vec![
                                Field::new("_1", DataType::Int8, true),
                                Field::new("_2", DataType::Utf8, true),
                            ])),
                            true,
                        ),
                        Field::new("_2", DataType::UInt8, true),
                    ])),
                    true,
                ),
            ])),
            true,
        ),
        Field::new(
            "_2",
            DataType::LargeList(Arc::new(Field::new_list_field(
                DataType::List(Arc::new(Field::new_list_field(
                    DataType::Struct(Fields::from(vec![
                        Field::new(
                            "_1",
                            DataType::Struct(Fields::from(vec![
                                Field::new("_1", DataType::Int8, true),
                                Field::new("_2", DataType::Int16, true),
                                Field::new("_3", DataType::Int32, true),
                            ])),
                            true,
                        ),
                        Field::new(
                            "_2",
                            DataType::List(Arc::new(Field::new(
                                "",
                                DataType::FixedSizeBinary(2),
                                true,
                            ))),
                            true,
                        ),
                    ])),
                    true,
                ))),
                true,
            ))),
            true,
        ),
    ];
    let schema = Schema::new(fields);
    Ok(create_random_batch(
        Arc::new(schema),
        size,
        null_density,
        true_density,
    )?)
}

fn write_batch_with_option(
    bench: &mut Bencher,
    batch: &RecordBatch,
    props: Option<WriterProperties>,
) -> Result<()> {
    let props = props.unwrap_or_default();

    bench.iter(|| {
        let mut file = Empty::default();
        let mut writer =
            ArrowWriter::try_new(&mut file, batch.schema(), Some(props.clone())).unwrap();
        writer.write(black_box(batch)).unwrap();
        black_box(writer.close()).unwrap();
    });

    Ok(())
}

/// High-null byte-array columns (Utf8 / LargeUtf8 / Utf8View / BinaryView).
/// Isolates sparse/nullable byte-array writes through `write_byte_values`.
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

/// FixedSizeBinary column → FIXED_LEN_BYTE_ARRAY. Exercises the raw FLBA path:
/// dense (`null_density == 0`) hits the contiguous raw-byte fast path; sparse
/// hits the selected per-value path.
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

/// Decimal128(38, 10) column → FIXED_LEN_BYTE_ARRAY(16) (precision 38 forces
/// FLBA). Exercises the *converted* FLBA path (i128 → big-endian fixed bytes):
/// dense hits the bulk conversion, sparse hits the per-value converted stream.
fn create_decimal128_bench_batch(size: usize, null_density: f32) -> Result<RecordBatch> {
    let values: Vec<i128> = (0..size).map(|i| i as i128).collect();
    let nulls = nulls_for_density(size, null_density);
    let array = Decimal128Array::new(values.into(), nulls).with_precision_and_scale(38, 10)?;
    Ok(RecordBatch::try_from_iter([(
        "col",
        Arc::new(array) as ArrayRef,
    )])?)
}

/// Nullable primitive with deterministic alternating validity runs. This
/// isolates compact level/range planning without random-data construction or
/// value-encoding variability.
fn create_nullable_runs_bench_batch(size: usize, run_len: usize) -> RecordBatch {
    assert!(run_len > 0);
    RecordBatch::try_from_iter([(
        "col",
        Arc::new(Int32Array::from_iter(
            (0..size).map(|idx| ((idx / run_len) % 2 == 0).then_some(idx as i32)),
        )) as ArrayRef,
    )])
    .unwrap()
}

struct BatchBenchmark {
    name: &'static str,
    logical_rows: u64,
    make_batch: Box<dyn Fn() -> RecordBatch>,
}

impl BatchBenchmark {
    fn new(
        name: &'static str,
        logical_rows: usize,
        make_batch: impl Fn() -> RecordBatch + 'static,
    ) -> Self {
        Self {
            name,
            logical_rows: logical_rows as u64,
            make_batch: Box::new(make_batch),
        }
    }
}

fn create_batches() -> Vec<BatchBenchmark> {
    const BATCH_SIZE: usize = 1024 * 1024;

    let mut batches = vec![];

    macro_rules! push_batch {
        ($name:expr, $batch:expr) => {
            batches.push(BatchBenchmark::new($name, BATCH_SIZE, || $batch));
        };
        ($name:expr, $logical_rows:expr, $batch:expr) => {
            batches.push(BatchBenchmark::new($name, $logical_rows, || $batch));
        };
    }

    push_batch!(
        "primitive",
        create_primitive_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "primitive_non_null",
        create_primitive_bench_batch_non_null(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "primitive_dictionary",
        create_primitive_dictionary_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "primitive_dictionary_1pct_cardinality",
        create_primitive_dictionary_bench_batch_1pct_cardinality(BATCH_SIZE, 0.25).unwrap()
    );

    push_batch!(
        "int64_dictionary",
        create_int64_dictionary_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "int64_dictionary_1pct_cardinality",
        create_int64_dictionary_bench_batch_1pct_cardinality(BATCH_SIZE, 0.25).unwrap()
    );

    push_batch!(
        "float64_dictionary",
        create_float64_dictionary_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "float64_dictionary_1pct_cardinality",
        create_float64_dictionary_bench_batch_1pct_cardinality(BATCH_SIZE, 0.25).unwrap()
    );

    push_batch!(
        "bool",
        create_bool_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "bool_non_null",
        create_bool_bench_batch_non_null(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    // Dictionary-encoded booleans: exercises the bool-dictionary write path
    // (materialize-vs-native decision).
    push_batch!("bool_dictionary", {
        let keys = Int8Array::from_iter((0..BATCH_SIZE).map(|i| {
            (i % 10 != 9).then_some((i % 2) as i8) // 10% null keys
        }));
        let values = BooleanArray::from(vec![false, true]);
        let dict: ArrayRef = Arc::new(DictionaryArray::new(keys, Arc::new(values) as ArrayRef));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "_1",
            dict.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![dict]).unwrap()
    });

    push_batch!(
        "string",
        create_string_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "short_string_non_null",
        create_short_string_bench_batch(BATCH_SIZE).unwrap()
    );

    // 1024 rows × 256 KiB = 256 MiB total. With the default 1 MiB page byte
    // limit, this is the case where the page-size fix kicks in: each value
    // needs its own page, and `write_batch_size = 1024` would otherwise
    // buffer all 256 MiB before the post-write check runs.
    push_batch!(
        "large_string_non_null",
        1024,
        create_large_string_bench_batch(1024, 256 * 1024).unwrap()
    );

    push_batch!(
        "string_and_binary_view",
        create_string_and_binary_view_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "string_dictionary",
        create_string_dictionary_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "string_dictionary_1pct_cardinality",
        create_string_dictionary_bench_batch_1pct_cardinality(BATCH_SIZE, 0.25).unwrap()
    );

    push_batch!(
        "string_non_null",
        create_string_bench_batch_non_null(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    // Run-end-encoded — high-cardinality / short-run regime (random runs).
    push_batch!(
        "string_ree",
        create_ree_bench_batch(DataType::Utf8, BATCH_SIZE, None, 0.75).unwrap()
    );

    push_batch!(
        "int32_ree",
        create_ree_bench_batch(DataType::Int32, BATCH_SIZE, None, 0.75).unwrap()
    );

    push_batch!(
        "string_ree_of_dict",
        create_ree_of_dict_bench_batch(BATCH_SIZE)
    );

    push_batch!(
        "numeric_ree_of_dict",
        create_ree_of_numeric_dict_bench_batch(BATCH_SIZE)
    );

    push_batch!(
        "ree_struct_of_dict",
        create_ree_struct_of_dict_bench_batch(BATCH_SIZE)
    );

    push_batch!(
        "bool_ree",
        create_ree_bench_batch(DataType::Boolean, BATCH_SIZE, None, 0.75).unwrap()
    );

    push_batch!(
        "fixed_size_binary_ree",
        create_ree_bench_batch(DataType::FixedSizeBinary(16), BATCH_SIZE, None, 0.75).unwrap()
    );

    push_batch!(
        "string_ree_95pct_null",
        create_ree_bench_batch(DataType::Utf8, BATCH_SIZE, Some(95), 0.75).unwrap()
    );

    push_batch!(
        "int32_ree_95pct_null",
        create_ree_bench_batch(DataType::Int32, BATCH_SIZE, Some(95), 0.75).unwrap()
    );

    // Run-end-encoded batches with 256-row runs and 16 distinct values, across
    // value families and run-end widths.
    push_batch!(
        "string_ree_low_cardinality",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("category_{:02}", i % 16)),
            ))
        })
    );
    // String run values with Int64 run ends.
    push_batch!(
        "string_ree_int64_run_ends",
        create_run_end_encoded_bench_batch(DataType::Int64, BATCH_SIZE, 256, |n| {
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("category_{:02}", i % 16)),
            ))
        })
    );
    // Numeric (Int64) run values.
    push_batch!(
        "int64_ree_low_cardinality",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
            Arc::new(Int64Array::from_iter_values(
                (0..n).map(|i| (i % 16) as i64),
            ))
        })
    );
    // Fixed-length byte-array run values (FLBA path).
    push_batch!(
        "fixed_size_binary_ree_low_cardinality",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
            Arc::new(
                FixedSizeBinaryArray::try_from_iter((0..n).map(|i| {
                    let mut b = [0u8; 16];
                    b[0] = (i % 16) as u8;
                    b
                }))
                .unwrap(),
            )
        })
    );

    // Non-leaf run values: struct-of-leaves (run fan-out path), long runs.
    push_batch!(
        "struct_ree",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
            make_ree_struct_values(n)
        })
    );
    // Non-leaf struct run values with short runs (per-run overhead regime).
    push_batch!(
        "struct_ree_short_runs",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 8, |n| {
            make_ree_struct_values(n)
        })
    );
    // Non-leaf run values: each run value is a list of 8 ints (block regime).
    push_batch!(
        "list_int32_ree",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
            make_ree_list_values(n)
        })
    );
    push_batch!(
        "list_int32_ree_short_runs",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 8, |n| {
            make_ree_list_values(n)
        })
    );
    // Byte-array leaves under replayed list blocks (dictionary index replay).
    push_batch!(
        "list_string_ree",
        create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
            make_ree_string_list_values(n)
        })
    );
    // Lists of REE-of-lists (block replay under the outer list, with
    // first-slot repetition overrides).
    push_batch!(
        "list_of_list_int32_ree",
        create_list_of_list_int32_ree_batch(BATCH_SIZE)
    );
    push_batch!(
        "ree_list_ree_list",
        create_ree_list_ree_list_batch(BATCH_SIZE)
    );
    // A list whose values child is run-end encoded (the fused list-of-runs
    // node).
    push_batch!(
        "list_of_int32_ree",
        create_list_of_int32_ree_batch(BATCH_SIZE, None)
    );
    // Alternating null/valid rows select the generic list plan with a native
    // REE leaf instead of a highly segmented fused plan.
    push_batch!(
        "list_of_int32_ree_alt_null",
        create_list_of_int32_ree_batch(BATCH_SIZE, Some(2))
    );
    push_batch!(
        "list_of_int32_ree_alt_empty",
        create_list_of_int32_ree_alternating_empty_batch(BATCH_SIZE)
    );
    push_batch!(
        "struct_of_list_of_int32_ree_alt_null",
        create_struct_of_list_of_int32_ree_batch(
            BATCH_SIZE,
            NullBuffer::from_iter((0..BATCH_SIZE).map(|i| i % 2 == 0)),
        )
    );
    push_batch!(
        "struct_of_list_of_int32_ree_single_null",
        create_struct_of_list_of_int32_ree_batch(
            BATCH_SIZE,
            NullBuffer::from_iter((0..BATCH_SIZE).map(|i| i != BATCH_SIZE / 2)),
        )
    );
    // Ordinary batches at both edges force finalization to append the open
    // tail to four already-completed segments.
    push_batch!(
        "struct_of_list_of_int32_ree_fragmented",
        create_struct_of_list_of_int32_ree_batch(
            BATCH_SIZE,
            NullBuffer::from_iter(
                (0..BATCH_SIZE).map(|i| i != 0 && i != BATCH_SIZE / 2 && i + 1 != BATCH_SIZE),
            ),
        )
    );

    push_batch!(
        "float_with_nans",
        create_float_bench_batch_with_nans(BATCH_SIZE, 0.5).unwrap()
    );

    push_batch!(
        "list_primitive",
        create_list_primitive_bench_batch(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "list_primitive_non_null",
        create_list_primitive_bench_batch_non_null(BATCH_SIZE, 0.25, 0.75).unwrap()
    );

    push_batch!(
        "primitive_sparse_99pct_null",
        create_primitive_bench_batch(BATCH_SIZE, 0.99, 0.75).unwrap()
    );

    push_batch!(
        "list_primitive_sparse_99pct_null",
        create_list_primitive_bench_batch(BATCH_SIZE, 0.99, 0.75).unwrap()
    );

    push_batch!(
        "primitive_all_null",
        create_primitive_bench_batch(BATCH_SIZE, 1.0, 0.75).unwrap()
    );

    push_batch!(
        "struct_non_null",
        create_struct_bench_batch(BATCH_SIZE, 0.0).unwrap()
    );

    push_batch!(
        "struct_sparse_99pct_null",
        create_struct_bench_batch(BATCH_SIZE, 0.99).unwrap()
    );

    push_batch!(
        "struct_all_null",
        create_struct_bench_batch(BATCH_SIZE, 1.0).unwrap()
    );

    push_batch!(
        "list_nested",
        create_nested_list_bench_batch(BATCH_SIZE, 0.25).unwrap()
    );

    push_batch!(
        "list_struct_with_list",
        create_list_struct_with_list_batch(BATCH_SIZE, 0.25).unwrap()
    );

    push_batch!(
        "byte_array_sparse_99pct_null",
        create_byte_array_sparse_bench_batch(BATCH_SIZE, 0.99).unwrap()
    );

    push_batch!(
        "fixed_size_binary_non_null",
        create_fixed_size_binary_bench_batch(BATCH_SIZE, 16, 0.0).unwrap()
    );

    push_batch!(
        "fixed_size_binary_sparse",
        create_fixed_size_binary_bench_batch(BATCH_SIZE, 16, 0.25).unwrap()
    );

    push_batch!(
        "decimal128_non_null",
        create_decimal128_bench_batch(BATCH_SIZE, 0.0).unwrap()
    );

    push_batch!(
        "decimal128_sparse",
        create_decimal128_bench_batch(BATCH_SIZE, 0.25).unwrap()
    );

    batches
}

fn create_writer_props() -> Vec<(&'static str, WriterProperties)> {
    let mut props = vec![];

    props.push(("default", Default::default()));

    let prop = WriterProperties::builder()
        .set_bloom_filter_enabled(true)
        .build();
    props.push(("bloom_filter", prop));

    let prop = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .build();
    props.push(("parquet_2", prop));

    let prop = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    props.push(("zstd", prop));

    let prop = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .build();
    props.push(("zstd_parquet_2", prop));

    let prop = WriterProperties::builder()
        .set_content_defined_chunking(Some(CdcOptions::default()))
        .build();
    props.push(("cdc", prop));

    props
}

fn bench_all_writers(c: &mut Criterion) {
    let batches = create_batches();
    let props = create_writer_props();

    for benchmark in &batches {
        let mut group = c.benchmark_group(benchmark.name);

        // Arrow's physical buffer size is not comparable between a dense
        // array and its run-end encoded equivalent. Logical rows are stable
        // across representations and don't require eagerly building input.
        group.throughput(Throughput::Elements(benchmark.logical_rows));

        // Criterion doesn't invoke a benchmark closure for a filtered-out
        // ID. Once any property is selected, retain one input for the whole
        // workload group without including its construction in `b.iter`.
        let batch = OnceCell::new();
        for (prop_name, prop) in &props {
            group.bench_function(*prop_name, |b| {
                let batch = batch.get_or_init(|| (benchmark.make_batch)());
                write_batch_with_option(b, batch, Some(prop.clone())).unwrap()
            });
        }
        group.finish();
    }
}

fn bench_level_planning(c: &mut Criterion) {
    const BATCH_SIZE: usize = 1024 * 1024;

    let batches = [
        BatchBenchmark::new("simple_nullable", BATCH_SIZE, || {
            RecordBatch::try_from_iter([(
                "col",
                Arc::new(Int32Array::from_iter(
                    (0..BATCH_SIZE).map(|i| (i % 8 != 7).then_some(i as i32)),
                )) as ArrayRef,
            )])
            .unwrap()
        }),
        BatchBenchmark::new("nullable_runs_64", BATCH_SIZE, || {
            create_nullable_runs_bench_batch(BATCH_SIZE, 64)
        }),
        BatchBenchmark::new("nested_list", BATCH_SIZE, || {
            create_nested_list_bench_batch(BATCH_SIZE, 0.25).unwrap()
        }),
        BatchBenchmark::new("ree_leaf", BATCH_SIZE, || {
            create_run_end_encoded_bench_batch(DataType::Int32, BATCH_SIZE, 256, |n| {
                Arc::new(Int64Array::from_iter_values(
                    (0..n).map(|i| (i % 16) as i64),
                ))
            })
        }),
        BatchBenchmark::new("list_of_ree_fused", BATCH_SIZE, || {
            create_list_of_int32_ree_batch(BATCH_SIZE, None)
        }),
        BatchBenchmark::new("list_of_ree_adversarial", BATCH_SIZE, || {
            create_list_of_int32_ree_alternating_empty_batch(BATCH_SIZE)
        }),
        BatchBenchmark::new("list_of_ree_fragmented", BATCH_SIZE, || {
            create_struct_of_list_of_int32_ree_batch(
                BATCH_SIZE,
                NullBuffer::from_iter(
                    (0..BATCH_SIZE).map(|i| i != 0 && i != BATCH_SIZE / 2 && i + 1 != BATCH_SIZE),
                ),
            )
        }),
    ];

    let mut group = c.benchmark_group("level_planning");
    for benchmark in &batches {
        group.throughput(Throughput::Elements(benchmark.logical_rows));
        let batch = OnceCell::new();
        group.bench_function(benchmark.name, |b| {
            let batch = batch.get_or_init(|| (benchmark.make_batch)());
            let schema = batch.schema();
            b.iter(|| {
                for (field, array) in schema.fields().iter().zip(batch.columns()) {
                    let leaves =
                        compute_leaves(black_box(field.as_ref()), black_box(array)).unwrap();
                    black_box(leaves);
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_all_writers, bench_level_planning);
criterion_main!(benches);
