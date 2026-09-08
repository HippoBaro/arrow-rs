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

//! Experimental encoder construction through the supported factory API.

#![cfg(feature = "experimental")]

use std::sync::Arc;

use parquet::basic::Encoding;
use parquet::data_type::{
    ByteArray, ByteArrayType, DataType, FixedLenByteArray, FixedLenByteArrayType,
};
use parquet::decoding::get_decoder;
use parquet::encoding::get_encoder;
use parquet::schema::types::{ColumnDescPtr, ColumnDescriptor, ColumnPath, Type as SchemaType};

fn column_descriptor<T: DataType>(type_length: i32) -> ColumnDescPtr {
    Arc::new(ColumnDescriptor::new(
        Arc::new(
            SchemaType::primitive_type_builder("value", T::get_physical_type())
                .with_length(type_length)
                .build()
                .unwrap(),
        ),
        0,
        0,
        ColumnPath::from("value"),
    ))
}

#[test]
fn delta_length_byte_array_factory_roundtrip() {
    let descriptor = column_descriptor::<ByteArrayType>(-1);
    let mut encoder =
        get_encoder::<ByteArrayType>(Encoding::DELTA_LENGTH_BYTE_ARRAY, &descriptor).unwrap();
    assert_eq!(encoder.encoding(), Encoding::DELTA_LENGTH_BYTE_ARRAY);
    let values: Vec<ByteArray> = ["", "prefix", "prefix-longer", "z", ""]
        .into_iter()
        .map(ByteArray::from)
        .collect();
    encoder.put(&values[..2]).unwrap();
    encoder.put(&[]).unwrap();
    encoder.put(&values[2..]).unwrap();
    let data = encoder.flush_buffer().unwrap();
    let mut decoder =
        get_decoder::<ByteArrayType>(descriptor, Encoding::DELTA_LENGTH_BYTE_ARRAY).unwrap();
    decoder.set_data(data, values.len()).unwrap();
    let mut decoded = vec![ByteArray::default(); values.len()];
    assert_eq!(decoder.get(&mut decoded).unwrap(), values.len());
    assert_eq!(decoded, values);
}

fn delta_byte_array_factory_pages<T: DataType>(type_length: i32, pages: &[Vec<T::T>]) {
    let descriptor = column_descriptor::<T>(type_length);
    let mut encoder = get_encoder::<T>(Encoding::DELTA_BYTE_ARRAY, &descriptor).unwrap();
    let mut decoder = get_decoder::<T>(descriptor.clone(), Encoding::DELTA_BYTE_ARRAY).unwrap();
    assert_eq!(encoder.encoding(), Encoding::DELTA_BYTE_ARRAY);
    for values in pages {
        let split = values.len().min(2);
        encoder.put(&values[..split]).unwrap();
        encoder.put(&[]).unwrap();
        encoder.put(&values[split..]).unwrap();
        let data = encoder.flush_buffer().unwrap();
        let mut fresh = get_encoder::<T>(Encoding::DELTA_BYTE_ARRAY, &descriptor).unwrap();
        fresh.put(values).unwrap();
        // Batch boundaries must not reset prefixes; page boundaries must reset them.
        assert_eq!(data, fresh.flush_buffer().unwrap());
        decoder.set_data(data, values.len()).unwrap();
        let mut decoded = vec![T::T::default(); values.len()];
        assert_eq!(decoder.get(&mut decoded).unwrap(), values.len());
        assert_eq!(&decoded, values);
    }
}

#[test]
fn delta_byte_array_factory_prefix_suffix_and_reset() {
    let values = ["prefix", "prefix-longer", "prefix", "", "other", "other"]
        .into_iter()
        .map(ByteArray::from)
        .collect::<Vec<_>>();
    delta_byte_array_factory_pages::<ByteArrayType>(-1, &[vec![], values.clone(), vec![], values]);
}

#[test]
fn delta_byte_array_factory_fixed_length_roundtrip() {
    let values = ["abcd", "abce", "abce", "xyzw"]
        .into_iter()
        .map(|value| FixedLenByteArray::from(ByteArray::from(value)))
        .collect::<Vec<_>>();
    delta_byte_array_factory_pages::<FixedLenByteArrayType>(4, &[vec![], values.clone(), values]);
}
