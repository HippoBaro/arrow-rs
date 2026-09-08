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

use parquet::basic::{Encoding, Type};
use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::decoding::get_decoder;
use parquet::encoding::get_encoder;
use parquet::schema::types::{ColumnDescPtr, ColumnDescriptor, ColumnPath, Type as SchemaType};

fn byte_array_descriptor() -> ColumnDescPtr {
    Arc::new(ColumnDescriptor::new(
        Arc::new(
            SchemaType::primitive_type_builder("value", Type::BYTE_ARRAY)
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
    let descriptor = byte_array_descriptor();
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
