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

use bytes::Bytes;
use parquet::basic::Encoding;
use parquet::column::reader::{ColumnReader, ColumnReaderImpl};
use parquet::column::writer::{ColumnWriter, ColumnWriterImpl};
use parquet::data_type::{
    BoolType, ByteArray, ByteArrayType, DataType, DoubleType, FixedLenByteArray,
    FixedLenByteArrayType, FloatType, Int32Type, Int64Type, Int96, Int96Type,
};
use parquet::decoding::get_decoder;
use parquet::encoding::{DictEncoder, Encoder, PlainEncoder, get_encoder};
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

fn generic_plain_factory<T: DataType>(descriptor: &ColumnDescPtr, values: &[T::T]) -> Bytes {
    let mut encoder = get_encoder::<T>(Encoding::PLAIN, descriptor).unwrap();
    assert_eq!(encoder.encoding(), Encoding::PLAIN);
    let split = values.len().min(2);
    encoder.put(&values[..split]).unwrap();
    encoder.put(&[]).unwrap();
    encoder.put(&values[split..]).unwrap();
    encoder.flush_buffer().unwrap()
}

fn generic_plain_and_dictionary<T: DataType>(type_length: i32, values: &[T::T]) -> Bytes {
    let descriptor = column_descriptor::<T>(type_length);
    let data = generic_plain_factory::<T>(&descriptor, values);
    let mut decoder = get_decoder::<T>(descriptor.clone(), Encoding::PLAIN).unwrap();
    decoder.set_data(data.clone(), values.len()).unwrap();
    let mut decoded = vec![T::T::default(); values.len()];
    assert_eq!(decoder.get(&mut decoded).unwrap(), values.len());
    assert_eq!(decoded, values);

    let mut dict = DictEncoder::<T>::new(descriptor.clone());
    dict.put(values).unwrap();
    let mut unique = Vec::new();
    for value in values {
        if !unique.contains(value) {
            unique.push(value.clone());
        }
    }
    assert_eq!(dict.num_entries(), unique.len());
    assert_eq!(
        dict.write_dict().unwrap(),
        generic_plain_factory::<T>(&descriptor, &unique)
    );
    data
}

fn supported_plain<T: DataType>(mut encoder: impl Encoder<T>, type_length: i32, values: &[T::T]) {
    for _ in 0..2 {
        let split = values.len().min(2);
        encoder.put(&values[..split]).unwrap();
        encoder.put(&[]).unwrap();
        encoder.put(&values[split..]).unwrap();
        assert_eq!(encoder.encoding(), Encoding::PLAIN);
        let size = encoder.estimated_data_encoded_size();
        assert!(encoder.estimated_memory_size() >= size);
        let data = encoder.flush_buffer().unwrap();
        assert_eq!(size, data.len());
        assert_eq!(encoder.estimated_data_encoded_size(), 0);
        assert_eq!(data, generic_plain_and_dictionary::<T>(type_length, values));
    }
}

#[test]
fn plain_encoder_supported_concrete_types() {
    supported_plain::<BoolType>(
        PlainEncoder::new(),
        -1,
        &[true, false, true, false, false, true, false, true, true],
    );
    supported_plain::<Int32Type>(PlainEncoder::new(), -1, &[i32::MIN, 0, i32::MAX, 0]);
    supported_plain::<Int64Type>(PlainEncoder::new(), -1, &[i64::MIN, 0, i64::MAX, 0]);
    supported_plain::<Int96Type>(
        PlainEncoder::new(),
        -1,
        &[Int96::from(vec![1, 2, 3]), Int96::from(vec![1, 2, 3])],
    );
    supported_plain::<FloatType>(PlainEncoder::new(), -1, &[-1.5, 0.0, 3.25, 0.0]);
    supported_plain::<DoubleType>(PlainEncoder::new(), -1, &[-1.5, 0.0, 3.25, 0.0]);
    supported_plain::<FixedLenByteArrayType>(
        PlainEncoder::new(),
        4,
        &[
            FixedLenByteArray::from(ByteArray::from("abcd")),
            FixedLenByteArray::from(ByteArray::from("abcd")),
            FixedLenByteArray::from(ByteArray::from("xyzw")),
        ],
    );
}

#[test]
fn plain_byte_array_generic_factory_and_dictionary() {
    let values = ["", "abc", "abc", "z", ""]
        .into_iter()
        .map(ByteArray::from)
        .collect::<Vec<_>>();
    let data = generic_plain_and_dictionary::<ByteArrayType>(-1, &values);
    assert_eq!(
        data.as_ref(),
        b"\x00\x00\x00\x00\x03\x00\x00\x00abc\x03\x00\x00\x00abc\x01\x00\x00\x00z\x00\x00\x00\x00"
    );
    assert!(generic_plain_and_dictionary::<ByteArrayType>(-1, &[]).is_empty());
}

struct CustomInt32Type;

impl DataType for CustomInt32Type {
    type T = i32;

    fn get_type_size() -> usize {
        std::mem::size_of::<Self::T>()
    }

    fn get_column_reader(_: ColumnReader) -> Option<ColumnReaderImpl<Self>> {
        None
    }

    fn get_column_writer(_: ColumnWriter<'_>) -> Option<ColumnWriterImpl<'_, Self>> {
        None
    }

    fn get_column_writer_ref<'a, 'b: 'a>(
        _: &'b ColumnWriter<'a>,
    ) -> Option<&'b ColumnWriterImpl<'a, Self>> {
        None
    }

    fn get_column_writer_mut<'a, 'b: 'a>(
        _: &'a mut ColumnWriter<'b>,
    ) -> Option<&'a mut ColumnWriterImpl<'b, Self>> {
        None
    }
}

#[test]
fn plain_encoder_custom_data_type_marker() {
    supported_plain::<CustomInt32Type>(PlainEncoder::new(), -1, &[1, 2, 1, 3]);
}
