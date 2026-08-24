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

//! Fixed-length byte-array sources, batches, and write-scoped encoding state.

use super::*;

/// Retains a source's physical fixed-length byte-array layout until the active encoder is known.
pub(crate) trait FixedLenByteArraySource {
    fn len(&self) -> usize;
    fn is_grouped(&self) -> bool {
        false
    }
    fn write_to(
        self,
        sink: &mut FixedLenByteArraySink<'_>,
        expected_width: Option<usize>,
    ) -> Result<()>;
}

enum FixedLenByteArraySinkTarget<'a> {
    Dictionary(&'a mut DictEncoder<FixedLenByteArrayType>),
    Fallback(&'a mut FixedLenByteArrayEncodingFamily),
}

/// Write-scoped fixed-length byte-array state shared by every bounded batch emitted for one input.
pub(crate) struct FixedLenByteArraySink<'a> {
    target: FixedLenByteArraySinkTarget<'a>,
    observer: FixedLenByteArrayObserver<'a>,
}

struct FixedLenByteArrayObserver<'a> {
    descr: &'a ColumnDescriptor,
    bloom: Option<&'a mut Sbbf>,
    scratch: &'a mut FixedLenByteArrayScratch,
    nan_count: &'a mut Option<u64>,
    collect_stats: bool,
    has_min: bool,
    has_max: bool,
}

/// Widest Arrow logical value computed into fixed-length bytes (decimal256).
#[cfg(feature = "arrow")]
pub(crate) const FIXED_LEN_BYTE_ARRAY_MAX_WIDTH: usize = 32;
/// Number of fixed-length values or run groups per stack batch.
#[cfg(feature = "arrow")]
pub(crate) const FIXED_LEN_BYTE_ARRAY_BATCH_VALUES: usize = 64;

#[cfg(feature = "arrow")]
pub(crate) struct FixedLenByteArrayBatchPacker<'sink, 'encoder> {
    sink: &'sink mut FixedLenByteArraySink<'encoder>,
    tile: [u8; FIXED_LEN_BYTE_ARRAY_BATCH_VALUES * FIXED_LEN_BYTE_ARRAY_MAX_WIDTH],
    width: usize,
    filled: usize,
}

#[cfg(feature = "arrow")]
impl<'sink, 'encoder> FixedLenByteArrayBatchPacker<'sink, 'encoder> {
    #[inline]
    pub(crate) fn new(sink: &'sink mut FixedLenByteArraySink<'encoder>, width: usize) -> Self {
        Self {
            sink,
            tile: [0; FIXED_LEN_BYTE_ARRAY_BATCH_VALUES * FIXED_LEN_BYTE_ARRAY_MAX_WIDTH],
            width,
            filled: 0,
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, fill: impl FnOnce(&mut [u8])) -> Result<()> {
        let offset = self.filled * self.width;
        let end = offset + self.width;
        fill(&mut self.tile[offset..end]);
        self.filled += 1;
        if self.filled == FIXED_LEN_BYTE_ARRAY_BATCH_VALUES {
            self.flush()?;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn finish(mut self) -> Result<()> {
        self.flush()
    }

    fn flush(&mut self) -> Result<()> {
        let len = self.filled * self.width;
        self.sink.push_batch(FixedLenByteArrayBatch::Packed(
            PackedFixedLenByteArrayBatch::new(&self.tile[..len], self.width, self.filled),
        ))?;
        self.filled = 0;
        Ok(())
    }
}

impl<D: DataType<T = FixedLenByteArray>> TypedColumnChunkEncoder<D> {
    #[inline]
    pub(crate) fn write_fixed_len_byte_array_source(
        &mut self,
        values: impl FixedLenByteArraySource,
    ) -> Result<()> {
        let len = values.len();
        let grouped = values.is_grouped();
        let expected_width = (self.fallback_encoding == Encoding::BYTE_STREAM_SPLIT)
            .then_some(self.descr.type_length() as usize);
        self.num_values += len;
        let target = match &mut self.encoding_family {
            FixedLenByteArrayEncodingFamily::Dictionary(dict) => {
                if !grouped {
                    dict.reserve(len);
                }
                FixedLenByteArraySinkTarget::Dictionary(dict)
            }
            encoder => {
                encoder.reserve_fixed_len(
                    (self.descr.type_length().max(0) as usize).saturating_mul(len),
                );
                FixedLenByteArraySinkTarget::Fallback(encoder)
            }
        };

        self.fixed_len_byte_array_scratch.min.clear();
        self.fixed_len_byte_array_scratch.max.clear();
        let collect_stats = self.statistics_enabled != EnabledStatistics::None
            && self.descr.converted_type() != ConvertedType::INTERVAL;
        let (has_min, has_max) = {
            let mut sink = FixedLenByteArraySink {
                target,
                observer: FixedLenByteArrayObserver {
                    descr: self.descr.as_ref(),
                    bloom: self.bloom_filter.as_mut(),
                    scratch: &mut self.fixed_len_byte_array_scratch,
                    nan_count: &mut self.nan_count,
                    collect_stats,
                    has_min: false,
                    has_max: false,
                },
            };
            values.write_to(&mut sink, expected_width)?;
            (sink.observer.has_min, sink.observer.has_max)
        };

        if has_min && has_max {
            let (min, max) = raw_fixed_len_min_max_values(
                &self.descr,
                &self.fixed_len_byte_array_scratch.min,
                &self.fixed_len_byte_array_scratch.max,
            );
            update_min(&self.descr, &min, &mut self.min_value);
            update_max(&self.descr, &max, &mut self.max_value);
        }
        Ok(())
    }
}

impl FixedLenByteArrayObserver<'_> {
    #[inline(always)]
    fn merge_extrema(&mut self, min: &[u8], max: &[u8], are_nan: bool) {
        let update_min = !self.has_min
            || match (is_nan_byte_array(self.descr, &self.scratch.min), are_nan) {
                (false, true) => false,
                (true, false) => true,
                _ => compare_greater_byte_array(self.descr, &self.scratch.min, min),
            };
        if update_min {
            self.scratch.min.clear();
            self.scratch.min.extend_from_slice(min);
            self.has_min = true;
        }
        let update_max = !self.has_max
            || match (is_nan_byte_array(self.descr, &self.scratch.max), are_nan) {
                (false, true) => false,
                (true, false) => true,
                _ => compare_greater_byte_array(self.descr, max, &self.scratch.max),
            };
        if update_max {
            self.scratch.max.clear();
            self.scratch.max.extend_from_slice(max);
            self.has_max = true;
        }
    }

    #[inline(always)]
    fn observe(&mut self, value: &[u8], multiplicity: usize) {
        if self.collect_stats {
            let value_is_nan = is_nan_byte_array(self.descr, value);
            if matches!(self.descr.logical_type_ref(), Some(LogicalType::Float16)) {
                let count = self.nan_count.get_or_insert(0);
                if value_is_nan {
                    *count += multiplicity as u64;
                }
            }
            self.merge_extrema(value, value, value_is_nan);
        }
        if let Some(bloom) = self.bloom.as_deref_mut() {
            bloom.insert(value);
        }
    }
}

impl FixedLenByteArraySink<'_> {
    #[inline(never)]
    fn encode_packed(&mut self, values: PackedFixedLenByteArrayBatch<'_>) -> Result<()> {
        if matches!(self.target, FixedLenByteArraySinkTarget::Dictionary(_)) {
            self.encode_dictionary_packed(values)
        } else {
            self.encode_fallback_packed(values)
        }
    }

    fn encode_dictionary_packed(&mut self, values: PackedFixedLenByteArrayBatch<'_>) -> Result<()> {
        for value in values.iter() {
            self.observer.observe(value, 1);
            let FixedLenByteArraySinkTarget::Dictionary(dict) = &mut self.target else {
                unreachable!()
            };
            dict.put_value_bytes(value, || value.to_vec().into())?;
        }
        Ok(())
    }

    #[inline(never)]
    fn encode_fallback_packed(&mut self, values: PackedFixedLenByteArrayBatch<'_>) -> Result<()> {
        debug_assert!(matches!(
            self.target,
            FixedLenByteArraySinkTarget::Fallback(_)
        ));
        if self.observer.collect_stats || self.observer.bloom.is_some() {
            let observer = &mut self.observer;
            let descr = observer.descr;
            let collect_stats = observer.collect_stats;
            let mut extrema: Option<(&[u8], &[u8], bool)> = None;
            let mut tile_nan_count = 0_u64;
            {
                let mut bloom = observer.bloom.as_deref_mut();
                for value in values.iter() {
                    if collect_stats {
                        let value_is_nan = is_nan_byte_array(descr, value);
                        tile_nan_count += value_is_nan as u64;
                        match extrema.as_mut() {
                            None => extrema = Some((value, value, value_is_nan)),
                            Some((min, max, are_nan)) => match (*are_nan, value_is_nan) {
                                (false, true) => {}
                                (true, false) => {
                                    *min = value;
                                    *max = value;
                                    *are_nan = false;
                                }
                                _ if compare_greater_byte_array(descr, min, value) => *min = value,
                                _ if compare_greater_byte_array(descr, value, max) => *max = value,
                                _ => {}
                            },
                        }
                    }
                    if let Some(bloom) = bloom.as_deref_mut() {
                        bloom.insert(value);
                    }
                }
            }
            if collect_stats {
                if matches!(descr.logical_type_ref(), Some(LogicalType::Float16)) {
                    *observer.nan_count.get_or_insert(0) += tile_nan_count;
                }
                if let Some((min, max, are_nan)) = extrema {
                    observer.merge_extrema(min, max, are_nan);
                }
            }
        }
        match &mut self.target {
            FixedLenByteArraySinkTarget::Fallback(encoder) => {
                encoder.put_fixed_len_byte_array_batch(values)
            }
            FixedLenByteArraySinkTarget::Dictionary(_) => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn push_selected<'a>(&mut self, values: impl ValueProducer<&'a [u8]>) -> Result<()> {
        let observer = &mut self.observer;
        match &mut self.target {
            FixedLenByteArraySinkTarget::Dictionary(dict) => values.try_for_each(|value| {
                observer.observe(value, 1);
                dict.put_value_bytes(value, || value.to_vec().into())
            }),
            FixedLenByteArraySinkTarget::Fallback(encoder) => values.try_for_each(|value| {
                observer.observe(value, 1);
                encoder.append_fixed_len_value(value)
            }),
        }
    }
}

#[cfg_attr(not(feature = "arrow"), allow(dead_code))]
pub(crate) enum FixedLenByteArrayBatch<'a> {
    Packed(PackedFixedLenByteArrayBatch<'a>),
}

impl<'batch> BatchSink<FixedLenByteArrayBatch<'batch>> for FixedLenByteArraySink<'_> {
    #[inline(never)]
    fn push_batch(&mut self, values: FixedLenByteArrayBatch<'batch>) -> Result<()> {
        match values {
            FixedLenByteArrayBatch::Packed(values) => self.encode_packed(values),
        }
    }
}

impl FixedLenByteArraySource for &[FixedLenByteArray] {
    fn len(&self) -> usize {
        <[FixedLenByteArray]>::len(self)
    }

    fn write_to(
        self,
        sink: &mut FixedLenByteArraySink<'_>,
        expected_width: Option<usize>,
    ) -> Result<()> {
        if let Some(expected) = expected_width
            && let Some(value) = self.iter().find(|value| value.data().len() != expected)
        {
            return Err(general_err!(
                "Mismatched FixedLenByteArray sizes: {} != {}",
                value.data().len(),
                expected
            ));
        }
        sink.push_selected(self)
    }
}

pub(super) fn encode_fixed_len_byte_array_slice<D: DataType<T = FixedLenByteArray>>(
    enc: &mut TypedColumnChunkEncoder<D>,
    values: &[FixedLenByteArray],
) -> Result<()> {
    enc.write_fixed_len_byte_array_source(values)
}

fn raw_fixed_len_min_max_values(
    _descr: &ColumnDescriptor,
    min: &[u8],
    max: &[u8],
) -> (FixedLenByteArray, FixedLenByteArray) {
    (min.to_vec().into(), max.to_vec().into())
}
