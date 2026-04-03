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

use bytes::Bytes;

use crate::basic::{Encoding, EncodingMask};
use crate::column::reader::run_level_buffer::RunLevelBuffer;
use crate::data_type::DataType;
use crate::encodings::{
    decoding::{Decoder, DictDecoder, PlainDecoder, get_decoder},
    rle::RleDecoder,
};
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;
use crate::util::bit_util::{BitReader, num_required_bits};

/// Decodes level data
pub trait ColumnLevelDecoder {
    type Buffer;

    /// Set data for this [`ColumnLevelDecoder`]
    fn set_data(&mut self, encoding: Encoding, data: Bytes) -> Result<()>;
}

pub trait RepetitionLevelDecoder: ColumnLevelDecoder {
    /// Read up to `max_records` of repetition level data into `out` returning the number
    /// of complete records and levels read
    ///
    /// A record only ends when the data contains a subsequent repetition level of 0,
    /// it is therefore left to the caller to delimit the final record in a column
    ///
    /// # Panics
    ///
    /// Implementations may panic if `range` overlaps with already written data
    fn read_rep_levels(
        &mut self,
        out: &mut Self::Buffer,
        num_records: usize,
        num_levels: usize,
    ) -> Result<(usize, usize)>;

    /// Skips over up to `num_levels` repetition levels corresponding to `num_records` records,
    /// where a record is delimited by a repetition level of 0
    ///
    /// Returns the number of records skipped, and the number of levels skipped
    ///
    /// A record only ends when the data contains a subsequent repetition level of 0,
    /// it is therefore left to the caller to delimit the final record in a column
    fn skip_rep_levels(&mut self, num_records: usize, num_levels: usize) -> Result<(usize, usize)>;

    /// Flush any partially read or skipped record
    fn flush_partial(&mut self) -> bool;
}

pub trait DefinitionLevelDecoder: ColumnLevelDecoder {
    /// Read up to `num_levels` definition levels into `out`.
    ///
    /// Returns the number of values read, and the number of levels read.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `range` overlaps with already written data
    fn read_def_levels(
        &mut self,
        out: &mut Self::Buffer,
        num_levels: usize,
    ) -> Result<(usize, usize)>;

    /// Skips over `num_levels` definition levels.
    ///
    /// Returns the number of values skipped, and the number of levels skipped.
    fn skip_def_levels(&mut self, num_levels: usize) -> Result<(usize, usize)>;
}

/// Decodes value data
pub trait ColumnValueDecoder {
    type Buffer;

    /// Create a new [`ColumnValueDecoder`]
    fn new(col: &ColumnDescPtr) -> Self;

    /// Set the current dictionary page
    fn set_dict(
        &mut self,
        buf: Bytes,
        num_values: u32,
        encoding: Encoding,
        is_sorted: bool,
    ) -> Result<()>;

    /// Set the current data page
    ///
    /// - `encoding` - the encoding of the page
    /// - `data` - a point to the page's uncompressed value data
    /// - `num_levels` - the number of levels contained within the page, i.e. values including nulls
    /// - `num_values` - the number of non-null values contained within the page (V2 page only)
    ///
    /// Note: data encoded with [`Encoding::RLE`] may not know its exact length, as the final
    /// run may be zero-padded. As such if `num_values` is not provided (i.e. `None`),
    /// subsequent calls to `ColumnValueDecoder::read` may yield more values than
    /// non-null definition levels within the page
    fn set_data(
        &mut self,
        encoding: Encoding,
        data: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
    ) -> Result<()>;

    /// Read up to `num_values` values into `out`
    ///
    /// # Panics
    ///
    /// Implementations may panic if `range` overlaps with already written data
    ///
    fn read(&mut self, out: &mut Self::Buffer, num_values: usize) -> Result<usize>;

    /// Skips over `num_values` values
    ///
    /// Returns the number of values skipped
    fn skip_values(&mut self, num_values: usize) -> Result<usize>;
}

/// Bucket-based storage for decoder instances keyed by `Encoding`.
///
/// This replaces `HashMap` lookups with direct indexing to avoid hashing overhead in the
/// hot decoding paths.
const ENCODING_SLOTS: usize = Encoding::MAX_DISCRIMINANT as usize + 1;

/// An implementation of [`ColumnValueDecoder`] for `[T::T]`
pub struct ColumnValueDecoderImpl<T: DataType> {
    descr: ColumnDescPtr,

    current_encoding: Option<Encoding>,

    /// Cache of decoders for existing encodings.
    /// Uses `EncodingMask` and dense storage keyed by encoding discriminant.
    decoder_mask: EncodingMask,
    decoders: [Option<Box<dyn Decoder<T>>>; ENCODING_SLOTS],
}

impl<T: DataType> ColumnValueDecoder for ColumnValueDecoderImpl<T> {
    type Buffer = Vec<T::T>;

    fn new(descr: &ColumnDescPtr) -> Self {
        Self {
            descr: descr.clone(),
            current_encoding: None,
            decoder_mask: EncodingMask::default(),
            decoders: std::array::from_fn(|_| None),
        }
    }

    fn set_dict(
        &mut self,
        buf: Bytes,
        num_values: u32,
        mut encoding: Encoding,
        _is_sorted: bool,
    ) -> Result<()> {
        if encoding == Encoding::PLAIN || encoding == Encoding::PLAIN_DICTIONARY {
            encoding = Encoding::RLE_DICTIONARY
        }

        if self.decoder_mask.is_set(encoding) {
            return Err(general_err!("Column cannot have more than one dictionary"));
        }

        if encoding == Encoding::RLE_DICTIONARY {
            let mut dictionary = PlainDecoder::<T>::new(self.descr.type_length());
            dictionary.set_data(buf, num_values as usize)?;

            let mut decoder = DictDecoder::new();
            decoder.set_dict(Box::new(dictionary))?;
            self.decoders[encoding as usize] = Some(Box::new(decoder));
            self.decoder_mask.insert(encoding);
            Ok(())
        } else {
            Err(nyi_err!(
                "Invalid/Unsupported encoding type for dictionary: {}",
                encoding
            ))
        }
    }

    fn set_data(
        &mut self,
        mut encoding: Encoding,
        data: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
    ) -> Result<()> {
        if encoding == Encoding::PLAIN_DICTIONARY {
            encoding = Encoding::RLE_DICTIONARY;
        }

        let decoder = if encoding == Encoding::RLE_DICTIONARY {
            self.decoders[encoding as usize]
                .as_mut()
                .expect("Decoder for dict should have been set")
        } else {
            let slot = encoding as usize;
            if self.decoders[slot].is_none() {
                let data_decoder = get_decoder::<T>(self.descr.clone(), encoding)?;
                self.decoders[slot] = Some(data_decoder);
                self.decoder_mask.insert(encoding);
            }
            self.decoders[slot]
                .as_mut()
                .expect("decoder should have been inserted")
        };

        decoder.set_data(data, num_values.unwrap_or(num_levels))?;
        self.current_encoding = Some(encoding);
        Ok(())
    }

    fn read(&mut self, out: &mut Self::Buffer, num_values: usize) -> Result<usize> {
        let encoding = self
            .current_encoding
            .expect("current_encoding should be set");

        let current_decoder = self.decoders[encoding as usize]
            .as_mut()
            .unwrap_or_else(|| panic!("decoder for encoding {encoding} should be set"));

        // TODO: Push vec into decoder (#5177)
        let start = out.len();
        out.resize(start + num_values, T::T::default());
        let read = current_decoder.get(&mut out[start..])?;
        out.truncate(start + read);
        Ok(read)
    }

    fn skip_values(&mut self, num_values: usize) -> Result<usize> {
        let encoding = self
            .current_encoding
            .expect("current_encoding should be set");

        let current_decoder = self.decoders[encoding as usize]
            .as_mut()
            .unwrap_or_else(|| panic!("decoder for encoding {encoding} should be set"));

        current_decoder.skip(num_values)
    }
}

const SKIP_BUFFER_SIZE: usize = 1024;

enum LevelDecoder {
    Packed(BitReader, u8),
    Rle(RleDecoder),
}

impl LevelDecoder {
    fn new(encoding: Encoding, data: Bytes, bit_width: u8) -> Result<Self> {
        match encoding {
            Encoding::RLE => {
                let mut decoder = RleDecoder::new(bit_width);
                decoder.set_data(data)?;
                Ok(Self::Rle(decoder))
            }
            #[allow(deprecated)]
            Encoding::BIT_PACKED => Ok(Self::Packed(BitReader::new(data), bit_width)),
            _ => unreachable!("invalid level encoding: {}", encoding),
        }
    }

    fn read(&mut self, out: &mut [i16]) -> Result<usize> {
        match self {
            Self::Packed(reader, bit_width) => {
                Ok(reader.get_batch::<i16>(out, *bit_width as usize))
            }
            Self::Rle(reader) => Ok(reader.get_batch(out)?),
        }
    }

    fn read_as_runs(
        &mut self,
        out: &mut Vec<(i16, u32)>,
        max_values: usize,
    ) -> Result<usize> {
        match self {
            Self::Rle(reader) => reader.get_batch_as_runs(out, max_values),
            Self::Packed(reader, bit_width) => {
                // Bit-packed: decode to a temp buffer, then convert to runs
                let mut buf = vec![0i16; max_values];
                let n = reader.get_batch::<i16>(&mut buf, *bit_width as usize);
                for &v in &buf[..n] {
                    if let Some(last) = out.last_mut().filter(|r| r.0 == v) {
                        last.1 += 1;
                    } else {
                        out.push((v, 1));
                    }
                }
                Ok(n)
            }
        }
    }
}

/// An implementation of [`DefinitionLevelDecoder`] for `[i16]`
pub struct DefinitionLevelDecoderImpl {
    decoder: Option<LevelDecoder>,
    bit_width: u8,
    max_level: i16,
}

impl DefinitionLevelDecoderImpl {
    pub fn new(max_level: i16) -> Self {
        let bit_width = num_required_bits(max_level as u64);
        Self {
            decoder: None,
            bit_width,
            max_level,
        }
    }
}

impl ColumnLevelDecoder for DefinitionLevelDecoderImpl {
    type Buffer = RunLevelBuffer;

    fn set_data(&mut self, encoding: Encoding, data: Bytes) -> Result<()> {
        self.decoder = Some(LevelDecoder::new(encoding, data, self.bit_width)?);
        Ok(())
    }
}

impl DefinitionLevelDecoder for DefinitionLevelDecoderImpl {
    fn read_def_levels(
        &mut self,
        out: &mut Self::Buffer,
        num_levels: usize,
    ) -> Result<(usize, usize)> {
        // Decode new runs into a temporary vector so we can count values
        // without worrying about merging with pre-existing runs.
        let mut new_runs = Vec::new();
        let levels_read = self
            .decoder
            .as_mut()
            .unwrap()
            .read_as_runs(&mut new_runs, num_levels)?;

        let values_read: usize = new_runs
            .iter()
            .filter(|(v, _)| *v == self.max_level)
            .map(|(_, c)| *c as usize)
            .sum();

        // Append new runs to the buffer, merging with the last existing run
        // if it has the same value.
        out.append_runs(&new_runs);
        let new_total = out.len() + levels_read;
        out.set_total_len(new_total);
        Ok((values_read, levels_read))
    }

    fn skip_def_levels(&mut self, num_levels: usize) -> Result<(usize, usize)> {
        let mut level_skip = 0;
        let mut value_skip = 0;
        let mut buf = RunLevelBuffer::new();
        while level_skip < num_levels {
            let remaining_levels = num_levels - level_skip;

            let to_read = remaining_levels.min(SKIP_BUFFER_SIZE);
            buf.clear();
            let (values_read, levels_read) = self.read_def_levels(&mut buf, to_read)?;
            if levels_read == 0 {
                // Reached end of page
                break;
            }

            level_skip += levels_read;
            value_skip += values_read;
        }

        Ok((value_skip, level_skip))
    }
}

pub(crate) const REPETITION_LEVELS_BATCH_SIZE: usize = 1024;

/// An implementation of [`RepetitionLevelDecoder`] that decodes directly
/// into [`RunLevelBuffer`] run pairs, preserving the RLE structure.
pub struct RepetitionLevelDecoderImpl {
    decoder: Option<LevelDecoder>,
    bit_width: u8,
    /// Staging buffer for runs decoded from the current page.
    run_buf: Vec<(i16, u32)>,
    /// Index of the current run being consumed in run_buf.
    run_idx: usize,
    /// Remaining count in run_buf[run_idx] (may be less than the full
    /// run count if a previous call partially consumed it).
    run_remaining: u32,
    has_partial: bool,
}

impl RepetitionLevelDecoderImpl {
    pub fn new(max_level: i16) -> Self {
        let bit_width = num_required_bits(max_level as u64);
        Self {
            decoder: None,
            bit_width,
            run_buf: Vec::new(),
            run_idx: 0,
            run_remaining: 0,
            has_partial: false,
        }
    }

    fn fill_buf(&mut self) -> Result<usize> {
        self.run_buf.clear();
        self.run_idx = 0;
        self.run_remaining = 0;
        let read = self
            .decoder
            .as_mut()
            .unwrap()
            .read_as_runs(&mut self.run_buf, REPETITION_LEVELS_BATCH_SIZE)?;
        if !self.run_buf.is_empty() {
            self.run_remaining = self.run_buf[0].1;
        }
        Ok(read)
    }

    /// Returns true if the run buffer has remaining data.
    fn has_buffered_data(&self) -> bool {
        self.run_idx < self.run_buf.len() && self.run_remaining > 0
    }

    /// Consume runs from the staging buffer, counting records and emitting
    /// the consumed runs to `out` (if Some). Stops when `records_to_read`
    /// complete records have been found or `num_levels` levels consumed.
    ///
    /// Returns (partial, records_read, levels_read).
    /// `partial` is true if we ran out of data without finding enough records.
    fn consume_runs(
        &mut self,
        mut out: Option<&mut RunLevelBuffer>,
        records_to_read: usize,
        num_levels: usize,
    ) -> (bool, usize, usize) {
        let mut records_read = 0;
        let mut levels_read = 0;

        while levels_read < num_levels && self.has_buffered_data() {
            let value = self.run_buf[self.run_idx].0;
            let available = self.run_remaining as usize;
            let budget = num_levels - levels_read;

            if value != 0 {
                // Non-zero run: continuation entries, no record boundaries.
                let take = available.min(budget);
                if let Some(out) = out.as_deref_mut() {
                    out.runs_mut().push((value, take as u32));
                }
                levels_read += take;
                self.run_remaining -= take as u32;
            } else {
                // Zero run: each entry is a potential record boundary.
                //
                // The first zero in the entire stream is NOT a boundary
                // (it starts the first record). This is tracked by
                // has_partial: if false and levels_read==0 and records_read==0,
                // the first zero is not a boundary.
                let skip_first = if !self.has_partial
                    && levels_read == 0
                    && records_read == 0
                {
                    1
                } else {
                    0
                };

                let take = available.min(budget);
                let boundaries = take.saturating_sub(skip_first);
                let remaining_records = records_to_read - records_read;

                if boundaries >= remaining_records {
                    // More boundaries than we need. Consume exactly enough:
                    // remaining_records boundaries + skip_first non-boundary entries.
                    let to_consume = remaining_records + skip_first;
                    // Emit only the consumed portion. But the terminating
                    // zero (the one that completed the last record) is NOT
                    // consumed — it stays in the buffer as the start of the
                    // next record. So we emit (to_consume) entries, which
                    // includes the skip_first + remaining_records zeros.
                    // However, the LAST zero we counted as a boundary must
                    // not be consumed (it's the start of the next record).
                    // So emit (to_consume - 1) entries and leave the rest.
                    //
                    // Wait — in the flat version, `count_records` returns
                    // `idx` which is the position of the terminating zero,
                    // and `levels_read = idx`. The caller consumes `idx`
                    // entries from buffer_offset, NOT including the zero.
                    //
                    // For runs: we need to consume (skip_first + remaining_records - 1)
                    // zeros. The last boundary zero is not consumed.
                    //
                    // Actually: rethinking. In the flat version:
                    // - Entry at idx=0 is not counted (skip_first=1)
                    // - Entry at idx=3 is a boundary → records_read=1
                    // - Entry at idx=5 is a boundary → records_read=2 = records_to_read
                    // - Returns (false, 2, 5). Consumes 5 levels (0..4). idx=5 not consumed.
                    //
                    // For a run of all zeros (0, N):
                    // - If skip_first=1: boundaries at positions 1..N-1
                    // - We want remaining_records boundaries
                    // - Consume (skip_first + remaining_records) entries
                    // - The entry at position (skip_first + remaining_records) is
                    //   the start of the NEXT record — not consumed.
                    // Consume skip_first non-boundary zeros + (remaining_records - 1)
                    // boundary zeros. The final boundary zero is NOT consumed —
                    // it stays as the start of the next record (matching the flat
                    // version where levels_read = idx of the terminating zero).
                    let consumed = skip_first + remaining_records - 1;
                    if let Some(out) = out.as_deref_mut() {
                        if consumed > 0 {
                            out.runs_mut().push((0, consumed as u32));
                        }
                    }
                    levels_read += consumed;
                    records_read += remaining_records;
                    self.run_remaining -= consumed as u32;
                    return (false, records_read, levels_read);
                }

                // Consume the whole run (or budget-limited portion)
                if let Some(out) = out.as_deref_mut() {
                    if take > 0 {
                        out.runs_mut().push((0, take as u32));
                    }
                }
                records_read += boundaries;
                levels_read += take;
                self.run_remaining -= take as u32;
            }

            if self.run_remaining == 0 {
                self.run_idx += 1;
                if self.run_idx < self.run_buf.len() {
                    self.run_remaining = self.run_buf[self.run_idx].1;
                }
            }
        }

        (true, records_read, levels_read)
    }
}

impl ColumnLevelDecoder for RepetitionLevelDecoderImpl {
    type Buffer = RunLevelBuffer;

    fn set_data(&mut self, encoding: Encoding, data: Bytes) -> Result<()> {
        self.decoder = Some(LevelDecoder::new(encoding, data, self.bit_width)?);
        self.run_buf.clear();
        self.run_idx = 0;
        self.run_remaining = 0;
        Ok(())
    }
}

impl RepetitionLevelDecoder for RepetitionLevelDecoderImpl {
    fn read_rep_levels(
        &mut self,
        out: &mut Self::Buffer,
        num_records: usize,
        num_levels: usize,
    ) -> Result<(usize, usize)> {
        let mut total_records_read = 0;
        let mut total_levels_read = 0;

        while total_records_read < num_records && total_levels_read < num_levels {
            if !self.has_buffered_data() {
                let read = self.fill_buf()?;
                if read == 0 {
                    break;
                }
            }

            let (partial, records_read, levels_read) = self.consume_runs(
                Some(out),
                num_records - total_records_read,
                num_levels - total_levels_read,
            );

            total_levels_read += levels_read;
            total_records_read += records_read;
            self.has_partial = partial;
        }
        let new_total = out.len() + total_levels_read;
        out.set_total_len(new_total);
        Ok((total_records_read, total_levels_read))
    }

    fn skip_rep_levels(&mut self, num_records: usize, num_levels: usize) -> Result<(usize, usize)> {
        let mut total_records_read = 0;
        let mut total_levels_read = 0;

        while total_records_read < num_records && total_levels_read < num_levels {
            if !self.has_buffered_data() {
                let read = self.fill_buf()?;
                if read == 0 {
                    break;
                }
            }

            let (partial, records_read, levels_read) = self.consume_runs(
                None,
                num_records - total_records_read,
                num_levels - total_levels_read,
            );

            total_levels_read += levels_read;
            total_records_read += records_read;
            self.has_partial = partial;
        }
        Ok((total_records_read, total_levels_read))
    }

    fn flush_partial(&mut self) -> bool {
        std::mem::take(&mut self.has_partial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encodings::rle::RleEncoder;
    use rand::{prelude::*, rng};

    #[test]
    fn test_skip_padding() {
        let mut encoder = RleEncoder::new(1, 1024);
        encoder.put(0);
        (0..3).for_each(|_| encoder.put(1));
        let data = Bytes::from(encoder.consume());

        let mut decoder = RepetitionLevelDecoderImpl::new(1);
        decoder.set_data(Encoding::RLE, data.clone()).unwrap();
        let (_, levels) = decoder.skip_rep_levels(100, 4).unwrap();
        assert_eq!(levels, 4);

        // The length of the final bit packed run is ambiguous, so without the correct
        // levels limit, it will decode zero padding
        let mut decoder = RepetitionLevelDecoderImpl::new(1);
        decoder.set_data(Encoding::RLE, data).unwrap();
        let (_, levels) = decoder.skip_rep_levels(100, 6).unwrap();
        assert_eq!(levels, 6);
    }

    #[test]
    fn test_skip_rep_levels() {
        for _ in 0..10 {
            let mut rng = rng();
            let total_len = 10000_usize;
            let mut encoded: Vec<i16> = (0..total_len).map(|_| rng.random_range(0..5)).collect();
            encoded[0] = 0;
            let mut encoder = RleEncoder::new(3, 1024);
            for v in &encoded {
                encoder.put(*v as _)
            }
            let data = Bytes::from(encoder.consume());

            let mut decoder = RepetitionLevelDecoderImpl::new(5);
            decoder.set_data(Encoding::RLE, data).unwrap();

            let total_records = encoded.iter().filter(|x| **x == 0).count();
            let mut remaining_records = total_records;
            let mut remaining_levels = encoded.len();
            loop {
                let skip = rng.random_bool(0.5);
                let records = rng.random_range(1..=remaining_records.min(5));
                let (records_read, levels_read) = if skip {
                    decoder.skip_rep_levels(records, remaining_levels).unwrap()
                } else {
                    let mut decoded = RunLevelBuffer::new();
                    let (records_read, levels_read) = decoder
                        .read_rep_levels(&mut decoded, records, remaining_levels)
                        .unwrap();

                    let decoded_flat = decoded.take_flat();
                    assert_eq!(
                        decoded_flat,
                        encoded[encoded.len() - remaining_levels..][..levels_read]
                    );
                    (records_read, levels_read)
                };

                remaining_levels = remaining_levels.checked_sub(levels_read).unwrap();
                if remaining_levels == 0 {
                    assert_eq!(records_read + 1, records);
                    assert_eq!(records, remaining_records);
                    break;
                }
                assert_eq!(records_read, records);
                remaining_records -= records;
                assert_ne!(remaining_records, 0);
            }
        }
    }
}
