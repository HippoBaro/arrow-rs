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

use super::ArrowPhysicalBridge;
#[cfg(test)]
use crate::column::value::GroupedSelectionRef;
use crate::column::value::{
    IndexSpan, PhysicalIndexPlan, Sink, ValueCursor, ValueSelectionRef, gather_tiled,
};
use crate::column::writer::encoder::ColumnValueEncoderImpl;
use crate::column::writer::{
    BYTE_ARRAY_BATCH_VALUES, ByteBatchSource, ByteBudgetTarget, ByteSink, FlatByteBatch,
};
use crate::data_type::ByteArrayType as ParquetByteArrayType;
use crate::errors::{ParquetError, Result};

use arrow_array::cast::AsArray;
use arrow_array::types::{BinaryType, ByteArrayType, LargeBinaryType, LargeUtf8Type, Utf8Type};
use arrow_array::{Array, BinaryViewArray, FixedSizeBinaryArray, OffsetSizeTrait, StringViewArray};
use arrow_buffer::Buffer;
use arrow_schema::DataType;

/// Type-specific access to the final physical byte values. Index composition is
/// deliberately absent here: [`PhysicalIndexPlan`] owns run and dictionary
/// mapping, leaving one Arrow-layout dispatch for both writing and page
/// budgeting.
trait ByteValueAccess<'a>: Copy {
    fn value(self, index: usize) -> &'a [u8];

    #[inline]
    fn value_len(self, index: usize) -> usize {
        self.value(index).len()
    }

    #[inline]
    fn try_for_each_range<E>(
        self,
        start: usize,
        len: usize,
        mut f: impl FnMut(&'a [u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        for index in start..start + len {
            f(self.value(index))?;
        }
        Ok(())
    }

    /// Exact encoded size of a contiguous range, when the layout has prefix
    /// offsets. Includes Parquet's four-byte PLAIN length prefix per value.
    #[inline]
    fn exact_range_encoded_size(self, _start: usize, _len: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn contiguous_batch(self, _start: usize, _len: usize) -> Option<FlatByteBatch<'a, 'a>> {
        None
    }

    /// Conservative encoded-size bound used to retain the O(1) direct-view
    /// fast path. Unlike an exact size, this is used only when it covers the
    /// entire remaining plan.
    #[inline]
    fn range_encoded_upper_bound(self, _start: usize, _len: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn is_variable_width(self) -> bool {
        true
    }
}

/// Access to offset-based byte arrays. Contiguous ranges iterate adjacent
/// offset windows and compute their total encoded size with one subtraction.
#[derive(Clone, Copy)]
struct OffsetByteAccess<'a, O: OffsetSizeTrait> {
    offsets: &'a [O],
    data: &'a [u8],
}

impl<'a, O: OffsetSizeTrait> OffsetByteAccess<'a, O> {
    #[inline]
    fn bind<T: ByteArrayType<Offset = O>>(values: &'a dyn Array) -> Self {
        let values = values.as_bytes::<T>();
        Self {
            offsets: values.value_offsets(),
            data: values.value_data(),
        }
    }
}

trait ByteOffset: OffsetSizeTrait {
    fn batch<'a>(offsets: &'a [Self], data: &'a [u8]) -> FlatByteBatch<'a, 'a>;
}

impl ByteOffset for i32 {
    #[inline]
    fn batch<'a>(offsets: &'a [Self], data: &'a [u8]) -> FlatByteBatch<'a, 'a> {
        FlatByteBatch::Offset32 { offsets, data }
    }
}

impl ByteOffset for i64 {
    #[inline]
    fn batch<'a>(offsets: &'a [Self], data: &'a [u8]) -> FlatByteBatch<'a, 'a> {
        FlatByteBatch::Offset64 { offsets, data }
    }
}

impl<'a, O: ByteOffset> ByteValueAccess<'a> for OffsetByteAccess<'a, O> {
    #[inline]
    fn value(self, index: usize) -> &'a [u8] {
        let start = self.offsets[index].as_usize();
        &self.data[start..self.offsets[index + 1].as_usize()]
    }

    #[inline]
    fn value_len(self, index: usize) -> usize {
        (self.offsets[index + 1] - self.offsets[index]).as_usize()
    }

    #[inline]
    fn try_for_each_range<E>(
        self,
        start: usize,
        len: usize,
        mut f: impl FnMut(&'a [u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let data = self.data;
        for window in self.offsets[start..start + len + 1].windows(2) {
            f(&data[window[0].as_usize()..window[1].as_usize()])?;
        }
        Ok(())
    }

    #[inline]
    fn exact_range_encoded_size(self, start: usize, len: usize) -> Option<usize> {
        let payload = (self.offsets[start + len] - self.offsets[start]).as_usize();
        Some(payload.saturating_add(len.saturating_mul(std::mem::size_of::<u32>())))
    }

    #[inline]
    fn contiguous_batch(self, start: usize, len: usize) -> Option<FlatByteBatch<'a, 'a>> {
        Some(O::batch(&self.offsets[start..start + len + 1], self.data))
    }
}

/// Access to view and fixed-size arrays through an already downcast accessor.
#[derive(Clone, Copy)]
struct IndexedByteAccess<'a, F> {
    get: F,
    views: Option<&'a [u128]>,
    max_encoded_size: Option<usize>,
    _marker: std::marker::PhantomData<&'a [u8]>,
}

impl<'a, F> IndexedByteAccess<'a, F>
where
    F: Fn(usize) -> &'a [u8] + Copy,
{
    #[inline]
    fn variable(get: F, views: &'a [u128], max_value_len: usize) -> Self {
        Self {
            get,
            views: Some(views),
            max_encoded_size: Some(max_value_len.saturating_add(std::mem::size_of::<u32>())),
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn fixed(get: F) -> Self {
        Self {
            get,
            views: None,
            max_encoded_size: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, F> ByteValueAccess<'a> for IndexedByteAccess<'a, F>
where
    F: Fn(usize) -> &'a [u8] + Copy + 'a,
{
    #[inline]
    fn value(self, index: usize) -> &'a [u8] {
        (self.get)(index)
    }

    #[inline]
    fn value_len(self, index: usize) -> usize {
        self.views.map_or_else(
            || (self.get)(index).len(),
            // View arrays store each value's length in the low 32 bits of
            // the u128 view word, so budgeting need not touch a data buffer.
            |views| views[index] as u32 as usize,
        )
    }

    #[inline]
    fn range_encoded_upper_bound(self, _start: usize, len: usize) -> Option<usize> {
        self.max_encoded_size
            .map(|per_value| per_value.saturating_mul(len))
    }

    #[inline]
    fn is_variable_width(self) -> bool {
        self.views.is_some()
    }
}

/// Byte-value cursor driven by a physical-index plan. Grouped dictionary input
/// remains compact through `for_each_run_group`.
#[derive(Clone, Copy)]
struct PhysicalByteValues<'a, A> {
    plan: PhysicalIndexPlan<'a>,
    values: A,
}

impl<'a, A> ValueCursor<&'a [u8]> for PhysicalByteValues<'a, A>
where
    A: ByteValueAccess<'a> + 'a,
{
    #[inline]
    fn len(self) -> usize {
        self.plan.len()
    }

    #[inline]
    fn try_for_each<E>(self, mut f: impl FnMut(&'a [u8]) -> Result<(), E>) -> Result<(), E> {
        if !self.plan.is_grouped() {
            return self
                .plan
                .try_for_each_index(|index| f(self.values.value(index)));
        }

        self.plan.try_for_each_span(|span| match span {
            IndexSpan::Range { start, len } => self.values.try_for_each_range(start, len, &mut f),
            IndexSpan::Repeat { index, count } => {
                let value = self.values.value(index);
                for _ in 0..count {
                    f(value)?;
                }
                Ok(())
            }
        })
    }

    #[inline]
    fn for_each_run_group<E>(
        self,
        mut f: impl FnMut(&'a [u8], usize) -> Result<(), E>,
    ) -> Result<(), E> {
        self.plan
            .try_for_each_index_group(|index, count| f(self.values.value(index), count))
    }
}

impl<'a, A> ByteBatchSource<'a> for PhysicalByteValues<'a, A>
where
    A: ByteValueAccess<'a> + 'a,
{
    #[inline]
    fn drive_flat(self, sink: &mut ByteSink<'a, '_>) -> Result<()> {
        if let Some(selection) = self.plan.unmapped_selection() {
            match selection {
                ValueSelectionRef::Empty => return Ok(()),
                ValueSelectionRef::Dense { offset, len } => {
                    if let Some(batch) = self.values.contiguous_batch(offset, len) {
                        return sink.commit(batch);
                    }
                }
                ValueSelectionRef::Ranges(ranges) => {
                    // Offset layouts can lend every contiguous range directly.
                    // Other layouts retain one gathered stream so short ranges
                    // still fill complete tiles.
                    if self.values.contiguous_batch(0, 0).is_some() {
                        return ranges.try_for_each_range(|offset, len| {
                            sink.commit(
                                self.values
                                    .contiguous_batch(offset, len)
                                    .expect("offset layout must lend every range"),
                            )
                        });
                    }
                }
                ValueSelectionRef::Sparse(_) => {}
                ValueSelectionRef::Grouped(_) => {}
            }
        } else if let Some(range) = self.plan.direct_physical_range() {
            if let Some(batch) = self.values.contiguous_batch(range.start, range.len()) {
                return sink.commit(batch);
            }
        }

        gather_tiled::<BYTE_ARRAY_BATCH_VALUES, _, _, _>(self, |values| {
            sink.commit(FlatByteBatch::Gathered(values))
        })
    }
}

/// The final, already-downcast Arrow byte layout. Wrapper lowering and layout
/// dispatch happen once when a leaf source is bound; page slicing, budgeting,
/// and encoding retain this small borrowed descriptor.
#[derive(Clone, Copy)]
pub(crate) struct ByteArrayStorage<'a>(ByteArrayStorageKind<'a>);

#[derive(Clone, Copy)]
enum ByteArrayStorageKind<'a> {
    Offset32(OffsetByteAccess<'a, i32>),
    Offset64(OffsetByteAccess<'a, i64>),
    Utf8View {
        values: &'a StringViewArray,
        max_value_len: usize,
    },
    BinaryView {
        values: &'a BinaryViewArray,
        max_value_len: usize,
    },
    Fixed(&'a FixedSizeBinaryArray),
}

impl<'a> ArrowPhysicalBridge<'a> for ByteArrayStorage<'a> {
    type Encoder = ColumnValueEncoderImpl<ParquetByteArrayType>;

    fn bind(values: &'a dyn Array) -> Result<Self> {
        let storage = match values.data_type() {
            DataType::Utf8 => Self(ByteArrayStorageKind::Offset32(OffsetByteAccess::bind::<
                Utf8Type,
            >(values))),
            DataType::LargeUtf8 => Self(ByteArrayStorageKind::Offset64(OffsetByteAccess::bind::<
                LargeUtf8Type,
            >(values))),
            DataType::Binary => Self(ByteArrayStorageKind::Offset32(OffsetByteAccess::bind::<
                BinaryType,
            >(values))),
            DataType::LargeBinary => {
                Self(ByteArrayStorageKind::Offset64(OffsetByteAccess::bind::<
                    LargeBinaryType,
                >(values)))
            }
            DataType::Utf8View => {
                let values = values.as_any().downcast_ref::<StringViewArray>().unwrap();
                Self(ByteArrayStorageKind::Utf8View {
                    values,
                    max_value_len: max_view_value_len(values.data_buffers()),
                })
            }
            DataType::BinaryView => {
                let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                Self(ByteArrayStorageKind::BinaryView {
                    values,
                    max_value_len: max_view_value_len(values.data_buffers()),
                })
            }
            DataType::FixedSizeBinary(_) => Self(ByteArrayStorageKind::Fixed(
                values
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap(),
            )),
            data_type => {
                return Err(ParquetError::General(format!(
                    "Cannot coerce final physical {data_type} to BYTE_ARRAY"
                )));
            }
        };
        Ok(storage)
    }

    fn write_values(self, encoder: &mut Self::Encoder, plan: PhysicalIndexPlan<'a>) -> Result<()> {
        self.dispatch(WritePhysicalBytes { encoder, plan })
    }

    fn count_variable_width_within_byte_budget(
        self,
        plan: PhysicalIndexPlan<'a>,
        budget: usize,
        target: ByteBudgetTarget,
    ) -> Option<usize> {
        if plan.len() == 0 {
            return None;
        }

        // Dictionary-key mappings do not model dictionary growth.
        if target == ByteBudgetTarget::DictionaryPage && plan.has_dictionary_mapping() {
            return None;
        }

        self.dispatch(CountPhysicalBytes { plan, budget })
    }
}

impl<'a> ByteArrayStorage<'a> {
    #[inline]
    fn dispatch<V: ByteLayoutVisitor<'a>>(self, visitor: V) -> V::Out {
        match self.0 {
            ByteArrayStorageKind::Offset32(values) => visitor.visit(values),
            ByteArrayStorageKind::Offset64(values) => visitor.visit(values),
            ByteArrayStorageKind::Utf8View {
                values,
                max_value_len,
            } => visitor.visit(IndexedByteAccess::variable(
                move |index| values.value(index).as_bytes(),
                values.views(),
                max_value_len,
            )),
            ByteArrayStorageKind::BinaryView {
                values,
                max_value_len,
            } => visitor.visit(IndexedByteAccess::variable(
                move |index| values.value(index),
                values.views(),
                max_value_len,
            )),
            ByteArrayStorageKind::Fixed(values) => {
                visitor.visit(IndexedByteAccess::fixed(move |index| values.value(index)))
            }
        }
    }
}

/// Continuation used by [`ByteArrayStorage::dispatch`]. Both the writer and
/// page budget calculator operate on the same concrete physical layout.
trait ByteLayoutVisitor<'a>: Sized {
    type Out;
    fn visit<A: ByteValueAccess<'a> + 'a>(self, values: A) -> Self::Out;
}

struct WritePhysicalBytes<'a, 'e> {
    encoder: &'e mut ColumnValueEncoderImpl<ParquetByteArrayType>,
    plan: PhysicalIndexPlan<'a>,
}

impl<'a> ByteLayoutVisitor<'a> for WritePhysicalBytes<'a, '_> {
    type Out = Result<()>;

    #[inline]
    fn visit<A: ByteValueAccess<'a> + 'a>(self, values: A) -> Self::Out {
        let values = PhysicalByteValues {
            plan: self.plan,
            values,
        };
        if self.plan.is_grouped() {
            self.encoder.write_byte_values_run_collapsed(values)
        } else {
            self.encoder.write_byte_values(values)
        }
    }
}

#[derive(Clone, Copy)]
struct CountPhysicalBytes<'a> {
    plan: PhysicalIndexPlan<'a>,
    budget: usize,
}

impl<'a> ByteLayoutVisitor<'a> for CountPhysicalBytes<'a> {
    type Out = Option<usize>;

    fn visit<A: ByteValueAccess<'a> + 'a>(self, values: A) -> Self::Out {
        // Fixed-size byte arrays with direct or mapped dictionary input are
        // bounded by value count, while run-driven FSB input has no
        // variable-width estimate.
        if !values.is_variable_width() {
            return (!self.plan.is_grouped() || !self.plan.has_dictionary_mapping())
                .then_some(self.plan.len());
        }
        Some(count_plan_within_byte_budget(
            self.plan,
            values,
            self.budget,
        ))
    }
}

/// Count the leading physical-plan values within a PLAIN byte budget. Range
/// spans over offset arrays cost O(1); repeat spans charge `(len + 4) * count`
/// without expanding the run. The result is a strict fit and may be zero; the
/// column chunker retries the first non-fitting value on a fresh page and only
/// then forces oversize-value progress.
fn count_plan_within_byte_budget<'a, A: ByteValueAccess<'a>>(
    plan: PhysicalIndexPlan<'a>,
    values: A,
    budget: usize,
) -> usize {
    let prefix = std::mem::size_of::<u32>();

    // A single direct range can use either exact offset accounting or the
    // conservative view bound without a second plan traversal.
    if let Some(range) = plan.direct_physical_range() {
        if values
            .exact_range_encoded_size(range.start, range.len())
            .or_else(|| values.range_encoded_upper_bound(range.start, range.len()))
            .is_some_and(|encoded| encoded <= budget)
        {
            return range.len();
        }
    }

    let mut remaining = budget;
    let mut count = 0usize;
    if let Some(ValueSelectionRef::Sparse(indices)) = plan.unmapped_selection() {
        for &index in indices {
            let encoded = values.value_len(index).saturating_add(prefix);
            if encoded > remaining {
                break;
            }
            remaining -= encoded;
            count += 1;
        }
        return count;
    }

    let _: std::result::Result<(), ()> = plan.try_for_each_span(|span| match span {
        IndexSpan::Range { start, len } => {
            if let Some(encoded) = values.exact_range_encoded_size(start, len) {
                if encoded <= remaining {
                    remaining -= encoded;
                    count += len;
                    return Ok(());
                }
            }

            for index in start..start + len {
                let encoded = values.value_len(index).saturating_add(prefix);
                if encoded > remaining {
                    return Err(());
                }
                remaining -= encoded;
                count += 1;
            }
            Ok(())
        }
        IndexSpan::Repeat {
            index,
            count: span_count,
        } => {
            let encoded = values.value_len(index).saturating_add(prefix);
            let span_bytes = encoded.saturating_mul(span_count);
            if span_bytes <= remaining {
                remaining -= span_bytes;
                count += span_count;
                return Ok(());
            }

            count += (remaining / encoded).min(span_count);
            Err(())
        }
    });
    count
}

/// Upper bound on any single value's byte length in a view array.
fn max_view_value_len(buffers: &[Buffer]) -> usize {
    /// Bytes that fit inline in a u128 view word (the rest is len + prefix).
    const MAX_INLINE_VIEW_LEN: usize = 12;
    // An out-of-line view's data is a contiguous slice of exactly one data
    // buffer, so it cannot exceed the largest buffer; inline views hold at
    // most `MAX_INLINE_VIEW_LEN`. Loose (a value is usually far smaller than
    // a whole buffer) but O(number of buffers) and always sound.
    buffers
        .iter()
        .map(|b| b.len())
        .max()
        .unwrap_or(0)
        .max(MAX_INLINE_VIEW_LEN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::value::{RangesSelectionRef, SelectionRange, ValueSelectionRef};
    use arrow_array::{LargeStringArray, StringArray};

    fn count_selection(
        values: &dyn Array,
        selection: ValueSelectionRef<'_>,
        budget: usize,
    ) -> usize {
        ByteArrayStorage::bind(values)
            .unwrap()
            .dispatch(CountPhysicalBytes {
                plan: PhysicalIndexPlan::identity(selection),
                budget,
            })
            .unwrap()
    }

    #[derive(Clone, Copy)]
    struct DirectBatch<'a> {
        plan: PhysicalIndexPlan<'a>,
    }

    impl<'a> ByteLayoutVisitor<'a> for DirectBatch<'a> {
        type Out = Option<FlatByteBatch<'a, 'a>>;

        fn visit<A: ByteValueAccess<'a> + 'a>(self, values: A) -> Self::Out {
            let range = self.plan.direct_physical_range()?;
            values.contiguous_batch(range.start, range.len())
        }
    }

    fn direct_batch<'a>(
        values: &'a dyn Array,
        selection: ValueSelectionRef<'a>,
    ) -> Option<FlatByteBatch<'a, 'a>> {
        ByteArrayStorage::bind(values)
            .unwrap()
            .dispatch(DirectBatch {
                plan: PhysicalIndexPlan::identity(selection),
            })
    }

    #[test]
    fn direct_offset_batches_preserve_selected_subrange() {
        let values = StringArray::from(vec!["a", "bb", "ccc", "dddd"]);
        let selection = ValueSelectionRef::Dense { offset: 1, len: 2 };
        let batch = direct_batch(&values, selection).unwrap();

        // Five payload bytes plus two four-byte PLAIN length prefixes.
        assert_eq!(batch.exact_plain_size(), Some((5, 13)));
        let FlatByteBatch::Offset32 { offsets, data } = batch else {
            panic!("small-offset input did not produce an Offset32 batch");
        };
        assert_eq!(offsets, &[1, 3, 6]);
        assert_eq!(data, b"abbcccdddd");
        let selected: Vec<_> = offsets
            .windows(2)
            .map(|offset| &data[offset[0] as usize..offset[1] as usize])
            .collect();
        assert_eq!(selected, [b"bb".as_slice(), b"ccc".as_slice()]);

        // Large offsets can represent a single value longer than a Parquet
        // byte-array prefix, so they retain the validating sizing pass.
        let values = LargeStringArray::from(vec!["a", "bb", "ccc", "dddd"]);
        let batch = direct_batch(&values, selection).unwrap();
        assert_eq!(batch.exact_plain_size(), None);
        let FlatByteBatch::Offset64 { offsets, data } = batch else {
            panic!("large-offset input did not produce an Offset64 batch");
        };
        assert_eq!(offsets, &[1, 3, 6]);
        let selected: Vec<_> = offsets
            .windows(2)
            .map(|offset| &data[offset[0] as usize..offset[1] as usize])
            .collect();
        assert_eq!(selected, [b"bb".as_slice(), b"ccc".as_slice()]);
    }

    #[test]
    fn range_byte_budget_cutoffs_are_strict() {
        // Encoded sizes are 5, 6, 8, and 7 bytes for the selected values.
        // A five-byte budget fits the first value exactly; the second value is
        // retried by the chunker on a fresh page.
        let values = StringArray::from(vec!["a", "bb", "cccc", "not selected", "ddd"]);
        let ranges = [SelectionRange::new(0..3, 3), SelectionRange::new(4..5, 4)];
        let selection = ValueSelectionRef::Ranges(RangesSelectionRef::new(&ranges, 4));
        assert_eq!(count_selection(&values, selection, 5), 1);

        // The first range costs 11 bytes. The first value in the second range
        // costs 7, exactly consuming the 18-byte budget; the following 8-byte
        // value is retried on a fresh page.
        let values = LargeStringArray::from(vec![
            "a",
            "bb",
            "not selected",
            "also not selected",
            "ccc",
            "dddd",
            "eeeee",
        ]);
        let ranges = [SelectionRange::new(0..2, 2), SelectionRange::new(4..7, 5)];
        let selection = ValueSelectionRef::Ranges(RangesSelectionRef::new(&ranges, 5));

        assert_eq!(count_selection(&values, selection, 18), 3);
    }

    #[test]
    fn grouped_dictionary_budget_charges_logical_occurrences() {
        let values = StringArray::from(vec!["a", "bbbb"]);
        let indices = [0usize, 1];
        let ends = [3, 5];
        let plan = PhysicalIndexPlan::identity(ValueSelectionRef::Grouped(
            GroupedSelectionRef::new(&indices, &ends),
        ));
        let storage = ByteArrayStorage::bind(&values).unwrap();

        assert_eq!(
            storage.count_variable_width_within_byte_budget(
                plan,
                15,
                ByteBudgetTarget::DictionaryPage,
            ),
            Some(3)
        );
        assert_eq!(
            storage.count_variable_width_within_byte_budget(
                plan,
                14,
                ByteBudgetTarget::DictionaryPage,
            ),
            Some(2)
        );
    }
}
