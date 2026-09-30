mod basic;
mod nested;

pub use basic::array_to_page;
pub use nested::array_to_page as nested_array_to_page;
use polars_arrow::array::{Array, FixedSizeBinaryArray, Float16Array, PrimitiveArray};
use polars_arrow::types::{NativeType, i256};
use polars_compute::min_max::MinMaxKernel;

use super::{EncodeNullability, StatisticsOptions};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::FixedLenStatistics;
use crate::parquet::types::NativeType as ParquetNativeType;

pub(crate) fn encode_plain(
    array: &FixedSizeBinaryArray,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    // append the non-null values
    if options.is_optional() && array.validity().is_some() {
        array.iter().for_each(|x| {
            if let Some(x) = x {
                buffer.extend_from_slice(x);
            }
        })
    } else {
        buffer.extend_from_slice(array.values());
    }
}

pub(super) fn build_statistics(
    array: &FixedSizeBinaryArray,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                #[expect(clippy::filter_map_identity)]
                array.iter().filter_map(|x| x).max().map(|x| x.to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                #[expect(clippy::filter_map_identity)]
                array.iter().filter_map(|x| x).min().map(|x| x.to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_float16(
    array: &Float16Array,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    // NaN is left out of the bounds, see the primitive `build_statistics`.
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .max_ignore_nan_kernel()
                    .filter(|x| !x.is_nan())
                    .map(|x| x.norm_max().to_le_bytes().as_ref().to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .min_ignore_nan_kernel()
                    .filter(|x| !x.is_nan())
                    .map(|x| x.norm_min().to_le_bytes().as_ref().to_vec())
            })
            .flatten(),
    }
}

/// Writes the bounds as big-endian bytes, using `T`'s own order. This is only a valid
/// parquet statistic if the column's declared sort order matches that order.
pub(super) fn build_statistics_big_endian<T>(
    array: &PrimitiveArray<T>,
    primitive_type: PrimitiveType,
    size: usize,
    options: &StatisticsOptions,
) -> FixedLenStatistics
where
    T: NativeType + Ord,
{
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .max()
                    .map(|x| x.to_be_bytes().as_ref()[16 - size..].to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .min()
                    .map(|x| x.to_be_bytes().as_ref()[16 - size..].to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_i256_big_endian_low(
    array: &PrimitiveArray<i256>,
    primitive_type: PrimitiveType,
    size: usize,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .max()
                    .map(|x| x.0.low().to_be_bytes()[16 - size..].to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .min()
                    .map(|x| x.0.low().to_be_bytes()[16 - size..].to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_i256_big_endian(
    array: &PrimitiveArray<i256>,
    primitive_type: PrimitiveType,
    size: usize,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .max()
                    .map(|x| x.0.to_be_bytes()[32 - size..].to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .min()
                    .map(|x| x.0.to_be_bytes()[32 - size..].to_vec())
            })
            .flatten(),
    }
}

#[cfg(test)]
mod tests {
    use polars_utils::float16::pf16;

    use super::*;
    use crate::parquet::schema::types::PhysicalType;

    fn check(values: &[Option<f32>], expected: Option<(f32, f32)>) {
        let array = Float16Array::from_iter(values.iter().map(|v| v.map(pf16::from)));
        let type_ = PrimitiveType::from_physical("a".into(), PhysicalType::FixedLenByteArray(2));
        let stats = build_statistics_float16(&array, type_, &StatisticsOptions::full());
        let bytes = |x: f32| pf16::from(x).to_le_bytes().to_vec();
        assert_eq!(
            stats.min_value,
            expected.map(|(l, _)| bytes(l)),
            "{values:?}"
        );
        assert_eq!(
            stats.max_value,
            expected.map(|(_, r)| bytes(r)),
            "{values:?}"
        );
    }

    #[test]
    fn float16_statistics_ignore_nan() {
        let nan = f32::NAN;
        check(&[Some(1.0), Some(nan), Some(3.0)], Some((1.0, 3.0)));
        check(&[Some(nan)], None);
        check(&[None, Some(nan), None], None);
        check(&[None, Some(nan), Some(2.0), None], Some((2.0, 2.0)));
    }
}
