use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::scalar::PrimitiveScalar;
use polars_arrow::types::NativeType;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::float::IsFloat;

use super::super::{WriteOptions, utils};
use crate::arrow::read::schema::is_nullable;
use crate::arrow::write::utils::ExactSizedIter;
use crate::parquet::encoding::Encoding;
use crate::parquet::encoding::delta_bitpacked::encode;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::PrimitiveStatistics;
use crate::parquet::types::NativeType as ParquetNativeType;
use crate::read::Page;
use crate::write::{EncodeNullability, StatisticsOptions};

pub(crate) fn encode_plain<T, P>(
    array: &PrimitiveArray<T>,
    options: EncodeNullability,
    mut buffer: Vec<u8>,
) -> Vec<u8>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
{
    let is_optional = options.is_optional();

    if is_optional {
        // append the non-null values
        let validity = array.validity();

        if let Some(validity) = validity {
            let null_count = validity.unset_bits();

            if null_count > 0 {
                let mut iter = validity.iter();
                let values = array.values().as_slice();

                buffer.reserve(size_of::<P::Bytes>() * (array.len() - null_count));

                let mut offset = 0;
                let mut remaining_valid = array.len() - null_count;
                while remaining_valid > 0 {
                    let num_valid = iter.take_leading_ones();
                    buffer.extend(
                        values[offset..offset + num_valid]
                            .iter()
                            .flat_map(|value| value.as_().to_le_bytes()),
                    );
                    remaining_valid -= num_valid;
                    offset += num_valid;

                    let num_invalid = iter.take_leading_zeros();
                    offset += num_invalid;
                }

                return buffer;
            }
        }
    }

    buffer.reserve(size_of::<P>() * array.len());
    buffer.extend(
        array
            .values()
            .iter()
            .flat_map(|value| value.as_().to_le_bytes()),
    );

    buffer
}

pub(crate) fn encode_delta<T, P>(
    array: &PrimitiveArray<T>,
    options: EncodeNullability,
    mut buffer: Vec<u8>,
) -> Vec<u8>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
    P: num_traits::AsPrimitive<i64>,
{
    let is_optional = options.is_optional();

    if is_optional {
        // append the non-null values
        let iterator = array.non_null_values_iter().map(|x| {
            let parquet_native: P = x.as_();
            let integer: i64 = parquet_native.as_();
            integer
        });
        let iterator = ExactSizedIter::new(iterator, array.len() - array.null_count());
        encode(iterator, &mut buffer, 1)
    } else {
        // append all values
        let iterator = array.values().iter().map(|x| {
            let parquet_native: P = x.as_();
            let integer: i64 = parquet_native.as_();
            integer
        });
        encode(iterator, &mut buffer, 1)
    }
    buffer
}

pub fn array_to_page_plain<T, P>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
) -> PolarsResult<DataPage>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P> + IsFloat,
{
    array_to_page(array, options, type_, Encoding::Plain, encode_plain)
}

pub fn array_to_page_integer<T, P>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
) -> PolarsResult<Page>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P> + IsFloat,
    P: num_traits::AsPrimitive<i64>,
{
    match encoding {
        Encoding::Plain => array_to_page(array, options, type_, encoding, encode_plain),
        Encoding::DeltaBinaryPacked => array_to_page(array, options, type_, encoding, encode_delta),
        other => polars_bail!(nyi = "Encoding integer as {other:?}"),
    }
    .map(Page::Data)
}

pub fn array_to_page<T, P, F: Fn(&PrimitiveArray<T>, EncodeNullability, Vec<u8>) -> Vec<u8>>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
    encode: F,
) -> PolarsResult<DataPage>
where
    T: NativeType,
    P: ParquetNativeType,
    // constraint required to build statistics
    T: num_traits::AsPrimitive<P> + IsFloat,
{
    let is_optional = is_nullable(&type_.field_info);
    let encode_options = EncodeNullability::new(is_optional);

    let validity = array.validity();

    let mut buffer = vec![];
    utils::write_def_levels(
        &mut buffer,
        is_optional,
        validity,
        array.len(),
        options.version,
    )?;

    let definition_levels_byte_length = buffer.len();

    let buffer = encode(array, encode_options, buffer);

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, type_.clone(), &options.statistics).serialize())
    } else {
        None
    };

    utils::build_plain_page(
        buffer,
        array.len(),
        array.len(),
        array.null_count(),
        0,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        encoding,
    )
}

pub fn build_statistics<T, P>(
    array: &PrimitiveArray<T>,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> PrimitiveStatistics<P>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P> + IsFloat,
{
    // Float bounds leave out NaN, as Parquet requires. The NaN-ignoring kernels only give NaN
    // if all non-null values are NaN, and then the bound is left out.
    let (min_value, max_value) = match (options.min_value, options.max_value) {
        (true, true) => {
            match polars_compute::min_max::dyn_array_min_max_ignore_nan(array as &dyn Array) {
                None => (None, None),
                Some((l, r)) => (Some(l), Some(r)),
            }
        },
        (true, false) => (
            polars_compute::min_max::dyn_array_min_ignore_nan(array as &dyn Array),
            None,
        ),
        (false, true) => (
            None,
            polars_compute::min_max::dyn_array_max_ignore_nan(array as &dyn Array),
        ),
        (false, false) => (None, None),
    };

    let min_value = min_value.and_then(|s| {
        s.as_any()
            .downcast_ref::<PrimitiveScalar<T>>()
            .unwrap()
            .value()
            .filter(|x| !x.is_nan())
            .map(|x| x.as_().norm_min())
    });
    let max_value = max_value.and_then(|s| {
        s.as_any()
            .downcast_ref::<PrimitiveScalar<T>>()
            .unwrap()
            .value()
            .filter(|x| !x.is_nan())
            .map(|x| x.as_().norm_max())
    });

    PrimitiveStatistics::<P> {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value,
        min_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::schema::types::PhysicalType;

    fn bounds<T>(values: &[Option<T>], physical_type: PhysicalType) -> (Option<T>, Option<T>)
    where
        T: NativeType + num_traits::AsPrimitive<T> + IsFloat + ParquetNativeType,
    {
        let array = PrimitiveArray::from(values.to_vec());
        let type_ = PrimitiveType::from_physical("a".into(), physical_type);
        let stats = build_statistics::<T, T>(&array, type_, &StatisticsOptions::full());
        (stats.min_value, stats.max_value)
    }

    fn check(values: &[Option<f64>], expected: Option<(f64, f64)>) {
        let (min, max) = bounds(values, PhysicalType::Double);
        assert_eq!(min.zip(max), expected, "{values:?}");

        let values: Vec<_> = values.iter().map(|v| v.map(|x| x as f32)).collect();
        let expected = expected.map(|(l, r)| (l as f32, r as f32));
        let (min, max) = bounds(&values, PhysicalType::Float);
        assert_eq!(min.zip(max), expected, "{values:?}");
    }

    #[test]
    fn float_statistics_ignore_nan() {
        let nan = f64::NAN;
        check(&[Some(1.0), Some(nan), Some(3.0)], Some((1.0, 3.0)));
        check(&[Some(nan), Some(1.0), Some(3.0)], Some((1.0, 3.0)));
        check(&[Some(1.0), Some(3.0), Some(nan)], Some((1.0, 3.0)));
        check(&[Some(nan)], None);
        check(&[None, Some(nan), None], None);
        check(&[None, Some(nan), Some(2.0), None], Some((2.0, 2.0)));
        check(&[None, None], None);
    }
}
