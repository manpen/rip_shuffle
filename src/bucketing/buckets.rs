#![allow(dead_code)]

use super::bucket::Bucket;
use arrayvec::ArrayVec;

pub type Buckets<'a, T, const N: usize> = ArrayVec<Bucket<'a, T>, N>;

pub fn split_slice_into_equally_sized_buckets<T, const N: usize>(
    mut data: &mut [T],
) -> Buckets<T, N> {
    let total_len = data.len();
    let mut buckets = ArrayVec::new();

    for i in 0..N {
        let start = i * total_len / N;
        let end = (i + 1) * total_len / N;
        let bucket_data;
        (bucket_data, data) = data.split_at_mut(end - start);
        buckets.push(Bucket::new(bucket_data));
    }

    buckets
}

pub fn compact_into_single_bucket<T, const N: usize>(mut buckets: Buckets<T, N>) -> Bucket<T> {
    let mut result = buckets.pop().unwrap();
    while let Some(bucket) = buckets.pop() {
        result = bucket.merge_with_right_neighbor(result);
    }
    result
}

pub fn split_each_bucket_in_half<'a, T, const N: usize>(
    buckets: &mut Buckets<'a, T, N>,
) -> Buckets<'a, T, N> {
    buckets
        .iter_mut()
        .map(|left| left.split_in_half())
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn split_slice_into_buckets() {
        let mut data: Vec<_> = (0..8).into_iter().collect();

        let buckets: Buckets<_, 2> = super::split_slice_into_equally_sized_buckets(&mut data);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets.as_slice()[0].len(), 4);
        assert_eq!(buckets.as_slice()[1].len(), 4);
    }

    #[test]
    fn compact_into_single_bucket() {
        for (((n0, n1), n2), n3) in (0..3)
            .cartesian_product(0..3)
            .cartesian_product(0..3)
            .cartesian_product(0..3)
        {
            let mut data: Vec<_> = vec![0; 12];
            let num_ones = n0 + n1 + n2 + n3;
            {
                let mut buckets: Buckets<_, 4> =
                    super::split_slice_into_equally_sized_buckets(&mut data);

                for (i, n) in [n0, n1, n2, n3].into_iter().enumerate() {
                    for _ in 0..n {
                        buckets[i].process_element();
                    }
                    buckets[i].data_processed_mut().fill(1);
                }

                let compact = super::compact_into_single_bucket(buckets);

                assert_eq!(compact.len(), 12);
                assert_eq!(compact.num_processed(), num_ones);
                assert_eq!(compact.num_unprocessed(), 12 - num_ones);

                assert_eq!(compact.data().iter().sum::<usize>(), num_ones);
                assert!(compact.data_processed().iter().all(|x| *x == 1));
                assert!(compact.data_unprocessed().iter().all(|x| *x == 0));
            }
        }
    }
}
