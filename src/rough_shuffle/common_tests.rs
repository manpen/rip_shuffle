#![allow(unused_macros)]

macro_rules! rough_shuffle_single_test {
    ($name : ident, $func : ident, $log_n : expr) => {
        mod $name {
            use super::*;
            use crate::bucketing::*;
            use itertools::Itertools;
            use rand::{Rng, SeedableRng};
            use rand_pcg::Pcg64Mcg;

            const LOG_NUM_BUCKETS: usize = $log_n;
            const NUM_BUCKETS: usize = 1 << LOG_NUM_BUCKETS;
            const SWAPS_PER_ROUND: usize = 64 / $log_n;
            type R = Pcg64Mcg;
            type T = usize;

            #[test]
            #[ignore]
            fn uniformity() {
                let mut rng = R::seed_from_u64(0x9654_3723_3489 + 23423 * NUM_BUCKETS as u64);
                for n in [10, 20, 30, 40] {
                    test_rough_shuffle_impl::<LOG_NUM_BUCKETS, NUM_BUCKETS>(
                        &mut rng,
                        n * NUM_BUCKETS,
                        10_000 * NUM_BUCKETS as u64,
                    );
                }
            }

            #[test]
            fn preserve_elements() {
                let mut rng = R::seed_from_u64(0x9654_3723_3489);
                for n in 1..500 {
                    let mut data: Vec<usize> = (0..n).into_iter().collect();

                    {
                        let mut buckets =
                        split_slice_into_equally_sized_buckets::<usize, NUM_BUCKETS>(&mut data);

                        $func :: <R, T, LOG_NUM_BUCKETS, NUM_BUCKETS, SWAPS_PER_ROUND>(
                            &mut rng,
                            &mut buckets,
                        );

                        if rng.gen_bool(0.25) {
                            compact_into_single_bucket(buckets);
                        }
                    }

                    data.sort();

                    assert!(data.iter().enumerate().all(|(i, &x)| i == x));
                }
            }

            fn test_rough_shuffle_impl<const LOG_NUM_BUCKETS: usize, const NUM_BUCKETS: usize>(
                rng: &mut impl Rng,
                num_elem: usize,
                num_iter: u64,
            ) {
                let num_min_processed = num_elem / NUM_BUCKETS - 1;

                let mut data = vec![0; num_elem];

                let mut counts = vec![vec![0u64; num_min_processed]; NUM_BUCKETS];

                let mut bucket_sizes = vec![0; NUM_BUCKETS];

                for _ in 0..num_iter {
                    let mut buckets = split_slice_into_equally_sized_buckets::<usize, NUM_BUCKETS>(&mut data);
                    for (bucket_idx, bucket) in buckets.iter_mut().enumerate() {
                        bucket.data_mut().fill(bucket_idx);
                        bucket_sizes[bucket_idx] = bucket.len();
                    }

                    $func::<_, _, LOG_NUM_BUCKETS, NUM_BUCKETS, SWAPS_PER_ROUND>(rng, &mut buckets);

                    let compact = compact_into_single_bucket(buckets);

                    assert!(compact.num_processed() >= num_min_processed);

                    for (idx, &origin_bucket) in compact
                        .data_processed()
                        .iter()
                        .enumerate()
                        .take(num_min_processed)
                    {
                        counts[origin_bucket][idx] += 1;
                    }
                }

                // self check: each position was counts `num_iter` times
                for i in 0..num_min_processed {
                    assert_eq!(counts.iter().map(|cnts| cnts[i]).sum::<u64>(), num_iter);
                }

                let significance = 0.001;
                // Bonferroni correction as we carry out `num_min_processed`-many independent trails
                let corrected_significance = significance / num_elem as f64;

                let nice_counts = (0..num_min_processed)
                    .into_iter()
                    .map(|pos| {
                        (0..NUM_BUCKETS)
                            .into_iter()
                            .map(|o| format!("{:>7}", counts[o][pos]))
                            .join(" ")
                    })
                    .join("\n");

                let nice_sums = counts
                    .iter()
                    .map(|c| c.iter().sum::<u64>())
                    .map(|s| format!("{s:>7}"))
                    .join(" ");

                for (origin_idx, origin) in counts.iter().enumerate() {
                    for (idx, &count) in origin.iter().enumerate() {
                        let p_value = compute_binomial_p_value(
                            num_iter,
                            bucket_sizes[origin_idx] as f64 / num_elem as f64,
                            count,
                        );
                        assert!(
                            p_value >= corrected_significance,
                            "origin_idx: {origin_idx} idx: {idx} of {num_elem} count: {count} \np_value: {p_value} corrected_sig: {corrected_significance} \nbucket sizes: {:?}\n{nice_counts}\n+----\n{nice_sums}",
                            &bucket_sizes
                        );
                    }
                }
            }

            fn compute_binomial_p_value(num_counts: u64, success_prob: f64, actual_count: u64) -> f64 {
                use statrs::{
                    distribution::{Binomial, DiscreteCDF},
                    statistics::Distribution,
                };

                let distr = Binomial::new(success_prob, num_counts).unwrap();
                let mean = distr.mean().unwrap();

                // compute two-sided p-value, i.e. the probability that more extreme values are
                // produced by the binomial distribution
                if mean >= actual_count as f64 {
                    2.0 * distr.cdf(actual_count)
                } else {
                    2.0 * (1.0 - distr.cdf(actual_count - 1))
                }
            }
        }
    };
}

macro_rules! rough_shuffle_tests {
    ($func : ident) => {
        use crate::rough_shuffle::common_tests::rough_shuffle_single_test;
        rough_shuffle_single_test!(test_2buckets, $func, 1);
        rough_shuffle_single_test!(test_4buckets, $func, 2);
        rough_shuffle_single_test!(test_8buckets, $func, 3);
        rough_shuffle_single_test!(test_16buckets, $func, 4);
    };
}

pub(super) use rough_shuffle_single_test;
pub(super) use rough_shuffle_tests;
