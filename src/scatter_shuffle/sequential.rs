use std::default::Default;
use std::marker::PhantomData;

use super::*;
use crate::bucketing::slicing::Slicing;
use crate::bucketing::*;
use crate::fisher_yates::noncontiguous::noncontiguous_fisher_yates;
use crate::prelude::*;
use crate::rough_shuffle::*;

use arrayvec::ArrayVec;
use rand::Rng;
use rand_distr::Distribution;

pub const LOG_NUM_BUCKETS: usize = 7;
pub const NUM_BUCKETS: usize = 1 << LOG_NUM_BUCKETS;
pub const BASE_CASE_SIZE: usize = 1 << 18;

#[derive(Clone, Copy, Default)]
struct DefaultConfiguration {}
implement_seq_config!(DefaultConfiguration, fisher_yates, 1 << 19);

pub fn seq_scatter_shuffle<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    SeqScatterShuffleImpl::<R, T, DefaultConfiguration, NUM_BUCKETS>::default()
        .shuffle_adaptive(rng, data)
}

pub struct SeqScatterShuffleImpl<R, T, C, const NUM_BUCKETS: usize> {
    config: C,
    _phantom_r: PhantomData<R>,
    _phantom_t: PhantomData<T>,
}

impl<R, T, C, const NUM_BUCKETS: usize> Default for SeqScatterShuffleImpl<R, T, C, NUM_BUCKETS>
where
    C: Default,
{
    fn default() -> Self {
        Self {
            config: Default::default(),
            _phantom_r: Default::default(),
            _phantom_t: Default::default(),
        }
    }
}

impl<R, T, C, const NUM_BUCKETS: usize> SeqScatterShuffleImpl<R, T, C, NUM_BUCKETS>
where
    R: Rng,
    T: Sized,
    C: SeqConfiguration,
    NumberOfBuckets<NUM_BUCKETS>: IsPowerOfTwo,
{
    pub fn new(config: C) -> Self {
        Self {
            config,
            _phantom_r: Default::default(),
            _phantom_t: Default::default(),
        }
    }

    pub fn shuffle_adaptive(&self, rng: &mut R, data: &mut [T]) {
        let num_buckets = data.len() / self.config.seq_base_case_size() * 2;

        if num_buckets <= 2 {
            return self.config.seq_base_case_shuffle(rng, data);
        }

        if num_buckets >= NUM_BUCKETS {
            return self.shuffle(rng, data);
        }

        let log_num_buckets = num_buckets.ilog2();

        macro_rules! call {
            ($log_n : expr) => {
                SeqScatterShuffleImpl::<R, T, C, { 1 << $log_n }>::new(self.config.clone())
                    .shuffle(rng, data)
            };
        }

        match log_num_buckets {
            1 => call!(1),
            2 => call!(2),
            3 => call!(3),
            4 => call!(4),
            5 => call!(5),
            6 => call!(6),
            7 => call!(7),
            8 => call!(8),
            9 => call!(9),
            10 => call!(10),
            _ => self.shuffle(rng, data),
        }
    }

    pub fn shuffle(&self, rng: &mut R, data: &mut [T]) {
        if data.len() <= self.config.seq_base_case_size() {
            return self.config.seq_base_case_shuffle(rng, data);
        }

        let mut buckets = split_slice_into_equally_sized_buckets(data);

        rough_shuffle(rng, &mut buckets);

        let num_unprocessed = buckets.iter().map(|b| b.num_unprocessed()).sum();

        let target_lengths = sample_final_bucket_size(rng, num_unprocessed, &buckets);
        move_buckets_to_fit_target_len(&mut buckets, &target_lengths);

        shuffle_stashes(rng, &mut buckets, |rng: &mut R, data: &mut [T]| {
            self.shuffle_adaptive(rng, data)
        });

        if !self.config.seq_disable_recursion() {
            for bucket in &mut buckets {
                self.shuffle_adaptive(rng, bucket.data_mut());
            }
        }
    }
}

pub fn shuffle_stashes<R: Rng, T, const NUM_BUCKETS: usize>(
    rng: &mut R,
    buckets: &mut Buckets<T, NUM_BUCKETS>,
    mut recurse: impl FnMut(&mut R, &mut [T]),
) -> usize {
    let stash_size = buckets.iter().map(|blk| blk.num_unprocessed()).sum();
    if stash_size <= buckets[NUM_BUCKETS - 1].len() {
        // typically the unprocessed items should easily fit the last bucket. Then, it's fastes
        // to compact all stashes into a contiguous range and recurse to shuffle them
        compact_ranges(buckets);
        recurse(rng, buckets[NUM_BUCKETS - 1].data_mut().suffix(stash_size));
        compact_ranges(buckets);
    } else {
        // however, for really small input (or astronomically unlikely cases), the number of
        // unprocessed items may be too large. It's really not worth the effort of doing something
        // clever/error-prone. We rather use the slow noncontigous Fisher Yates implementation.
        let mut unprocessed: ArrayVec<&mut [T], NUM_BUCKETS> = buckets
            .iter_mut()
            .map(|blk| blk.data_unprocessed_mut())
            .collect();

        noncontiguous_fisher_yates(rng, &mut unprocessed);
    }
    stash_size
}

pub fn compact_ranges<T>(buckets: &mut [Bucket<T>]) -> usize {
    let (acceptor, doners) = buckets.split_last_mut().unwrap();

    let mut num_accepted = acceptor.num_unprocessed();
    let mut space_available = acceptor.num_processed();

    for bucket in doners.iter_mut().rev() {
        if bucket.num_unprocessed() == 0 {
            continue;
        }

        let to_accept = bucket.num_unprocessed();

        debug_assert!(to_accept <= space_available);

        bucket.data_unprocessed_mut().swap_with_slice(
            acceptor
                .data_mut()
                .suffix(num_accepted + to_accept)
                .prefix(to_accept),
        );

        num_accepted += to_accept;
        space_available -= to_accept;
    }

    num_accepted
}

pub fn move_buckets_to_fit_target_len<T, const NUM_BUCKETS: usize>(
    buckets: &mut Buckets<T, NUM_BUCKETS>,
    target_lengths: &[usize; NUM_BUCKETS],
) {
    shrink_sweep_to_right(buckets, target_lengths);
    shrink_sweep_to_left(buckets, target_lengths);

    debug_assert!(buckets
        .iter()
        .zip(target_lengths.iter())
        .all(|(blk, &target)| blk.len() == target));
}

fn shrink_sweep_to_right<T, const NUM_BUCKETS: usize>(
    buckets: &mut Buckets<T, NUM_BUCKETS>,
    target_lengths: &[usize; NUM_BUCKETS],
) {
    // compute exclusive prefix sum of the iterator above
    let mut growth_needed_left = 0;

    let mut buckets = buckets.as_mut_slice();

    for &target_length in &target_lengths[0..NUM_BUCKETS - 1] {
        let this_bucket;
        (this_bucket, buckets) = buckets.split_first_mut().unwrap();

        let reservation_for_left = growth_needed_left.max(0) as usize;
        let target_with_reservation = target_length + reservation_for_left;

        if this_bucket.len() > target_with_reservation {
            let num_to_move = this_bucket.len() - target_with_reservation;

            this_bucket.shrink_to_right(buckets.first_mut().unwrap(), num_to_move);
        }

        growth_needed_left += target_length as isize - this_bucket.len() as isize;
    }
}

fn shrink_sweep_to_left<T, const NUM_BUCKETS: usize>(
    buckets: &mut Buckets<T, NUM_BUCKETS>,
    target_lengths: &[usize; NUM_BUCKETS],
) {
    let mut buckets = buckets.as_mut_slice();
    for &target in target_lengths[1..].iter().rev() {
        let this_bucket;
        (this_bucket, buckets) = buckets.split_last_mut().unwrap();

        if this_bucket.len() <= target {
            continue;
        }

        let too_long_by = this_bucket.len() - target;
        buckets
            .last_mut()
            .unwrap()
            .grow_from_right(this_bucket, too_long_by);
    }
}

pub fn sample_final_bucket_size<R: Rng, T, const NUM_BUCKETS: usize>(
    rng: &mut R,
    num_unprocessed: usize,
    buckets: &Buckets<T, NUM_BUCKETS>,
) -> [usize; NUM_BUCKETS] {
    fn multinomial<R: Rng>(
        rng: &mut R,
        num_bins: usize,
        mut num_balls: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        (0..num_bins).into_iter().map(move |i| {
            let remaining_bins = num_bins - i;
            let into_this_bin =
                rand_distr::Binomial::new(num_balls as u64, 1.0 / (remaining_bins as f64))
                    .unwrap()
                    .sample(rng) as usize;
            num_balls -= into_this_bin;
            into_this_bin
        })
    }

    let mut target_len = [0usize; NUM_BUCKETS];

    for (target, (bucket, additional)) in target_len.iter_mut().zip(
        buckets
            .iter()
            .zip(multinomial(rng, NUM_BUCKETS, num_unprocessed)),
    ) {
        *target = bucket.num_processed() + additional;
    }

    target_len
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::{seq::SliceRandom, SeedableRng};
    use rand_pcg::Pcg64;

    use super::*;

    macro_rules! invoke_for_pot {
        ($func : ident) => {
            let mut rng = Pcg64::seed_from_u64(12345);
            $func::<1>(&mut rng);
            $func::<2>(&mut rng);
            $func::<4>(&mut rng);
            $func::<8>(&mut rng);
            $func::<16>(&mut rng);
            $func::<32>(&mut rng);
            $func::<64>(&mut rng);
            $func::<128>(&mut rng);
        };
    }

    macro_rules! invoke_with_random_buckets {
        ($func:ident) => {
            fn generate_random_data<const NUM_BUCKETS: usize>(rng: &mut impl Rng) {
                let mut data = Vec::new();

                for _ in 0..10 {
                    let buckets = generate_random_buckets::<NUM_BUCKETS>(rng, &mut data);
                    let num_unprocessed = buckets.iter().map(|blk| blk.num_unprocessed()).sum();
                    let target_lengths: [usize; NUM_BUCKETS] =
                        sample_final_bucket_size(rng, num_unprocessed, &buckets);

                    $func(rng, buckets, target_lengths);
                }
            }

            invoke_for_pot!(generate_random_data);
        };
    }

    #[test]
    fn draw_unprocessed_distribution() {
        fn test_impl<const NUM_BUCKETS: usize>(rng: &mut impl Rng) {
            let total_length = 10 * NUM_BUCKETS;
            let mut data = vec![0; total_length];
            let mut buckets = split_slice_into_equally_sized_buckets(&mut data);
            for bucket in &mut buckets {
                bucket.set_num_processed(bucket.len());
            }

            let num_unprocessed = rng.gen_range(NUM_BUCKETS..total_length);
            for _ in 0..num_unprocessed {
                loop {
                    let bucket = buckets.choose_mut(rng).unwrap();
                    if bucket.num_processed() > 0 {
                        bucket.set_num_processed(bucket.num_processed() - 1);
                        break;
                    }
                }
            }

            let target_lengths: [usize; NUM_BUCKETS] =
                super::sample_final_bucket_size(rng, num_unprocessed, &buckets);

            assert!(buckets
                .iter()
                .zip_eq(&target_lengths)
                .all(|(blk, &target)| target >= blk.num_processed()));

            assert_eq!(target_lengths.iter().sum::<usize>(), total_length);
        }

        invoke_for_pot!(test_impl);
    }

    #[test]
    fn compact_ranges() {
        fn test_impl<const NUM_BUCKETS: usize>(
            _rng: &mut impl Rng,
            mut buckets: Buckets<usize, NUM_BUCKETS>,
            _target_lengths: [usize; NUM_BUCKETS],
        ) {
            let num_stash: usize = buckets.iter().map(|r| r.num_unprocessed()).sum();
            if num_stash > buckets.last().unwrap().len() {
                return;
            }

            mark_unprocessed_data(&mut buckets);

            super::compact_ranges(&mut buckets);

            assert_eq!(
                buckets.iter().map(|r| r.num_unprocessed()).sum::<usize>(),
                num_stash
            );

            assert!(buckets
                .last()
                .unwrap()
                .data()
                .suffix(num_stash)
                .iter()
                .all(|x| *x != 0));

            let data = merge_data(&buckets);

            assert_eq!(
                sort_dedup(&data),
                (0..=num_stash).into_iter().collect_vec(),
                "num_stash = {num_stash}"
            );
        }

        invoke_with_random_buckets!(test_impl);
    }

    macro_rules! shrink_sweep_test_skeleton {
        ($sweep : ident) => {
            fn test_impl<const NUM_BUCKETS: usize>(
                _rng: &mut impl Rng,
                mut buckets: Buckets<usize, NUM_BUCKETS>,
                target_lengths: [usize; NUM_BUCKETS],
            ) {
                mark_unprocessed_data(&mut buckets);

                assert_processed_are_zero(&buckets);
                assert_unprocessed_are_non_zero(&buckets);
                let unprocessed_before = sort_dedup(&merge_data(&buckets));

                $sweep(&mut buckets, &target_lengths);

                assert_processed_are_zero(&buckets);
                assert_unprocessed_are_non_zero(&buckets);
                let unprocessed_after = sort_dedup(&merge_data(&buckets));

                assert_eq!(unprocessed_before, unprocessed_after,);
            }

            invoke_with_random_buckets!(test_impl);
        };
    }

    #[test]
    fn move_buckets_to_fit_target_len() {
        fn sweep<const NUM_BUCKETS: usize>(
            buckets: &mut Buckets<usize, NUM_BUCKETS>,
            target_lengths: &[usize; NUM_BUCKETS],
        ) {
            super::move_buckets_to_fit_target_len(buckets, target_lengths);
            for (bucket_idx, (bucket, &target)) in buckets.iter().zip(target_lengths).enumerate() {
                assert!(bucket.len() == target, "bucket_idx = {bucket_idx}");
            }
        }

        shrink_sweep_test_skeleton!(sweep);
    }

    #[test]
    fn shrink_sweep_to_left() {
        fn sweep<const NUM_BUCKETS: usize>(
            buckets: &mut Buckets<usize, NUM_BUCKETS>,
            target_lengths: &[usize; NUM_BUCKETS],
        ) {
            super::shrink_sweep_to_left(buckets, target_lengths);
            for (bucket_idx, (bucket, &target)) in
                buckets.iter().zip(target_lengths).enumerate().skip(1)
            {
                assert!(bucket.len() <= target, "bucket_idx = {bucket_idx}");
            }
        }

        shrink_sweep_test_skeleton!(sweep);
    }

    fn generate_random_buckets<'a, const NUM_BUCKETS: usize>(
        rng: &mut impl Rng,
        storage: &'a mut Vec<usize>,
    ) -> Buckets<'a, usize, NUM_BUCKETS> {
        let sizes = (0..NUM_BUCKETS)
            .into_iter()
            .map(|_| (rng.gen_range(0..30), rng.gen_range(0..10)))
            .collect_vec();
        storage.resize(sizes.iter().map(|(a, b)| a + b).sum(), 0);

        let mut data = storage.as_mut_slice();
        let mut buckets = Vec::new();
        for sze in &sizes {
            let bucket;
            (bucket, data) = data.split_at_mut(sze.0 + sze.1);
            buckets.push(Bucket::new_with_num_unprocessed(bucket, sze.1));
        }

        buckets.into_iter().collect()
    }

    fn mark_unprocessed_data(buckets: &mut [Bucket<usize>]) {
        buckets
            .iter_mut()
            .for_each(|blk| blk.data_processed_mut().fill(0));

        for (idx, value) in buckets
            .iter_mut()
            .flat_map(|r| r.data_unprocessed_mut().iter_mut())
            .enumerate()
        {
            *value = idx + 1;
        }
    }

    fn assert_processed_are_zero(buckets: &[Bucket<usize>]) {
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            for (idx, &dat) in bucket.data_processed().iter().enumerate() {
                assert_eq!(dat, 0, "bucket_idx={bucket_idx} i={idx}");
            }
        }
    }

    fn assert_unprocessed_are_non_zero(buckets: &[Bucket<usize>]) {
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            for (idx, &dat) in bucket.data_unprocessed().iter().enumerate() {
                assert_ne!(dat, 0, "bucket_idx={bucket_idx} i={idx}");
            }
        }
    }

    fn merge_data<T: Copy>(buckets: &[Bucket<T>]) -> Vec<T> {
        buckets
            .iter()
            .flat_map(|blk| blk.data().iter().copied())
            .collect()
    }

    fn sort_dedup<T: Copy + Ord>(data: &[T]) -> Vec<T> {
        let mut data: Vec<T> = data.into();
        data.sort();
        data.dedup();
        data
    }
}

#[cfg(test)]
mod integration_test {
    use super::*;

    pub fn inplace_scatter_shuffle_test<R: Rng + SeedableRng, T: Send>(
        rng: &mut R,
        data: &mut [T],
    ) {
        const NUM_BUCKETS: usize = 4;

        #[derive(Clone, Copy, Default)]
        struct TestConfiguration {}
        implement_seq_config!(TestConfiguration, fisher_yates, NUM_BUCKETS * 4);

        SeqScatterShuffleImpl::<R, T, _, NUM_BUCKETS>::new(TestConfiguration::default())
            .shuffle(rng, data)
    }

    crate::statistical_tests::test_shuffle_algorithm!(inplace_scatter_shuffle_test);
}
