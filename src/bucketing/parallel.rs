use std::marker::PhantomData;

use super::*;
use crate::buckets::*;
use crate::prelude::fisher_yates;
use crate::rough_shuffle::*;

use rand::Rng;
use rand::SeedableRng;

#[derive(Clone, Copy, Default)]
struct DefaultConfiguration {}

implement_seq_config!(DefaultConfiguration, fisher_yates, 1 << 16); // not relevant, as we do not use SeqScatterShuffle

impl ParConfiguration for DefaultConfiguration {
    fn par_base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]) {
        fisher_yates(rng, data);
    }

    fn par_base_case_size(&self) -> usize {
        1 << 18
    }

    fn par_number_of_subproblems(&self, n: usize) -> usize {
        (n / self.par_base_case_size()).clamp(1, 2040)
    }
}

pub fn par_scatter_shuffle<R: Rng + SeedableRng + Send + Sync, T: Send + Sync + Sized>(
    rng: &mut R,
    data: &mut [T],
) {
    let num_bytes = data.len() * std::mem::size_of::<T>();

    if num_bytes <= (1 << 23) {
        return fisher_yates(rng, data);
    }

    if num_bytes < (1 << 27) {
        const NUM_BUCKETS: usize = 64;
        let algo = ParScatterShuffleImpl::<R, T, DefaultConfiguration, NUM_BUCKETS>::default();
        algo.shuffle(rng, data);
    } else {
        const NUM_BUCKETS: usize = 256;
        let algo = ParScatterShuffleImpl::<R, T, DefaultConfiguration, NUM_BUCKETS>::default();
        algo.shuffle(rng, data);
    }
}

pub struct ParScatterShuffleImpl<R, T, C, const NUM_BUCKETS: usize> {
    config: C,
    _phantom_r: PhantomData<R>,
    _phantom_t: PhantomData<T>,
}

impl<R, T, C, const NUM_BUCKETS: usize> Default for ParScatterShuffleImpl<R, T, C, NUM_BUCKETS>
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

impl<R, T, C, const NUM_BUCKETS: usize> ParScatterShuffleImpl<R, T, C, NUM_BUCKETS>
where
    R: Rng + SeedableRng + Send + Sync,
    T: Send + Sync + Sized,
    C: ParConfiguration,
    NumberOfBlocks<num_buckets>: IsPowerOfTwo,
{
    pub fn new(config: C) -> Self {
        Self {
            config,
            _phantom_r: Default::default(),
            _phantom_t: Default::default(),
        }
    }

    pub fn shuffle(&self, rng: &mut R, data: &mut [T]) {
        let n = data.len();

        if n <= self.config.par_base_case_size() {
            return self.config.par_base_case_shuffle(rng, data);
        }

        let mut buckets = split_slice_into_equally_sized_buckets(data);
        Self::invoke_rough_shuffle(rng, &mut buckets, self.config.par_number_of_subproblems(n));
        let num_unprocessed =
            sequential::shuffle_stashes(rng, &mut buckets, |r: &mut R, d: &mut [T]| {
                self.shuffle(r, d)
            });

        let target_lengths = sequential::draw_target_lengths(rng, num_unprocessed, &buckets);
        sequential::move_buckets_to_fit_target_len(&mut buckets, &target_lengths);

        if !self.config.par_disable_recursion() {
            self.recurse(rng, &mut buckets);
        }
    }

    fn invoke_rough_shuffle(
        rng: &mut R,
        buckets: &mut Buckets<T, num_buckets>,
        num_problems: usize,
    ) {
        if num_problems == 1 {
            return rough_shuffle(rng, buckets);
        }

        let mut right_rng: R = seed_new_rng(rng);
        let mut right_halves = split_each_bucket_in_half(buckets);

        rayon::join(
            || Self::invoke_rough_shuffle(rng, buckets, num_problems / 2),
            || {
                Self::invoke_rough_shuffle(
                    &mut right_rng,
                    &mut right_halves,
                    (num_problems + 1) / 2,
                )
            },
        );

        buckets
            .iter_mut()
            .zip(right_halves.iter_mut())
            .for_each(|(left, right)| {
                let left_taken = std::mem::take(left);
                let right = std::mem::take(right);
                *left = left_taken.merge_with_right_neighbor(right)
            });

        rough_shuffle(rng, buckets)
    }

    fn recurse(&self, rng: &mut R, buckets: &mut [Bucket<T>]) {
        if buckets.len() == 1 {
            return self.shuffle(rng, buckets[0].data_mut());
        }

        let (left_buckets, right_buckets) = buckets.split_at_mut(buckets.len() / 2);

        let mut right_rng: R = seed_new_rng(rng);
        let left_rng = rng;

        rayon::join(
            || self.recurse(left_rng, left_buckets),
            || self.recurse(&mut right_rng, right_buckets),
        );
    }
}

pub fn seed_new_rng<RIn: Rng, ROut: SeedableRng>(base: &mut RIn) -> ROut {
    let mut seed = ROut::Seed::default();
    base.try_fill_bytes(seed.as_mut()).unwrap();
    ROut::from_seed(seed)
}

#[cfg(test)]
mod integration_test {
    use super::*;

    const NUM_BUCKETS: usize = 4;

    #[derive(Clone, Copy, Default)]
    struct TestConfiguration {}

    implement_seq_config!(TestConfiguration, fisher_yates, 2);

    impl ParConfiguration for TestConfiguration {
        fn par_base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]) {
            sequential::scatter_shuffle_impl::<R, T, _, num_buckets>(rng, data, self)
        }

        fn par_base_case_size(&self) -> usize {
            1 << 18
        }

        fn par_number_of_subproblems(&self, n: usize) -> usize {
            (n / 2 / self.par_base_case_size()).max(1024)
        }
    }

    pub fn inplace_scatter_shuffle_test<
        R: Rng + SeedableRng + Send + Sync,
        T: Send + Sync + Sized,
    >(
        rng: &mut R,
        data: &mut [T],
    ) {
        let algo = ParScatterShuffleImpl::<R, T, TestConfiguration, num_buckets>::default();
        algo.shuffle(rng, data);
    }

    crate::statistical_tests::test_shuffle_algorithm!(inplace_scatter_shuffle_test);
}
