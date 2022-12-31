use std::marker::PhantomData;

use super::sequential;
use crate::blocked::*;
use crate::rough_shuffle::*;

use rand::Rng;
use rand::SeedableRng;

pub const LOG_NUM_BLOCKS: usize = 7;
pub const NUM_BLOCKS: usize = 1 << LOG_NUM_BLOCKS;

pub trait Configuration: Send + Sync {
    fn base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]);
    fn sequential_base_case_size(&self) -> usize;
    fn number_problems(&self, n: usize) -> usize;

    fn try_base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]) -> bool {
        if data.len() < self.sequential_base_case_size() {
            self.base_case_shuffle(rng, data);
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Copy, Default)]
struct DefaultConfiguration {}
impl Configuration for DefaultConfiguration {
    fn base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]) {
        const FY_BASE_CASE: usize = 1 << 16;
        sequential::scatter_shuffle_impl::<R, T, NUM_BLOCKS, FY_BASE_CASE>(rng, data)
    }

    fn sequential_base_case_size(&self) -> usize {
        1 << 18
    }

    fn number_problems(&self, n: usize) -> usize {
        (n / 2 / self.sequential_base_case_size()).max(128)
    }
}

pub fn par_scatter_shuffle<R: Rng + SeedableRng + Send + Sync, T: Send + Sync + Sized>(
    rng: &mut R,
    data: &mut [T],
) {
    let algo = ParScatterShuffleImpl::<R, T, DefaultConfiguration, NUM_BLOCKS>::default();
    algo.shuffle(rng, data);
}

pub struct ParScatterShuffleImpl<R, T, C, const NUM_BLOCKS: usize> {
    config: C,
    _phantom_r: PhantomData<R>,
    _phantom_t: PhantomData<T>,
}

impl<R, T, C, const NUM_BLOCKS: usize> Default for ParScatterShuffleImpl<R, T, C, NUM_BLOCKS>
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

impl<R, T, C, const NUM_BLOCKS: usize> ParScatterShuffleImpl<R, T, C, NUM_BLOCKS>
where
    R: Rng + SeedableRng + Send + Sync,
    T: Send + Sync + Sized,
    C: Configuration,
    NumberOfBlocks<NUM_BLOCKS>: IsPowerOfTwo,
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

        if self.config.try_base_case_shuffle(rng, data) {
            return;
        }

        let mut blocks = split_slice_into_blocks(data);

        Self::rough_shuffle(rng, &mut blocks, self.config.number_problems(n));

        let num_unprocessed =
            sequential::shuffle_stashes(rng, &mut blocks, |r: &mut R, d: &mut [T]| {
                self.shuffle(r, d)
            });
        let target_lengths = sequential::draw_target_lengths(rng, num_unprocessed, &blocks);
        sequential::move_blocks_to_fit_target_len(&mut blocks, &target_lengths);

        self.recurse(rng, &mut blocks);
    }

    fn rough_shuffle(rng: &mut R, blocks: &mut Blocks<T, NUM_BLOCKS>, num_problems: usize) {
        if num_problems == 1 {
            return rough_shuffle(rng, blocks);
        }

        let mut right_halves = split_each_block_in_half(blocks);
        let mut right_rng: R = seed_new_rng(rng);

        rayon::join(
            || Self::rough_shuffle(rng, blocks, num_problems / 2),
            || Self::rough_shuffle(&mut right_rng, &mut right_halves, (num_problems + 1) / 2),
        );

        blocks
            .iter_mut()
            .zip(right_halves.iter_mut())
            .for_each(|(left, right)| {
                let left_taken = std::mem::take(left);
                let right = std::mem::take(right);
                *left = left_taken.merge_with_right_neighbor(right)
            });

        rough_shuffle(rng, blocks)
    }

    fn recurse(&self, rng: &mut R, blocks: &mut [Block<T>]) {
        if blocks.len() == 1 {
            return self.shuffle(rng, blocks[0].data_mut());
        }

        let (left_blocks, right_blocks) = blocks.split_at_mut(blocks.len() / 2);

        let mut right_rng: R = seed_new_rng(rng);
        let left_rng = rng;

        rayon::join(
            || self.recurse(left_rng, left_blocks),
            || self.recurse(&mut right_rng, right_blocks),
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

    const NUM_BLOCKS: usize = 4;

    #[derive(Clone, Copy, Default)]
    struct TestConfiguration {}
    impl Configuration for TestConfiguration {
        fn base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]) {
            const FY_BASE_CASE: usize = 2;
            sequential::scatter_shuffle_impl::<R, T, NUM_BLOCKS, FY_BASE_CASE>(rng, data)
        }

        fn sequential_base_case_size(&self) -> usize {
            1 << 18
        }

        fn number_problems(&self, n: usize) -> usize {
            (n / 2 / self.sequential_base_case_size()).max(1024)
        }
    }

    pub fn inplace_scatter_shuffle_test<
        R: Rng + SeedableRng + Send + Sync,
        T: Send + Sync + Sized,
    >(
        rng: &mut R,
        data: &mut [T],
    ) {
        let algo = ParScatterShuffleImpl::<R, T, TestConfiguration, NUM_BLOCKS>::default();
        algo.shuffle(rng, data);
    }

    crate::statistical_tests::test_shuffle_algorithm!(inplace_scatter_shuffle_test);
}
