use crate::uniform_index;

use super::prelude::*;

use super::blocked::*;
use super::rough_shuffle::*;

use rand::Rng;
use std::fmt::Debug;

const DEFAULT_NUM_BLOCKS: usize = 128;
const DEFAULT_FY_BASE_CASE: usize = 1 << 16;

pub fn merge_shuffle<R: Rng, T: Debug>(rng: &mut R, data: &mut [T]) {
    merge_shuffle_impl::<R, T, DEFAULT_NUM_BLOCKS, DEFAULT_FY_BASE_CASE>(rng, data)
}

fn merge_shuffle_impl<R: Rng, T: Debug, const NUM_BLOCKS: usize, const FY_BASE_CASE: usize>(
    rng: &mut R,
    data: &mut [T],
) where
    Number<NUM_BLOCKS>: IsPowerOfTwo,
{
    if data.len() < FY_BASE_CASE {
        return fisher_yates(rng, data);
    }

    let mut blocks: Blocks<T, NUM_BLOCKS> = split_slice_into_blocks(data);

    for block in &mut blocks {
        merge_shuffle_impl::<R, T, NUM_BLOCKS, FY_BASE_CASE>(rng, block.data_mut());
    }

    rough_shuffle(rng, &mut blocks);

    let mut recombined = compact_into_single_block(blocks);

    insertion_shuffle(rng, &mut recombined);
}

pub fn insertion_shuffle<R: Rng, T: std::fmt::Debug>(rng: &mut R, block: &mut Block<T>) {
    let unprocessed_range = block.num_processed()..block.len();
    let data = block.data_mut();

    for item in unprocessed_range {
        let partner = uniform_index::gen_index(rng, item + 1);
        data.swap(item, partner);
    }
}

#[cfg(test)]
mod test_ms2 {
    use super::*;

    fn merge_shuffle_test<R: Rng, T: std::fmt::Debug>(rng: &mut R, data: &mut [T]) {
        merge_shuffle_impl::<R, T, 2, 2>(rng, data)
    }

    crate::statistical_tests::test_shuffle_algorithm!(merge_shuffle_test);
    crate::statistical_tests::test_shuffle_algorithm_deterministic!(merge_shuffle_test);
}

#[cfg(test)]
mod test_ms4 {
    use super::*;

    fn merge_shuffle_test<R: Rng, T: std::fmt::Debug>(rng: &mut R, data: &mut [T]) {
        merge_shuffle_impl::<R, T, 4, 2>(rng, data)
    }

    crate::statistical_tests::test_shuffle_algorithm!(merge_shuffle_test);
    crate::statistical_tests::test_shuffle_algorithm_deterministic!(merge_shuffle_test);
}
