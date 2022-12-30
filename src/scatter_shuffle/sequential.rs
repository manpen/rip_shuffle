use crate::blocked::slicing::Slicing;
use crate::blocked::*;
use crate::fisher_yates::noncontiguous::noncontiguous_fisher_yates;
use crate::prelude::*;
use crate::rough_shuffle::*;

#[cfg(feature = "timeiss")]
use std::time::Instant;

use arrayvec::ArrayVec;
use rand::Rng;
use rand_distr::Distribution;

pub const LOG_NUM_BLOCKS: usize = 7;
pub const NUM_BLOCKS: usize = 1 << LOG_NUM_BLOCKS;
pub const BASE_CASE_SIZE: usize = 1 << 18;

pub fn seq_scatter_shuffle<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    scatter_shuffle_impl::<R, T, NUM_BLOCKS, BASE_CASE_SIZE>(rng, data)
}

pub fn scatter_shuffle_impl<R, T, const NUM_BLOCKS: usize, const BASE_CASE_SIZE: usize>(
    rng: &mut R,
    data: &mut [T],
) where
    R: Rng,
    T: Sized,
    NumberOfBlocks<NUM_BLOCKS>: IsPowerOfTwo,
{
    if data.len() <= BASE_CASE_SIZE {
        return fisher_yates(rng, data);
    }

    let recurse = |rng: &mut R, data: &mut [T]| {
        scatter_shuffle_impl::<R, T, NUM_BLOCKS, BASE_CASE_SIZE>(rng, data)
    };

    let mut blocks = split_slice_into_blocks(data);

    rough_shuffle(rng, &mut blocks);

    let num_unprocessed = shuffle_stashes(rng, &mut blocks, recurse);
    let target_lengths = draw_target_lengths(rng, num_unprocessed, &blocks);
    move_blocks_to_fit_target_len(&mut blocks, &target_lengths);

    for block in &mut blocks {
        recurse(rng, block.data_mut());
    }
}

fn shuffle_stashes<R: Rng, T, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
    mut recurse: impl FnMut(&mut R, &mut [T]),
) -> usize {
    let stash_size = blocks.iter().map(|blk| blk.num_unprocessed()).sum();
    if stash_size <= blocks[NUM_BLOCKS - 1].len() {
        // typically the unprocessed items should easily fit the last block. Then, it's fastes
        // to compact all stashes into a contiguous range and recurse to shuffle them
        compact_ranges(blocks);
        recurse(rng, blocks[NUM_BLOCKS - 1].data_mut().suffix(stash_size));
        compact_ranges(blocks);
    } else {
        // however, for really small input (or astronomically unlikely cases), the number of
        // unprocessed items may be too large. It's really not worth the effort of doing something
        // clever/error-prone. We rather use the slow noncontigous Fisher Yates implementation.
        let mut unprocessed: ArrayVec<&mut [T], NUM_BLOCKS> = blocks
            .iter_mut()
            .map(|blk| blk.data_unprocessed_mut())
            .collect();

        noncontiguous_fisher_yates(rng, &mut unprocessed);
    }
    stash_size
}

pub fn compact_ranges<T>(blocks: &mut [Block<T>]) -> usize {
    let (acceptor, doners) = blocks.split_last_mut().unwrap();

    let mut num_accepted = acceptor.num_unprocessed();
    let mut space_available = acceptor.num_processed();

    for block in doners.iter_mut().rev() {
        if block.num_unprocessed() == 0 {
            continue;
        }

        let to_accept = block.num_unprocessed();

        debug_assert!(to_accept <= space_available);

        block.data_unprocessed_mut().swap_with_slice(
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

fn move_blocks_to_fit_target_len<T, const NUM_BLOCKS: usize>(
    blocks: &mut Blocks<T, NUM_BLOCKS>,
    target_lengths: &[usize; NUM_BLOCKS],
) {
    shrink_sweep_to_right(blocks, target_lengths);
    shrink_sweep_to_left(blocks, target_lengths);

    debug_assert!(blocks
        .iter()
        .zip(target_lengths)
        .all(|(blk, &target)| blk.len() == target));
}

fn shrink_sweep_to_right<T, const NUM_BLOCKS: usize>(
    blocks: &mut Blocks<T, NUM_BLOCKS>,
    target_lengths: &[usize; NUM_BLOCKS],
) {
    let mut blocks = blocks.as_mut_slice();
    for &target in target_lengths.iter().take(NUM_BLOCKS - 1) {
        let this_block;
        (this_block, blocks) = blocks.split_first_mut().unwrap();

        if this_block.len() <= target {
            continue;
        }

        let too_long_by = this_block.len() - target;
        this_block.shrink_to_right(&mut blocks[0], too_long_by);
    }
}

fn shrink_sweep_to_left<T, const NUM_BLOCKS: usize>(
    blocks: &mut Blocks<T, NUM_BLOCKS>,
    target_lengths: &[usize; NUM_BLOCKS],
) {
    let mut blocks = blocks.as_mut_slice();
    for &target in target_lengths[1..].iter().rev() {
        let this_block;
        (this_block, blocks) = blocks.split_last_mut().unwrap();

        if this_block.len() <= target {
            continue;
        }

        let too_long_by = this_block.len() - target;
        blocks
            .last_mut()
            .unwrap()
            .grow_from_right(this_block, too_long_by);
    }
}

fn draw_target_lengths<R: Rng, T, const NUM_BLOCKS: usize>(
    rng: &mut R,
    num_unprocessed: usize,
    blocks: &Blocks<T, NUM_BLOCKS>,
) -> [usize; NUM_BLOCKS] {
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

    let mut target_len = [0usize; NUM_BLOCKS];

    for (target, (block, additional)) in target_len.iter_mut().zip(blocks.iter().zip(multinomial(
        rng,
        NUM_BLOCKS,
        num_unprocessed,
    ))) {
        *target = block.num_processed() + additional;
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

    macro_rules! invoke_with_random_blocks {
        ($func:ident) => {
            fn generate_random_data<const NUM_BLOCKS: usize>(rng: &mut impl Rng) {
                let mut data = Vec::new();

                for _ in 0..10 {
                    let blocks = generate_random_blocks::<NUM_BLOCKS>(rng, &mut data);
                    let num_unprocessed = blocks.iter().map(|blk| blk.num_unprocessed()).sum();
                    let target_lengths: [usize; NUM_BLOCKS] =
                        draw_target_lengths(rng, num_unprocessed, &blocks);

                    $func(rng, blocks, target_lengths);
                }
            }

            invoke_for_pot!(generate_random_data);
        };
    }

    #[test]
    fn draw_unprocessed_distribution() {
        fn test_impl<const NUM_BLOCKS: usize>(rng: &mut impl Rng) {
            let total_length = 10 * NUM_BLOCKS;
            let mut data = vec![0; total_length];
            let mut blocks = split_slice_into_blocks(&mut data);
            for block in &mut blocks {
                block.set_num_processed(block.len());
            }

            let num_unprocessed = rng.gen_range(NUM_BLOCKS..total_length);
            for _ in 0..num_unprocessed {
                loop {
                    let block = blocks.choose_mut(rng).unwrap();
                    if block.num_processed() > 0 {
                        block.set_num_processed(block.num_processed() - 1);
                        break;
                    }
                }
            }

            let target_lengths: [usize; NUM_BLOCKS] =
                super::draw_target_lengths(rng, num_unprocessed, &blocks);

            assert!(blocks
                .iter()
                .zip_eq(&target_lengths)
                .all(|(blk, &target)| target >= blk.num_processed()));

            assert_eq!(target_lengths.iter().sum::<usize>(), total_length);
        }

        invoke_for_pot!(test_impl);
    }

    #[test]
    fn compact_ranges() {
        fn test_impl<const NUM_BLOCKS: usize>(
            _rng: &mut impl Rng,
            mut blocks: Blocks<usize, NUM_BLOCKS>,
            _target_lengths: [usize; NUM_BLOCKS],
        ) {
            let num_stash: usize = blocks.iter().map(|r| r.num_unprocessed()).sum();
            if num_stash > blocks.last().unwrap().len() {
                return;
            }

            mark_unprocessed_data(&mut blocks);

            super::compact_ranges(&mut blocks);

            assert_eq!(
                blocks.iter().map(|r| r.num_unprocessed()).sum::<usize>(),
                num_stash
            );

            assert!(blocks
                .last()
                .unwrap()
                .data()
                .suffix(num_stash)
                .iter()
                .all(|x| *x != 0));

            let data = merge_data(&blocks);

            assert_eq!(
                sort_dedup(&data),
                (0..=num_stash).into_iter().collect_vec(),
                "num_stash = {num_stash}"
            );
        }

        invoke_with_random_blocks!(test_impl);
    }

    macro_rules! shrink_sweep_test_skeleton {
        ($sweep : ident) => {
            fn test_impl<const NUM_BLOCKS: usize>(
                _rng: &mut impl Rng,
                mut blocks: Blocks<usize, NUM_BLOCKS>,
                target_lengths: [usize; NUM_BLOCKS],
            ) {
                mark_unprocessed_data(&mut blocks);

                assert_processed_are_zero(&blocks);
                assert_unprocessed_are_non_zero(&blocks);
                let unprocessed_before = sort_dedup(&merge_data(&blocks));

                $sweep(&mut blocks, &target_lengths);

                assert_processed_are_zero(&blocks);
                assert_unprocessed_are_non_zero(&blocks);
                let unprocessed_after = sort_dedup(&merge_data(&blocks));

                assert_eq!(unprocessed_before, unprocessed_after,);
            }

            invoke_with_random_blocks!(test_impl);
        };
    }

    #[test]
    fn shrink_sweep_to_left() {
        fn sweep<const NUM_BLOCKS: usize>(
            blocks: &mut Blocks<usize, NUM_BLOCKS>,
            target_lengths: &[usize; NUM_BLOCKS],
        ) {
            super::shrink_sweep_to_left(blocks, target_lengths);
            for (block_idx, (block, &target)) in
                blocks.iter().zip(target_lengths).enumerate().skip(1)
            {
                assert!(block.len() <= target, "block_idx = {block_idx}");
            }
        }

        shrink_sweep_test_skeleton!(sweep);
    }

    #[test]
    fn shrink_sweep_to_right() {
        fn sweep<const NUM_BLOCKS: usize>(
            blocks: &mut Blocks<usize, NUM_BLOCKS>,
            target_lengths: &[usize; NUM_BLOCKS],
        ) {
            super::shrink_sweep_to_right(blocks, target_lengths);
            for (block_idx, (block, &target)) in blocks
                .iter()
                .zip(target_lengths)
                .take(NUM_BLOCKS - 1)
                .enumerate()
            {
                assert!(block.len() <= target, "block_idx = {block_idx}");
            }
        }

        shrink_sweep_test_skeleton!(sweep);
    }

    #[test]
    fn move_blocks_to_fit_target_len() {
        fn sweep<const NUM_BLOCKS: usize>(
            blocks: &mut Blocks<usize, NUM_BLOCKS>,
            target_lengths: &[usize; NUM_BLOCKS],
        ) {
            super::move_blocks_to_fit_target_len(blocks, target_lengths);
            for (block_idx, (block, &target)) in blocks.iter().zip(target_lengths).enumerate() {
                assert!(block.len() == target, "block_idx = {block_idx}");
            }
        }

        shrink_sweep_test_skeleton!(sweep);
    }

    fn generate_random_blocks<'a, const NUM_BLOCKS: usize>(
        rng: &mut impl Rng,
        storage: &'a mut Vec<usize>,
    ) -> Blocks<'a, usize, NUM_BLOCKS> {
        let sizes = (0..NUM_BLOCKS)
            .into_iter()
            .map(|_| (rng.gen_range(0..30), rng.gen_range(0..10)))
            .collect_vec();
        storage.resize(sizes.iter().map(|(a, b)| a + b).sum(), 0);

        let mut data = storage.as_mut_slice();
        let mut blocks = Vec::new();
        for sze in &sizes {
            let block;
            (block, data) = data.split_at_mut(sze.0 + sze.1);
            blocks.push(Block::new_with_num_unprocessed(block, sze.1));
        }

        blocks.into_iter().collect()
    }

    fn mark_unprocessed_data(blocks: &mut [Block<usize>]) {
        blocks
            .iter_mut()
            .for_each(|blk| blk.data_processed_mut().fill(0));

        for (idx, value) in blocks
            .iter_mut()
            .flat_map(|r| r.data_unprocessed_mut().iter_mut())
            .enumerate()
        {
            *value = idx + 1;
        }
    }

    fn assert_processed_are_zero(blocks: &[Block<usize>]) {
        for (block_idx, block) in blocks.iter().enumerate() {
            for (idx, &dat) in block.data_processed().iter().enumerate() {
                assert_eq!(dat, 0, "block_idx={block_idx} i={idx}");
            }
        }
    }

    fn assert_unprocessed_are_non_zero(blocks: &[Block<usize>]) {
        for (block_idx, block) in blocks.iter().enumerate() {
            for (idx, &dat) in block.data_unprocessed().iter().enumerate() {
                assert_ne!(dat, 0, "block_idx={block_idx} i={idx}");
            }
        }
    }

    fn merge_data<T: Copy>(blocks: &[Block<T>]) -> Vec<T> {
        blocks
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
        const NUM_BLOCKS: usize = 4;
        const BASE_CASE_SIZE: usize = NUM_BLOCKS * 4;
        scatter_shuffle_impl::<R, T, NUM_BLOCKS, BASE_CASE_SIZE>(rng, data)
    }

    crate::statistical_tests::test_shuffle_algorithm!(inplace_scatter_shuffle_test);
}
