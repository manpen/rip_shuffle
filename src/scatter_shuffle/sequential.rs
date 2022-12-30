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
    let target_lengths = draw_target_lengths(rng, num_unprocessed, &mut blocks);
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
    blocks: &mut Blocks<T, NUM_BLOCKS>,
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

    for (target, (block, additional)) in target_len.iter_mut().zip(
        blocks
            .iter_mut()
            .zip(multinomial(rng, NUM_BLOCKS, num_unprocessed)),
    ) {
        *target = block.num_processed() + additional;
    }

    target_len
}

#[cfg(test)]
mod test {
    use rand::{seq::SliceRandom, SeedableRng};
    use rand_pcg::Pcg64;

    use super::*;
    /*
    #[test]
    fn draw_unprocessed_distribution() {
        fn helper<const NUM_BLOCKS: usize>(rng: &mut impl Rng) {
            let total_length = 10 * NUM_BLOCKS;
            let mut data = vec![0; total_length];
            let mut blocks = split_slice_into_blocks(&mut data);

            let num_unprocessed = rng.gen_range(NUM_BLOCKS..total_length);
            for _ in 0..num_unprocessed {
                loop {
                    let block = blocks.choose_mut(&mut rng).unwrap();
                    if block.len() > block.num_unprocessed {
                        block.num_unprocessed += 1;
                        break;
                    }
                }
            }

            super::draw_unprocessed_distribution(&mut rng, num_unprocessed, &mut blocks);

            assert!(blocks
                .iter()
                .all(|blk| blk.target_len >= blk.num_processed()));

            assert_eq!(
                blocks.iter().map(|blk| blk.target_len).sum::<usize>(),
                total_length
            );
        }

        let mut rng = Pcg64::seed_from_u64(12345);
        helper::<1>(&mut rng);
        helper::<2>(&mut rng);
        helper::<4>(&mut rng);
        helper::<8>(&mut rng);
        helper::<32>(&mut rng);
        helper::<128>(&mut rng);
    }
    */

    /*
    #[test]
    fn compact_ranges() {
        let mut rng = Pcg64::seed_from_u64(12345);

        for i in 1..1000 {
            let blocks = generate_random_blocks(&mut rng, i);
            let len = blocks.last().unwrap().end + rng.gen_range(0..10);
            let mut data = generate_data_with_unprocessed_marked(len, &blocks);

            let num_stash: usize = blocks.iter().map(|r| r.num_unprocessed).sum();

            super::compact_ranges(&mut data, &blocks);
            assert!(data[..data.len() - num_stash].iter().all(|x| *x == 0));

            assert_eq!(
                sort_dedup(&data),
                (0..=num_stash).into_iter().collect_vec(),
                "i={}, ranges={:?}",
                i,
                &blocks
            );
        }
    }
    /*
    #[test]
    fn shrink_sweep_to_left() {
        let mut rng = Pcg64::seed_from_u64(123456);
        shrink_sweep_test_skeleton(&mut rng, |data, blocks| {
            super::shrink_sweep_to_left(data, blocks);
            for (block_idx, block) in blocks.iter().skip(1).enumerate() {
                assert!(
                    block.len() <= block.target_len,
                    "block_idx = {block_idx}, block = {block:?}"
                );
            }
        });
    }

    #[test]
    fn shrink_sweep_to_right() {
        let mut rng = Pcg64::seed_from_u64(1234567);
        shrink_sweep_test_skeleton(&mut rng, |data, blocks| {
            super::shrink_sweep_to_right(data, blocks);
            let num_blocks = blocks.len();
            for (block_idx, block) in blocks.iter().take(num_blocks - 1).enumerate() {
                assert!(
                    block.len() <= block.target_len,
                    "block_idx = {block_idx}, block = {block:?}"
                );
            }
        });
    }

    #[test]
    fn move_blocks_to_fit_target_len() {
        let mut rng = Pcg64::seed_from_u64(123456789);
        for _ in 0..10 {
            shrink_sweep_test_skeleton(&mut rng, |data, blocks| {
                super::move_blocks_to_fit_target_len(data, blocks);
                assert!(blocks.iter().all(|blk| blk.len() == blk.target_len));
            });
        }
    } */

    ///////////////// HELPER

    fn shrink_sweep_test_skeleton<R: Rng, T, F: Fn(&mut [usize], &mut [Block<T>])>(
        rng: &mut R,
        sweep: F,
    ) {
        for num_blocks in 2..100 {
            let mut blocks = generate_random_blocks_with_target_length(rng, num_blocks);
            let mut data =
                generate_data_with_unprocessed_marked(blocks.last().unwrap().end, &blocks);

            assert_processed_are_zero(&data, &blocks);
            assert_unprocessed_are_non_zero(&data, &blocks);
            let unprocessed_before = sort_dedup(&data);

            sweep(&mut data, &mut blocks);
            assert_processed_are_zero(&data, &blocks);
            assert_unprocessed_are_non_zero(&data, &blocks);
            let unprocessed_after = sort_dedup(&data);

            assert_eq!(
                unprocessed_before, unprocessed_after,
                "num_blocks={num_blocks}"
            );
        }
    }


    fn generate_random_blocks<R: Rng>(rng: &mut R, num_blocks: usize) -> Vec<Block> {
        let mut blocks = Vec::with_capacity(num_blocks);

        let mut begin = 0;
        for _ in 0..num_blocks {
            let num_proc = rng.gen_range(0..30);
            let num_unproc = rng.gen_range(0..10);
            let end = begin + num_proc + num_unproc;

            let mut block: Block = (begin..end).into();
            block.num_unprocessed = num_unproc;

            blocks.push(block);
            begin = end;
        }

        blocks
    }

    fn generate_random_blocks_with_target_length<R: Rng>(
        rng: &mut R,
        num_blocks: usize,
    ) -> Vec<Block> {
        let mut blocks = generate_random_blocks(rng, num_blocks);
        let total_num_unprocessed = blocks.iter().map(|blk| blk.num_unprocessed).sum();
        for (blk, add) in blocks
            .iter_mut()
            .zip(multinomial(rng, num_blocks, total_num_unprocessed))
        {
            blk.target_len = blk.num_processed() + add;
        }
        blocks
    }

    fn generate_data_with_unprocessed_marked(len: usize, blocks: &[Block]) -> Vec<usize> {
        let mut data = vec![0; len];
        for (val, idx) in blocks
            .iter()
            .flat_map(|r| r.range_unprocessed())
            .enumerate()
        {
            data[idx] = val + 1;
        }
        data
    }

    fn assert_processed_are_zero<T: Debug + Zero>(data: &[T], blocks: &[Block]) {
        for (block_idx, block) in blocks.iter().enumerate() {
            for i in block.range_processed() {
                assert!(
                    data[i].is_zero(),
                    "block_idx={block_idx} block={block:?} i={i}"
                );
            }
        }
    }

    fn assert_unprocessed_are_non_zero<T: PartialEq + Debug + Zero>(data: &[T], blocks: &[Block]) {
        for (block_idx, block) in blocks.iter().enumerate() {
            for i in block.range_unprocessed() {
                assert!(
                    !data[i].is_zero(),
                    "block_idx={block_idx} block={block:?} i={i}"
                );
            }
        }
    }

    fn sort_dedup<T: Copy + Ord>(data: &[T]) -> Vec<T> {
        let mut data: Vec<T> = data.into();
        data.sort();
        data.dedup();
        data
    }
    */
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
