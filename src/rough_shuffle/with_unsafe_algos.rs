#![allow(clippy::needless_range_loop)]

use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr::copy_nonoverlapping,
};

use crate::prefetch::*;

use super::*;

pub(super) fn rough_shuffle<
    R: Rng,
    T,
    const LOG_NUM_BLOCKS: usize,
    const NUM_BLOCKS: usize,
    const SWAPS_PER_ROUND: usize,
>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) {
    let mut first_unprocessed_in_block = BlockBasePointers::new(blocks);

    loop {
        let rounds = first_unprocessed_in_block.length_of_shortest_block() / 2 / SWAPS_PER_ROUND;
        if rounds <= 1 {
            break;
        }

        let seed_for_stash: *mut T = first_unprocessed_in_block.fetch_and_increment(0);
        let mut stash = Stash::new(unsafe { &mut *seed_for_stash });

        for _ in 0..rounds {
            let pointers_to_swap0 = prefetch::<_, _, LOG_NUM_BLOCKS, NUM_BLOCKS, SWAPS_PER_ROUND>(
                rng,
                &mut first_unprocessed_in_block,
            );
            let pointers_to_swap1 = prefetch::<_, _, LOG_NUM_BLOCKS, NUM_BLOCKS, SWAPS_PER_ROUND>(
                rng,
                &mut first_unprocessed_in_block,
            );

            // since we execute two swaps per iteration, we're always sure that the first
            // swap will be with the first stash lane
            for k in 0..SWAPS_PER_ROUND {
                stash.swap_assume_read_from::<0>(unsafe { &mut *pointers_to_swap0[k] });
                stash.swap_assume_read_from::<1>(unsafe { &mut *pointers_to_swap1[k] });
            }
        }

        unsafe {
            let current_base = first_unprocessed_in_block.fetch_and_decrement(0);
            copy_nonoverlapping(current_base, seed_for_stash, 1);
            stash.deconstruct(&mut *current_base);
        }

        first_unprocessed_in_block.synchronize_blocks(blocks);
    }
}

fn prefetch<
    R: Rng,
    T,
    const LOG_NUM_BLOCKS: usize,
    const NUM_BLOCKS: usize,
    const SWAPS_PER_ROUND: usize,
>(
    rng: &mut R,
    first_unprocessed_in_block: &mut BlockBasePointers<T, NUM_BLOCKS>,
) -> [*mut T; SWAPS_PER_ROUND] {
    let mask = (1usize << LOG_NUM_BLOCKS) - 1;
    let rand: u64 = rng.gen();

    let mut buffer: [MaybeUninit<*mut T>; SWAPS_PER_ROUND] =
        unsafe { MaybeUninit::uninit().assume_init() };

    // compute and prefetch indices
    for k in 0..SWAPS_PER_ROUND {
        let index = (rand >> (k * LOG_NUM_BLOCKS)) as usize & mask;

        let target_ptr = first_unprocessed_in_block.fetch_and_increment(index);

        prefetch_write_data(unsafe { &mut *target_ptr });

        buffer[k].write(target_ptr);
    }

    unsafe { std::mem::transmute_copy(&ManuallyDrop::new(buffer)) }
}

struct Stash<T> {
    data: [MaybeUninit<T>; 2],
    read_idx: usize,
}

impl<T> Stash<T> {
    fn new(elem: &T) -> Self {
        let mut stash = Self {
            data: [MaybeUninit::<T>::uninit(), MaybeUninit::<T>::uninit()],
            read_idx: 0,
        };

        unsafe {
            copy_nonoverlapping(elem, stash.data[0].as_mut_ptr(), 1);
        }

        stash
    }

    #[allow(dead_code)]
    fn swap(&mut self, elem: &mut T) {
        let write_idx = (self.read_idx == 0) as usize;
        unsafe {
            copy_nonoverlapping(elem, self.data[write_idx].as_mut_ptr(), 1);
            copy_nonoverlapping(self.data[self.read_idx].as_ptr(), elem as *mut T, 1);
        }
        self.read_idx = write_idx;
    }

    fn swap_assume_read_from<const N: usize>(&mut self, elem: &mut T) {
        debug_assert_eq!(self.read_idx, N);
        unsafe {
            copy_nonoverlapping(elem, self.data[1 - N].as_mut_ptr(), 1);
            copy_nonoverlapping(self.data[N].as_ptr(), elem as *mut T, 1);
        }
        self.read_idx = 1 - N;
    }

    fn deconstruct(self, elem: &mut T) {
        unsafe {
            copy_nonoverlapping(self.data[self.read_idx].as_ptr(), elem as *mut T, 1);
        }
    }
}

struct BlockBasePointers<T, const NUM_BLOCKS: usize> {
    pointers: [*mut T; NUM_BLOCKS],
    length_of_shortest_block: usize,
}

impl<T, const NUM_BLOCKS: usize> BlockBasePointers<T, NUM_BLOCKS> {
    fn new(blocks: &mut Blocks<T, NUM_BLOCKS>) -> Self {
        let mut pointers: [MaybeUninit<*mut T>; NUM_BLOCKS] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut length_of_shortest_block = usize::MAX;

        for (ptr, block) in pointers.iter_mut().zip(blocks.iter_mut()) {
            ptr.write(block.data_unprocessed_mut().as_mut_ptr());
            length_of_shortest_block = length_of_shortest_block.min(block.num_unprocessed());
        }

        Self {
            pointers: unsafe { std::mem::transmute_copy(&ManuallyDrop::new(pointers)) },
            length_of_shortest_block,
        }
    }

    fn synchronize_blocks(&mut self, blocks: &mut Blocks<T, NUM_BLOCKS>) {
        for (block, ptr) in blocks.iter_mut().zip(self.pointers.iter()) {
            let num_processed = unsafe { ptr.offset_from(block.data().as_ptr()) } as usize;
            assert!(num_processed <= block.len());
            block.set_num_processed(num_processed);

            self.length_of_shortest_block =
                self.length_of_shortest_block.min(block.num_unprocessed());
        }
    }

    fn length_of_shortest_block(&self) -> usize {
        self.length_of_shortest_block
    }

    fn fetch_and_increment(&mut self, idx: usize) -> *mut T {
        let result = self.pointers[idx];
        self.pointers[idx] = unsafe { result.add(1) };
        result
    }

    fn fetch_and_decrement(&mut self, idx: usize) -> *mut T {
        let result = self.pointers[idx];
        self.pointers[idx] = unsafe { result.sub(1) };
        result
    }
}

#[cfg(test)]
mod test {
    use super::{common_tests, rough_shuffle};

    common_tests::rough_shuffle_tests!(rough_shuffle);
}
