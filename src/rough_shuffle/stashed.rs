#![allow(dead_code)]
use super::*;
use crate::random_bits::RandomBitsSource;
use std::{mem::MaybeUninit, ptr::copy_nonoverlapping};

pub(super) fn rough_shuffle<R: Rng, T, const LOG_NUM_BLOCKS: usize, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) {
    if blocks.iter().any(|blk| blk.is_fully_processed()) {
        return;
    }

    assert_eq!(1 << LOG_NUM_BLOCKS, NUM_BLOCKS);

    rough_shuffle_stashed_impl::<R, T, LOG_NUM_BLOCKS, NUM_BLOCKS>(rng, blocks);
}

fn rough_shuffle_stashed_impl<R: Rng, T, const LOG_NUM_BLOCKS: usize, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) {
    let mut rbs = RandomBitsSource::new();

    let initial_doner = rng.gen_range(0..NUM_BLOCKS);
    let stashed_element = blocks[initial_doner].pop().unwrap() as *mut T;

    let mut stash = Stash::new(unsafe { &*stashed_element });

    loop {
        let block = &mut blocks[rbs.gen_const_bits::<LOG_NUM_BLOCKS>(rng) as usize];

        if let Some(elem) = block.first_unprocessed() {
            stash.swap(elem);
            block.process_element();
        } else {
            break;
        }
    }

    stash.deconstruct(unsafe { &mut *stashed_element });
    blocks[initial_doner].push(unsafe { &mut *stashed_element });
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

    fn swap(&mut self, elem: &mut T) {
        let write_idx = (self.read_idx == 0) as usize;
        unsafe {
            copy_nonoverlapping(elem, self.data[write_idx].as_mut_ptr(), 1);
            copy_nonoverlapping(self.data[self.read_idx].as_ptr(), elem as *mut T, 1);
        }
        self.read_idx = write_idx;
    }

    fn deconstruct(self, elem: &mut T) {
        unsafe {
            copy_nonoverlapping(self.data[self.read_idx].as_ptr(), elem as *mut T, 1);
        }
    }
}

#[cfg(test)]
mod test {
    use super::{common_tests, rough_shuffle};

    common_tests::rough_shuffle_tests!(rough_shuffle);
}
