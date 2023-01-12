#![allow(clippy::needless_range_loop)]

use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::copy_nonoverlapping,
};

use super::*;
use crate::prefetch::*;

pub(super) fn rough_shuffle<R: Rng, T, const LOG_N: usize, const N: usize, const SWAPS: usize>(
    rng: &mut R,
    buckets: &mut Buckets<T, N>,
) {
    RoughShuffle::<R, T, LOG_N, N, SWAPS>::new(buckets).rough_shuffle(rng)
}

pub struct RoughShuffle<'a, 'b, R, T, const LOG_N: usize, const N: usize, const SWAPS: usize> {
    buckets: &'a mut Buckets<'b, T, N>,
    first_staged: BlockBasePointers<T, N>,
    _phantom_r: PhantomData<R>,
}

impl<'a, 'b, R: Rng, T, const LOG_N: usize, const N: usize, const SWAPS: usize>
    RoughShuffle<'a, 'b, R, T, LOG_N, N, SWAPS>
{
    fn new(buckets: &'a mut Buckets<'b, T, N>) -> Self {
        let first_staged = BlockBasePointers::new(buckets);
        Self {
            buckets,
            first_staged,
            _phantom_r: Default::default(),
        }
    }

    fn rough_shuffle(&mut self, rng: &mut R) {
        loop {
            let rounds = self.first_staged.length_of_shortest_bucket() / 2 / SWAPS;
            if rounds <= 1 {
                break;
            }

            let seed_for_stash: *mut T = self.first_staged.fetch_and_increment(0);
            let mut stash = Stash::new(unsafe { &mut *seed_for_stash });

            for _ in 0..rounds {
                let pointers_to_swap0 = self.prefetch(rng);
                let pointers_to_swap1 = self.prefetch(rng);

                // since we execute two swaps per iteration, we're always sure that the first
                // swap will be with the first stash lane
                for k in 0..SWAPS {
                    stash.swap_assume_read_from::<0>(unsafe { &mut *pointers_to_swap0[k] });
                    stash.swap_assume_read_from::<1>(unsafe { &mut *pointers_to_swap1[k] });
                }
            }

            unsafe {
                let current_base = self.first_staged.fetch_and_decrement(0);
                copy_nonoverlapping(current_base, seed_for_stash, 1);
                stash.deconstruct(&mut *current_base);
            }

            self.first_staged.synchronize_buckets(&mut self.buckets);
        }
    }

    fn prefetch(&mut self, rng: &mut R) -> [*mut T; SWAPS] {
        let mask = (1usize << LOG_N) - 1;
        let rand: u64 = rng.gen();

        let mut buffer: [MaybeUninit<*mut T>; SWAPS] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // compute and prefetch indices
        for k in 0..SWAPS {
            let index = (rand >> (k * LOG_N)) as usize & mask;

            let target_ptr = self.first_staged.fetch_and_increment(index);

            prefetch_write_data(unsafe { &mut *target_ptr });

            buffer[k].write(target_ptr);
        }

        unsafe { std::mem::transmute_copy(&ManuallyDrop::new(buffer)) }
    }
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

struct BlockBasePointers<T, const NUM_BUCKETS: usize> {
    pointers: [*mut T; NUM_BUCKETS],
    length_of_shortest_bucket: usize,
}

impl<T, const NUM_BUCKETS: usize> BlockBasePointers<T, NUM_BUCKETS> {
    fn new(buckets: &mut Buckets<T, NUM_BUCKETS>) -> Self {
        let mut pointers: [MaybeUninit<*mut T>; NUM_BUCKETS] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut length_of_shortest_bucket = usize::MAX;

        for (ptr, bucket) in pointers.iter_mut().zip(buckets.iter_mut()) {
            ptr.write(bucket.data_unprocessed_mut().as_mut_ptr());
            length_of_shortest_bucket = length_of_shortest_bucket.min(bucket.num_unprocessed());
        }

        Self {
            pointers: unsafe { std::mem::transmute_copy(&ManuallyDrop::new(pointers)) },
            length_of_shortest_bucket,
        }
    }

    fn synchronize_buckets(&mut self, buckets: &mut Buckets<T, NUM_BUCKETS>) {
        for (bucket, ptr) in buckets.iter_mut().zip(self.pointers.iter()) {
            let num_processed = unsafe { ptr.offset_from(bucket.data().as_ptr()) } as usize;
            assert!(num_processed <= bucket.len());
            bucket.set_num_processed(num_processed);

            self.length_of_shortest_bucket =
                self.length_of_shortest_bucket.min(bucket.num_unprocessed());
        }
    }

    fn length_of_shortest_bucket(&self) -> usize {
        self.length_of_shortest_bucket
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
