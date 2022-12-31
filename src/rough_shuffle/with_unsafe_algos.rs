use std::{
    mem::MaybeUninit,
    ptr::{copy_nonoverlapping, NonNull},
};

use super::*;

#[allow(dead_code)]
pub(super) fn rough_shuffle<R: Rng, T, const LOG_NUM_BLOCKS: usize, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) {
    let mask = (1usize << LOG_NUM_BLOCKS) - 1;
    let num_swaps_per_round = 64 / LOG_NUM_BLOCKS;

    let mut swap_stash = [MaybeUninit::<T>::uninit(), MaybeUninit::<T>::uninit()];

    // one optimization is that we do not check in-situ, whether a range got full and stop.
    // we rather pessimitically check before hand how many rounds we can safely carry out.
    // thereby we save a check per swap and also do not have to store end-of-range pointers
    // anymore. Thus, we represend each range only by its base pointer:
    let mut base_ptrs = [NonNull::<T>::dangling().as_ptr(); NUM_BLOCKS];
    for (ptr, block) in base_ptrs.iter_mut().zip(blocks.iter_mut()) {
        *ptr = block.data_unprocessed_mut().as_mut_ptr();
    }

    // we can safely carry out MIN swaps where MIN is the length of the shortest range. Since in each
    // round we perform `num_swaps_per_round` many swaps, number of rounds `rounds` is divided by this size.
    let mut rounds = blocks
        .iter()
        .map(|blk| blk.num_unprocessed())
        .min()
        .unwrap()
        / num_swaps_per_round;

    while rounds >= 8 {
        // ^^ threshold found experimentally and not too critical

        // a classical `std::mem::swap(a,b)` operation involves there data movements: tmp <- a, a <- b, and b <- tmp.
        // here, we are a little bit more clever and use only two + epsilon many. During the run we use the following
        // invariant: the value to be assigned next is in the `swap_stash[x]` where x is either 0 or 1; before overwriting
        // the value in `data`, we move it into `swap_stash[1-x]` and then do the same for next swap with x=1-x.

        // To get the ball rolling, we copy the first element of the first range (arbitrary choice!) into the stash and
        // later put the last remaining item remaining in the stash back into that position.

        // Memory safty: The only call within a round that may panic is `rng.gen()`. Observe, however, that our swapping
        // algorithm only copies values bitwise to and fro the `swap_stash`. We additionally have the invariant, that each
        // value in the stash at the begin of a round is still 'somewhere' in data. So we won't lose original data. Additionally,
        // the stash is `MaybeUninit` which prevents the copies there from being dropped.
        let first_element: *mut T = base_ptrs[0];
        unsafe {
            copy_nonoverlapping(first_element, swap_stash[0].as_mut_ptr(), 1);
            base_ptrs[0] = base_ptrs[0].add(1);
        };

        for _ in 0..rounds {
            let rand: u64 = rng.gen();

            for k in 0..num_swaps_per_round {
                let index = (rand >> (LOG_NUM_BLOCKS * k)) as usize & mask;
                let target_ptr = base_ptrs[index];

                unsafe {
                    base_ptrs[index] = target_ptr.add(1);
                    copy_nonoverlapping(target_ptr, swap_stash[1 - (k % 2)].as_mut_ptr(), 1);
                    copy_nonoverlapping(swap_stash[k % 2].as_ptr(), target_ptr, 1);
                }
            }

            // we even share the stash in between rounds. if there's an odd number of swaps per round,
            // the initially populated position in the stash would need to change. Instead, we make sure that
            // the last element in the stash is always in `swap_stash[0]`.
            if num_swaps_per_round % 2 != 0 {
                unsafe {
                    copy_nonoverlapping(swap_stash[1].as_ptr(), swap_stash[0].as_mut_ptr(), 1)
                }
            }
        }

        unsafe {
            copy_nonoverlapping(swap_stash[0].as_mut_ptr(), first_element, 1);
            std::ptr::swap(first_element, base_ptrs[0]);
            base_ptrs[0] = base_ptrs[0].sub(1);
        };

        // We updated the base_ptrs array during our switches. We now need to mirror the updates back into
        // our input data structure. At the same time, we compute the length of the shortest range to set
        // the number of rounds to be carried out in the next run.
        let mut shortest_range = usize::MAX;
        for (block, ptr) in blocks.iter_mut().zip(base_ptrs.iter()) {
            let num_processed = unsafe { ptr.offset_from(block.data().as_ptr()) } as usize;
            block.set_num_processed(num_processed);
            shortest_range = shortest_range.min(block.num_unprocessed());
        }
        rounds = shortest_range / num_swaps_per_round;
    }
}

#[cfg(test)]
mod test {
    use super::{common_tests, rough_shuffle};

    common_tests::rough_shuffle_tests!(rough_shuffle);
}
