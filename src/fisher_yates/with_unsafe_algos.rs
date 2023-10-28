use super::*;
use std::intrinsics::prefetch_write_data;

const DEFAULT_PREFETCH_WIDTH: usize = 16;
const LOCALITY: i32 = 1;

pub fn fisher_yates_u32<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    assert!(data.len() < u32::MAX as usize);
    fisher_yates_impl::<R, T, DEFAULT_PREFETCH_WIDTH>(rng, data)
}

pub fn fisher_yates_impl<R: Rng, T, const PREFETCH_WIDTH: usize>(rng: &mut R, data: &mut [T]) {
    let n = data.len();

    if PREFETCH_WIDTH == 0 || n <= 2 * PREFETCH_WIDTH {
        return super::naive::fisher_yates(rng, data);
    }

    // this is an ultra-compact ring buffer
    let mut enqueue = {
        let mut ring_buf = [0usize; PREFETCH_WIDTH];
        let mut ring_buf_idx = 0;

        move |new_val| -> usize {
            let old;
            unsafe {
                let bucket = ring_buf.as_mut_ptr().add(ring_buf_idx);
                old = *bucket;
                *bucket = new_val;
            }

            ring_buf_idx = (ring_buf_idx + 1) % PREFETCH_WIDTH;
            old
        }
    };

    // generate new random index and prefetch its address
    let draw_and_fetch_init = |rng: &mut R, data: &[T], initial: u32, ub: usize| -> usize {
        let new_idx = uniform_index::impl_u32::gen_index_impl(rng, initial, ub as u32) as usize;
        unsafe { prefetch_write_data(data.as_ptr().add(new_idx), LOCALITY) };
        new_idx
    };

    let draw_and_fetch = |rng: &mut R, data: &[T], ub: usize| -> usize {
        let initial = rng.gen();
        draw_and_fetch_init(rng, data, initial, ub)
    };

    for i in (n - PREFETCH_WIDTH..n).rev() {
        enqueue(draw_and_fetch(rng, data, i + 1));
    }

    let mut init: u64 = rng.gen();

    for i in (PREFETCH_WIDTH + 1..n).rev() {
        if i % 2 == 0 {
            init = rng.gen();
        } else {
            init >>= 32;
        }

        let j = enqueue(draw_and_fetch_init(
            rng,
            data,
            init as u32,
            i - PREFETCH_WIDTH + 1,
        ));
        unsafe {
            let ptr = data.as_mut_ptr();
            std::ptr::swap(ptr.add(i), ptr.add(j));
        }
    }

    for i in (1..PREFETCH_WIDTH + 1).rev() {
        let j = enqueue(0);
        unsafe {
            let ptr = data.as_mut_ptr();
            std::ptr::swap(ptr.add(i), ptr.add(j));
        }
    }
}

#[cfg(test)]
mod test {
    use super::fisher_yates_u32;

    crate::statistical_tests::test_shuffle_algorithm!(fisher_yates_u32);
    crate::statistical_tests::test_shuffle_algorithm_deterministic!(fisher_yates_u32);
}
