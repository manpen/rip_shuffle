use super::*;
use std::intrinsics::prefetch_write_data;

const DEFAULT_PREFETCH_WIDTH: usize = 16;

pub fn fisher_yates<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    if data.len() < uniform_index::U32_MAX_UPPER_BOUND as usize {
        fisher_yates_u32(rng, data);
    } else {
        fisher_yates_u64(rng, data);
    }
}

pub fn fisher_yates_u32<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    fisher_yates_impl::<DEFAULT_PREFETCH_WIDTH, R, T, _>(
        rng,
        |rng: &mut R, ub: usize| uniform_index::impl_u32::gen_index(rng, ub as u32) as usize,
        data,
    );
}

pub fn fisher_yates_u64<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    fisher_yates_impl::<DEFAULT_PREFETCH_WIDTH, R, T, _>(
        rng,
        |rng: &mut R, ub: usize| uniform_index::impl_u64::gen_index(rng, ub as u64) as usize,
        data,
    );
}

fn fisher_yates_impl<const N: usize, R: Rng, T, D: Fn(&mut R, usize) -> usize>(
    rng: &mut R,
    distr: D,
    data: &mut [T],
) {
    let n = data.len();

    if N == 0 || n <= 2 * N {
        return super::naive::fisher_yates(rng, data);
    }

    // this is an ultra-compact ring buffer
    let mut enqueue = {
        let mut ring_buf = [0usize; N];
        let mut ring_buf_idx = 0;

        move |new_val| -> usize {
            let old = std::mem::replace(&mut ring_buf[ring_buf_idx], new_val);
            ring_buf_idx = (ring_buf_idx + 1) % N;
            old
        }
    };

    // generate new random index and prefetch its address
    let mut draw_and_fetch = |data: &[T], ub: usize| -> usize {
        let new_idx = distr(rng, ub);
        const LOCALITY: i32 = 1;
        unsafe { prefetch_write_data(data.as_ptr().add(new_idx), LOCALITY) };
        new_idx
    };

    for i in (n - N..n).rev() {
        enqueue(draw_and_fetch(data, i + 1));
    }

    for i in (N + 1..n).rev() {
        let j = enqueue(draw_and_fetch(data, i - N + 1));
        data.swap(i, j);
    }

    for i in (1..N + 1).rev() {
        let j = enqueue(0);
        data.swap(i, j);
    }
}

#[cfg(test)]
mod test {
    mod test_u32 {
        use super::super::fisher_yates_u32;

        crate::statistical_tests::test_shuffle_algorithm!(fisher_yates_u32);
        crate::statistical_tests::test_shuffle_algorithm_deterministic!(fisher_yates_u32);
    }

    mod test_u64 {
        use super::super::fisher_yates_u64;

        crate::statistical_tests::test_shuffle_algorithm!(fisher_yates_u64);
        crate::statistical_tests::test_shuffle_algorithm_deterministic!(fisher_yates_u64);
    }
}
