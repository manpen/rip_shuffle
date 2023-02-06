use crate::{
    bucketing::slicing::Slicing, random_bits::RandomBitsSource,
    scatter_shuffle::parallel::seed_new_rng,
};

use super::{fisher_yates::fisher_yates, uniform_index};
use rand::{Rng, SeedableRng};

const FY_BASE_CASE: usize = 1 << 18;

pub fn seq_merge_shuffle<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    let n = data.len();
    if n < FY_BASE_CASE {
        return fisher_yates(rng, data);
    }

    let (left, right) = data.split_at_mut(n / 2);

    seq_merge_shuffle(rng, left);
    seq_merge_shuffle(rng, right);
    random_merge(rng, left, right);
}

pub fn par_merge_shuffle<R: Rng + SeedableRng + Send + Sync, T: std::marker::Send>(
    rng: &mut R,
    data: &mut [T],
) {
    let n = data.len();
    if n < FY_BASE_CASE {
        return fisher_yates(rng, data);
    }

    let (left, right) = data.split_at_mut(n / 2);

    let mut right_rng: R = seed_new_rng(rng);

    rayon::join(
        || par_merge_shuffle(rng, left),
        || par_merge_shuffle(&mut right_rng, right),
    );

    random_merge(rng, left, right);
}

fn random_merge<R: Rng, T>(rng: &mut R, left: &mut [T], right: &mut [T]) {
    assert!(left.is_left_neighbor_of(&right));

    let num_rough_merged = {
        #[cfg(feature = "unsafe_algos")]
        unsafe {
            unsafe_rough_random_merge(rng, left, right)
        }

        #[cfg(not(feature = "unsafe_algos"))]
        safe_rough_random_merge(rng, left, right)
    };

    insertion_shuffle(rng, left.merge_with_right_neighbor(right), num_rough_merged);
}

#[allow(dead_code)]
fn safe_rough_random_merge<R: Rng, T>(rng: &mut R, left: &mut [T], right: &mut [T]) -> usize {
    let mut begin = 0;
    let mut mid = left.len();
    let data = left.merge_with_right_neighbor(right);
    let end = data.len();

    let mut rbs = RandomBitsSource::default();

    loop {
        if rbs.gen_bool(rng) {
            if mid == end {
                break;
            }

            data.swap(begin, mid);
            mid += 1;
        } else if begin == mid {
            break;
        }

        begin += 1;
    }

    begin
}

#[allow(dead_code)]
unsafe fn unsafe_rough_random_merge<R: Rng, T>(
    rng: &mut R,
    left: &mut [T],
    right: &mut [T],
) -> usize {
    let mut begin = left.as_mut_ptr();
    let mut mid = right.as_mut_ptr();
    let end = right.as_mut_ptr_range().end;

    while end.offset_from(mid).min(mid.offset_from(begin)) >= 64 {
        let rand: u64 = rng.gen();

        const STEPS: usize = 16;
        for i in (0..64).into_iter().step_by(STEPS) {
            (begin, mid) = unsafe_uncheck_iterations::<T, 16>((rand >> i) as usize, begin, mid);
        }
    }

    let mut rbs = RandomBitsSource::default();
    loop {
        if rbs.gen_bool(rng) {
            if mid == end {
                break;
            }

            std::ptr::swap(begin, mid);
            mid = mid.add(1);
        } else if begin == mid {
            break;
        }

        begin = begin.add(1);
    }

    begin.offset_from(left.as_ptr()) as usize
}

unsafe fn unsafe_uncheck_iterations<T, const N: usize>(
    rand: usize,
    mut begin: *mut T,
    mut mid: *mut T,
) -> (*mut T, *mut T) {
    for i in 0..N {
        let bit = (rand >> i) & 1;

        let partner = if bit == 1 { begin } else { mid };
        std::ptr::swap(begin, partner);

        mid = mid.add(bit);
        begin = begin.add(1);
    }

    (begin, mid)
}

fn insertion_shuffle<R: Rng, T>(rng: &mut R, data: &mut [T], num_merged: usize) {
    let end = data.len();
    for left in num_merged..end {
        let partner = uniform_index::gen_index(rng, end - left);
        data.swap(left, partner);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! impl_merge_test {
        ($shuffle : ident) => {
            mod $shuffle {
                use super::*;
                use rand::seq::SliceRandom;

                fn shuffle<T>(rng: &mut impl Rng, data: &mut [T]) {
                    let n = data.len();
                    let (left, right) = data.split_at_mut(n / 2);
                    left.shuffle(rng);
                    right.shuffle(rng);
                    #[allow(unused_unsafe)]
                    let rough = unsafe { super::super::$shuffle(rng, left, right) };
                    insertion_shuffle(rng, data, rough);
                }

                crate::statistical_tests::test_shuffle_algorithm!(shuffle);
            }
        };
    }

    impl_merge_test!(safe_rough_random_merge);
    impl_merge_test!(unsafe_rough_random_merge);
}

#[cfg(test)]
mod integration_test {
    use super::*;

    mod seq {
        use super::*;
        crate::statistical_tests::test_shuffle_algorithm!(seq_merge_shuffle);
    }

    mod par {
        use super::*;
        crate::statistical_tests::test_shuffle_algorithm!(par_merge_shuffle);
    }
}
