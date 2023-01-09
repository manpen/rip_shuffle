use crate::uniform_index::impl_u32;

use super::*;

macro_rules! maybe_unchecked_swap {
    ($slice : ident, $i : expr, $j : expr) => {
        #[cfg(not(feature = "unsafe_algos"))]
        $slice.swap($i, $j);

        #[cfg(feature = "unsafe_algos")]
        unsafe {
            $slice.swap_unchecked($i, $j);
        }
    };
}

pub fn fisher_yates_u32<R: Rng, T>(rng: &mut R, mut data: &mut [T]) {
    const UNROLL: usize = 4; // HAS TO MATCH the number of step! calls!!!
    const ELEMS_PER_ROUND: usize = 2 * UNROLL;

    while data.len() >= 2 * ELEMS_PER_ROUND {
        let n = data.len() as u32;

        macro_rules! step {
            ($i : expr) => {
                let (i0, i1) = impl_u32::gen_index_pair(rng, (n - $i, n - 1 - $i));
                maybe_unchecked_swap!(data, $i, $i + i0 as usize);
                maybe_unchecked_swap!(data, $i + 1, $i + i1 as usize);
            };
        }

        step!(0);
        step!(2);
        step!(4);
        step!(6);

        data = &mut data[ELEMS_PER_ROUND..];
    }

    super::naive::fisher_yates(rng, data)
}

#[cfg(test)]
mod test {
    mod test_u32 {
        use super::super::fisher_yates_u32;

        crate::statistical_tests::test_shuffle_algorithm!(fisher_yates_u32);
        crate::statistical_tests::test_shuffle_algorithm_deterministic!(fisher_yates_u32);
    }
}
