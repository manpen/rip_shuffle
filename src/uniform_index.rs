use rand::Rng;

/// While [`impl_32::gen_index`] supports producing indices up to
/// `u32::MAX`, in practice, the rejection rate rises significantly
/// for large upper bounds and it's then typically faster to use
/// [`impl_64::gen_index`] in these regimes. This constant gives
/// the recommended size when to switch.
pub const U32_MAX_UPPER_BOUND: u32 = u32::MAX / 16;

/// Generates an index from the exclusive range `0..ub`
/// uniformly at random. It is functionally equivalent to
/// `rng.gen_range(0..ub)` but is much faster.
///
/// # Warning
/// The upper bound must be strictly positive. This is not
/// checked in release builds!
///
/// # Example
/// ```
/// use rip_shuffle::uniform_index::gen_index;
/// use rand::prelude::*;
///
/// for i in 1..100 {
///   let rand = gen_index(&mut rand::thread_rng(), i);
///   assert!(rand < i);
/// }
/// ```
pub fn gen_index(rng: &mut impl Rng, exclusive_ub: usize) -> usize {
    if exclusive_ub <= U32_MAX_UPPER_BOUND as usize {
        impl_u32::gen_index(rng, exclusive_ub as u32) as usize
    } else {
        impl_u64::gen_index(rng, exclusive_ub as u64) as usize
    }
}

macro_rules! impl_gen_index {
    ( $t : ty) => {
        use super::*;

        #[inline]
        pub fn gen_index(rng: &mut impl Rng, exclusive_ub: $t) -> $t {
            let initial = rng.gen();
            gen_index_impl(rng, initial, exclusive_ub)
        }

        #[inline]
        pub fn gen_index_impl(rng: &mut impl Rng, initial: $t, exclusive_ub: $t) -> $t {
            debug_assert!(exclusive_ub != 0);

            let (mut lo, mut hi) = initial.widening_mul(exclusive_ub);

            if lo >= exclusive_ub {
                return hi;
            }

            let t = exclusive_ub.wrapping_neg() % exclusive_ub;

            loop {
                if lo >= t {
                    return hi;
                }

                let rand: $t = rng.gen();
                (lo, hi) = rand.widening_mul(exclusive_ub);
            }
        }
    };
}

pub mod impl_u16 {
    impl_gen_index!(u16);
}

pub mod impl_u32 {
    impl_gen_index!(u32);

    #[inline]
    pub fn gen_index_pair(rng: &mut impl Rng, exclusive_ub: (u32, u32)) -> (u32, u32) {
        let rand: u64 = rng.gen();

        let r0 = rand as u32;
        let r1 = (rand >> 32) as u32;

        let (lo0, hi0) = r0.widening_mul(exclusive_ub.0);
        let (lo1, hi1) = r1.widening_mul(exclusive_ub.1);

        if (lo0 < exclusive_ub.0) | (lo1 < exclusive_ub.1) {
            (
                gen_index_impl(rng, r0, exclusive_ub.0),
                gen_index_impl(rng, r1, exclusive_ub.1),
            )
        } else {
            (hi0, hi1)
        }
    }
}

pub mod impl_u64 {
    impl_gen_index!(u64);
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! impl_tests {
        ($n : expr, $t : ty) => {
            use super::*;
            use rand::SeedableRng;
            use rand_pcg::Pcg64;

            #[test]
            fn below_lower() {
                let mut rng = Pcg64::seed_from_u64(1234);

                for ub in [1, 2, 5, 10, 1000] {
                    for _ in 0..1000 {
                        assert!($n(&mut rng, ub) < ub);
                    }
                }
            }

            #[test]
            fn match_expected() {
                let mut rng = Pcg64::seed_from_u64(12345);
                const ITERATIONS: u64 = 1000;

                for ub in [100, 1000, 10000, <$t>::MAX] {
                    let sum: u128 = (0..ITERATIONS).map(|_| $n(&mut rng, ub) as u128).sum();

                    assert!(sum > ITERATIONS as u128 * (ub as u128) / 4);
                    assert!(sum < ITERATIONS as u128 * (ub as u128) * 3 / 4);
                }
            }
        };
    }

    mod test_u32 {
        impl_tests!(impl_u32::gen_index, u32);
    }
    mod test_u64 {
        impl_tests!(impl_u64::gen_index, u64);
    }

    mod test_usize {
        impl_tests!(gen_index, usize);
    }
}
