use rand::Rng;

/// While [`impl_32::gen_index`] supports producing indices up to
/// `u32::MAX`, in practice, the rejection rate rises significantly
/// for large upper bounds and it's then typically faster to use
/// [`impl_64::gen_index`] in these regimes. This constant gives
/// the recommended size when to switch.
pub const U32_MAX_UPPER_BOUND: u32 = u32::MAX / 256;

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
    ($n : ident, $t : ty) => {
        pub mod $n {
            use super::*;

            #[inline]
            pub fn gen_index(rng: &mut impl Rng, exclusive_ub: $t) -> $t {
                let initial = rng.gen();
                gen_index_impl(rng, initial, exclusive_ub)
            }

            #[inline]
            pub fn gen_index_impl(rng: &mut impl Rng, initial: $t, exclusive_ub: $t) -> $t {
                debug_assert!(exclusive_ub != 0);

                let (lo, hi) = initial.widening_mul(exclusive_ub);

                if lo >= exclusive_ub {
                    return hi;
                }

                let t = exclusive_ub.wrapping_neg() % exclusive_ub;
                let mut lo = lo;
                let mut hi = hi;

                loop {
                    if lo >= t {
                        return hi;
                    }

                    let rand: $t = rng.gen();
                    (lo, hi) = rand.widening_mul(exclusive_ub);
                }
            }
        }
    };
}

impl_gen_index!(impl_u32, u32);
impl_gen_index!(impl_u64, u64);

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
