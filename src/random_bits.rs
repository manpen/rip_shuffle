use rand::Rng;

/// Accelerator to repeatedly sample a small number of bits
#[derive(Default)]
pub struct RandomBitsSource {
    random_bits: u64,
    num_available: usize,
}

impl RandomBitsSource {
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce up to `num_bits <= 64` random bits and return them in the
    /// `num_bits` least significant positions of the returned value. The
    /// unused bits are cached and may speed up subsequent calls.
    #[inline]
    pub fn gen_bits(&mut self, rng: &mut impl Rng, num_bits: usize) -> u64 {
        if num_bits == 64 {
            return rng.gen();
        }

        let mask = (1u64 << num_bits) - 1;

        let random = if num_bits > self.num_available {
            debug_assert!(num_bits < 64);

            let rand: u64 = rng.gen();
            self.random_bits |= (rand >> num_bits) << self.num_available;
            self.num_available = (self.num_available + 64 - num_bits).min(64);

            rand
        } else {
            let rand = self.random_bits;
            self.random_bits >>= num_bits;
            self.num_available -= num_bits;
            rand
        };

        random & mask
    }

    /// Produce up to `N <= 32` random bits and return them in the
    /// `N` least significant positions of the returned value. The
    /// unused bits are cached and may speed up subsequent calls.
    ///
    /// In contrast to [`RandomBitsSource::gen_bits`] the number of bits
    /// is known at compile time and may lead to more efficient code.
    /// Also, we avoid some costly shifts, which may waste a few
    /// random bits, but eventually pays off in our benchmarks.
    #[inline]
    pub fn gen_const_bits<const N: usize>(&mut self, rng: &mut impl Rng) -> u32 {
        if self.num_available < N {
            self.random_bits = rng.gen();
            self.num_available = 64;
        }

        let mask = (1u64 << N) - 1;
        let rand = self.random_bits & mask;
        self.random_bits >>= N;
        self.num_available -= N;

        rand as u32
    }

    #[inline]
    pub fn gen_bool(&mut self, rng: &mut impl Rng) -> bool {
        self.gen_const_bits::<1>(rng) == 0
    }
}

pub type FairCoin = RandomBitsSource;

#[cfg(test)]
mod test {
    use super::*;

    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn gen_bits_below_lower() {
        let mut rng = Pcg64::seed_from_u64(1234789);
        let mut rbs = RandomBitsSource::new();

        for num_bits in [1, 2, 5, 10, 20, 50, 63] {
            let exclusive_upper_bound = 1u64 << num_bits;

            for _ in 0..1000 {
                assert!(rbs.gen_bits(&mut rng, num_bits) < exclusive_upper_bound);
            }
        }
    }

    #[test]
    fn gen_bits_expected_num_bits() {
        const NUM_ITERATIONS: u64 = 10_000;
        let mut rng = Pcg64::seed_from_u64(234789);
        let mut rbs = RandomBitsSource::new();

        for num_bits in [1u64, 2, 5, 10, 20, 50, 63] {
            let bit_sum: u64 = (0..NUM_ITERATIONS)
                .into_iter()
                .map(|_| rbs.gen_bits(&mut rng, num_bits as usize).count_ones() as u64)
                .sum();

            assert!(4 * bit_sum > NUM_ITERATIONS * num_bits);
            assert!(4 * bit_sum < 3 * NUM_ITERATIONS * num_bits);
        }
    }

    #[test]
    fn gen_const_bits_expected_num_bits() {
        test_const_bits_expected_num_bits::<1>();
        test_const_bits_expected_num_bits::<2>();
        test_const_bits_expected_num_bits::<5>();
        test_const_bits_expected_num_bits::<10>();
        test_const_bits_expected_num_bits::<16>();
        test_const_bits_expected_num_bits::<31>();
    }

    #[test]
    fn gen_const_bits_below_lower() {
        test_const_bits_below_lower::<1>();
        test_const_bits_below_lower::<2>();
        test_const_bits_below_lower::<5>();
        test_const_bits_below_lower::<10>();
        test_const_bits_below_lower::<16>();
        test_const_bits_below_lower::<31>();
    }

    fn test_const_bits_below_lower<const N: usize>() {
        let mut rng = Pcg64::seed_from_u64(1234789 * N as u64 + 3242);
        let mut rbs = RandomBitsSource::new();

        let exclusive_upper_bound = 1u32 << N;

        for _ in 0..1000 {
            assert!(rbs.gen_const_bits::<N>(&mut rng) < exclusive_upper_bound);
        }
    }

    fn test_const_bits_expected_num_bits<const N: usize>() {
        const NUM_ITERATIONS: u64 = 10_000;
        let mut rng = Pcg64::seed_from_u64(23474789 * N as u64 + 3242);
        let mut rbs = RandomBitsSource::new();

        let bit_sum: u64 = (0..NUM_ITERATIONS)
            .into_iter()
            .map(|_| rbs.gen_const_bits::<N>(&mut rng).count_ones() as u64)
            .sum();

        assert!(4 * bit_sum > NUM_ITERATIONS * N as u64);
        assert!(4 * bit_sum < 3 * NUM_ITERATIONS * N as u64);
    }

    #[test]
    fn gen_bool_expected_num_bits() {
        const NUM_ITERATIONS: u64 = 10_000;
        let mut rng = Pcg64::seed_from_u64(234747893);
        let mut rbs = RandomBitsSource::new();

        let bit_sum: u64 = (0..NUM_ITERATIONS)
            .into_iter()
            .map(|_| rbs.gen_bool(&mut rng) as u64)
            .sum();

        assert!(4 * bit_sum > NUM_ITERATIONS);
        assert!(4 * bit_sum < 3 * NUM_ITERATIONS);
    }
}
