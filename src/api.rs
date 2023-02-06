use super::*;
use rand::{Rng, SeedableRng};

pub trait RipShuffleSequential {
    /// Rearranges the input in a random permutation, such that any order appears
    /// with equal probability. The permutation only depends on the random number
    /// generator. If a deterministic sequence is provided, the output is the same
    /// each run with the same build on the same machine.
    ///
    /// # Warning
    /// We might change the algorithm or fine-tune the its parameters. Therefore,
    /// the emitted order might change with future revisions of the code.
    ///
    /// # Example
    /// ```
    /// use rip_shuffle::RipShuffleSequential;
    /// let mut data : Vec<_> = (0..100).into_iter().collect();
    /// let org = data.clone();
    ///
    /// data.seq_shuffle(&mut rand::thread_rng());
    ///
    /// assert_ne!(data, org); // might fail with probility 1 / 100!
    /// ```
    fn seq_shuffle<R: Rng>(&mut self, rng: &mut R);
}

pub trait RipShuffleParallel: Send + Sync {
    /// Rearranges the input in a random permutation, such that any order appears
    /// with equal probability. The permutation only depends on the random number
    /// generator. If a deterministic sequence is provided, the output is the same
    /// each run with the same build on the same machine.
    ///
    /// In contrast to [`RipShuffleSequential::seq_shuffle`], this implementation
    /// uses a rayon worker pool to balance the work over multiple threads (if the
    /// input is sufficiently large.)
    ///
    /// # Remarks
    /// This implementation requires a random number generator that is both seedable
    /// (implements [`rand::SeedableRng`]) and can be exchanged between threads
    /// (implements [`std::marker::Send`] and [`std::marker::Sync`]).
    ///
    /// Amongst others, this does not apply for [`rand::rngs::ThreadRng`]. If this
    /// is your default source of randomness, consider seeding a compatible RNG as
    /// shown in the example. We suggest the very fast [`rand_pcg::Pcg64Mcg`].
    /// If you enable the `seed_with` flag for this crate, you can use the
    /// [`RipShuffleParallel::par_shuffle_seed_with`] short-hand.
    ///
    /// # Warning
    /// We might change the algorithm or fine-tune the its parameters. Therefore,
    /// the emitted order might change with future revisions of the code.
    ///
    /// # Example
    /// ```
    /// use rip_shuffle::RipShuffleParallel;
    /// use rand::prelude::*;
    /// let mut data : Vec<_> = (0..1_000_000).into_iter().collect();
    /// let org = data.clone();
    ///
    /// let mut rng = StdRng::from_rng(thread_rng()).unwrap();
    /// data.par_shuffle(&mut rng);
    ///
    /// assert_ne!(data, org); // might fail with probility 1 / 100!
    /// ```
    fn par_shuffle<R: SeedableRng + Rng + Send + Sync>(&mut self, rng: &mut R);

    /// Invokes [`RipShuffleParallel::par_shuffle`] with a compatible RNG that
    /// is seeded with an arbitrary RNG provided.
    ///
    /// # Example
    /// ```
    /// use rip_shuffle::RipShuffleParallel;
    /// use rand::prelude::*;
    /// let mut data : Vec<_> = (0..1_000_000).into_iter().collect();
    /// let org = data.clone();
    ///
    /// data.par_shuffle_seed_with(&mut thread_rng());
    ///
    /// assert_ne!(data, org); // might fail with probility 1 / 100!
    /// ```
    #[cfg(feature = "seed_with")]
    fn par_shuffle_seed_with<R: Rng>(&mut self, rng: &mut R) {
        let mut pcg = rand_pcg::Pcg64Mcg::from_rng(rng).unwrap();
        self.par_shuffle(&mut pcg);
    }
}

impl<T> RipShuffleSequential for [T] {
    fn seq_shuffle<R: Rng>(&mut self, rng: &mut R) {
        scatter_shuffle::sequential::seq_scatter_shuffle(rng, self)
    }
}

impl<T: Send + Sync> RipShuffleParallel for [T] {
    fn par_shuffle<R: SeedableRng + Rng + Send + Sync>(&mut self, rng: &mut R) {
        scatter_shuffle::parallel::par_scatter_shuffle(rng, self)
    }
}
