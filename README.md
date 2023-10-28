# Rust In-Place Shuffle (rip_shuffle)

This crate contains several efficient in-place shuffling algorithms to generate random permutations.
Their design and performances is analyzed in detail in the paper "Engineering Shared-Memory Parallel Shuffling to Generate Random Permutations In-Place" [M. Penschuck].

At time of writing, the default sequential implementation is 1.5 to 4 times faster than `rand::shuffling`.
The parallel implementation can get several orders of magnitute faster.
All implementations are in-place and do not use heap allocations (though, the parallel algorithms may set up a Rayon worker pool, if it's not already the case).

## Usage

Include the following into your `Cargo.toml` file:

```toml
[dependencies]
rip_shuffle={version="0.1"}
```

For general use cases, we export the two traits [`RipShuffleSequential`] and [`RipShuffleParallel`] which
expose the functions `seq_shuffle` and `par_shuffle`, respectively. The sequential variant `seq_shuffle`
can be used as a drop-in replacement for `rand::shuffle`:

```rust
use rip_shuffle::RipShuffleSequential;
let mut data : Vec<_> = (0..100).into_iter().collect();

data.seq_shuffle(&mut rand::thread_rng());
```

The parallel variant imposes some constraints on the random number generator: it needs to be a [`rand::SeedableRng`] and
support [`std::marker::Send`] and [`std::marker::Sync`]. Most prominently, this is not the case for [`rand::rngs::ThreadRng`].
However, you can seed a compatible instace (e.g., [`rand::rngs::StdRng`] or [`rand_pcg::Pcg64`]) from [`rand::rngs::ThreadRng`] and then pass them:

```rust
use rip_shuffle::RipShuffleParallel;
use rand::prelude::*;

let mut data : Vec<_> = (0..1_000_000).into_iter().collect();

let mut rng = StdRng::from_rng(thread_rng()).unwrap();
data.par_shuffle(&mut rng);
```

As a short-hand you can use `RipShuffleParallel::par_shuffle_seed_with`. This methods supports arbitrary `Rng`s
to seed a `Pcg64Mcg` from them:

```ignore
use rip_shuffle::RipShuffleParallel;
let mut data : Vec<_> = (0..1_000_000).into_iter().collect();

data.par_shuffle_seed_with(&mut rand::thread_rng());
```

## Features

This crate has two default feature sets which should be appropriate for most cases and do not change the API.

- `default` is supposed to work with all recent rust compilers
- `unstable_default` requires nightly features but may yield slightly faster binaries.
  If you are using an non-stable compiler consider enabling this feature.

This crate supports the following features, which are all enable by default:

- `unsafe_algos` this feature enables algorithms that rely on pointer arithmetic, but are faster than their safe variants
- `seed_with` adds a dependency to [`rand_pcg`] and offers the [`RipShuffleParallel::par_shuffle_seed_with`] short-hand.
- `prefetch` enables explicit prefetching via [`std::intrinsics::prefetch_write_data`] to speed-up shuffling.
  This feature does require a **nightly-channel** compiler.


To disable these feature, you can adopt the `dependency` in your `Cargo.toml`, for instace:

```toml
rip_shuffle={version="0.1", default-features = false, features = ["seed_with"]}
```
