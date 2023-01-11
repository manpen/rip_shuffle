#![feature(bigint_helper_methods)]
#![feature(core_intrinsics)]
#![feature(slice_swap_unchecked)]
#![feature(slice_split_at_unchecked)]

pub mod fisher_yates;
pub mod random_bits;
pub mod rough_shuffle;
pub mod scatter_shuffle;
pub mod uniform_index;

pub mod prelude {
    pub use super::fisher_yates::fisher_yates;
    pub use super::rough_shuffle::{IsPowerOfTwo, NumberOfBuckets};
    pub use super::scatter_shuffle::parallel::par_scatter_shuffle;
    pub use super::scatter_shuffle::sequential::seq_scatter_shuffle;
    pub use super::scatter_shuffle::{ParConfiguration, SeqConfiguration};
}

mod bucketing;
mod prefetch;

#[cfg(test)]
mod statistical_tests;
