#![feature(bigint_helper_methods)]
#![feature(core_intrinsics)]
#![feature(slice_swap_unchecked)]

pub mod fisher_yates;
pub mod random_bits;
pub mod scatter_shuffle;
pub mod uniform_index;

pub mod prelude {
    pub use super::fisher_yates::fisher_yates;
    pub use super::scatter_shuffle::parallel::par_scatter_shuffle;
    pub use super::scatter_shuffle::sequential::seq_scatter_shuffle;
}

mod blocked;
mod prefetch;
mod rough_shuffle;

#[cfg(test)]
mod statistical_tests;
