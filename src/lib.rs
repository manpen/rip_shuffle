#![feature(bigint_helper_methods)]
#![feature(core_intrinsics)]
#![feature(slice_swap_unchecked)]

pub mod fisher_yates;
pub mod merge_shuffle;
pub mod random_bits;
pub mod uniform_index;

pub mod prelude {
    pub use super::fisher_yates::fisher_yates;
    pub use super::merge_shuffle::merge_shuffle;
}

mod blocked;
mod rough_shuffle;

#[cfg(test)]
mod statistical_tests;
