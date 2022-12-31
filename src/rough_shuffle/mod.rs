use super::blocked::*;
use rand::Rng;

#[cfg(test)]
mod common_tests;

mod naive;
mod stashed;

#[cfg(feature = "unsafe_algos")]
pub mod with_unsafe_algos;

pub struct NumberOfBlocks<const N: usize> {}

pub trait IsPowerOfTwo {
    const N: usize;
    const LOG2: usize;
}

macro_rules! impl_index_bits_trait {
    ($log_n : expr, $n : expr) => {
        impl IsPowerOfTwo for NumberOfBlocks<$n> {
            const N: usize = $n;
            const LOG2: usize = $log_n;
        }
    };
    ($log_n : expr) => {
        impl_index_bits_trait!($log_n, { 1 << $log_n });
    };
}

impl_index_bits_trait!(1);
impl_index_bits_trait!(2);
impl_index_bits_trait!(3);
impl_index_bits_trait!(4);
impl_index_bits_trait!(5);
impl_index_bits_trait!(6);
impl_index_bits_trait!(7);
impl_index_bits_trait!(8);
impl_index_bits_trait!(9);
impl_index_bits_trait!(10);

pub fn rough_shuffle<R: Rng, T, const N: usize>(rng: &mut R, blocks: &mut Blocks<T, N>)
where
    NumberOfBlocks<N>: IsPowerOfTwo,
{
    macro_rules! entry {
        ($log_n : expr) => {{
            #[cfg(feature = "unsafe_algos")]
            with_unsafe_algos::rough_shuffle::<R, T, $log_n, N>(rng, blocks);

            // the unsafe algo may terminate early. then the naive algo takes over.
            naive::rough_shuffle::<R, T, $log_n, N>(rng, blocks);
        }};
    }

    match N {
        2 => entry!(1),
        4 => entry!(2),
        8 => entry!(3),
        16 => entry!(4),
        32 => entry!(5),
        64 => entry!(6),
        128 => entry!(7),
        256 => entry!(8),
        512 => entry!(9),
        1024 => entry!(10),
        _ => panic!(), // cannot be reached due to the PoT type check
    }
}
