use super::uniform_index;
use rand::Rng;

pub mod naive;

#[cfg(feature = "prefetch")]
pub mod with_prefetch;

#[cfg(feature = "prefetch")]
#[cfg(feature = "unsafe_algos")]
pub mod with_unsafe_algos;

pub fn fisher_yates<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    #[cfg(feature = "prefetch")]
    #[cfg(feature = "unsafe_algos")]
    if data.len() < uniform_index::U32_MAX_UPPER_BOUND as usize {
        return with_unsafe_algos::fisher_yates_u32(rng, data);
    }

    #[cfg(feature = "prefetch")]
    with_prefetch::fisher_yates(rng, data);

    #[cfg(not(feature = "prefetch"))]
    naive::fisher_yates(rng, data);
}
