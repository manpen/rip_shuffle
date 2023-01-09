use super::uniform_index;
use rand::Rng;

pub mod naive;
pub(crate) mod noncontiguous;

#[cfg(feature = "prefetch")]
pub mod with_prefetch;

#[cfg(feature = "prefetch")]
pub mod with_prefetch_alt;

#[cfg(feature = "prefetch")]
#[cfg(feature = "unsafe_algos")]
pub mod with_unsafe_algos;

#[allow(unreachable_code)]
pub fn fisher_yates<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    #[cfg(feature = "prefetch")]
    #[cfg(feature = "unsafe_algos")]
    if data.len() < uniform_index::U32_MAX_UPPER_BOUND as usize {
        return with_unsafe_algos::fisher_yates_u32(rng, data);
    }

    #[cfg(feature = "prefetch")]
    return with_prefetch::fisher_yates(rng, data);

    naive::fisher_yates(rng, data);
}
