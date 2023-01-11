use super::*;
use crate::random_bits::RandomBitsSource;

pub(super) fn rough_shuffle<
    R: Rng,
    T,
    const LOG_NUM_BUCKETS: usize,
    const NUM_BUCKETS: usize,
    const _SWAPS_PER_ROUND: usize,
>(
    rng: &mut R,
    buckets: &mut Buckets<T, NUM_BUCKETS>,
) {
    if buckets.iter().any(|blk| blk.is_fully_processed()) {
        return;
    }

    assert_eq!(1 << LOG_NUM_BUCKETS, NUM_BUCKETS);

    rough_shuffle_impl::<R, T, LOG_NUM_BUCKETS, NUM_BUCKETS>(rng, buckets);
}

fn rough_shuffle_impl<R: Rng, T, const LOG_NUM_BUCKETS: usize, const NUM_BUCKETS: usize>(
    rng: &mut R,
    buckets: &mut Buckets<T, NUM_BUCKETS>,
) -> Option<()> {
    let mut rbs = RandomBitsSource::new();
    let (active_bucket, partners) = buckets.as_mut_slice().split_first_mut().unwrap();

    let mut active_element = active_bucket.first_unprocessed().unwrap();

    loop {
        let partner_bucket_idx = rbs.gen_const_bits::<LOG_NUM_BUCKETS>(rng) as usize;

        if let Some(partner_bucket) = partners.get_mut(partner_bucket_idx) {
            let partner_element = partner_bucket.first_unprocessed().unwrap();

            std::mem::swap(active_element, partner_element);

            partner_bucket.process_element()?;
        } else {
            assert_eq!(partner_bucket_idx, NUM_BUCKETS - 1);
            active_element = active_bucket.process_element()?;
        }
    }
}

#[cfg(test)]
mod test {
    use super::{common_tests, rough_shuffle};

    common_tests::rough_shuffle_tests!(rough_shuffle);
}
