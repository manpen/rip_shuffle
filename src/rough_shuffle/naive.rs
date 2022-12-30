use super::*;
use crate::random_bits::RandomBitsSource;

pub(super) fn rough_shuffle<R: Rng, T, const LOG_NUM_BLOCKS: usize, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) {
    if blocks.iter().any(|blk| blk.is_fully_processed()) {
        return;
    }

    assert_eq!(1 << LOG_NUM_BLOCKS, NUM_BLOCKS);

    rough_shuffle_impl::<R, T, LOG_NUM_BLOCKS, NUM_BLOCKS>(rng, blocks);
}

fn rough_shuffle_impl<R: Rng, T, const LOG_NUM_BLOCKS: usize, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) -> Option<()> {
    let mut rbs = RandomBitsSource::new();
    let (active_block, partners) = blocks.as_mut_slice().split_first_mut().unwrap();

    let mut active_element = active_block.first_unprocessed().unwrap();

    loop {
        let partner_block_idx = rbs.gen_const_bits::<LOG_NUM_BLOCKS>(rng) as usize;

        if let Some(partner_block) = partners.get_mut(partner_block_idx) {
            let partner_element = partner_block.first_unprocessed().unwrap();

            std::mem::swap(active_element, partner_element);

            partner_block.process_element()?;
        } else {
            assert_eq!(partner_block_idx, NUM_BLOCKS - 1);
            active_element = active_block.process_element()?;
        }
    }
}

#[cfg(test)]
mod test {
    use super::{common_tests, rough_shuffle};

    common_tests::rough_shuffle_tests!(rough_shuffle);
}
