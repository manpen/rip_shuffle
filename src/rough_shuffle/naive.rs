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
    let _ = rough_shuffle_impl::<R, T, LOG_NUM_BLOCKS, NUM_BLOCKS>(rng, blocks);
}

fn rough_shuffle_impl<R: Rng, T, const LOG_NUM_BLOCKS: usize, const NUM_BLOCKS: usize>(
    rng: &mut R,
    blocks: &mut Blocks<T, NUM_BLOCKS>,
) -> Option<()> {
    let (active_block, partners) = blocks.as_mut_slice().split_last_mut().unwrap();
    let mut rbs = RandomBitsSource::new();

    let mut active_element = active_block.peek_next_element_to_be_processed().unwrap();

    println!();

    loop {
        let partner_block_idx = rbs.gen_const_bits::<LOG_NUM_BLOCKS>(rng) as usize;
        print!("{partner_block_idx} ");

        if let Some(partner_block) = partners.get_mut(partner_block_idx) {
            let partner_element = partner_block.peek_next_element_to_be_processed().unwrap();

            std::mem::swap(active_element, partner_element);

            partner_block.process_element()?;
        } else {
            active_element = active_block.process_element()?;
        }
    }
}
