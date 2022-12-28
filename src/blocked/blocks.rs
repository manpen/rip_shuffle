#![allow(dead_code)]

use super::block::Block;
use arrayvec::ArrayVec;

pub type Blocks<'a, T, const N: usize> = ArrayVec<Block<'a, T>, N>;

pub fn split_slice_into_blocks<T, const N: usize>(mut data: &mut [T]) -> Blocks<T, N> {
    let total_len = data.len();
    let mut blocks = ArrayVec::new();

    for i in 0..N {
        let start = i * total_len / N;
        let end = (i + 1) * total_len / N;
        let block_data;
        (block_data, data) = data.split_at_mut(end - start);
        blocks.push(Block::new(block_data));
    }

    blocks
}

pub fn compact_into_single_block<T, const N: usize>(mut blocks: Blocks<T, N>) -> Block<T> {
    let mut result = blocks.pop().unwrap();
    while let Some(block) = blocks.pop() {
        result = block.merge_with_right_neighbor(result);
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn split_slice_into_blocks() {
        let mut data: Vec<_> = (0..8).into_iter().collect();

        let blocks: Blocks<_, 2> = super::split_slice_into_blocks(&mut data);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.as_slice()[0].len(), 4);
        assert_eq!(blocks.as_slice()[1].len(), 4);
    }

    #[test]
    fn compact_into_single_block() {
        for (((n0, n1), n2), n3) in (0..3)
            .cartesian_product(0..3)
            .cartesian_product(0..3)
            .cartesian_product(0..3)
        {
            let mut data: Vec<_> = vec![0; 12];
            let num_ones = n0 + n1 + n2 + n3;
            {
                let mut blocks: Blocks<_, 4> = super::split_slice_into_blocks(&mut data);

                for (i, n) in [n0, n1, n2, n3].into_iter().enumerate() {
                    for _ in 0..n {
                        *blocks[i].peek_next_element_to_be_processed().unwrap() = 1;
                        blocks[i].process_element();
                    }
                }

                let compact = super::compact_into_single_block(blocks);

                assert_eq!(compact.len(), 12);
                assert_eq!(compact.num_processed(), num_ones);
                assert_eq!(compact.num_unprocessed(), 12 - num_ones);

                assert_eq!(compact.data().iter().sum::<usize>(), num_ones);
                assert!(compact.data_processed().iter().all(|x| *x == 1));
                assert!(compact.data_unprocessed().iter().all(|x| *x == 0));
            }
        }
    }
}
