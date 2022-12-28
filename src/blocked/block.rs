#![allow(dead_code)]
use super::*;

pub struct Block<'a, T> {
    data: &'a mut [T],
    num_unprocessed: usize,
}

impl<'a, T> Default for Block<'a, T> {
    fn default() -> Self {
        Self {
            data: &mut [],
            num_unprocessed: 0,
        }
    }
}

impl<'a, T> Block<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self::new_with_num_unprocessed(data, data.len())
    }

    pub fn new_with_num_unprocessed(data: &'a mut [T], num_unprocessed: usize) -> Self {
        assert!(num_unprocessed <= data.len());
        Self {
            data,
            num_unprocessed,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_fully_processed(&self) -> bool {
        self.num_unprocessed == 0
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn merge_with_right_neighbor(mut self, mut rhs: Self) -> Self {
        assert!(self.is_left_neighbor_of(&rhs));

        self.move_stash_to_right_neighbor(&mut rhs);

        let merged_num_unprocessed = self.num_unprocessed + rhs.num_unprocessed;
        let merged_data = self.data.merge_with_right_neighbor(rhs.data);

        Self::new_with_num_unprocessed(merged_data, merged_num_unprocessed)
    }

    pub fn set_num_processed(&mut self, num: usize) {
        self.num_unprocessed = self.len() - num;
    }

    pub fn num_processed(&self) -> usize {
        self.len() - self.num_unprocessed
    }

    pub fn num_unprocessed(&self) -> usize {
        self.num_unprocessed
    }

    pub fn data(&self) -> &[T] {
        self.data
    }

    pub fn data_unprocessed(&self) -> &[T] {
        self.data().suffix(self.num_unprocessed)
    }

    pub fn data_processed(&self) -> &[T] {
        self.data().prefix(self.num_processed())
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    pub fn data_unprocessed_mut(&mut self) -> &mut [T] {
        self.data.suffix(self.num_unprocessed)
    }

    pub fn data_processed_mut(&mut self) -> &mut [T] {
        self.data.prefix(self.num_processed())
    }

    pub fn peek_next_element_to_be_processed(&mut self) -> Option<&mut T> {
        self.data_unprocessed_mut().first_mut()
    }

    pub fn process_element(&mut self) -> Option<&mut T> {
        self.num_unprocessed -= 1;
        self.peek_next_element_to_be_processed()
    }

    pub fn move_stash_to_right_neighbor(&mut self, rhs: &mut Self) {
        assert!(self.is_left_neighbor_of(rhs));

        let num_elements_to_move = self.num_unprocessed.min(rhs.num_processed());
        self.data_unprocessed_mut()
            .prefix(num_elements_to_move)
            .swap_with_slice(rhs.data_processed_mut().suffix(num_elements_to_move));

        rhs.num_unprocessed += num_elements_to_move;
        self.num_unprocessed -= num_elements_to_move;
    }

    pub fn is_left_neighbor_of(&self, rhs: &Self) -> bool {
        self.data.is_left_neighbor_of(&rhs.data)
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::Block;

    #[test]
    fn len() {
        for i in 0..4 {
            let mut data = vec![0; i];
            let block = Block::new(&mut data);

            assert_eq!(block.len(), i);
            assert_eq!(block.is_empty(), i == 0);
        }
    }

    #[test]
    fn is_fully_processed() {
        for i in 0usize..4 {
            let mut data: Vec<_> = (0..i).collect();
            let mut block = Block::new(&mut data);

            for j in 0..i {
                assert!(!block.is_fully_processed());
                assert_eq!(block.num_processed(), j);
                assert_eq!(block.num_unprocessed(), i - j);
                assert_eq!(*block.peek_next_element_to_be_processed().unwrap(), j);
                assert_eq!(block.process_element().is_some(), j + 1 < i);
            }

            assert!(block.is_fully_processed());
        }
    }

    #[test]
    fn data() {
        fn ref_vec(len: usize, processed: usize) -> Vec<usize> {
            (1..=processed + 1)
                .into_iter()
                .chain(std::iter::repeat(0))
                .take(len)
                .collect()
        }

        for i in 0usize..4 {
            let mut data: Vec<_> = vec![0; i];
            let mut block = Block::new(&mut data);

            for j in 0..i {
                *block.peek_next_element_to_be_processed().unwrap() = j + 1;
                block.process_element();

                // data
                assert_eq!(Vec::from(block.data()), ref_vec(i, j));
                assert_eq!(Vec::from(block.data_mut()), ref_vec(i, j));

                // data_processed
                assert_eq!(block.data_processed().len(), j + 1);
                assert_eq!(block.data_processed_mut().len(), j + 1);
                assert!(block
                    .data_processed()
                    .iter()
                    .enumerate()
                    .all(|(i, x)| i + 1 == *x));
                assert!(block
                    .data_processed_mut()
                    .iter()
                    .enumerate()
                    .all(|(i, x)| i + 1 == *x));

                // data_unprocessed
                assert_eq!(block.data_unprocessed().len(), i - j - 1);
                assert_eq!(block.data_unprocessed_mut().len(), i - j - 1);
                assert!(block.data_unprocessed().iter().all(|x| *x == 0));
                assert!(block.data_unprocessed_mut().iter().all(|x| *x == 0));
            }

            assert!(block.is_fully_processed());
        }
    }

    #[test]
    fn move_stash_to_right_neighbor() {
        for total_len in [1usize, 2, 3, 5, 8, 12] {
            for left_len in 0..total_len {
                let right_len = total_len - left_len;

                for (left_stash, right_stash) in (0..=left_len).cartesian_product(0..=right_len) {
                    let mut data = vec![0; total_len];

                    let (left_data, right_data) = data.as_mut_slice().split_at_mut(left_len);

                    let mut left_block = Block::new_with_num_unprocessed(left_data, left_stash);
                    let mut right_block = Block::new_with_num_unprocessed(right_data, right_stash);

                    left_block.data_processed_mut().fill(1);
                    right_block.data_processed_mut().fill(2);

                    let merged = left_block.merge_with_right_neighbor(right_block);

                    assert_eq!(merged.len(), total_len);
                    assert_eq!(merged.num_unprocessed, left_stash + right_stash);

                    assert!(merged.data_processed().iter().all(|x| *x > 0));
                    assert!(merged.data_unprocessed().iter().all(|x| *x == 0));

                    assert_eq!(
                        merged.data_processed().iter().sum::<usize>(),
                        (left_len - left_stash) + 2 * (right_len - right_stash)
                    );
                }
            }
        }
    }
}
