#![allow(dead_code)]
use super::*;

pub struct Block<'a, T> {
    data: &'a mut [T],
    num_processed: usize,
}

impl<'a, T> Default for Block<'a, T> {
    fn default() -> Self {
        Self {
            data: &mut [],
            num_processed: 0,
        }
    }
}

impl<'a, T> Block<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self {
            data,
            num_processed: 0,
        }
    }

    pub fn new_with_num_unprocessed(data: &'a mut [T], num_unprocessed: usize) -> Self {
        assert!(num_unprocessed <= data.len());
        let n = data.len();
        Self {
            data,
            num_processed: n - num_unprocessed,
        }
    }

    pub fn pop(&mut self) -> Option<&mut T> {
        let data = std::mem::take(&mut self.data);
        if let Some((elem, slice)) = data.split_last_mut() {
            self.data = slice;
            Some(elem)
        } else {
            None
        }
    }

    pub fn push(&mut self, elem: &'a mut T) {
        let rhs = std::slice::from_mut(elem);
        let data = std::mem::take(&mut self.data);
        self.data = data.merge_with_right_neighbor(rhs);
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_fully_processed(&self) -> bool {
        self.num_processed == self.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn merge_with_right_neighbor(mut self, mut rhs: Self) -> Self {
        assert!(self.is_left_neighbor_of(&rhs));

        self.move_stash_to_right_neighbor(&mut rhs);

        let merged_num_processed = self.num_processed + rhs.num_processed;
        let merged_data = self.data.merge_with_right_neighbor(rhs.data);

        Self {
            data: merged_data,
            num_processed: merged_num_processed,
        }
    }

    pub fn set_num_processed(&mut self, num: usize) {
        self.num_processed = num;
    }

    pub fn num_processed(&self) -> usize {
        self.num_processed
    }

    pub fn num_unprocessed(&self) -> usize {
        self.len() - self.num_processed
    }

    pub fn data(&self) -> &[T] {
        self.data
    }

    pub fn data_unprocessed(&self) -> &[T] {
        &self.data()[self.num_processed..]
    }

    pub fn data_processed(&self) -> &[T] {
        self.data().prefix(self.num_processed)
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    pub fn data_unprocessed_mut(&mut self) -> &mut [T] {
        &mut self.data[self.num_processed..]
    }

    pub fn data_processed_mut(&mut self) -> &mut [T] {
        self.data.prefix(self.num_processed)
    }

    pub fn first_unprocessed(&mut self) -> Option<&mut T> {
        self.data.get_mut(self.num_processed)
    }

    pub fn process_element(&mut self) -> Option<&mut T> {
        self.num_processed += 1;
        self.first_unprocessed()
    }

    pub fn move_stash_to_right_neighbor(&mut self, rhs: &mut Self) {
        assert!(self.is_left_neighbor_of(rhs));

        let num_elements_to_move = self.num_unprocessed().min(rhs.num_processed());
        self.data_unprocessed_mut()
            .prefix(num_elements_to_move)
            .swap_with_slice(rhs.data_processed_mut().suffix(num_elements_to_move));

        rhs.num_processed -= num_elements_to_move;
        self.num_processed += num_elements_to_move;
    }

    pub fn is_left_neighbor_of(&self, rhs: &Self) -> bool {
        self.data.is_left_neighbor_of(&rhs.data)
    }

    pub fn shrink_to_right(&mut self, rhs: &mut Self, num: usize) {
        assert!(self.is_left_neighbor_of(rhs));
        assert!(num <= self.num_unprocessed());

        // it holds this_block.num_unprocessed() >= too_long_by >= to_move
        let to_move = rhs.num_processed().min(num);

        // actual data transfer
        {
            let left = self.data_mut().suffix(num).prefix(to_move);
            let right = rhs.data_processed_mut().suffix(to_move);
            left.swap_with_slice(right);
        }

        self.data.give_to_right_neighbor(&mut rhs.data, num);
    }

    pub fn grow_from_right(&mut self, rhs: &mut Self, num: usize) {
        assert!(self.is_left_neighbor_of(rhs));
        assert!(num <= rhs.num_unprocessed());

        rhs.data.give_to_left_neighbor(&mut self.data, num);

        let to_move = rhs.num_processed().min(num);

        // actual data transfer
        {
            let left = self.data_mut().suffix(num).prefix(to_move);
            let right = rhs.data_processed_mut().suffix(to_move);
            left.swap_with_slice(right);
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::Block;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum Item {
        Unint,
        ProcL,
        ProcR,
        Stash,
    }

    #[test]
    fn shrink_to_right() {
        const TOTAL_LEN: usize = 10;
        let mut data = vec![Item::Unint; TOTAL_LEN];
        for left_len in 0..TOTAL_LEN {
            let right_len = TOTAL_LEN - left_len;

            for left_stash_len in 0..left_len {
                for left_shrinkage in 0..left_stash_len {
                    for right_stash_len in 0..right_len {
                        let (mut left, mut right) = {
                            let (left, right) = data.as_mut_slice().split_at_mut(left_len);
                            (Block::new(left), Block::new(right))
                        };

                        left.set_num_processed(left_len - left_stash_len);
                        right.set_num_processed(right_len - right_stash_len);

                        left.data_processed_mut().fill(Item::ProcL);
                        left.data_unprocessed_mut().fill(Item::Stash);
                        right.data_processed_mut().fill(Item::ProcR);
                        right.data_unprocessed_mut().fill(Item::Stash);

                        left.shrink_to_right(&mut right, left_shrinkage);

                        assert_eq!(left.len(), left_len - left_shrinkage);
                        assert_eq!(right.len(), right_len + left_shrinkage);

                        assert_eq!(left.num_unprocessed(), left_stash_len - left_shrinkage);
                        assert_eq!(right.num_unprocessed(), right_stash_len + left_shrinkage);

                        assert!(left.data_processed().iter().all(|&x| x == Item::ProcL));
                        assert!(right.data_processed().iter().all(|&x| x == Item::ProcR));

                        assert!(left
                            .data_unprocessed_mut()
                            .iter()
                            .all(|&x| x == Item::Stash));

                        assert!(right
                            .data_unprocessed_mut()
                            .iter()
                            .all(|&x| x == Item::Stash));
                    }
                }
            }
        }
    }

    #[test]
    fn grow_from_right() {
        const TOTAL_LEN: usize = 10;
        let mut data = vec![Item::Unint; TOTAL_LEN];
        for left_len in 0..TOTAL_LEN {
            let right_len = TOTAL_LEN - left_len;

            for left_stash_len in 0..left_len {
                for right_stash_len in 0..right_len {
                    for left_growth in 0..right_stash_len {
                        let (mut left, mut right) = {
                            let (left, right) = data.as_mut_slice().split_at_mut(left_len);
                            (Block::new(left), Block::new(right))
                        };

                        left.set_num_processed(left_len - left_stash_len);
                        right.set_num_processed(right_len - right_stash_len);

                        left.data_processed_mut().fill(Item::ProcL);
                        left.data_unprocessed_mut().fill(Item::Stash);
                        right.data_processed_mut().fill(Item::ProcR);
                        right.data_unprocessed_mut().fill(Item::Stash);

                        left.grow_from_right(&mut right, left_growth);

                        assert_eq!(left.len(), left_len + left_growth);
                        assert_eq!(right.len(), right_len - left_growth);

                        assert_eq!(left.num_unprocessed(), left_stash_len + left_growth);
                        assert_eq!(right.num_unprocessed(), right_stash_len - left_growth);

                        assert!(left.data_processed().iter().all(|&x| x == Item::ProcL));
                        assert!(right.data_processed().iter().all(|&x| x == Item::ProcR));

                        assert!(left
                            .data_unprocessed_mut()
                            .iter()
                            .all(|&x| x == Item::Stash));

                        assert!(right
                            .data_unprocessed_mut()
                            .iter()
                            .all(|&x| x == Item::Stash));
                    }
                }
            }
        }
    }

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
                assert_eq!(*block.first_unprocessed().unwrap(), j);
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
                *block.first_unprocessed().unwrap() = j + 1;
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
                    assert_eq!(merged.num_unprocessed(), left_stash + right_stash);

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
