pub trait Slicing: Sized {
    fn prefix(self, n: usize) -> Self;
    fn suffix(self, n: usize) -> Self;

    fn partial_merge_with_right_neighbor(&mut self, rhs: &mut Self, n: usize);
    fn merge_with_right_neighbor(self, rhs: Self) -> Self;
    fn is_left_neighbor_of(&self, rhs: &Self) -> bool;
}

macro_rules! slicing_impl {
    ($to_ptr : ident, $split_at : ident, $from_raw : path) => {
        fn prefix(self, n: usize) -> Self {
            self.$split_at(n).0
        }

        fn suffix(self, n: usize) -> Self {
            let total_len = self.len();
            let start = total_len - n;
            self.$split_at(start).1
        }

        fn partial_merge_with_right_neighbor(&mut self, rhs: &mut Self, n: usize) {
            assert!(self.is_left_neighbor_of(&rhs));
            assert!(self.len() >= n);

            let left_len = self.len() - n;
            let right_len = rhs.len() + n;

            unsafe {
                let begin = self.$to_ptr();
                *self = $from_raw(begin, left_len);
                *rhs = $from_raw(begin.add(left_len), right_len);
            }
        }

        fn merge_with_right_neighbor(self, rhs: Self) -> Self {
            assert!(self.is_left_neighbor_of(&rhs));

            // it's safe since we asserted that both slices are adjacent
            unsafe { $from_raw(self.$to_ptr(), self.len() + rhs.len()) }
        }

        fn is_left_neighbor_of(&self, rhs: &Self) -> bool {
            std::ptr::eq(self.as_ptr_range().end, rhs.as_ptr())
        }
    };
}

impl<'a, T> Slicing for &'a mut [T] {
    slicing_impl!(as_mut_ptr, split_at_mut, std::slice::from_raw_parts_mut);
}

impl<'a, T> Slicing for &'a [T] {
    slicing_impl!(as_ptr, split_at, std::slice::from_raw_parts);
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::Slicing;

    #[test]
    fn prefix_and_suffix() {
        const N: usize = 5;

        for len_pref in 0..N {
            let len_suffix = N - len_pref;

            let data: Vec<_> = (0..N)
                .into_iter()
                .map(|i| (i < len_pref) as usize)
                .collect();

            assert_eq!(data.as_slice().prefix(len_pref).len(), len_pref);
            assert!(data.as_slice().prefix(len_pref).iter().all(|x| *x == 1));

            assert_eq!(data.as_slice().suffix(len_suffix).len(), len_suffix);
            assert!(data.as_slice().suffix(len_suffix).iter().all(|x| *x == 0));
        }
    }

    #[test]
    fn is_neighbor() {
        const N: usize = 8;
        let data = [0; N];

        for (begin0, begin1) in (0..N).cartesian_product(0..N) {
            for (end0, end1) in (begin0..N).cartesian_product(begin1..N) {
                let slice0 = &data[begin0..end0];
                let slice1 = &data[begin1..end1];

                assert_eq!(slice0.is_left_neighbor_of(&slice1), end0 == begin1);
                assert_eq!(slice1.is_left_neighbor_of(&slice0), end1 == begin0);
            }
        }
    }

    #[test]
    fn merge_with_right_neighbor() {
        for total_len in [1, 2, 3, 10] {
            for left_len in 0..total_len {
                let mut data = vec![0usize; total_len];

                let (left, right) = data.as_mut_slice().split_at_mut(left_len);
                right.iter_mut().for_each(|x| *x = 1);

                let left = left as &[usize];
                let right = right as &[usize];

                let merged = left.merge_with_right_neighbor(right);

                assert_eq!(merged.len(), total_len);
                assert!(merged.prefix(left_len).iter().all(|x| *x == 0));
                assert_eq!(merged.iter().sum::<usize>(), total_len - left_len);
            }
        }
    }

    #[test]
    fn partial_merge_with_right_neighbor() {
        for total_len in [1, 2, 3, 10] {
            let data: Vec<_> = (0..total_len).into_iter().collect();
            for left_len in 0..total_len {
                let right_len = total_len - left_len;

                for to_move in 0..left_len {
                    let (mut left, mut right) = data.as_slice().split_at(left_len);

                    left.partial_merge_with_right_neighbor(&mut right, to_move);

                    assert_eq!(left.len(), left_len - to_move);
                    assert_eq!(right.len(), right_len + to_move);

                    assert!(left.is_left_neighbor_of(&right));

                    assert!(left.iter().enumerate().all(|(i, &x)| i == x));
                    assert!(right
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| i + left_len - to_move == x));
                }
            }
        }
    }
}

#[cfg(test)]
mod test_mut {
    use super::Slicing;

    #[test]
    fn prefix() {
        let mut data = vec![0; 5];
        data.as_mut_slice()
            .prefix(2)
            .iter_mut()
            .for_each(|x| *x = 1);

        assert_eq!(data, vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn suffix() {
        let mut data = vec![0; 5];
        data.as_mut_slice()
            .suffix(2)
            .iter_mut()
            .for_each(|x| *x = 1);

        assert_eq!(data, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn combined() {
        let mut data = vec![0; 5];
        data.as_mut_slice()
            .suffix(4)
            .prefix(2)
            .iter_mut()
            .for_each(|x| *x = 1);

        assert_eq!(data, vec![0, 1, 1, 0, 0]);
    }

    #[test]
    fn is_neighbor() {
        const N: usize = 8;
        let mut data = [0; N];

        for (begin0, end0) in (0..N).flat_map(|i| std::iter::repeat(i).zip((i + 1)..N)) {
            for (begin1, end1) in (end0..N).flat_map(|i| std::iter::repeat(i).zip((i + 1)..N)) {
                let mut slice = data.as_mut_slice();
                let slice0;

                slice = slice.split_at_mut(begin0).1;
                (slice0, slice) = slice.split_at_mut(end0 - begin0);
                slice = slice.split_at_mut(begin1 - end0).1;
                let slice1 = slice.split_at_mut(end1 - begin1).0;

                assert_eq!(slice0.len(), end0 - begin0);
                assert_eq!(slice1.len(), end1 - begin1);

                assert_eq!(slice0.is_left_neighbor_of(&slice1), end0 == begin1);
            }
        }
    }

    #[test]
    fn merge_with_right_neighbor() {
        for total_len in [1, 2, 3, 10] {
            for left_len in 0..total_len {
                let mut data = vec![0; total_len];

                let (left, right) = data.as_mut_slice().split_at_mut(left_len);
                right.iter_mut().for_each(|x| *x = 1);

                let merged = left.merge_with_right_neighbor(right);

                assert_eq!(merged.len(), total_len);
                assert!(merged.prefix(left_len).iter().all(|x| *x == 0));
                assert_eq!(merged.iter().sum::<usize>(), total_len - left_len);
            }
        }
    }

    #[test]
    fn partial_merge_with_right_neighbor() {
        for total_len in [1, 2, 3, 10] {
            let mut data: Vec<_> = (0..total_len).into_iter().collect();
            for left_len in 0..total_len {
                let right_len = total_len - left_len;

                for to_move in 0..left_len {
                    let (mut left, mut right) = data.as_mut_slice().split_at_mut(left_len);

                    left.partial_merge_with_right_neighbor(&mut right, to_move);

                    assert_eq!(left.len(), left_len - to_move);
                    assert_eq!(right.len(), right_len + to_move);

                    assert!(left.is_left_neighbor_of(&right));

                    assert!(left.iter().enumerate().all(|(i, &x)| i == x));
                    assert!(right
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| i + left_len - to_move == x));
                }
            }
        }
    }
}
