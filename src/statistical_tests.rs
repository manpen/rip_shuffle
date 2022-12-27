#![allow(unused_macros)]

#[cfg(tarpaulin)]
macro_rules! test_shuffle_algorithm {
    ($func : ident) => {};
}

#[cfg(not(tarpaulin))]
macro_rules! test_shuffle_algorithm {
    ($func : ident) => {
        use rand::SeedableRng;
        use rand_pcg::Pcg64Mcg;

        /// This test asserts that the shuffling algorithm maintains a permutation of the
        /// input elements, i.e. no elements are modified, removed, or added.
        #[test]
        fn preserve_elements() {
            let mut rng = Pcg64Mcg::seed_from_u64(1234);

            for n in 0..1000 {
                let mut data: Vec<_> = (0..n).into_iter().map(|x| 3 * x).collect();
                $func(&mut rng, &mut data);
                data.sort();

                for (idx, &val) in data.iter().enumerate() {
                    assert_eq!(3 * idx, val, "n={}", n);
                }
            }
        }

        /// This tests produces a large number of random permutations of the same input and asserts,
        /// that each element is spotted in each position.  This basically boils down to the
        /// coupon-collector problem and therefore Theta(n*log(n)) runs per input of length n suffice.
        #[test]
        fn test_1_independence() {
            let mut rng = Pcg64Mcg::seed_from_u64(12345);

            for n in [2, 3, 4, 5, 10, 13, 29, 33, 50] {
                let runs = 5 * n * ((n as f64).ln().ceil() as usize);

                let mut positions: Vec<Vec<usize>> = (0..n)
                    .into_iter()
                    .map(|_| Vec::with_capacity(runs))
                    .collect();

                for run in 0..runs {
                    let mut data: Vec<_> = (0usize..n).into_iter().collect();
                    $func(&mut rng, &mut data);
                    for (i, &x) in data.iter().enumerate() {
                        assert_eq!(positions[x].len(), run);
                        positions[x].push(i);
                    }
                }

                for (x, ranks) in positions.iter_mut().enumerate() {
                    ranks.sort();
                    ranks.dedup();

                    let missing = if ranks.len() == n {
                        Vec::new()
                    } else {
                        (0..n).into_iter().filter(|x| !ranks.contains(x)).collect()
                    };

                    assert_eq!(
                        ranks.len(),
                        n,
                        "x = {}, n = {}, missin = {:?}",
                        x,
                        n,
                        missing
                    );
                }
            }
        }

        /// Analogously to `test_1_inpendence` but this time, we consider all pairs of input elements
        /// and assert that each pair of input elements reaches any of the `n*(n-1)` possible indicies.
        /// We therefore need `Theta(n*n*log(n))` many rounds per input sequence.
        #[test]
        fn test_2_independence() {
            let mut rng = Pcg64Mcg::seed_from_u64(2345);

            for n in [5usize, 17, 23] {
                let num_items = (n as f64).powi(2);
                let runs = (3.0 * num_items * num_items.ln()).ceil() as usize;

                let mut positions: Vec<Vec<usize>> = (0..n * n)
                    .into_iter()
                    .map(|_| Vec::with_capacity(runs))
                    .collect();

                for run in 0..runs {
                    let mut data: Vec<_> = (0usize..n).into_iter().collect();
                    $func(&mut rng, &mut data);
                    for (i, &x) in data.iter().enumerate() {
                        for (j, &y) in data.iter().enumerate() {
                            if i == j {
                                continue;
                            }
                            let pair = x + n * y;
                            let rank = i + n * j;
                            assert_eq!(positions[pair].len(), run);
                            positions[pair].push(rank);
                        }
                    }
                }

                for (pair, ranks) in positions.iter_mut().enumerate() {
                    if pair % n == pair / n {
                        assert!(ranks.is_empty());
                        continue;
                    }

                    ranks.sort();
                    ranks.dedup();

                    assert_eq!(
                        ranks.len(),
                        n * n - n,
                        "n = {}, pair = {:?}",
                        n,
                        (pair % n, pair / n)
                    );
                }
            }
        }
    };
}

macro_rules! test_shuffle_algorithm_deterministic {
    ($func : ident) => {
        /// This test asserts that the shuffling algorithm maintains a permutation of the
        /// input elements, i.e. no elements are modified, removed, or added.
        #[test]
        fn deterministic() {
            for num in [2, 5, 10, 13, 29, 50] {
                let rng = Pcg64Mcg::seed_from_u64(1234 * num);

                let runs: Vec<Vec<_>> = (0..10)
                    .map(|_| {
                        let mut data: Vec<_> = (0..num).into_iter().map(|x| 3 * x).collect();
                        let mut rng = rng.clone();
                        $func(&mut rng, &mut data);
                        data
                    })
                    .collect();

                for i in 1..runs.len() {
                    assert_eq!(runs[0], runs[i]);
                }
            }
        }
    };
}

pub(crate) use test_shuffle_algorithm;
pub(crate) use test_shuffle_algorithm_deterministic;
