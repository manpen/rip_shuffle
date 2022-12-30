use super::*;

/// A Fisher-Yates implementation that does not assume that the input consists
/// of an contigous array. The implementation is relatively slow and assume that
/// the input ranges have roughly equal size (otherwise it get's even slower).
///
/// # Warning
/// For performance reasons, you should avoid using this function.
pub fn noncontiguous_fisher_yates<R: Rng, T>(rng: &mut R, ranges: &mut [&mut [T]]) {
    if ranges.is_empty() {
        return;
    }

    let mut max_len = ranges.iter().map(|r| r.len()).max().unwrap();
    let mut max_len_tol = ranges.len() * max_len / 2;

    for i_range in (0..ranges.len()).rev() {
        let i_start = if i_range == 0 { 1 } else { 0 };

        for i in (i_start..ranges[i_range].len()).rev() {
            loop {
                let ub = if i_range == 0 {
                    i
                } else {
                    if max_len_tol == 0 {
                        max_len = ranges
                            .iter()
                            .take(i_range + 1)
                            .map(|r| r.len())
                            .max()
                            .unwrap();
                        max_len_tol = (i_range + 1) * max_len / 2;
                    } else {
                        max_len_tol -= 1;
                    }
                    max_len
                };

                let j_range = uniform_index::gen_index(rng, i_range + 1);
                let j = uniform_index::gen_index(rng, ub + 1);

                if j < ranges[j_range].len() {
                    unsafe {
                        let i_ptr = ranges[i_range].as_mut_ptr().add(i);
                        let j_ptr = ranges[j_range].as_mut_ptr().add(j);

                        core::ptr::swap(i_ptr, j_ptr);
                    }

                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test_split {
        ($mod:ident, $func:ident) => {
            mod $mod {
                use super::*;

                pub fn split_adapter<R: Rng, T>(rng: &mut R, mut data: &mut [T]) {
                    let mut ranges: Vec<&mut [T]> = Vec::new();

                    while data.len() > 1 {
                        let prefix;
                        (prefix, data) = data.split_at_mut(rng.gen_range(1..data.len()));
                        ranges.push(prefix);
                    }

                    ranges.push(data);

                    $func(rng, &mut ranges);
                }

                crate::statistical_tests::test_shuffle_algorithm!(split_adapter);
                crate::statistical_tests::test_shuffle_algorithm_deterministic!(split_adapter);
            }
        };
    }

    test_split!(reject, noncontiguous_fisher_yates);
}
