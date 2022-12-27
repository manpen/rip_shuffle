use super::*;

pub fn fisher_yates<R: Rng, T>(rng: &mut R, data: &mut [T]) {
    for i in (1..data.len()).rev() {
        let j = uniform_index::gen_index(rng, i + 1);
        data.swap(i, j);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    crate::statistical_tests::test_shuffle_algorithm!(fisher_yates);
    crate::statistical_tests::test_shuffle_algorithm_deterministic!(fisher_yates);
}
