use rand::Rng;

pub mod parallel;
pub mod sequential;

pub trait SeqConfiguration {
    fn seq_base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]);
    fn seq_base_case_size(&self) -> usize;
    fn seq_disable_recursion(&self) -> bool {
        false
    }
}

pub trait ParConfiguration: Send + Sync + SeqConfiguration {
    fn par_base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]);
    fn par_base_case_size(&self) -> usize;
    fn par_number_of_subproblems(&self, n: usize) -> usize;
    fn par_disable_recursion(&self) -> bool {
        false
    }
}

macro_rules! implement_seq_config {
    ($config : ty, $base_algo : path, $size : expr) => {
        impl SeqConfiguration for $config {
            fn seq_base_case_shuffle<R: Rng, T: Sized>(&self, rng: &mut R, data: &mut [T]) {
                $base_algo(rng, data)
            }

            fn seq_base_case_size(&self) -> usize {
                $size
            }
        }
    };
}

pub(crate) use implement_seq_config;
