use rand::Rng;

pub mod parallel;
pub mod sequential;

pub trait SeqConfiguration: Clone {
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

    type Profiler: Profiler;
    fn get_profiler(&self) -> &Self::Profiler;
}

#[macro_export]
macro_rules! implement_no_profiler {
    () => {
        type Profiler = $crate::profiler::no_profiler::NoProfiler;
        fn get_profiler(&self) -> &Self::Profiler {
            &$crate::profiler::no_profiler::NoProfiler {}
        }
    };
}

pub use implement_no_profiler;

#[macro_export]
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

pub use implement_seq_config;

use crate::profiler::Profiler;
