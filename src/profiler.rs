pub trait Profiler {
    type Frame: ProfilerFrame;

    fn start(&self, region: &'static str) -> Self::Frame;
}

pub trait ProfilerFrame {
    fn new_region(&mut self, name: &'static str);
}

pub mod no_profiler {
    use super::*;

    #[derive(Default)]
    pub struct NoProfiler();

    pub struct NoProfileFrame {}
    impl ProfilerFrame for NoProfileFrame {
        fn new_region(&mut self, _name: &'static str) {}
    }
    impl Drop for NoProfileFrame {
        fn drop(&mut self) {}
    }

    impl Profiler for NoProfiler {
        type Frame = NoProfileFrame;

        fn start(&self, _region: &'static str) -> Self::Frame {
            Self::Frame {}
        }
    }
}

pub mod par_profile {}
