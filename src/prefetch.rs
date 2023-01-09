#![allow(dead_code)]

#[cfg(feature = "prefetch")]
pub mod avail {
    pub const SUPPORTED: bool = true;

    #[inline(always)]
    pub fn prefetch_write_data<T>(item: &mut T) {
        unsafe {
            std::intrinsics::prefetch_write_data(item as *mut T, 1);
        }
    }
}

pub use avail::*;

#[cfg(not(feature = "prefetch"))]
mod mock {
    pub const SUPPORTED: bool = false;

    pub fn prefetch_write_data<T>(item: &mut T) {}
}

#[cfg(not(feature = "prefetch"))]
pub use mock::*;
