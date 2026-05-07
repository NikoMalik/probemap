//! Allocator API shim for stable Rust via `allocator-api2`.
//!
//! Two-tier cfg
//! 1. `feature = "allocator-api2"` (default) — polyfill crate
//! 2. Neither — minimal custom trait + Global fallback

use core::ptr::NonNull;
use std::alloc::Layout;

// --- Tier 1: allocator-api2 (default) ----------------------------------------

#[cfg(feature = "allocator-api2")]
pub(crate) use allocator_api2::alloc::{Allocator, Global};

/// Unified allocation entry point.
/// Returns `Result<NonNull<u8>, ()>` instead of `Result<NonNull<[u8]>, AllocError>`.
#[cfg(feature = "allocator-api2")]
pub(crate) fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
    match alloc.allocate(layout) {
        Ok(ptr) => Ok(ptr.cast()),
        Err(_) => Err(()),
    }
}

// --- Tier 2: fallback — minimal Allocator trait ------------------------------

#[cfg(not(feature = "allocator-api2"))]
mod fallback {
    use core::ptr::NonNull;
    use std::alloc::{self, Layout};

    /// Minimal Allocator trait — allocate/deallocate only.
    ///
    /// # Safety
    /// Implementations must return valid pointers with correct layout.
    pub unsafe trait Allocator {
        /// Allocate a block of memory.
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, ()>;

        /// Deallocate a block of memory.
        ///
        /// # Safety
        /// `ptr` must have been allocated by this allocator with the same `layout`.
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
    }

    /// Global allocator — wraps `std::alloc::alloc/dealloc`.
    #[derive(Copy, Clone, Default)]
    pub struct Global;

    unsafe impl Allocator for Global {
        #[inline]
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, ()> {
            unsafe { NonNull::new(alloc::alloc(layout)).ok_or(()) }
        }

        #[inline]
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }

    pub(crate) fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        alloc.allocate(layout)
    }
}

#[cfg(not(feature = "allocator-api2"))]
pub use fallback::{Allocator, Global, do_alloc};
