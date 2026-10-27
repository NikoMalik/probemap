// NEON and generic SWAR groups adapted from hashbrown (MIT OR Apache-2.0), changed for EMPTY=0x80 and DELETED=0xFE

#[cfg(all(target_arch = "x86_64", not(probemap_generic)))]
mod imp {
    use crate::ctrl;
    use core::arch::x86_64::*;

    use super::BitMask;

    pub(crate) type BitMaskWord = u16;
    pub(crate) const BITMASK_STRIDE: u32 = 1;

    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub(crate) struct Group {
        ctrl: __m128i,
    }

    impl Group {
        #[inline(always)]
        pub(crate) unsafe fn load(ptr: *const u8) -> Self {
            Group {
                // SAFETY: ptr is valid for GROUP_WIDTH bytes (caller invariant of `load`).
                ctrl: unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) },
            }
        }

        #[inline(always)]
        pub(crate) unsafe fn load_aligned(ptr: *const u8) -> Self {
            Group {
                // SAFETY: ptr is valid and aligned for GROUP_WIDTH bytes (caller invariant).
                ctrl: unsafe { _mm_load_si128(ptr.cast::<__m128i>()) },
            }
        }

        #[inline(always)]
        pub(crate) fn match_byte(&self, byte: u8) -> BitMask {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe {
                let cmp = _mm_cmpeq_epi8(self.ctrl, _mm_set1_epi8(byte as i8));
                BitMask(_mm_movemask_epi8(cmp) as u16)
            }
        }

        #[inline(always)]
        pub(crate) fn match_empty(&self) -> BitMask {
            self.match_byte(ctrl::EMPTY)
        }

        #[inline(always)]
        pub(crate) fn match_empty_or_deleted(&self) -> BitMask {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe { BitMask(_mm_movemask_epi8(self.ctrl) as u16) }
        }

        #[inline(always)]
        pub(crate) fn match_full(&self) -> BitMask {
            BitMask(!self.match_empty_or_deleted().0)
        }
    }
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little",
    not(probemap_generic)
))]
mod imp {
    use crate::ctrl;
    use core::arch::aarch64 as neon;

    use super::BitMask;

    pub(crate) type BitMaskWord = u64;
    pub(crate) const BITMASK_STRIDE: u32 = 8;
    const ONE_BIT_PER_BYTE: u64 = 0x8080_8080_8080_8080;

    #[derive(Clone, Copy)]
    pub(crate) struct Group(neon::uint8x8_t);

    impl Group {
        #[inline(always)]
        pub(crate) unsafe fn load(ptr: *const u8) -> Self {
            // SAFETY: ptr is valid for GROUP_WIDTH bytes (caller invariant of `load`).
            Group(unsafe { neon::vld1_u8(ptr) })
        }

        #[inline(always)]
        pub(crate) unsafe fn load_aligned(ptr: *const u8) -> Self {
            // SAFETY: ptr is valid for GROUP_WIDTH bytes (caller invariant of `load`).
            Group(unsafe { neon::vld1_u8(ptr) })
        }

        #[inline(always)]
        fn to_mask(lanes: neon::uint8x8_t) -> BitMask {
            // SAFETY: NEON is enabled by the cfg on this module.
            BitMask(
                unsafe { neon::vget_lane_u64(neon::vreinterpret_u64_u8(lanes), 0) }
                    & ONE_BIT_PER_BYTE,
            )
        }

        #[inline(always)]
        pub(crate) fn match_byte(&self, byte: u8) -> BitMask {
            // SAFETY: NEON is enabled by the cfg on this module.
            Self::to_mask(unsafe { neon::vceq_u8(self.0, neon::vdup_n_u8(byte)) })
        }

        #[inline(always)]
        pub(crate) fn match_empty(&self) -> BitMask {
            self.match_byte(ctrl::EMPTY)
        }

        #[inline(always)]
        pub(crate) fn match_empty_or_deleted(&self) -> BitMask {
            // SAFETY: NEON is enabled by the cfg on this module.
            Self::to_mask(unsafe { neon::vcltz_s8(neon::vreinterpret_s8_u8(self.0)) })
        }

        #[inline(always)]
        pub(crate) fn match_full(&self) -> BitMask {
            // SAFETY: NEON is enabled by the cfg on this module.
            Self::to_mask(unsafe { neon::vcgez_s8(neon::vreinterpret_s8_u8(self.0)) })
        }
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", not(probemap_generic)),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little",
        not(probemap_generic)
    )
)))]
mod imp {
    use super::BitMask;
    use core::ptr;

    pub(crate) type BitMaskWord = u64;
    pub(crate) const BITMASK_STRIDE: u32 = 8;

    #[inline(always)]
    const fn repeat(byte: u8) -> u64 {
        u64::from_ne_bytes([byte; 8])
    }

    const HIGH_BITS: u64 = repeat(0x80);

    #[derive(Clone, Copy)]
    pub(crate) struct Group(u64);

    impl Group {
        #[inline(always)]
        pub(crate) unsafe fn load(ptr: *const u8) -> Self {
            // SAFETY: ptr is valid for GROUP_WIDTH bytes (caller invariant of `load`).
            Group(unsafe { ptr::read_unaligned(ptr.cast::<u64>()) })
        }

        #[inline(always)]
        pub(crate) unsafe fn load_aligned(ptr: *const u8) -> Self {
            // SAFETY: ptr is valid and aligned for GROUP_WIDTH bytes (caller invariant).
            Group(unsafe { ptr::read(ptr.cast::<u64>()) })
        }

        #[inline(always)]
        pub(crate) fn match_byte(&self, byte: u8) -> BitMask {
            let cmp = self.0 ^ repeat(byte);
            BitMask((cmp.wrapping_sub(repeat(0x01)) & !cmp & HIGH_BITS).to_le())
        }

        #[inline(always)]
        pub(crate) fn match_empty(&self) -> BitMask {
            BitMask((self.0 & !(self.0 << 1) & HIGH_BITS).to_le())
        }

        #[inline(always)]
        pub(crate) fn match_empty_or_deleted(&self) -> BitMask {
            BitMask((self.0 & HIGH_BITS).to_le())
        }

        #[inline(always)]
        pub(crate) fn match_full(&self) -> BitMask {
            BitMask((!self.0 & HIGH_BITS).to_le())
        }
    }
}

pub(crate) use imp::Group;
use imp::{BITMASK_STRIDE, BitMaskWord};

pub(crate) const GROUP_WIDTH: usize = core::mem::size_of::<Group>();

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct BitMask(pub(crate) BitMaskWord);

impl BitMask {
    pub(crate) const NONE: Self = BitMask(0);

    #[inline(always)]
    pub(crate) fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub(crate) fn lowest_set_bit(self) -> usize {
        (self.0.trailing_zeros() / BITMASK_STRIDE) as usize
    }

    #[inline(always)]
    pub(crate) fn remove_lowest_bit(&mut self) {
        self.0 &= self.0 - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctrl;

    fn naive(bytes: &[u8], pred: impl Fn(u8) -> bool) -> Vec<usize> {
        (0..GROUP_WIDTH).filter(|&i| pred(bytes[i])).collect()
    }

    fn bits(mut m: BitMask) -> Vec<usize> {
        let mut out = Vec::new();
        while m.any() {
            out.push(m.lowest_set_bit());
            m.remove_lowest_bit();
        }
        out
    }

    #[repr(align(16))]
    struct Aligned([u8; 16]);

    #[test]
    fn group_matches_agree_with_naive() {
        let mut x: u64 = 0x1234_5678_9ABC_DEF1;
        for _ in 0..20_000 {
            let mut buf = Aligned([0; 16]);
            for b in buf.0.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = match x % 4 {
                    0 => ctrl::EMPTY,
                    1 => ctrl::DELETED,
                    _ => (x >> 8) as u8 & 0x7F,
                };
            }
            let bytes = &buf.0;
            // SAFETY: buf is 16 bytes, aligned to 16, GROUP_WIDTH <= 16.
            let g = unsafe { Group::load_aligned(bytes.as_ptr()) };
            assert_eq!(
                bits(g.match_empty()),
                naive(bytes, |b| b == ctrl::EMPTY),
                "{bytes:x?}"
            );
            assert_eq!(
                bits(g.match_empty_or_deleted()),
                naive(bytes, |b| b & 0x80 != 0),
                "{bytes:x?}"
            );
            assert_eq!(
                bits(g.match_full()),
                naive(bytes, ctrl::is_full),
                "{bytes:x?}"
            );
            let tag = bytes[(x % GROUP_WIDTH as u64) as usize] & 0x7F;
            let got = bits(g.match_byte(tag));
            let want = naive(bytes, |b| b == tag);
            assert!(
                want.iter().all(|i| got.contains(i)),
                "tag {tag:x} {bytes:x?}"
            );
            assert!(
                got.iter().all(|&i| ctrl::is_full(bytes[i])),
                "tag {tag:x} {bytes:x?}"
            );
        }
    }
}
