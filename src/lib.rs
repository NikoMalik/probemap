//! # SwissTable
//!
//! A flat hash map based on Google's SwissTable design (used in Abseil C++),
//! Key design points:
//! 1. **Flat storage** — contiguous array, cache-friendly.
//! 2. **Control bytes** — 1-byte metadata per slot (7-bit h2 hash or marker).
//! 3. **SIMD probing** — 16 control bytes checked in one SSE2 instruction.
//! 4. **Linear group probing** — step by GROUP_WIDTH (16) on collision.
//!

/// Allocator API shim — see `alloc.rs`.
mod alloc;
use crate::alloc::{Allocator, Global, do_alloc};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::ptr::NonNull;
use std::alloc::Layout;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;

/// `#[cold]` marks the function as rarely called, making the opposite branch
/// the predicted path. Used to emulate `likely`/`unlikely` on stable.
#[inline(always)]
#[cold]
fn cold_path() {}

/// Hint: condition is likely true.
#[inline(always)]
fn likely(b: bool) -> bool {
    if b {
        true
    } else {
        cold_path();
        false
    }
}

/// Hint: condition is likely false.
#[inline(always)]
fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
        true
    } else {
        false
    }
}

/// FxHash — multiply-rotate hash from Firefox / rustc.
/// Non-cryptographic, very fast for hash tables.
pub struct FxHasher {
    hash: u64,
}

/// Magic constant with good bit distribution for multiply-based mixing.
const SEED: u64 = 0x517cc1b727220a95;

impl FxHasher {
    /// Create a hasher with the given seed.
    #[inline]
    pub fn with_seed(seed: u64) -> Self {
        FxHasher { hash: seed }
    }

    /// Core operation: rotate-xor-multiply.
    #[inline]
    fn add_to_hash(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in chunks.by_ref() {
            let word = u64::from_ne_bytes(chunk.try_into().unwrap());
            self.add_to_hash(word);
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut buf = [0u8; 8];
            buf[..remainder.len()].copy_from_slice(remainder);
            self.add_to_hash(u64::from_ne_bytes(buf));
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_to_hash(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_to_hash(i as u64);
    }
}

/// `BuildHasher` factory for `FxHasher`.
///
/// Allows plugging FxHash into any `HashMap`:
/// ```
/// use probemap::{SwissTable, PairExtract, FxBuildHasher};
/// let map = SwissTable::<PairExtract<u64, String>, FxBuildHasher>::with_hasher(FxBuildHasher);
/// ```
#[derive(Clone, Default)]
pub struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::with_seed(SEED)
    }
}

/// Hybrid hasher: FxHash (rotate-xor-multiply) for integer writes,
/// buthash `low_level_hash` for byte slices
pub struct AutoHasher {
    hash: u64,
}

impl AutoHasher {
    /// Create with seed.
    #[inline]
    pub fn with_seed(seed: u64) -> Self {
        AutoHasher { hash: seed }
    }

    /// FxHash core: rotate-xor-multiply. Fast single-word mixing.
    #[inline]
    fn fx_mix(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl std::hash::Hasher for AutoHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    /// Byte slices -> buthash low_level_hash.
    /// Uses 128-bit multiply mixing for strong distribution at all sizes,
    /// and 64-byte-per-iteration bulk processing for large inputs.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.hash = buthash::hash::low_level_hash(self.hash, bytes);
    }

    /// Integers -> FxHash (single multiply, no loops).
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.fx_mix(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.fx_mix(i as u64);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.fx_mix(i as u64);
    }
}

/// Default `BuildHasher` for probemap.
/// Creates `AutoHasher` — FxHash for ints, buthash for bytes.
#[derive(Clone, Default)]
pub struct AutoBuildHasher;

impl std::hash::BuildHasher for AutoBuildHasher {
    type Hasher = AutoHasher;

    #[inline]
    fn build_hasher(&self) -> AutoHasher {
        AutoHasher::with_seed(SEED)
    }
}

/// Default `BuildHasher` for probemap — `AutoBuildHasher`.
pub type DefaultHashBuilder = AutoBuildHasher;

// =============================================================================
// KeyExtract trait
// =============================================================================

/// Strategy for extracting a key from a stored value.
///
/// Allows storing only values when the key is embedded in the value.
///
/// For simple `(K, V)` pairs, use [`PairExtract`].
///
/// ```
/// use probemap::KeyExtract;
///
/// struct User { id: u64, name: String }
///
/// struct UserById;
/// impl KeyExtract for UserById {
///     type Key = u64;
///     type Value = User;
///     fn extract(value: &User) -> &u64 { &value.id }
/// }
/// ```
pub trait KeyExtract {
    /// Key type. Must be hashable and comparable.
    type Key: std::hash::Hash + PartialEq;
    /// Value type stored in the table.
    type Value;

    /// Extract a key reference from a value.
    fn extract(value: &Self::Value) -> &Self::Key;
}

/// Standard `(K, V)` extraction — key is the first element.
pub struct PairExtract<K, V>(PhantomData<(K, V)>);

impl<K: std::hash::Hash + PartialEq, V> KeyExtract for PairExtract<K, V> {
    type Key = K;
    type Value = (K, V);

    #[inline]
    fn extract(value: &(K, V)) -> &K {
        &value.0
    }
}

/// Control byte constants
///
/// Values:
/// - `0x00..=0x7F` — FULL: slot occupied, stores h2 (top 7 bits of hash)
/// - `0x80` — EMPTY: slot never used
/// - `0xFE` — DELETED: tombstone
///
/// The last `GROUP_WIDTH` bytes of the control array mirror the first
/// `GROUP_WIDTH` bytes. This allows unaligned SIMD group loads near the end
/// of the array to wrap around correctly without missing any slots.
mod ctrl {
    pub(crate) const EMPTY: u8 = 0x80;
    pub(crate) const DELETED: u8 = 0xFE;
    /// Threshold for the `match_empty_or_deleted` signed comparison trick.
    /// As i8: EMPTY(-128) < -1 ✓, DELETED(-2) < -1 ✓, FULL(0..127) < -1 ✗
    pub(crate) const SPECIAL_THRESHOLD: u8 = 0xFF;

    /// Returns true if control byte indicates an occupied slot (bit 7 clear).
    #[inline(always)]
    pub(crate) fn is_full(c: u8) -> bool {
        (c & 0x80) == 0
    }

    /// Extract h2 from hash — top 7 bits (bits 57..63).
    /// These bits are independent from the lower bits used for positioning.
    #[inline(always)]
    pub(crate) fn h2(hash: u64) -> u8 {
        (hash >> 57) as u8
    }
}

/// Group width — 16 bytes (SSE2 `__m128i` register width).
const GROUP_WIDTH: usize = 16;

/// 16 control bytes loaded into an SSE2 register.
///
/// All operations are O(1) via SIMD:
/// - `match_byte(h2)` — find all slots matching a given h2
/// - `match_empty()` — find all empty slots
/// - `match_empty_or_deleted()` — find empty and deleted slots (for insertion)
#[cfg(target_arch = "x86_64")]
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Group {
    ctrl: __m128i,
}

#[cfg(target_arch = "x86_64")]
impl Group {
    /// Unaligned load of 16 control bytes into SSE2 register.
    #[inline]
    unsafe fn load(ptr: *const u8) -> Self {
        Group {
            ctrl: unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) },
        }
    }

    /// Find all positions where control byte == `byte`.
    ///
    /// 1. `_mm_set1_epi8` — broadcast byte to all 16 lanes
    /// 2. `_mm_cmpeq_epi8` — per-byte compare (0xFF if equal, 0x00 if not)
    /// 3. `_mm_movemask_epi8` — extract MSB of each byte -> 16-bit mask
    #[inline]
    fn match_byte(&self, byte: u8) -> BitMask {
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.ctrl, _mm_set1_epi8(byte as i8));
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Find all EMPTY slots.
    #[inline]
    fn match_empty(&self) -> BitMask {
        self.match_byte(ctrl::EMPTY)
    }

    /// Find all EMPTY or DELETED slots.
    ///
    /// Uses signed comparison: ctrl[i] < -1 is true for
    /// EMPTY(-128) and DELETED(-2), but not for FULL(0..127).
    #[inline]
    fn match_empty_or_deleted(&self) -> BitMask {
        unsafe {
            let sentinel = _mm_set1_epi8(ctrl::SPECIAL_THRESHOLD as i8);
            let cmp = _mm_cmpgt_epi8(sentinel, self.ctrl);
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Find all FULL (occupied) slots — bit 7 clear means h2 value (0x00-0x7F).
    /// Single movemask + invert.
    #[inline]
    fn match_full(&self) -> BitMask {
        unsafe {
            let mask = _mm_movemask_epi8(self.ctrl) as u16;
            BitMask(!mask)
        }
    }
}

/// Scalar fallback: 16 control bytes in a plain array.
#[cfg(not(target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct Group {
    ctrl: [u8; GROUP_WIDTH],
}

#[cfg(not(target_arch = "x86_64"))]
impl Group {
    #[inline]
    unsafe fn load(ptr: *const u8) -> Self {
        Group {
            ctrl: unsafe { ptr::read(ptr as *const [u8; GROUP_WIDTH]) },
        }
    }

    #[inline]
    fn match_byte(&self, byte: u8) -> BitMask {
        let mut mask: u16 = 0;
        for i in 0..GROUP_WIDTH {
            if self.ctrl[i] == byte {
                mask |= 1 << i;
            }
        }
        BitMask(mask)
    }

    #[inline]
    fn match_empty(&self) -> BitMask {
        self.match_byte(ctrl::EMPTY)
    }

    /// Signed comparison: EMPTY(-128) and DELETED(-2) are both < -1 as i8.
    #[inline]
    fn match_empty_or_deleted(&self) -> BitMask {
        let mut mask: u16 = 0;
        for i in 0..GROUP_WIDTH {
            if (self.ctrl[i] as i8) < (ctrl::SPECIAL_THRESHOLD as i8) {
                mask |= 1 << i;
            }
        }
        BitMask(mask)
    }

    /// Find all FULL (occupied) slots — bit 7 clear.
    #[inline]
    fn match_full(&self) -> BitMask {
        let mut mask: u16 = 0;
        for i in 0..GROUP_WIDTH {
            if (self.ctrl[i] & 0x80) == 0 {
                mask |= 1 << i;
            }
        }
        BitMask(mask)
    }
}

/// 16-bit mask — one bit per slot in a group.
///
/// Iteration via `trailing_zeros()` (TZCNT) + `x & (x-1)` (Kernighan's trick).
#[derive(Clone, Copy)]
#[repr(transparent)]
struct BitMask(u16);
const _: () = assert!(std::mem::size_of::<BitMask>() == 2);

impl BitMask {
    /// Any bits set?
    #[inline]
    fn any(self) -> bool {
        self.0 != 0
    }

    /// Index of lowest set bit (0..15).
    #[inline]
    fn lowest_set_bit(self) -> usize {
        self.0.trailing_zeros() as usize
    }

    /// Clear the lowest set bit.
    #[inline]
    fn remove_lowest_bit(&mut self) {
        self.0 &= self.0 - 1;
    }
}

/// Prefetch data into L1 cache.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn prefetch_read(ptr: *const u8) {
    unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0) };
}

/// Prefetch data into L2+ cache.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn prefetch_read_l2(ptr: *const u8) {
    unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T1) };
}

/// No-op prefetch on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn prefetch_read(_ptr: *const u8) {}

/// No-op prefetch on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn prefetch_read_l2(_ptr: *const u8) {}

// =============================================================================
// SwissTable
// =============================================================================

/// Open-addressing hash table with SIMD probing.
///
/// ## Type parameters
/// - `E: KeyExtract` — key extraction strategy
/// - `S: BuildHasher` — hash builder (defaults to `AutoBuildHasher`)
/// - `A: Allocator` — memory allocator (defaults to `Global`)
///
/// ## Layout
/// - `ctrl`: control bytes array, length = cap + GROUP_WIDTH (overflow sentinel zone)
/// - `slots`: value array (`MaybeUninit` — not all slots are initialized)
/// - Capacity is always a power of 2
pub struct SwissTable<E: KeyExtract, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    pub(crate) ctrl: NonNull<u8>,
    pub(crate) slots: NonNull<MaybeUninit<E::Value>>,
    pub(crate) cap: usize,
    pub(crate) len: usize,
    pub(crate) hash_builder: S,
    pub(crate) alloc: A,
    pub(crate) _marker: PhantomData<E::Value>,
}

impl<E: KeyExtract, S: BuildHasher + Clone, A: Allocator + Clone> Clone for SwissTable<E, S, A>
where
    E::Value: Clone,
{
    fn clone(&self) -> Self {
        self.clone_table()
    }
}

/// Load factor threshold: grow when `len >= cap * 87 / 100`.
const LOAD_FACTOR_NUMERATOR: usize = 87;

impl<E: KeyExtract, S: BuildHasher + Default, A: Allocator + Default> Default
    for SwissTable<E, S, A>
{
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

/// Convenience constructors with default hasher and global allocator.
impl<E: KeyExtract> SwissTable<E> {
    /// Create an empty table. Does not allocate.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Create a table with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

/// Constructors with custom hasher.
impl<E: KeyExtract, S: BuildHasher> SwissTable<E, S, Global> {
    /// Create an empty table with a custom hash builder.
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            ctrl: NonNull::dangling(),
            slots: NonNull::dangling(),
            cap: 0,
            len: 0,
            hash_builder,
            alloc: Global,
            _marker: PhantomData,
        }
    }

    /// Create a table with capacity and custom hash builder.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        let mut table = Self::with_hasher(hash_builder);
        if capacity > 0 {
            let cap = capacity
                .max(GROUP_WIDTH)
                .checked_next_power_of_two()
                .expect("capacity overflow");
            unsafe {
                table.ctrl = Self::alloc_ctrl(cap, &table.alloc).unwrap();
                table.slots = Self::alloc_slots(cap, &table.alloc).unwrap();
                let cp = table.ctrl_ptr();
                ptr::write_bytes(cp, ctrl::EMPTY, cap + GROUP_WIDTH);
                table.cap = cap;
            }
        }
        table
    }
}

impl<E: KeyExtract, S: BuildHasher + Default, A: Allocator> SwissTable<E, S, A> {
    /// Create an empty table with a custom allocator. Does not allocate.
    pub fn new_in(alloc: A) -> Self {
        Self::with_hasher_and_alloc(S::default(), alloc)
    }

    /// Create a table with the given capacity and custom allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let mut table = Self::new_in(alloc);
        if capacity > 0 {
            let cap = capacity
                .max(GROUP_WIDTH)
                .checked_next_power_of_two()
                .expect("capacity overflow");
            unsafe {
                table.ctrl = Self::alloc_ctrl(cap, &table.alloc).unwrap();
                table.slots = Self::alloc_slots(cap, &table.alloc).unwrap();
                let cp = table.ctrl_ptr();
                ptr::write_bytes(cp, ctrl::EMPTY, cap + GROUP_WIDTH);
                table.cap = cap;
            }
        }
        table
    }
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> SwissTable<E, S, A> {
    /// Create an empty table with custom hasher and allocator.
    pub fn with_hasher_and_alloc(hash_builder: S, alloc: A) -> Self {
        Self {
            ctrl: NonNull::dangling(),
            slots: NonNull::dangling(),
            cap: 0,
            len: 0,
            hash_builder,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the underlying allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns a reference to the hash builder.
    #[inline]
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Hash a key through the stored BuildHasher.
    #[inline]
    fn make_hash(&self, key: &E::Key) -> u64 {
        self.hash_builder.hash_one(key)
    }

    /// Raw pointer to control bytes.
    #[inline(always)]
    fn ctrl_ptr(&self) -> *mut u8 {
        self.ctrl.as_ptr()
    }

    /// Raw pointer to slots array.
    #[inline(always)]
    fn slots_ptr(&self) -> *mut MaybeUninit<E::Value> {
        self.slots.as_ptr()
    }

    /// Write a control byte and maintain the trailing mirror.
    ///
    /// The first `GROUP_WIDTH` bytes are duplicated at `ctrl[cap..]` so that
    /// unaligned SIMD loads near the end of the array see the correct wrapped
    /// values instead of stale data.
    #[inline(always)]
    unsafe fn set_ctrl(&self, idx: usize, value: u8) {
        unsafe {
            *self.ctrl_ptr().add(idx) = value;
            if idx < GROUP_WIDTH {
                *self.ctrl_ptr().add(self.cap + idx) = value;
            }
        }
    }

    /// Lazy-init: allocate minimum capacity in-place (without moving the allocator).
    fn ensure_capacity(&mut self) {
        debug_assert_eq!(self.cap, 0);
        let cap = GROUP_WIDTH;
        unsafe {
            self.ctrl = Self::alloc_ctrl(cap, &self.alloc).unwrap();
            self.slots = Self::alloc_slots(cap, &self.alloc).unwrap();
            let cp = self.ctrl_ptr();
            ptr::write_bytes(cp, ctrl::EMPTY, cap + GROUP_WIDTH);
            self.cap = cap;
        }
    }

    /// Allocate control bytes array, aligned to GROUP_WIDTH (16).
    unsafe fn alloc_ctrl<AA: Allocator>(cap: usize, a: &AA) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(cap + GROUP_WIDTH, GROUP_WIDTH).unwrap();
        do_alloc(a, layout).ok()
    }

    /// Allocate slots array.
    unsafe fn alloc_slots<AA: Allocator>(
        cap: usize,
        a: &AA,
    ) -> Option<NonNull<MaybeUninit<E::Value>>> {
        let layout = Layout::array::<MaybeUninit<E::Value>>(cap).unwrap();
        let ptr = do_alloc(a, layout).ok()?;
        Some(ptr.cast::<MaybeUninit<E::Value>>())
    }

    /// Free control bytes array via allocator.
    unsafe fn dealloc_ctrl(ctrl: NonNull<u8>, cap: usize, alloc: &A) {
        let layout = Layout::from_size_align(cap + GROUP_WIDTH, GROUP_WIDTH).unwrap();
        unsafe { alloc.deallocate(ctrl, layout) };
    }

    /// Free slots array via allocator.
    unsafe fn dealloc_slots(slots: NonNull<MaybeUninit<E::Value>>, cap: usize, alloc: &A) {
        let layout = Layout::array::<MaybeUninit<E::Value>>(cap).unwrap();
        unsafe { alloc.deallocate(slots.cast::<u8>(), layout) };
    }

    /// Initial probe position from lower bits of hash.
    /// `hash & (cap - 1)` — fast modulo for powers of two.
    #[inline(always)]
    fn probe_start(&self, hash: u64) -> usize {
        (hash as usize) & (self.cap - 1)
    }

    /// Next group in probe chain: `(pos + GROUP_WIDTH) & (cap - 1)`.
    #[inline(always)]
    fn next_group(&self, pos: usize) -> usize {
        (pos + GROUP_WIDTH) & (self.cap - 1)
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the table is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Current capacity (number of slots, not accounting for load factor).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Insert a value (or update existing by key).
    ///
    /// If the key already exists, the value is overwritten.
    /// Grows automatically when load factor is exceeded.
    pub fn insert(&mut self, value: E::Value) {
        if unlikely(self.cap == 0) {
            self.ensure_capacity();
        }
        if unlikely(self.len >= (self.cap * LOAD_FACTOR_NUMERATOR) / 100) {
            self.grow();
        }
        self.insert_inner(value);
    }

    /// Fast insert for rehash — no duplicate check, no deleted slots.
    /// Only valid on a freshly allocated table (all EMPTY, no DELETED).
    fn insert_assume_unique(&mut self, value: E::Value) {
        let hash = self.make_hash(E::extract(&value));
        let h2 = ctrl::h2(hash);
        let mask = self.cap - 1;
        let mut pos = self.probe_start(hash);

        loop {
            let group = unsafe { Group::load(self.ctrl_ptr().add(pos)) };
            let empty = group.match_empty();
            if empty.any() {
                let idx = (pos + empty.lowest_set_bit()) & mask;
                unsafe {
                    self.set_ctrl(idx, h2);
                    ptr::write(self.slots_ptr().add(idx), MaybeUninit::new(value));
                }
                self.len += 1;
                return;
            }
            pos = self.next_group(pos);
        }
    }

    /// Inner insert with duplicate check.
    #[inline]
    fn insert_inner(&mut self, value: E::Value) {
        let key = E::extract(&value);
        let hash = self.make_hash(key);
        let h2 = ctrl::h2(hash);
        let cap_mask = self.cap - 1;
        let mut pos = self.probe_start(hash);
        let mut first_empty: usize = usize::MAX; // sentinel: no candidate yet

        loop {
            let group = unsafe { Group::load(self.ctrl_ptr().add(pos)) };

            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                let slot_ptr = unsafe { self.slots_ptr().add(idx) };
                let val_ptr = slot_ptr as *const E::Value;
                let slot_ref = unsafe { &*val_ptr };
                if E::extract(slot_ref) == key {
                    unsafe {
                        ptr::drop_in_place(slot_ptr.cast::<E::Value>());
                        ptr::write(slot_ptr, MaybeUninit::new(value));
                    }
                    return;
                }
                mask.remove_lowest_bit();
            }

            // Track first empty/deleted slot in probe order
            if unlikely(first_empty == usize::MAX) {
                let empty_del = group.match_empty_or_deleted();
                if empty_del.any() {
                    first_empty = (pos + empty_del.lowest_set_bit()) & cap_mask;
                }
            }

            if likely(group.match_empty().any()) {
                let idx = first_empty;
                unsafe {
                    self.set_ctrl(idx, h2);
                    ptr::write(self.slots_ptr().add(idx), MaybeUninit::new(value));
                }
                self.len += 1;
                return;
            }

            pos = self.next_group(pos);
        }
    }

    #[inline]
    fn get_inner(&self, key: &E::Key) -> Option<*const E::Value> {
        if unlikely(self.cap == 0) {
            return None;
        }
        let hash = self.make_hash(key);
        let h2 = ctrl::h2(hash);
        let cap_mask = self.cap - 1;
        let mut pos = self.probe_start(hash);

        loop {
            let group = unsafe { Group::load(self.ctrl_ptr().add(pos)) };
            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                let val_ptr = unsafe { self.slots_ptr().add(idx) } as *const E::Value;
                let slot_ref = unsafe { &*val_ptr };
                if E::extract(slot_ref) == key {
                    return Some(val_ptr);
                }
                mask.remove_lowest_bit();
            }
            if likely(group.match_empty().any()) {
                return None;
            }
            pos = self.next_group(pos);
        }
    }

    /// Get a reference to the value for the given key.
    pub fn get(&self, key: &E::Key) -> Option<&E::Value> {
        unsafe { self.get_inner(key).map(|p| &*p) }
    }

    /// Get a mutable reference to the value for the given key.
    pub fn get_mut(&mut self, key: &E::Key) -> Option<&mut E::Value> {
        unsafe { self.get_inner(key).map(|p| &mut *p.cast_mut()) }
    }

    /// Remove by key. Uses DELETED tombstone to preserve probe chains.
    fn remove_inner(&mut self, key: &E::Key) -> bool {
        if unlikely(self.cap == 0) {
            return false;
        }
        let hash = self.make_hash(key);
        let h2 = ctrl::h2(hash);
        let cap_mask = self.cap - 1;
        let mut pos = self.probe_start(hash);

        loop {
            let group = unsafe { Group::load(self.ctrl_ptr().add(pos)) };
            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                let val_ptr = unsafe { self.slots_ptr().add(idx) } as *const E::Value;
                let slot_ref = unsafe { &*val_ptr };
                if E::extract(slot_ref) == key {
                    unsafe {
                        ptr::drop_in_place(self.slots_ptr().add(idx).cast::<E::Value>());
                        self.set_ctrl(idx, ctrl::DELETED);
                    }
                    self.len -= 1;
                    return true;
                }
                mask.remove_lowest_bit();
            }
            if likely(group.match_empty().any()) {
                return false;
            }
            pos = self.next_group(pos);
        }
    }

    /// Remove an element by key. Returns `true` if the key was found.
    pub fn remove(&mut self, key: &E::Key) -> bool {
        self.remove_inner(key)
    }

    /// Returns true if the key exists.
    pub fn contains(&self, key: &E::Key) -> bool {
        self.get(key).is_some()
    }

    /// Get or insert: returns `&mut` to existing value if key found,
    /// otherwise inserts `value` and returns `&mut` to the new entry.
    pub fn get_or_insert(&mut self, value: E::Value) -> &mut E::Value {
        if unlikely(self.cap == 0) {
            self.ensure_capacity();
        }
        if unlikely(self.len >= (self.cap * LOAD_FACTOR_NUMERATOR) / 100) {
            self.grow();
        }
        self.get_or_insert_inner(value)
    }

    fn get_or_insert_inner(&mut self, value: E::Value) -> &mut E::Value {
        let hash = self.make_hash(E::extract(&value));
        let h2 = ctrl::h2(hash);
        let cap_mask = self.cap - 1;
        let mut pos = self.probe_start(hash);
        let mut first_empty: usize = usize::MAX;

        loop {
            let group = unsafe { Group::load(self.ctrl_ptr().add(pos)) };
            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                let val_ptr = unsafe { self.slots_ptr().add(idx) }.cast::<E::Value>();
                let slot_ref = unsafe { &mut *val_ptr };
                if E::extract(slot_ref) == E::extract(&value) {
                    return slot_ref;
                }
                mask.remove_lowest_bit();
            }

            if unlikely(first_empty == usize::MAX) {
                let empty_del = group.match_empty_or_deleted();
                if empty_del.any() {
                    first_empty = (pos + empty_del.lowest_set_bit()) & cap_mask;
                }
            }

            if likely(group.match_empty().any()) {
                let idx = first_empty;
                unsafe {
                    self.set_ctrl(idx, h2);
                    ptr::write(self.slots_ptr().add(idx), MaybeUninit::new(value));
                }
                self.len += 1;
                return unsafe { &mut *(self.slots_ptr().add(idx).cast::<E::Value>()) };
            }

            pos = self.next_group(pos);
        }
    }

    /// Drop all values and mark all slots as EMPTY.
    pub fn clear(&mut self) {
        if self.cap == 0 {
            return;
        }
        unsafe {
            for i in 0..self.cap {
                if ctrl::is_full(*self.ctrl_ptr().add(i)) {
                    ptr::drop_in_place(self.slots_ptr().add(i).cast::<E::Value>());
                }
            }
            ptr::write_bytes(self.ctrl_ptr(), ctrl::EMPTY, self.cap + GROUP_WIDTH);
        }
        self.len = 0;
    }

    /// Deep-copy the table.
    ///
    /// For `Copy` types (no drop glue) this is a single `memcpy` of the entire
    /// slots array. For non-Copy types each occupied slot is cloned individually.
    pub fn clone_table(&self) -> Self
    where
        E::Value: Clone,
        S: Clone,
        A: Clone,
    {
        if self.cap == 0 {
            return Self::with_hasher_and_alloc(self.hash_builder.clone(), self.alloc.clone());
        }
        unsafe {
            let ctrl = Self::alloc_ctrl(self.cap, &self.alloc).unwrap();
            let slots = Self::alloc_slots(self.cap, &self.alloc).unwrap();

            let cp = ctrl.as_ptr();
            let sp = slots.as_ptr();

            // Control bytes are POD — plain memcpy
            ptr::copy_nonoverlapping(self.ctrl_ptr(), cp, self.cap + GROUP_WIDTH);

            if !std::mem::needs_drop::<E::Value>() {
                // Copy type — bulk memcpy of entire slots array.
                // Unoccupied slots contain garbage MaybeUninit, which is fine
                // because we never read them without checking ctrl first.
                ptr::copy_nonoverlapping(self.slots_ptr(), sp, self.cap);
            } else {
                for i in 0..self.cap {
                    if ctrl::is_full(*self.ctrl_ptr().add(i)) {
                        let src = &*(self.slots_ptr().add(i) as *const E::Value);
                        ptr::write(sp.add(i), MaybeUninit::new(src.clone()));
                    }
                }
            }

            Self {
                ctrl,
                slots,
                cap: self.cap,
                len: self.len,
                hash_builder: self.hash_builder.clone(),
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            }
        }
    }

    /// Iterator over all values. Uses SIMD group scan to skip empty regions.
    pub fn iter(&self) -> Iter<'_, E, S, A> {
        let bitmask = if self.cap > 0 {
            unsafe { Group::load(self.ctrl_ptr()) }.match_full()
        } else {
            BitMask(0)
        };
        Iter {
            table: self,
            group_pos: 0,
            bitmask,
        }
    }

    /// Prefetch ctrl + slots for a key into CPU cache.
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch(&self, key: &E::Key) {
        if self.cap == 0 {
            return;
        }
        let hash = self.make_hash(key);
        let pos = self.probe_start(hash);
        unsafe {
            prefetch_read(self.ctrl_ptr().add(pos));
            prefetch_read(self.slots_ptr().add(pos) as *const u8);
            let next_pos = self.next_group(pos);
            prefetch_read_l2(self.ctrl_ptr().add(next_pos));
        }
    }

    /// Prefetch (no-op on non-x86_64).
    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch(&self, _key: &E::Key) {}

    /// Double capacity and rehash all elements.
    /// O(n). Deleted slots are cleaned up during rehash.
    fn grow(&mut self) {
        let new_cap = if self.cap == 0 {
            GROUP_WIDTH
        } else {
            self.cap.checked_mul(2).expect("capacity overflow on grow")
        };

        unsafe {
            let new_ctrl = Self::alloc_ctrl(new_cap, &self.alloc).unwrap();
            let new_slots = Self::alloc_slots(new_cap, &self.alloc).unwrap();
            let ncp = new_ctrl.as_ptr();
            ptr::write_bytes(ncp, ctrl::EMPTY, new_cap + GROUP_WIDTH);

            let old_ctrl = self.ctrl;
            let old_slots = self.slots;
            let old_cap = self.cap;

            self.ctrl = new_ctrl;
            self.slots = new_slots;
            self.cap = new_cap;
            self.len = 0;

            // Rehash: move (not clone) all FULL elements
            if old_cap > 0 {
                let ocp = old_ctrl.as_ptr();
                let osp = old_slots.as_ptr();
                for i in 0..old_cap {
                    if ctrl::is_full(*ocp.add(i)) {
                        let value = ptr::read(osp.add(i) as *const E::Value);
                        self.insert_assume_unique(value);
                    }
                }
                Self::dealloc_ctrl(old_ctrl, old_cap, &self.alloc);
                Self::dealloc_slots(old_slots, old_cap, &self.alloc);
            }
        }
    }
}

// =============================================================================
// Drop & Iterator
// =============================================================================

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Drop for SwissTable<E, S, A> {
    fn drop(&mut self) {
        if unlikely(self.cap == 0) {
            return;
        }
        unsafe {
            for i in 0..self.cap {
                if ctrl::is_full(*self.ctrl_ptr().add(i)) {
                    ptr::drop_in_place(self.slots_ptr().add(i).cast::<E::Value>());
                }
            }
            Self::dealloc_ctrl(self.ctrl, self.cap, &self.alloc);
            Self::dealloc_slots(self.slots, self.cap, &self.alloc);
        }
    }
}

/// SIMD-accelerated iterator — scans 16 control bytes per group via bitmask.
pub struct Iter<'a, E: KeyExtract, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    table: &'a SwissTable<E, S, A>,
    group_pos: usize,
    bitmask: BitMask,
}

impl<'a, E: KeyExtract, S: BuildHasher, A: Allocator> Iterator for Iter<'a, E, S, A> {
    type Item = &'a E::Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            loop {
                if self.bitmask.any() {
                    let bit = self.bitmask.lowest_set_bit();
                    self.bitmask.remove_lowest_bit();
                    let idx = self.group_pos + bit;
                    return Some(&*(self.table.slots_ptr().add(idx).cast::<E::Value>()));
                }
                self.group_pos += GROUP_WIDTH;
                if self.group_pos >= self.table.cap {
                    return None;
                }
                let group = Group::load(self.table.ctrl_ptr().add(self.group_pos));
                self.bitmask = group.match_full();
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestValue {
        key: u32,
        data: &'static str,
    }

    struct TestExtract;
    impl KeyExtract for TestExtract {
        type Key = u32;
        type Value = TestValue;
        #[inline]
        fn extract(value: &TestValue) -> &u32 {
            &value.key
        }
    }

    type TestMap = SwissTable<TestExtract>;

    // Identity hash — forces collisions for keys with same lower bits.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct IdentityKey(u32);

    impl std::hash::Hash for IdentityKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write_u32(self.0);
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct IdentityValue {
        key: IdentityKey,
        data: &'static str,
    }

    struct IdentityExtract;
    impl KeyExtract for IdentityExtract {
        type Key = IdentityKey;
        type Value = IdentityValue;
        #[inline]
        fn extract(value: &IdentityValue) -> &IdentityKey {
            &value.key
        }
    }

    type IdentityMap = SwissTable<IdentityExtract>;

    #[test]
    fn test_basic_put_and_get() {
        let mut map = TestMap::with_capacity(64);
        assert_eq!(map.capacity(), 64);
        assert_eq!(map.len(), 0);

        map.insert(TestValue {
            key: 1,
            data: "one",
        });
        map.insert(TestValue {
            key: 2,
            data: "two",
        });
        assert_eq!(map.len(), 2);

        assert_eq!(map.get(&1).unwrap().data, "one");
        assert_eq!(map.get(&2).unwrap().data, "two");
        assert!(map.get(&3).is_none());
    }

    #[test]
    fn test_update_existing() {
        let mut map = TestMap::with_capacity(64);
        map.insert(TestValue {
            key: 1,
            data: "one",
        });
        map.insert(TestValue {
            key: 2,
            data: "two",
        });
        assert_eq!(map.len(), 2);

        map.insert(TestValue {
            key: 1,
            data: "updated",
        });
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1).unwrap().data, "updated");
    }

    #[test]
    fn test_remove() {
        let mut map = TestMap::with_capacity(64);
        map.insert(TestValue {
            key: 1,
            data: "one",
        });
        map.insert(TestValue {
            key: 2,
            data: "two",
        });

        assert!(map.remove(&1));
        assert_eq!(map.len(), 1);
        assert!(map.get(&1).is_none());
        assert!(!map.remove(&1));
    }

    #[test]
    fn test_iterator() {
        let mut map = TestMap::with_capacity(64);
        map.insert(TestValue {
            key: 2,
            data: "two",
        });
        map.insert(TestValue {
            key: 3,
            data: "three",
        });

        let mut found_two = false;
        let mut found_three = false;
        let mut count = 0;

        for value in map.iter() {
            count += 1;
            match value.key {
                2 => {
                    found_two = true;
                    assert_eq!(value.data, "two");
                }
                3 => {
                    found_three = true;
                    assert_eq!(value.data, "three");
                }
                _ => panic!("unexpected key {}", value.key),
            }
        }

        assert_eq!(count, 2);
        assert!(found_two);
        assert!(found_three);
    }

    #[test]
    fn test_clear() {
        let mut map = TestMap::with_capacity(64);
        map.insert(TestValue {
            key: 1,
            data: "one",
        });
        map.insert(TestValue {
            key: 2,
            data: "two",
        });

        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.get(&1).is_none());
        assert!(map.get(&2).is_none());
    }

    #[test]
    fn test_clone() {
        let mut map = TestMap::with_capacity(64);
        map.insert(TestValue {
            key: 1,
            data: "one",
        });
        map.insert(TestValue {
            key: 2,
            data: "two",
        });
        map.insert(TestValue {
            key: 3,
            data: "three",
        });
        map.remove(&2);
        map.insert(TestValue {
            key: 4,
            data: "four",
        });

        let mut copy = map.clone_table();
        assert_eq!(copy.len(), map.len());
        assert_eq!(copy.capacity(), map.capacity());
        assert_eq!(copy.get(&1).unwrap().data, "one");
        assert!(copy.get(&2).is_none());
        assert_eq!(copy.get(&3).unwrap().data, "three");
        assert_eq!(copy.get(&4).unwrap().data, "four");

        copy.insert(TestValue {
            key: 5,
            data: "five",
        });
        assert!(map.get(&5).is_none());
    }

    /// Regression: put after delete must not duplicate the key.
    /// A deleted slot earlier in the probe chain must not cause
    /// a duplicate insertion when the key already exists further along.
    #[test]
    fn test_put_after_delete_no_duplicate() {
        let mut map = IdentityMap::with_capacity(16);

        for i in 0..16u32 {
            map.insert(IdentityValue {
                key: IdentityKey(i),
                data: "filler",
            });
        }
        assert_eq!(map.len(), 16);

        map.insert(IdentityValue {
            key: IdentityKey(16),
            data: "original",
        });
        assert!(map.remove(&IdentityKey(0)));
        assert_eq!(map.len(), 16);

        map.insert(IdentityValue {
            key: IdentityKey(16),
            data: "updated",
        });

        assert_eq!(map.len(), 16);
        assert_eq!(map.get(&IdentityKey(16)).unwrap().data, "updated");

        let count_16 = map.iter().filter(|v| v.key == IdentityKey(16)).count();
        assert_eq!(count_16, 1);
    }

    #[test]
    fn test_contains() {
        let mut map = TestMap::with_capacity(64);
        map.insert(TestValue { key: 42, data: "x" });
        assert!(map.contains(&42));
        assert!(!map.contains(&99));
    }

    #[test]
    fn test_get_or_insert() {
        let mut map = TestMap::with_capacity(64);

        let val = map.get_or_insert(TestValue {
            key: 1,
            data: "first",
        });
        assert_eq!(val.data, "first");

        let val = map.get_or_insert(TestValue {
            key: 1,
            data: "second",
        });
        assert_eq!(val.data, "first");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_grow() {
        let mut map = TestMap::with_capacity(16);
        for i in 0..100u32 {
            map.insert(TestValue {
                key: i,
                data: "val",
            });
        }
        assert_eq!(map.len(), 100);
        for i in 0..100u32 {
            assert!(map.get(&i).is_some(), "missing key {i} after grow");
        }
    }

    #[test]
    fn test_lazy_alloc() {
        let mut map = TestMap::new();
        assert_eq!(map.capacity(), 0);
        assert!(map.get(&1).is_none());
        assert!(!map.remove(&1));

        map.insert(TestValue {
            key: 1,
            data: "one",
        });
        assert!(map.capacity() >= GROUP_WIDTH);
        assert_eq!(map.get(&1).unwrap().data, "one");
    }

    #[test]
    fn test_pair_extract() {
        type PairMap = SwissTable<PairExtract<u64, String>>;
        let mut map = PairMap::with_capacity(32);
        map.insert((42, "hello".to_string()));
        map.insert((99, "world".to_string()));

        let val = map.get(&42).unwrap();
        assert_eq!(val.0, 42);
        assert_eq!(val.1, "hello");

        assert!(map.remove(&42));
        assert!(map.get(&42).is_none());
    }

    #[test]
    fn test_drop_with_heap_values() {
        type StringMap = SwissTable<PairExtract<u32, String>>;

        let mut map = StringMap::with_capacity(16);
        for i in 0..50u32 {
            map.insert((i, format!("value_{i}")));
        }
        drop(map);
    }
}
