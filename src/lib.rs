#![warn(clippy::undocumented_unsafe_blocks)]

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

mod group;
use crate::group::{BitMask, GROUP_WIDTH, Group};

pub mod hash_map;
pub mod hash_set;
pub use crate::hash_map::HashMap;
pub use crate::hash_set::HashSet;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::ptr::NonNull;

use std::alloc::Layout;
use std::fmt;
use std::hash::BuildHasher;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;

/// Key equivalence trait.
///
/// This trait defines the function used to compare the input value with the
/// map keys (or set values) during a lookup operation such as [`HashMap::get`]
/// or [`HashSet::contains`].
/// It is provided with a blanket implementation based on the
/// [`Borrow`](core::borrow::Borrow) trait.
///
/// # Correctness
///
/// Equivalent values must hash to the same value.
pub trait Equivalent<K: ?Sized> {
    /// Checks if this value is equivalent to the given key.
    ///
    /// Returns `true` if both values are equivalent, and `false` otherwise.
    fn equivalent(&self, key: &K) -> bool;
}

impl<Q: ?Sized, K: ?Sized> Equivalent<K> for Q
where
    Q: Eq,
    K: core::borrow::Borrow<Q>,
{
    #[inline(always)]
    fn equivalent(&self, key: &K) -> bool {
        self == key.borrow()
    }
}

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
        self.hash.rotate_left(26)
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
#[repr(transparent)]
pub struct AutoHasher {
    hash: u64,
}
const _: () = assert!(std::mem::size_of::<AutoHasher>() == 8);

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

    #[inline]
    fn fx_bytes(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in chunks.by_ref() {
            let word = u64::from_ne_bytes(chunk.try_into().unwrap());
            self.fx_mix(word);
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut buf = [0u8; 8];
            buf[..remainder.len()].copy_from_slice(remainder);
            self.fx_mix(u64::from_ne_bytes(buf));
        }
    }
}

impl std::hash::Hasher for AutoHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
    // Bytes uses also fxhash
    /// TODO: find good replace for bytes handle
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.fx_bytes(bytes);
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
pub struct PairExtract<K, V>(PhantomData<fn() -> (K, V)>);

impl<K: std::hash::Hash + PartialEq, V> KeyExtract for PairExtract<K, V> {
    type Key = K;
    type Value = (K, V);

    #[inline]
    fn extract(value: &(K, V)) -> &K {
        &value.0
    }
}

pub struct SelfExtract<T>(PhantomData<fn() -> T>);

impl<T: std::hash::Hash + PartialEq> KeyExtract for SelfExtract<T> {
    type Key = T;
    type Value = T;

    #[inline]
    fn extract(value: &T) -> &T {
        value
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

    /// Returns true if control byte indicates an occupied slot (bit 7 clear).
    #[inline(always)]
    pub(crate) const fn is_full(c: u8) -> bool {
        (c & EMPTY) == 0
    }

    /// Extract h2 from hash — top 7 bits (bits 57..63).
    /// These bits are independent from the lower bits used for positioning.
    #[inline(always)]
    pub(crate) const fn h2(hash: u64) -> u8 {
        (hash >> 57) as u8
    }
}

/// Prefetch data into L1 cache.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn prefetch_read(ptr: *const u8) {
    // SAFETY: _mm_prefetch is a hint only with no memory safety requirements; ptr may be invalid.
    unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0) };
}

/// Prefetch with non-temporal hint – data used once, discard quickly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn prefetch_nta(ptr: *const u8) {
    // SAFETY: _mm_prefetch is a hint only with no memory safety requirements; ptr may be invalid.
    unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_NTA) };
}

/// Prefetch data into L2+ cache.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn prefetch_read_l2(ptr: *const u8) {
    // SAFETY: _mm_prefetch is a hint only with no memory safety requirements; ptr may be invalid.
    unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T2) };
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn prefetch_nta(_ptr: *const u8) {}

// =============================================================================
// SwissTable
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TryReserveError {
    CapacityOverflow,
    AllocError { layout: Layout },
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TryReserveError::CapacityOverflow => f.write_str("capacity overflow"),
            TryReserveError::AllocError { layout } => {
                write!(f, "memory allocation of {} bytes failed", layout.size())
            }
        }
    }
}

impl std::error::Error for TryReserveError {}

#[cold]
#[inline(never)]
fn handle_reserve_error(e: TryReserveError) -> ! {
    match e {
        TryReserveError::CapacityOverflow => panic!("capacity overflow"),
        TryReserveError::AllocError { layout } => std::alloc::handle_alloc_error(layout),
    }
}

fn capacity_to_buckets(items: usize) -> Option<usize> {
    if items < 8 {
        return Some(MIN_BUCKETS);
    }
    (items.checked_mul(8)? / 7).checked_next_power_of_two()
}

const MIN_BUCKETS: usize = 8;

#[inline(always)]
unsafe fn fix_insert_slot(ctrl: *const u8, mask: usize, idx: usize) -> usize {
    if unlikely(mask < GROUP_WIDTH - 1) {
        // SAFETY: forwarded caller invariant.
        return unsafe { fix_insert_slot_small(ctrl, idx) };
    }
    idx
}

#[cold]
#[inline(never)]
unsafe fn fix_insert_slot_small(ctrl: *const u8, idx: usize) -> usize {
    // SAFETY: idx < cap and ctrl is a live table with cap + GROUP_WIDTH bytes (caller invariant).
    unsafe {
        if ctrl::is_full(*ctrl.add(idx)) {
            Group::load_aligned(ctrl).match_empty_or_deleted().lowest_set_bit()
        } else {
            idx
        }
    }
}

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
    pub(crate) slots: NonNull<u8>,
    /// Capacity mask: `cap - 1` when non-empty (cap is always a power of 2),
    /// `0` when empty (minimum non-empty cap is GROUP_WIDTH=16, so mask=0 is unambiguous).
    /// Stored instead of `cap` to avoid recomputing `cap - 1` on every probe.
    pub(crate) mask: usize,
    pub(crate) len: usize,
    /// Slots remaining before a grow is needed.
    /// Decremented on each new insertion; grow triggers when this reaches 0.
    pub(crate) growth_left: usize,
    pub(crate) hash_builder: S,
    pub(crate) alloc: A,
    pub(crate) _marker: PhantomData<fn() -> E::Value>,
}

/// Write a control byte and maintain the trailing mirror.
///
/// The first `GROUP_WIDTH` bytes are duplicated at `ctrl[cap..]` so that
/// unaligned SIMD loads near the end of the array see the correct wrapped
/// values instead of stale data.
///
/// Branchless: for `idx < GROUP_WIDTH` the second write hits the mirror zone
/// `ctrl[cap + idx]`; for `idx >= GROUP_WIDTH` it harmlessly rewrites
/// `ctrl[idx]` (the same byte).
#[inline(always)]
unsafe fn set_ctrl(ctrl: *mut u8, mask: usize, idx: usize, value: u8) {
    // SAFETY: idx < cap (caller invariant), ctrl array has cap + GROUP_WIDTH bytes allocated.
    // Mirror index: (idx.wrapping_sub(GROUP_WIDTH) & mask) + GROUP_WIDTH
    //   idx < 16 → wraps high, mask keeps low bits → cap + idx - 16 + 16 = cap + idx (mirror)
    //   idx >= 16 → (idx - 16) & mask + 16 = idx (harmless self-write)
    unsafe {
        let mirror = (idx.wrapping_sub(GROUP_WIDTH) & mask) + GROUP_WIDTH;
        *ctrl.add(idx) = value;
        *ctrl.add(mirror) = value;
    }
}

impl<E, S, A> PartialEq for SwissTable<E, S, A>
where
    E: KeyExtract,
    E::Key: PartialEq + Eq + std::hash::Hash,
    E::Value: PartialEq,
    S: BuildHasher,
    A: Allocator,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        self.iter().all(|v| {
            let k = E::extract(v);
            other.get(k).is_some_and(|ov| ov == v)
        })
    }
}

impl<E: KeyExtract, S: BuildHasher + Clone, A: Allocator + Clone> Clone for SwissTable<E, S, A>
where
    E::Value: Clone,
{
    fn clone(&self) -> Self {
        self.clone_table()
    }
}

/// Compute how many slots can be filled before growing.
/// Load factor ≈ 7/8 (87.5%), matching hashbrown.
/// `(cap / 8) * 7` compiles to shift + subtract - no division.
#[inline(always)]
fn bucket_mask_to_capacity(mask: usize) -> usize {
    if mask < 8 { mask } else { ((mask + 1) / 8) * 7 }
}

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
            mask: 0,
            len: 0,
            growth_left: 0,
            hash_builder,
            alloc: Global,
            _marker: PhantomData,
        }
    }

    /// Create a table with capacity and custom hash builder.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        let mut table = Self::with_hasher(hash_builder);
        table.reserve(capacity);
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
        table.reserve(capacity);
        table
    }
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> SwissTable<E, S, A> {
    /// Create an empty table with custom hasher and allocator.
    pub fn with_hasher_and_alloc(hash_builder: S, alloc: A) -> Self {
        Self {
            ctrl: NonNull::dangling(),
            slots: NonNull::dangling(),
            mask: 0,
            len: 0,
            growth_left: 0,
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
    #[inline(always)]
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
    fn slots_ptr(&self) -> *mut u8 {
        self.slots.as_ptr()
    }

    /// Helper: get a pointer to the `MaybeUninit` slot by index.
    #[inline(always)]
    fn slot_mut(&self, idx: usize) -> *mut MaybeUninit<E::Value> {
        // SAFETY:
        // - `idx` must be less than the current capacity (`self.mask + -1`).
        // - The memory block pointed to by `self.slots` must have been allocated
        //   for `cap` elements of type `MaybeUninit<E::Value>` (including correct
        //   size and alignment). This is guaranteed by `Self::alloc_table`.
        // - The caller must ensure that the returned pointer is not used after
        //   the table is deallocated or reallocated (e.g., after `grow`).
        // - The pointer may be written to or read from only when the corresponding
        //   control byte indicates a FULL slot (for reads) or when the slot is known
        //   to be uninitialized (for writes).
        unsafe {
            self.slots
                .as_ptr()
                .add(idx * std::mem::size_of::<E::Value>())
                .cast::<MaybeUninit<E::Value>>()
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn slot_ref(&self, idx: usize) -> *const MaybeUninit<E::Value> {
        self.slot_mut(idx).cast()
    }

    /// Lazy-init: allocate minimum capacity in-place (without moving the allocator).
    #[cold]
    #[inline(never)]
    fn ensure_capacity(&mut self) {
        debug_assert_eq!(self.mask, 0);
        let cap = MIN_BUCKETS;
        // SAFETY: alloc_table succeeded (unwrap), writing EMPTY to freshly allocated memory.
        unsafe {
            let (ctrl, slots) =
                Self::alloc_table(cap, &self.alloc).unwrap_or_else(|e| handle_reserve_error(e));
            self.ctrl = ctrl;
            self.slots = slots;
            let cp = self.ctrl_ptr();
            ptr::write_bytes(cp, ctrl::EMPTY, cap + GROUP_WIDTH);
            self.mask = cap - 1;
            self.growth_left = bucket_mask_to_capacity(cap - 1);
        }
    }

    /// Compute the combined layout for `[ctrl | padding | slots]`.
    /// Returns `(layout, slots_offset)`.
    #[inline(always)]
    fn table_layout(cap: usize) -> Option<(Layout, usize)> {
        let ctrl = Layout::from_size_align(cap.checked_add(GROUP_WIDTH)?, GROUP_WIDTH).ok()?;
        let slots = Layout::array::<MaybeUninit<E::Value>>(cap).ok()?;
        ctrl.extend(slots).ok()
    }

    /// Allocate a single block for ctrl + slots.
    /// Returns `(ctrl_ptr, slots_ptr)`.
    #[allow(clippy::type_complexity)]
    #[inline]
    unsafe fn alloc_table<AA: Allocator>(
        cap: usize,
        a: &AA,
    ) -> Result<(NonNull<u8>, NonNull<u8>), TryReserveError> {
        let (layout, slots_offset) =
            Self::table_layout(cap).ok_or(TryReserveError::CapacityOverflow)?;
        let ptr = do_alloc(a, layout).map_err(|()| TryReserveError::AllocError { layout })?;
        // SAFETY: slots_offset is within the allocation computed by Layout::extend.
        let slots = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(slots_offset)) };
        Ok((ptr, slots))
    }

    /// Free the combined ctrl + slots allocation.
    #[inline(always)]
    unsafe fn dealloc_table(ctrl: NonNull<u8>, cap: usize, alloc: &A) {
        let (layout, _) = Self::table_layout(cap).expect("layout was valid when allocated");
        // SAFETY: layout matches the one used in alloc_table, ctrl was allocated by this allocator.
        unsafe { alloc.deallocate(ctrl, layout) };
    }

    /// Initial probe position from lower bits of hash.
    /// `hash & mask` — fast modulo for powers of two (mask = cap - 1).
    #[inline(always)]
    fn probe_start(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Next group in probe chain: `(pos + GROUP_WIDTH) & mask`.
    #[inline(always)]
    fn next_group(&self, pos: usize) -> usize {
        (pos + GROUP_WIDTH) & self.mask
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
        if unlikely(self.mask == 0) {
            0
        } else {
            self.mask + 1
        }
    }

    /// Insert a value (or update existing by key).
    ///
    /// If the key already exists, the value is overwritten.
    /// Grows automatically when load factor is exceeded.
    pub fn insert(&mut self, value: E::Value) {
        if unlikely(self.mask == 0) {
            self.ensure_capacity();
            self.insert_inner(value);
            return;
        }
        self.insert_inner(value);
    }

    pub fn replace(&mut self, value: E::Value) -> Option<E::Value> {
        if unlikely(self.mask == 0) {
            self.ensure_capacity();
        }
        self.insert_inner(value)
    }

    /// Fast insert for rehash — no duplicate check, no deleted slots.
    /// Only valid on a freshly allocated table (all EMPTY, no DELETED).
    /// Returns a mutable reference to the inserted value.
    #[inline(always)]
    fn insert_assume_unique_return(&mut self, value: E::Value) -> &mut E::Value {
        let hash = self.make_hash(E::extract(&value));
        let h2 = ctrl::h2(hash);
        let mut pos = self.probe_start(hash);
        let ctrl = self.ctrl_ptr();
        let mask = self.mask;

        loop {
            // SAFETY: pos < cap, ctrl has cap + GROUP_WIDTH bytes, so pos..pos+GROUP_WIDTH is within allocation.
            let group = unsafe { Group::load(ctrl.add(pos)) };
            let empty = group.match_empty_or_deleted();
            if likely(empty.any()) {
                // SAFETY: idx < cap (masked), ctrl is the live table.
                let idx =
                    unsafe { fix_insert_slot(ctrl, mask, (pos + empty.lowest_set_bit()) & mask) };
                // SAFETY: idx < cap (masked), slot is valid uninitialized memory from alloc_slots.
                unsafe {
                    set_ctrl(ctrl, mask, idx, h2);
                    ptr::write(self.slot_mut(idx), MaybeUninit::new(value));
                }
                self.len += 1;
                self.growth_left -= 1;
                // SAFETY: slot initialized
                return unsafe { &mut *self.slot_mut(idx).cast::<E::Value>() };
            }
            pos = self.next_group(pos);
        }
    }

    // /// Inner insert with duplicate check.
    #[inline]
    fn insert_inner(&mut self, value: E::Value) -> Option<E::Value> {
        let key = E::extract(&value);
        let hash = self.make_hash(key);
        let h2 = ctrl::h2(hash);
        let cap_mask = self.mask;
        let mut pos = self.probe_start(hash);
        let mut insert_slot: usize = usize::MAX; // sentinel: no candidate yet
        let ctrl = self.ctrl_ptr();
        // // SAFETY: hint only with no memory touches
        unsafe {
            // Prefetch the heap-allocated memory region to resolve potential TLB and
            // cache misses. This is intended to overlap with execution of calculating the
            // hash for a key.
            prefetch_nta(ctrl);
        }

        loop {
            // SAFETY: pos < cap, ctrl has cap + GROUP_WIDTH bytes, so pos..pos+GROUP_WIDTH is within allocation.
            let group = unsafe { Group::load(ctrl.add(pos)) };

            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                // SAFETY: idx < cap (masked), slot is initialized (ctrl byte matched h2, so slot is FULL).
                let slot_ptr = self.slot_mut(idx);
                let val_ptr = slot_ptr.cast::<E::Value>();
                // SAFETY: val_ptr points to initialized Value (slot is FULL, confirmed by h2 match).
                let slot_ref = unsafe { &*val_ptr };
                if unlikely(E::extract(slot_ref) == key) {
                    // SAFETY: idx is a valid FULL slot, the old value is moved out to the caller.
                    return Some(unsafe { ptr::replace(val_ptr, value) });
                }
                mask.remove_lowest_bit();
            }

            // Check for EMPTY — ends the probe chain.
            // On the common path (no tombstones) this is the only check needed,
            // saving the match_empty_or_deleted SIMD op.
            let empty = group.match_empty();
            if likely(empty.any()) {
                let idx = if unlikely(insert_slot != usize::MAX) {
                    insert_slot
                } else {
                    // SAFETY: idx < cap (masked), ctrl is the live table.
                    unsafe {
                        fix_insert_slot(ctrl, cap_mask, (pos + empty.lowest_set_bit()) & cap_mask)
                    }
                };
                // SAFETY: idx < cap (masked).
                let was_empty = (likely(insert_slot == usize::MAX)
                    && likely(cap_mask >= GROUP_WIDTH - 1))
                    || unsafe { *ctrl.add(idx) } == ctrl::EMPTY;

                if unlikely(was_empty && self.growth_left == 0) {
                    self.grow_and_insert(value);
                    return None;
                }

                // SAFETY: idx < cap (masked). Writing to empty/deleted slot.
                unsafe {
                    set_ctrl(ctrl, cap_mask, idx, h2);
                    ptr::write(self.slot_mut(idx), MaybeUninit::new(value));
                }
                self.len += 1;
                self.growth_left -= was_empty as usize;
                return None;
            }

            // No EMPTY in this group - probe chain continues.
            // Track first DELETED slot as potential insert position.
            if unlikely(insert_slot == usize::MAX) {
                let empty_del = group.match_empty_or_deleted();
                if likely(empty_del.any()) {
                    insert_slot = (pos + empty_del.lowest_set_bit()) & cap_mask;
                }
            }

            pos = self.next_group(pos);
        }
    }

    #[inline]
    fn get_inner<Q>(&self, key: &Q) -> Option<*const E::Value>
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        if unlikely(self.mask == 0) {
            return None;
        }
        let idx = self.find_hashed(self.hash_builder.hash_one(key), key)?;
        Some(self.slot_mut(idx).cast::<E::Value>().cast_const())
    }

    #[inline(always)]
    fn find_hashed<Q>(&self, hash: u64, key: &Q) -> Option<usize>
    where
        Q: Equivalent<E::Key> + ?Sized,
    {
        let h2 = ctrl::h2(hash);
        let cap_mask = self.mask;
        let mut pos = self.probe_start(hash);
        let ctrl = self.ctrl_ptr();
        // SAFETY: hint only with no memory touches
        unsafe {
            // Prefetch the heap-allocated memory region to resolve potential TLB and
            // cache misses. This is intended to overlap with execution of calculating the
            // hash for a key.

            prefetch_nta(ctrl);
        }

        loop {
            // SAFETY: pos < cap, ctrl has cap + GROUP_WIDTH bytes, so pos..pos+GROUP_WIDTH is within allocation.
            let group = unsafe { Group::load(ctrl.add(pos)) };
            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                // SAFETY: idx < cap (masked), slot is FULL (h2 matched).
                let val_ptr = self.slot_mut(idx).cast::<E::Value>();
                // SAFETY: val_ptr points to initialized Value (FULL slot).
                let slot_ref = unsafe { &*val_ptr };
                if likely(key.equivalent(E::extract(slot_ref))) {
                    return Some(idx);
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
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&E::Value>
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        // SAFETY: get_inner returns pointer to initialized Value in a FULL slot; lifetime tied to &self.

        unsafe { self.get_inner(key).map(|p| &*p) }
    }

    /// Get a mutable reference to the value for the given key.
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut E::Value>
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        // SAFETY: get_inner returns pointer to initialized Value in a FULL slot; &mut self ensures exclusive access.
        unsafe { self.get_inner(key).map(|p| &mut *p.cast_mut()) }
    }

    /// Remove by key. Uses DELETED tombstone to preserve probe chains.
    fn remove_inner<Q>(&mut self, key: &Q) -> bool
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        if unlikely(self.mask == 0) {
            return false;
        }
        let Some(idx) = self.find_hashed(self.hash_builder.hash_one(key), key) else {
            return false;
        };
        // SAFETY: find_hashed returned a FULL slot.
        unsafe {
            if core::mem::needs_drop::<E::Value>() {
                ptr::drop_in_place(self.slot_mut(idx).cast::<E::Value>());
            }
            self.erase_no_drop(idx);
        }
        true
    }

    #[inline(always)]
    unsafe fn erase_no_drop(&mut self, idx: usize) {
        // SAFETY: idx < cap (caller invariant).
        unsafe { set_ctrl(self.ctrl_ptr(), self.mask, idx, ctrl::DELETED) };
        self.len -= 1;
    }

    #[inline(always)]
    unsafe fn take_at(&mut self, idx: usize) -> E::Value {
        // SAFETY: idx is a FULL slot (caller invariant), marked DELETED so it is read only once.
        unsafe {
            self.erase_no_drop(idx);
            ptr::read(self.slot_mut(idx).cast::<E::Value>())
        }
    }

    pub fn take<Q>(&mut self, key: &Q) -> Option<E::Value>
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        if unlikely(self.mask == 0) {
            return None;
        }
        let idx = self.find_hashed(self.hash_builder.hash_one(key), key)?;
        // SAFETY: find_hashed returned a FULL slot.
        Some(unsafe { self.take_at(idx) })
    }

    pub fn get_or_insert_with<Q>(&mut self, key: &Q, f: impl FnOnce() -> E::Value) -> &mut E::Value
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        if unlikely(self.mask == 0) {
            self.ensure_capacity();
        }
        let hash = self.hash_builder.hash_one(key);
        if let Some(idx) = self.find_hashed(hash, key) {
            // SAFETY: FULL slot, &mut self gives exclusive access.
            return unsafe { &mut *self.slot_mut(idx).cast::<E::Value>() };
        }
        let value = f();
        assert!(
            key.equivalent(E::extract(&value)),
            "get_or_insert_with: new value key is not equivalent to the lookup key"
        );
        // SAFETY: insert_new_hashed returns a pointer to the value it just wrote.
        unsafe { &mut *self.insert_new_hashed(hash, value) }
    }

    pub(crate) fn insert_new_hashed(&mut self, hash: u64, value: E::Value) -> *mut E::Value {
        if unlikely(self.mask == 0) {
            self.ensure_capacity();
        }
        let mut idx = self.find_insert_slot(hash);
        // SAFETY: idx < cap (masked).
        let mut was_empty = unsafe { *self.ctrl_ptr().add(idx) } == ctrl::EMPTY;
        if unlikely(was_empty && self.growth_left == 0) {
            self.grow();
            idx = self.find_insert_slot(hash);
            was_empty = true;
        }
        let slot = self.slot_mut(idx);
        // SAFETY: idx < cap, slot is EMPTY or DELETED so it holds no live value.
        unsafe {
            set_ctrl(self.ctrl_ptr(), self.mask, idx, ctrl::h2(hash));
            ptr::write(slot, MaybeUninit::new(value));
        }
        self.len += 1;
        self.growth_left -= was_empty as usize;
        slot.cast::<E::Value>()
    }

    pub fn reserve(&mut self, additional: usize) {
        if unlikely(additional > self.growth_left)
            && let Err(e) = self.reserve_rehash(additional)
        {
            handle_reserve_error(e);
        }
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        if additional > self.growth_left {
            self.reserve_rehash(additional)
        } else {
            Ok(())
        }
    }

    pub fn shrink_to_fit(&mut self) {
        if self.mask == 0 {
            return;
        }
        if self.len == 0 {
            // SAFETY: no FULL slots remain, the table was allocated with cap = mask + 1.
            unsafe { Self::dealloc_table(self.ctrl, self.mask + 1, &self.alloc) };
            self.ctrl = NonNull::dangling();
            self.slots = NonNull::dangling();
            self.mask = 0;
            self.growth_left = 0;
            return;
        }
        if capacity_to_buckets(self.len).is_some_and(|b| b < self.mask + 1)
            && let Err(e) = self.resize(self.len)
        {
            handle_reserve_error(e);
        }
    }

    pub fn retain(&mut self, mut f: impl FnMut(&mut E::Value) -> bool) {
        for value in self.extract_if(|v| !f(v)) {
            drop(value);
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, E, S, A> {
        IterMut {
            raw: RawIter::new(
                self.ctrl_ptr(),
                self.slots_ptr().cast(),
                self.mask,
                self.len,
            ),
            _table: PhantomData,
            _marker: PhantomData,
        }
    }

    pub fn drain(&mut self) -> Drain<'_, E, S, A> {
        Drain {
            raw: RawIter::new(
                self.ctrl_ptr(),
                self.slots_ptr().cast(),
                self.mask,
                self.len,
            ),
            table: self,
        }
    }

    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, E, F, S, A>
    where
        F: FnMut(&mut E::Value) -> bool,
    {
        ExtractIf {
            raw: RawIter::new(
                self.ctrl_ptr(),
                self.slots_ptr().cast(),
                self.mask,
                self.len,
            ),
            table: self,
            pred,
        }
    }

    /// Remove an element by key. Returns `true` if the key was found.
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        self.remove_inner(key)
    }

    /// Returns true if the key exists.
    #[inline]
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: std::hash::Hash + Equivalent<E::Key> + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Get or insert: returns `&mut` to existing value if key found,
    /// otherwise inserts `value` and returns `&mut` to the new entry.
    #[inline]
    pub fn get_or_insert(&mut self, value: E::Value) -> &mut E::Value {
        if unlikely(self.mask == 0) {
            self.ensure_capacity();
            return self.get_or_insert_inner(value);
        }
        self.get_or_insert_inner(value)
    }

    #[inline]
    fn get_or_insert_inner(&mut self, value: E::Value) -> &mut E::Value {
        let hash = self.make_hash(E::extract(&value));
        let h2 = ctrl::h2(hash);
        let cap_mask = self.mask;
        let mut pos = self.probe_start(hash);
        let mut insert_slot: usize = usize::MAX;
        let ctrl = self.ctrl_ptr();
        // SAFETY: hint only with no memory touches
        unsafe {
            prefetch_nta(ctrl);
        }

        loop {
            // SAFETY: pos < cap, ctrl has cap + GROUP_WIDTH bytes, so pos..pos+GROUP_WIDTH is within allocation.
            let group = unsafe { Group::load(ctrl.add(pos)) };
            let mut mask = group.match_byte(h2);
            while mask.any() {
                let bit = mask.lowest_set_bit();
                let idx = (pos + bit) & cap_mask;
                // SAFETY: idx < cap (masked), slot is FULL (h2 matched).
                let val_ptr = self.slot_mut(idx).cast::<E::Value>();
                // SAFETY: FULL slot, &mut self ensures exclusive access.
                let slot_ref = unsafe { &mut *val_ptr };
                if E::extract(slot_ref) == E::extract(&value) {
                    return slot_ref;
                }
                mask.remove_lowest_bit();
            }

            let empty = group.match_empty();
            if likely(empty.any()) {
                let idx = if unlikely(insert_slot != usize::MAX) {
                    insert_slot
                } else {
                    // SAFETY: idx < cap (masked), ctrl is the live table.
                    unsafe {
                        fix_insert_slot(ctrl, cap_mask, (pos + empty.lowest_set_bit()) & cap_mask)
                    }
                };
                // SAFETY: idx < cap (masked).
                let was_empty = (likely(insert_slot == usize::MAX)
                    && likely(cap_mask >= GROUP_WIDTH - 1))
                    || unsafe { *ctrl.add(idx) } == ctrl::EMPTY;

                if unlikely(was_empty && self.growth_left == 0) {
                    return self.grow_and_insert(value);
                }

                let slot = self.slot_mut(idx);
                // SAFETY: idx < cap (masked). Writing to empty/deleted slot.
                unsafe {
                    set_ctrl(ctrl, cap_mask, idx, h2);
                    ptr::write(slot, MaybeUninit::new(value));
                }
                self.len += 1;
                self.growth_left -= was_empty as usize;
                // SAFETY: just wrote value at idx, it is initialized.
                return unsafe { &mut *slot.cast::<E::Value>() };
            }

            if unlikely(insert_slot == usize::MAX) {
                let empty_del = group.match_empty_or_deleted();
                if empty_del.any() {
                    insert_slot = (pos + empty_del.lowest_set_bit()) & cap_mask;
                }
            }

            pos = self.next_group(pos);
        }
    }

    /// Drop all values and mark all slots as EMPTY.
    #[inline]
    pub fn clear(&mut self) {
        if unlikely(self.mask == 0) {
            return;
        }
        let ctrl = self.ctrl_ptr();

        let cap = self.mask + 1;
        // SAFETY: mask > 0 so cap > 0, dropping FULL slots via SIMD group scan, then resetting ctrl.
        unsafe {
            if core::mem::needs_drop::<E::Value>() {
                let mut group_pos = 0;
                while group_pos < cap {
                    let mut full = Group::load_aligned(ctrl.add(group_pos)).match_full();
                    while full.any() {
                        let bit = full.lowest_set_bit();
                        if core::mem::needs_drop::<E::Value>() {
                            ptr::drop_in_place(self.slot_mut(group_pos + bit).cast::<E::Value>());
                        }

                        full.remove_lowest_bit();
                    }
                    group_pos += GROUP_WIDTH;
                }
            }
            ptr::write_bytes(ctrl, ctrl::EMPTY, cap + GROUP_WIDTH);
        }
        self.len = 0;
        self.growth_left = bucket_mask_to_capacity(self.mask);
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
        if unlikely(self.mask == 0) {
            return Self::with_hasher_and_alloc(self.hash_builder.clone(), self.alloc.clone());
        }
        let cap = self.mask + 1;
        // SAFETY: freshly allocated table via alloc_table, source table valid.
        unsafe {
            let (ctrl, slots) =
                Self::alloc_table(cap, &self.alloc).unwrap_or_else(|e| handle_reserve_error(e));

            let cp = ctrl.as_ptr();
            let sp = slots.as_ptr();

            // Control bytes are POD — plain memcpy
            ptr::copy_nonoverlapping(self.ctrl_ptr(), cp, cap + GROUP_WIDTH);

            if !std::mem::needs_drop::<E::Value>() {
                // Copy type — bulk memcpy of entire slots array.
                // Unoccupied slots contain garbage MaybeUninit, which is fine
                // because we never read them without checking ctrl first.
                ptr::copy_nonoverlapping(
                    self.slots_ptr(),
                    sp,
                    cap * std::mem::size_of::<E::Value>(),
                );
            } else {
                let mut raw = RawIter::new(
                    self.ctrl_ptr(),
                    self.slots_ptr().cast::<E::Value>(),
                    self.mask,
                    self.len,
                );
                while let Some((i, src)) = raw.next() {
                    let dst = sp.cast::<E::Value>().add(i);
                    ptr::write(dst, (*src).clone());
                }
            }

            Self {
                ctrl,
                slots,
                mask: self.mask,
                len: self.len,
                growth_left: self.growth_left,
                hash_builder: self.hash_builder.clone(),
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            }
        }
    }

    /// Iterator over all values. Uses SIMD group scan to skip empty regions.
    pub fn iter(&self) -> Iter<'_, E, S, A> {
        Iter {
            _table: PhantomData,
            raw: RawIter::new(
                self.ctrl_ptr(),
                self.slots_ptr().cast(),
                self.mask,
                self.len,
            ),
        }
    }

    /// Prefetch ctrl + slots for a key into CPU cache.
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch(&self, key: &E::Key) {
        if self.mask == 0 {
            return;
        }
        let hash = self.make_hash(key);
        let pos = self.probe_start(hash);
        // SAFETY: prefetch is a hint; pos derived from hash & mask so within ctrl/slots bounds.
        unsafe {
            prefetch_read(self.ctrl_ptr().add(pos));
            prefetch_read(self.slots_ptr().add(pos).cast());
            let next_pos = self.next_group(pos);
            prefetch_read_l2(self.ctrl_ptr().add(next_pos));
        }
    }

    /// Prefetch (no-op on non-x86_64).
    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch(&self, _key: &E::Key) {}

    #[cold]
    #[inline(never)]
    fn grow_and_insert(&mut self, value: E::Value) -> &mut E::Value {
        self.grow();
        self.insert_assume_unique_return(value)
    }

    #[inline(never)]
    fn grow(&mut self) {
        if let Err(e) = self.reserve_rehash(1) {
            handle_reserve_error(e);
        }
    }

    #[cold]
    #[inline(never)]
    fn reserve_rehash(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let new_items = self
            .len
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;
        let full_cap = bucket_mask_to_capacity(self.mask);
        if self.mask != 0 && new_items <= full_cap / 2 {
            self.rehash_in_place();
            return Ok(());
        }
        self.resize(new_items.max(full_cap + 1))
    }

    fn resize(&mut self, capacity: usize) -> Result<(), TryReserveError> {
        let new_cap = capacity_to_buckets(capacity).ok_or(TryReserveError::CapacityOverflow)?;
        let old_cap = self.capacity();
        let size = std::mem::size_of::<E::Value>();

        // SAFETY: the new table is freshly allocated and filled with EMPTY, every FULL old slot
        // is copied once into an EMPTY new slot, self switches to the new table only after
        // all moves, so a panicking hasher leaves the old table intact.
        unsafe {
            let (new_ctrl, new_slots) = Self::alloc_table(new_cap, &self.alloc)?;
            let ncp = new_ctrl.as_ptr();
            let nsp = new_slots.as_ptr();
            let new_mask = new_cap - 1;
            ptr::write_bytes(ncp, ctrl::EMPTY, new_cap + GROUP_WIDTH);

            let mut raw = RawIter::new(
                self.ctrl_ptr(),
                self.slots_ptr().cast::<E::Value>(),
                self.mask,
                self.len,
            );
            while let Some((_, src)) = raw.next() {
                let hash = self.hash_builder.hash_one(E::extract(&*src));
                let mut pos = (hash as usize) & new_mask;
                let idx = loop {
                    let empty = Group::load(ncp.add(pos)).match_empty_or_deleted();
                    if likely(empty.any()) {
                        break fix_insert_slot(
                            ncp,
                            new_mask,
                            (pos + empty.lowest_set_bit()) & new_mask,
                        );
                    }
                    pos = (pos + GROUP_WIDTH) & new_mask;
                };
                set_ctrl(ncp, new_mask, idx, ctrl::h2(hash));
                ptr::copy_nonoverlapping(src.cast::<u8>(), nsp.add(idx * size), size);
            }

            if old_cap > 0 {
                Self::dealloc_table(self.ctrl, old_cap, &self.alloc);
            }
            self.ctrl = new_ctrl;
            self.slots = new_slots;
            self.mask = new_mask;
            self.growth_left = bucket_mask_to_capacity(new_mask) - self.len;
        }
        Ok(())
    }

    #[inline(always)]
    fn find_insert_slot(&self, hash: u64) -> usize {
        let ctrl = self.ctrl_ptr();
        let mut pos = self.probe_start(hash);
        loop {
            // SAFETY: pos < cap, ctrl has cap + GROUP_WIDTH bytes.
            let m = unsafe { Group::load(ctrl.add(pos)) }.match_empty_or_deleted();
            if likely(m.any()) {
                // SAFETY: idx < cap (masked), ctrl is the live table.
                return unsafe {
                    fix_insert_slot(ctrl, self.mask, (pos + m.lowest_set_bit()) & self.mask)
                };
            }
            pos = self.next_group(pos);
        }
    }

    #[inline(always)]
    fn is_in_same_group(&self, a: usize, b: usize, hash: u64) -> bool {
        let start = self.probe_start(hash);
        let ga = (a.wrapping_sub(start) & self.mask) / GROUP_WIDTH;
        let gb = (b.wrapping_sub(start) & self.mask) / GROUP_WIDTH;
        ga == gb
    }

    #[cold]
    #[inline(never)]
    fn rehash_in_place(&mut self) {
        let cap = self.mask + 1;
        let size = std::mem::size_of::<E::Value>();
        let guard = RehashGuard(self);
        let table = &mut *guard.0;
        let ctrl = table.ctrl_ptr();
        let mask = table.mask;
        // SAFETY: all indexes are < cap, ctrl has cap + GROUP_WIDTH bytes, slots marked
        // DELETED below hold live values until moved or marked FULL.
        unsafe {
            for i in 0..cap {
                let c = ctrl.add(i);
                *c = if ctrl::is_full(*c) {
                    ctrl::DELETED
                } else {
                    ctrl::EMPTY
                };
            }
            ptr::copy_nonoverlapping(ctrl, ctrl.add(cap.max(GROUP_WIDTH)), cap.min(GROUP_WIDTH));

            let slots = table.slots_ptr();
            'outer: for i in 0..cap {
                if *ctrl.add(i) != ctrl::DELETED {
                    continue;
                }
                let src = slots.add(i * size);
                loop {
                    let hash = table.make_hash(E::extract(&*src.cast::<E::Value>()));
                    let h2 = ctrl::h2(hash);
                    let new_i = table.find_insert_slot(hash);

                    if likely(table.is_in_same_group(i, new_i, hash)) {
                        set_ctrl(ctrl, mask, i, h2);
                        continue 'outer;
                    }

                    let prev = *ctrl.add(new_i);
                    set_ctrl(ctrl, mask, new_i, h2);
                    let dst = slots.add(new_i * size);
                    if prev == ctrl::EMPTY {
                        set_ctrl(ctrl, mask, i, ctrl::EMPTY);
                        ptr::copy_nonoverlapping(src, dst, size);
                        continue 'outer;
                    }
                    debug_assert_eq!(prev, ctrl::DELETED);
                    ptr::swap_nonoverlapping(src, dst, size);
                }
            }
        }
        table.growth_left = bucket_mask_to_capacity(mask) - table.len;
    }
}

struct RehashGuard<'a, E: KeyExtract, S: BuildHasher, A: Allocator>(&'a mut SwissTable<E, S, A>);

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Drop for RehashGuard<'_, E, S, A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        let t = &mut *self.0;
        let ctrl = t.ctrl_ptr();
        for i in 0..=t.mask {
            // SAFETY: i < cap, a DELETED slot during rehash still holds a live value.
            unsafe {
                if *ctrl.add(i) == ctrl::DELETED {
                    set_ctrl(ctrl, t.mask, i, ctrl::EMPTY);
                    ptr::drop_in_place(t.slot_mut(i).cast::<E::Value>());
                    t.len -= 1;
                }
            }
        }
        t.growth_left = bucket_mask_to_capacity(t.mask) - t.len;
    }
}

// =============================================================================
// Drop & Iterator
// =============================================================================

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Drop for SwissTable<E, S, A> {
    fn drop(&mut self) {
        if unlikely(self.mask == 0) {
            return;
        }
        let ctrl = self.ctrl_ptr();
        let cap = self.mask + 1;
        // SAFETY: mask > 0 so cap > 0, dropping FULL slots via SIMD group scan, then deallocating.
        unsafe {
            if std::mem::needs_drop::<E::Value>() {
                let mut raw = RawIter::new(
                    ctrl,
                    self.slots_ptr().cast::<E::Value>(),
                    self.mask,
                    self.len,
                );
                while let Some((_, p)) = raw.next() {
                    ptr::drop_in_place(p);
                }
            }
            Self::dealloc_table(self.ctrl, cap, &self.alloc);
        }
    }
}

struct RawIter<T> {
    ctrl: *const u8,
    data: *mut T,
    group_pos: usize,
    bitmask: BitMask,
    items: usize,
}

// SAFETY: RawIter only holds positions inside a table, the iterator that wraps it carries the
// borrow or ownership of that table and so decides Send and Sync.
unsafe impl<T> Send for RawIter<T> {}
// SAFETY: see Send above.
unsafe impl<T> Sync for RawIter<T> {}

impl<T> RawIter<T> {
    #[inline(always)]
    fn new(ctrl: *const u8, data: *mut T, mask: usize, items: usize) -> Self {
        let bitmask = if likely(mask > 0) {
            // SAFETY: mask > 0 so the ctrl array is allocated with cap + GROUP_WIDTH bytes.
            unsafe { Group::load_aligned(ctrl) }.match_full()
        } else {
            BitMask::NONE
        };
        RawIter {
            ctrl,
            data,
            group_pos: 0,
            bitmask,
            items,
        }
    }

    #[inline(always)]
    unsafe fn next(&mut self) -> Option<(usize, *mut T)> {
        if self.items == 0 {
            return None;
        }
        loop {
            if likely(self.bitmask.any()) {
                let bit = self.bitmask.lowest_set_bit();
                self.bitmask.remove_lowest_bit();
                self.items -= 1;
                // SAFETY: bit < GROUP_WIDTH and the group lies inside the table.
                return Some((self.group_pos + bit, unsafe { self.data.add(bit) }));
            }
            // SAFETY: items > 0 means a FULL slot is still ahead, so the next group is
            // inside the ctrl and slot arrays (caller keeps the table alive and unchanged).
            unsafe {
                self.ctrl = self.ctrl.add(GROUP_WIDTH);
                self.data = self.data.add(GROUP_WIDTH);
                self.bitmask = Group::load_aligned(self.ctrl).match_full();
            }
            self.group_pos += GROUP_WIDTH;
        }
    }
}

pub struct Iter<'a, E: KeyExtract, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    _table: PhantomData<&'a SwissTable<E, S, A>>,
    raw: RawIter<E::Value>,
}

impl<'a, E: KeyExtract, S: BuildHasher, A: Allocator> Iterator for Iter<'a, E, S, A> {
    type Item = &'a E::Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: RawIter yields FULL slots of the borrowed table.
        unsafe { self.raw.next().map(|(_, p)| &*p) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.raw.items, Some(self.raw.items))
    }
}

pub struct IterMut<'a, E: KeyExtract, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    _table: PhantomData<&'a SwissTable<E, S, A>>,
    raw: RawIter<E::Value>,
    _marker: PhantomData<&'a mut E::Value>,
}

impl<'a, E: KeyExtract, S: BuildHasher, A: Allocator> Iterator for IterMut<'a, E, S, A> {
    type Item = &'a mut E::Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: RawIter yields each FULL slot once, the table is mutably borrowed for 'a.
        unsafe { self.raw.next().map(|(_, p)| &mut *p) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.raw.items, Some(self.raw.items))
    }
}

pub struct IntoIter<E: KeyExtract, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    table: SwissTable<E, S, A>,
    raw: RawIter<E::Value>,
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Iterator for IntoIter<E, S, A> {
    type Item = E::Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: RawIter yields FULL slots, take_at marks them DELETED so Drop skips them.
        unsafe {
            let (idx, _) = self.raw.next()?;
            Some(self.table.take_at(idx))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.raw.items, Some(self.raw.items))
    }
}

pub struct Drain<'a, E: KeyExtract, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    table: &'a mut SwissTable<E, S, A>,
    raw: RawIter<E::Value>,
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Iterator for Drain<'_, E, S, A> {
    type Item = E::Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: RawIter yields FULL slots, take_at marks them DELETED.
        unsafe {
            let (idx, _) = self.raw.next()?;
            Some(self.table.take_at(idx))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.raw.items, Some(self.raw.items))
    }
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Drop for Drain<'_, E, S, A> {
    fn drop(&mut self) {
        self.table.clear();
    }
}

pub struct ExtractIf<
    'a,
    E: KeyExtract,
    F,
    S: BuildHasher = DefaultHashBuilder,
    A: Allocator = Global,
> {
    table: &'a mut SwissTable<E, S, A>,
    raw: RawIter<E::Value>,
    pred: F,
}

impl<E, F, S, A> Iterator for ExtractIf<'_, E, F, S, A>
where
    E: KeyExtract,
    F: FnMut(&mut E::Value) -> bool,
    S: BuildHasher,
    A: Allocator,
{
    type Item = E::Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: RawIter yields FULL slots, take_at marks them DELETED.
        unsafe {
            loop {
                let (idx, p) = self.raw.next()?;
                if (self.pred)(&mut *p) {
                    return Some(self.table.take_at(idx));
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.raw.items))
    }
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> ExactSizeIterator for Iter<'_, E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> FusedIterator for Iter<'_, E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> ExactSizeIterator for IterMut<'_, E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> FusedIterator for IterMut<'_, E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> ExactSizeIterator for IntoIter<E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> FusedIterator for IntoIter<E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> ExactSizeIterator for Drain<'_, E, S, A> {}
impl<E: KeyExtract, S: BuildHasher, A: Allocator> FusedIterator for Drain<'_, E, S, A> {}
impl<E, F, S, A> FusedIterator for ExtractIf<'_, E, F, S, A>
where
    E: KeyExtract,
    F: FnMut(&mut E::Value) -> bool,
    S: BuildHasher,
    A: Allocator,
{
}

// SAFETY: the table owns its values like a Vec, so it can move to another thread when they can.
unsafe impl<E: KeyExtract, S: BuildHasher + Send, A: Allocator + Send> Send for SwissTable<E, S, A> where
    E::Value: Send
{
}

// SAFETY: &SwissTable only gives out &E::Value, &S and &A.
unsafe impl<E: KeyExtract, S: BuildHasher + Sync, A: Allocator + Sync> Sync for SwissTable<E, S, A> where
    E::Value: Sync
{
}

// SAFETY: IterMut hands out &mut E::Value, like std's IterMut it is Send when the values are.
unsafe impl<E: KeyExtract, S: BuildHasher, A: Allocator> Send for IterMut<'_, E, S, A> where
    E::Value: Send
{
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> IntoIterator for SwissTable<E, S, A> {
    type Item = E::Value;
    type IntoIter = IntoIter<E, S, A>;

    fn into_iter(self) -> IntoIter<E, S, A> {
        IntoIter {
            raw: RawIter::new(
                self.ctrl_ptr(),
                self.slots_ptr().cast(),
                self.mask,
                self.len,
            ),
            table: self,
        }
    }
}

impl<'a, E: KeyExtract, S: BuildHasher, A: Allocator> IntoIterator for &'a SwissTable<E, S, A> {
    type Item = &'a E::Value;
    type IntoIter = Iter<'a, E, S, A>;

    fn into_iter(self) -> Iter<'a, E, S, A> {
        self.iter()
    }
}

impl<'a, E: KeyExtract, S: BuildHasher, A: Allocator> IntoIterator for &'a mut SwissTable<E, S, A> {
    type Item = &'a mut E::Value;
    type IntoIter = IterMut<'a, E, S, A>;

    fn into_iter(self) -> IterMut<'a, E, S, A> {
        self.iter_mut()
    }
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> Extend<E::Value> for SwissTable<E, S, A> {
    fn extend<I: IntoIterator<Item = E::Value>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let hint = iter.size_hint().0;
        self.reserve(if self.is_empty() {
            hint
        } else {
            hint.div_ceil(2)
        });
        for value in iter {
            self.insert(value);
        }
    }
}

impl<E: KeyExtract, S: BuildHasher + Default, A: Allocator + Default> FromIterator<E::Value>
    for SwissTable<E, S, A>
{
    fn from_iter<I: IntoIterator<Item = E::Value>>(iter: I) -> Self {
        let mut table = Self::new_in(A::default());
        table.extend(iter);
        table
    }
}

impl<E: KeyExtract, S: BuildHasher, A: Allocator> fmt::Debug for SwissTable<E, S, A>
where
    E::Value: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
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
        assert!(map.capacity() >= 64);
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
        assert_eq!(map.capacity(), MIN_BUCKETS);
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

    #[test]
    fn test_covariance() {
        fn build<'a>() -> SwissTable<PairExtract<&'a str, u8>> {
            let mut map = SwissTable::<PairExtract<&'a str, u8>>::new();

            map.insert(("one", 1));
            map.insert(("two", 2));

            map
        }

        let map = build();

        let key: &str = "one";
        let value = map.get(&key);

        assert_eq!(value.map(|v| v.1), Some(1));
    }

    #[test]
    fn test_covariance_usage() {
        fn takes<'a>(map: &SwissTable<PairExtract<&'a str, u8>>) {
            let key: &'a str = "one";
            let _ = map.get(&key);
        }

        let map = {
            let mut m = SwissTable::<PairExtract<&'static str, u8>>::new();
            m.insert(("one", 1));
            m
        };

        takes(&map);
    }

    #[test]
    #[doc = "Regression test for issue #1"]
    fn test_invariants() {
        fn number_names() -> SwissTable<PairExtract<&'static str, u8>> {
            let mut map = SwissTable::new();
            map.insert(("one", 1));
            map.insert(("two", 2));
            map
        }

        let names = number_names();
        let arg = String::from("one");

        let value = names.get(arg.as_str());

        assert_eq!(value.map(|v| v.1), Some(1));
    }

    #[test]
    #[doc = "Regression test for Table::retrieve"]
    fn test_table_trait_retrieve() {
        use super::*;

        trait Table<V> {
            fn retrieve(&self, key: &str) -> Option<&V>;
        }

        impl<V> Table<V> for SwissTable<PairExtract<&str, V>> {
            fn retrieve(&self, key: &str) -> Option<&V> {
                self.get(key).map(|(_, v)| v)
            }
        }

        let mut names = SwissTable::<PairExtract<&str, u8>>::new();
        names.insert(("one", 1));
        names.insert(("two", 2));

        // owned -> borrowed coercion
        let arg = String::from("one");

        let result = names.retrieve(&arg);

        assert_eq!(result, Some(&1));
    }

    #[test]
    fn test_tombstone_churn_stays_bounded() {
        let mut map = TestMap::with_capacity(16);
        for round in 0..10_000u32 {
            let k = round & 0xF;
            map.insert(TestValue { key: k, data: "v" });
            assert!(map.get(&k).is_some());
            assert!(map.remove(&k));
            assert!(map.get(&k).is_none());
        }
        assert_eq!(map.len(), 0);
        assert!(map.capacity() <= 64, "capacity {}", map.capacity());
    }

    #[test]
    fn test_rehash_in_place_matches_model() {
        let mut map = SwissTable::<PairExtract<u32, String>>::new();
        let mut model = std::collections::HashMap::new();
        let mut x: u64 = 0xF00D_CAFE;
        for op in 0..300_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let k = (x % 4001) as u32;
            if x % 100 < 60 {
                assert_eq!(
                    map.remove(&k),
                    model.remove(&k).is_some(),
                    "op {op} key {k}"
                );
            } else {
                let v = format!("{k}-{op}");
                map.insert((k, v.clone()));
                model.insert(k, v);
            }
        }
        assert_eq!(map.len(), model.len());
        assert_eq!(map.iter().count(), model.len());
        for (k, v) in &model {
            assert_eq!(map.get(k).map(|(_, mv)| mv), Some(v), "key {k}");
        }
        assert!(map.capacity() <= 16384, "capacity {}", map.capacity());
    }

    #[test]
    fn test_rehash_in_place_panic_in_hash() {
        use std::cell::Cell;

        thread_local!(static ARMED: Cell<bool> = const { Cell::new(false) });

        #[derive(PartialEq, Eq)]
        struct Bomb(u32, std::rc::Rc<()>);
        impl std::hash::Hash for Bomb {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                if ARMED.with(Cell::get) && self.0 == 3 {
                    panic!("hash bomb");
                }
                self.0.hash(state);
            }
        }

        let rc = std::rc::Rc::new(());
        let mut map = SwissTable::<PairExtract<Bomb, ()>>::with_capacity(64);
        for i in 0..40 {
            map.insert((Bomb(i, rc.clone()), ()));
        }
        ARMED.with(|a| a.set(true));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map.rehash_in_place();
        }));
        ARMED.with(|a| a.set(false));
        assert!(r.is_err());
        assert!(map.len() < 40);
        assert_eq!(map.iter().count(), map.len());
        assert_eq!(std::rc::Rc::strong_count(&rc), map.len() + 1);
        drop(map);
        assert_eq!(std::rc::Rc::strong_count(&rc), 1);
    }

    type PairMap = SwissTable<PairExtract<u32, String>>;

    fn pair_map(n: u32) -> PairMap {
        (0..n).map(|i| (i, i.to_string())).collect()
    }

    #[test]
    fn test_iterators_and_collections() {
        let mut map = pair_map(100);
        assert_eq!(map.len(), 100);

        for (_, v) in map.iter_mut() {
            v.push('!');
        }
        for (k, v) in &map {
            assert_eq!(*v, format!("{k}!"));
        }
        for (_, v) in &mut map {
            v.pop();
        }

        map.extend((100..200).map(|i| (i, i.to_string())));
        assert_eq!(map.len(), 200);

        let mut items: Vec<(u32, String)> = map.into_iter().collect();
        items.sort();
        let want: Vec<(u32, String)> = (0..200).map(|i| (i, i.to_string())).collect();
        assert_eq!(items, want);

        let small: PairMap = std::iter::once((7, "seven".to_owned())).collect();
        assert_eq!(format!("{small:?}"), r#"{(7, "seven")}"#);
    }

    #[test]
    fn test_into_iter_partial_drops_rest() {
        let rc = std::rc::Rc::new(());
        let map: SwissTable<PairExtract<u32, std::rc::Rc<()>>> =
            (0..50).map(|i| (i, rc.clone())).collect();
        let mut it = map.into_iter();
        let first = it.next();
        assert!(first.is_some());
        assert_eq!(std::rc::Rc::strong_count(&rc), 51);
        drop(it);
        assert_eq!(std::rc::Rc::strong_count(&rc), 2);
        drop(first);
        assert_eq!(std::rc::Rc::strong_count(&rc), 1);
    }

    #[test]
    fn test_reserve_and_shrink() {
        for n in [1usize, 7, 14, 15, 100, 1000, 28_672, 28_673] {
            let mut map = TestMap::with_capacity(n);
            let cap = map.capacity();
            for i in 0..n as u32 {
                map.insert(TestValue { key: i, data: "v" });
            }
            assert_eq!(map.capacity(), cap, "with_capacity({n}) grew");
        }

        let mut map = PairMap::new();
        map.reserve(500);
        let cap = map.capacity();
        map.extend((0..500).map(|i| (i, i.to_string())));
        assert_eq!(map.capacity(), cap);

        assert_eq!(
            map.try_reserve(usize::MAX),
            Err(TryReserveError::CapacityOverflow)
        );

        map.retain(|(k, _)| *k < 10);
        assert_eq!(map.len(), 10);
        map.shrink_to_fit();
        assert!(map.capacity() < cap, "capacity {}", map.capacity());
        for i in 0..10 {
            assert_eq!(
                map.get(&i).map(|(_, v)| v.as_str()),
                Some(i.to_string().as_str())
            );
        }

        map.clear();
        map.shrink_to_fit();
        assert_eq!(map.capacity(), 0);
        map.insert((1, "one".to_owned()));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_take_retain_drain_extract_if() {
        let mut map = pair_map(100);
        assert_eq!(map.take(&5), Some((5, "5".to_owned())));
        assert_eq!(map.take(&5), None);
        assert_eq!(map.len(), 99);

        map.retain(|(k, _)| *k % 2 == 0);
        assert_eq!(map.len(), 50);
        assert!(map.iter().all(|(k, _)| k % 2 == 0));

        let mut big: Vec<u32> = map.extract_if(|(k, _)| *k >= 50).map(|(k, _)| k).collect();
        big.sort_unstable();
        assert_eq!(big, (50..100).step_by(2).collect::<Vec<_>>());
        assert_eq!(map.len(), 25);
        assert!(map.iter().all(|(k, _)| *k < 50));

        let cap = map.capacity();
        let mut drained: Vec<u32> = map.drain().map(|(k, _)| k).collect();
        drained.sort_unstable();
        assert_eq!(drained, (0..50).step_by(2).collect::<Vec<_>>());
        assert!(map.is_empty());
        assert_eq!(map.capacity(), cap);
        assert_eq!(map.iter().count(), 0);

        let mut map = pair_map(30);
        let mut d = map.drain();
        assert!(d.next().is_some());
        drop(d);
        assert!(map.is_empty());
        map.insert((1, "one".to_owned()));
        assert_eq!(map.get(&1).map(|(_, v)| v.as_str()), Some("one"));
    }

    #[test]
    fn test_get_or_insert_with() {
        let mut map = SwissTable::<PairExtract<String, u32>>::new();
        let mut calls = 0;
        for _ in 0..3 {
            let v = map.get_or_insert_with("a", || {
                calls += 1;
                ("a".to_owned(), 1)
            });
            v.1 += 1;
        }
        assert_eq!(calls, 1);
        assert_eq!(map.get("a"), Some(&("a".to_owned(), 4)));

        for i in 0..1000 {
            let k = i.to_string();
            map.get_or_insert_with(k.as_str(), || (k.clone(), i));
        }
        assert_eq!(map.len(), 1001);
        assert_eq!(map.get("999").map(|(_, v)| *v), Some(999));
    }

    #[test]
    #[should_panic(expected = "not equivalent")]
    fn test_get_or_insert_with_wrong_key_panics() {
        let mut map = SwissTable::<PairExtract<String, u32>>::new();
        map.get_or_insert_with("a", || ("b".to_owned(), 1));
    }

    #[test]
    fn test_strided_keys_spread_over_table() {
        let mask = (1u64 << 18) - 1;
        for stride in [1u64, 8, 64, 4096, 1 << 20] {
            for hasher in ["auto", "fx"] {
                let mut starts = std::collections::HashSet::new();
                for i in 0..100_000u64 {
                    let h = if hasher == "auto" {
                        AutoBuildHasher.hash_one(i * stride)
                    } else {
                        FxBuildHasher.hash_one(i * stride)
                    };
                    starts.insert(h & mask);
                }
                assert!(
                    starts.len() > 80_000,
                    "{hasher} stride {stride}: only {} distinct start positions",
                    starts.len()
                );
            }
        }
    }

    #[test]
    fn test_small_tables_match_model() {
        let mut x: u64 = 0x5EED;
        for target in 0..20u32 {
            let mut map = SwissTable::<PairExtract<u32, u32>>::new();
            let mut model = std::collections::HashMap::new();
            for op in 0..2_000u32 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let k = (x % (u64::from(target) + 1)) as u32;
                match x % 5 {
                    0 | 1 => {
                        map.insert((k, op));
                        model.insert(k, op);
                    }
                    2 => assert_eq!(map.take(&k).map(|(_, v)| v), model.remove(&k), "take {k}"),
                    3 => {
                        let v = map.get_or_insert_with(&k, || (k, op)).1;
                        assert_eq!(v, *model.entry(k).or_insert(op), "get_or_insert_with {k}");
                    }
                    _ => {
                        if op % 97 == 0 {
                            map.shrink_to_fit();
                        }
                    }
                }
                assert_eq!(map.len(), model.len());
                assert_eq!(map.iter().count(), model.len());
                for (mk, mv) in &model {
                    assert_eq!(
                        map.get(mk).map(|(_, v)| *v),
                        Some(*mv),
                        "target {target} key {mk}"
                    );
                }
            }
            if model.len() < 4 {
                map.shrink_to_fit();
                assert!(map.capacity() <= MIN_BUCKETS, "capacity {}", map.capacity());
            }
        }
    }
}
