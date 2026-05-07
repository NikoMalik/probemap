# probemap

A SwissTable hash map with direct SSE2 SIMD — works on **stable Rust**.

## Features

- **Direct SSE2 intrinsics** on x86_64, **scalar fallback** on other architectures
- **`KeyExtract` trait** — store only values when the key is embedded in the value
- **Pluggable `BuildHasher`** — default `AutoHasher` (FxHash for ints, buthash for byte slices), swap for any hasher
- **Custom allocator support** via `allocator-api2` (stable polyfill for the Allocator API)
- **Zero-cost empty table** — `new()` does not allocate; first `insert` triggers allocation
- **SIMD-accelerated iterator** — scans 16 control bytes per group via bitmask
- **Bulk `memcpy` clone** for `Copy` types
- **87% load factor** with linear group probing

## Usage

```rust
use probemap::{KeyExtract, SwissTable, PairExtract};

// Simple (K, V) usage — like HashMap<K, V>
type MyMap = SwissTable<PairExtract<u64, String>>;
let mut map = MyMap::new();
map.insert((42, "hello".to_string()));
assert_eq!(map.get(&42).unwrap().1, "hello");

// key_from_value pattern — key embedded in value
struct User { id: u64, name: String }

struct UserById;
impl KeyExtract for UserById {
    type Key = u64;
    type Value = User;
    fn extract(value: &User) -> &u64 { &value.id }
}

let mut users = SwissTable::<UserById>::with_capacity(64);
users.insert(User { id: 1, name: "Alice".into() });
assert_eq!(users.get(&1).unwrap().name, "Alice");
```

### Custom hasher

```rust
use probemap::{SwissTable, PairExtract, FxBuildHasher};
use std::collections::hash_map::RandomState;

// Use FxHash only (no buthash)
let map = SwissTable::<PairExtract<u64, u64>, FxBuildHasher>::with_hasher(FxBuildHasher);

// Use SipHash (DoS-resistant)
let map = SwissTable::<PairExtract<String, u64>, RandomState>::with_hasher(RandomState::new());
```

## API

| Method | Description |
|---|---|
| `new()` / `new_in(alloc)` | Empty table (no allocation) |
| `with_capacity(n)` / `with_capacity_in(n, alloc)` | Pre-allocate for `n` elements |
| `with_hasher(S)` | Empty table with custom hash builder |
| `with_capacity_and_hasher(n, S)` | Pre-allocate with custom hash builder |
| `with_hasher_and_alloc(S, A)` | Custom hash builder + custom allocator |
| `insert(value)` | Insert or update by key |
| `get(key)` / `get_mut(key)` | Lookup by key |
| `remove(key)` | Remove by key (returns `bool`) |
| `contains(key)` | Key existence check |
| `get_or_insert(value)` | Get existing or insert new |
| `clear()` | Drop all values, keep allocation |
| `clone_table()` / `clone()` | Deep copy |
| `iter()` | Iterator over all values |
| `prefetch(key)` | Hint CPU to preload cache lines |
| `hasher()` | Reference to the hash builder |
| `allocator()` | Reference to the allocator |
| `len()` / `is_empty()` / `capacity()` | Size queries |

## Benchmarks (vs hashbrown+fxhash, n=1000)

| Benchmark | probemap | hb+fxhash | hb+foldhash | vs hb+fxhash |
|---|---|---|---|---|
| insert (random) | **6.03 µs** | 6.51 µs | 6.89 µs | **-7%** |
| insert (serial) | **4.55 µs** | 5.90 µs | 6.64 µs | **-23%** |
| grow_insert | **19.8 µs** | 21.7 µs | 27.3 µs | **-9%** |
| lookup (random) | **2.83 µs** | 4.11 µs | 4.47 µs | **-31%** |
| lookup (serial) | **2.73 µs** | 4.11 µs | 4.58 µs | **-34%** |
| lookup_fail | **2.71 µs** | 3.17 µs | 3.54 µs | **-15%** |
| insert_erase | **10.0 µs** | 15.4 µs | 16.0 µs | **-35%** |
| iter | **1.21 µs** | 1.25 µs | 1.25 µs | **-3%** |
| clone | **1.19 µs** | 1.25 µs | 1.29 µs | **-5%** |



## Why faster than hashbrown?

Both probemap and hashbrown implement the same SwissTable algorithm and use the
same SSE2 intrinsics. The performance difference comes from how the surrounding
code is structured.

**Monomorphized key comparison vs `dyn FnMut`.**
hashbrown's inner lookup (`find_inner`) takes the key comparator as
`&mut dyn FnMut(usize) -> bool` — a trait object. This means every key
comparison in the probe loop is an indirect call through a vtable pointer.
The CPU cannot predict it, and LLVM cannot inline through it. This is a
deliberate trade-off in hashbrown to reduce binary size from monomorphization.
probemap calls `E::extract(slot) == key` directly — the compiler monomorphizes
the comparison for each concrete type and inlines it into the probe loop,
eliminating the indirect call overhead entirely. This is likely the single
biggest contributor to the ~30% lookup speedup.

**Forward slot layout.**
hashbrown stores slots in reverse order relative to control bytes
(`base.sub(index)`). probemap uses forward order (`slots.add(index)`). Modern
CPUs prefetch forward; reverse indexing works against the hardware prefetcher.

**Flatter data path.**
hashbrown has four layers: `HashMap` -> `HashTable` -> `RawTable` ->
`RawTableInner`. `RawTableInner` type-erases the value to `*u8` and reconstructs
`Bucket<T>` through pointer arithmetic. probemap is a single struct with inline
methods — no type erasure, no `Bucket` wrapper, no intermediate layers.

**Linear group probing.**
hashbrown uses triangular probing (`stride += 16; pos += stride`) — two
additions and an extra field per step. probemap uses linear probing with
mirrored control bytes: `pos = (pos + 16) & mask` — one add, one and. The
mirror (first 16 control bytes duplicated at the end of the array) lets
unaligned SIMD loads wrap around without sentinel bytes or branches.

**`insert_assume_unique` during grow.**
When rehashing into a fresh table, there can be no duplicates. probemap skips
the h2 match + key comparison loop and only scans for empty slots. hashbrown
also optimizes this path but carries extra bookkeeping (`growth_left` updates)
in the hot loop.

**SIMD iterator with `match_full()`.**
The iterator loads 16 control bytes, extracts a bitmask of occupied slots with
`_mm_movemask_epi8`, and pops bits with `trailing_zeros`. Empty groups are
skipped in one branch. The struct is flat — `(table, group_pos, bitmask)`.

**Bulk `memcpy` clone.**
`clone_table()` checks `needs_drop::<V>()` at compile time. For `Copy` types
the entire slots array is a single `memcpy` — no per-element loop, no rehash.

## Requirements

- **Rust edition 2024** (1.85+)
- Stable toolchain (no nightly features required)
- SSE2 SIMD on x86_64, scalar fallback on other architectures

## License
Apache License
