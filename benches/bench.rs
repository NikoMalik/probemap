//! Benchmarks: probemap vs hashbrown (same and different hashers).
//!
//! Three variants:
//! - `probemap`    — our SwissTable + FxHasher
//! - `hb_fxhash`   — hashbrown + our FxHasher (fair table-vs-table comparison)
//! - `hb_foldhash`  — hashbrown + its default foldhash (production baseline)
//!
//! Run: `cargo bench`
//! HTML report: target/criterion/report/index.html

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use hashbrown::HashMap as HbMap;

use probemap::{FxBuildHasher, KeyExtract, PairExtract, SwissTable};

const SIZE: usize = 10000;

type HbFxMap<K, V> = HbMap<K, V, FxBuildHasher>;
type HbFoldMap<K, V> = HbMap<K, V>;

#[derive(Clone, Copy)]
struct RandomKeys {
    state: usize,
}

impl RandomKeys {
    fn new() -> Self {
        Self { state: 0 }
    }
}

impl Iterator for RandomKeys {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        self.state = self.state.wrapping_add(1).wrapping_mul(3_787_392_781);
        Some(self.state)
    }
}

struct PairExt;
impl KeyExtract for PairExt {
    type Key = usize;
    type Value = (usize, usize);
    #[inline]
    fn extract(value: &(usize, usize)) -> &usize {
        &value.0
    }
}

type ProbeMap = SwissTable<PairExt>;

fn random_keys(n: usize) -> Vec<usize> {
    RandomKeys::new().take(n).collect()
}

fn serial_keys(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn bench_insert(c: &mut Criterion) {
    let keys = random_keys(SIZE);
    let mut g = c.benchmark_group("insert");

    g.bench_function("probemap", |b| {
        let mut m = ProbeMap::with_capacity(SIZE);
        b.iter(|| {
            m.clear();
            for &k in &keys {
                m.insert((k, k));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_fxhash", |b| {
        let mut m = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
        b.iter(|| {
            m.clear();
            for &k in &keys {
                m.insert(k, k);
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_foldhash", |b| {
        let mut m = HbFoldMap::with_capacity(SIZE);
        b.iter(|| {
            m.clear();
            for &k in &keys {
                m.insert(k, k);
            }
            black_box(&mut m);
        });
    });

    g.finish();
}

fn bench_insert_serial(c: &mut Criterion) {
    let keys = serial_keys(SIZE);
    let mut g = c.benchmark_group("insert_serial");

    g.bench_function("probemap", |b| {
        let mut m = ProbeMap::with_capacity(SIZE);
        b.iter(|| {
            m.clear();
            for &k in &keys {
                m.insert((k, k));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_fxhash", |b| {
        let mut m = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
        b.iter(|| {
            m.clear();
            for &k in &keys {
                m.insert(k, k);
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_foldhash", |b| {
        let mut m = HbFoldMap::with_capacity(SIZE);
        b.iter(|| {
            m.clear();
            for &k in &keys {
                m.insert(k, k);
            }
            black_box(&mut m);
        });
    });

    g.finish();
}

fn bench_grow_insert(c: &mut Criterion) {
    let keys = random_keys(SIZE);
    let mut g = c.benchmark_group("grow_insert");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            let mut m = ProbeMap::new();
            for &k in &keys {
                m.insert((k, k));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            let mut m = HbFxMap::with_hasher(FxBuildHasher);
            for &k in &keys {
                m.insert(k, k);
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            let mut m = HbFoldMap::new();
            for &k in &keys {
                m.insert(k, k);
            }
            black_box(&mut m);
        });
    });

    g.finish();
}

fn bench_lookup(c: &mut Criterion) {
    let keys = random_keys(SIZE);

    let mut probe = ProbeMap::with_capacity(SIZE);
    let mut hb_fx = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldMap::with_capacity(SIZE);
    for &k in &keys {
        probe.insert((k, k));
        hb_fx.insert(k, k);
        hb_fold.insert(k, k);
    }

    let mut g = c.benchmark_group("lookup");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(probe.get(&k));
            }
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(hb_fx.get(&k));
            }
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(hb_fold.get(&k));
            }
        });
    });

    g.finish();
}

fn bench_lookup_serial(c: &mut Criterion) {
    let keys = serial_keys(SIZE);

    let mut probe = ProbeMap::with_capacity(SIZE);
    let mut hb_fx = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldMap::with_capacity(SIZE);
    for &k in &keys {
        probe.insert((k, k));
        hb_fx.insert(k, k);
        hb_fold.insert(k, k);
    }

    let mut g = c.benchmark_group("lookup_serial");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(probe.get(&k));
            }
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(hb_fx.get(&k));
            }
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(hb_fold.get(&k));
            }
        });
    });

    g.finish();
}

fn bench_lookup_fail(c: &mut Criterion) {
    let keys = random_keys(SIZE);
    let miss_keys: Vec<usize> = RandomKeys::new().skip(SIZE).take(SIZE).collect();

    let mut probe = ProbeMap::with_capacity(SIZE);
    let mut hb_fx = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldMap::with_capacity(SIZE);
    for &k in &keys {
        probe.insert((k, k));
        hb_fx.insert(k, k);
        hb_fold.insert(k, k);
    }

    let mut g = c.benchmark_group("lookup_fail");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            for &k in &miss_keys {
                black_box(probe.get(&k));
            }
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            for &k in &miss_keys {
                black_box(hb_fx.get(&k));
            }
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            for &k in &miss_keys {
                black_box(hb_fold.get(&k));
            }
        });
    });

    g.finish();
}

fn bench_insert_erase(c: &mut Criterion) {
    let keys = random_keys(SIZE);
    let extra_keys: Vec<usize> = RandomKeys::new().skip(SIZE).take(SIZE).collect();

    let mut g = c.benchmark_group("insert_erase");

    g.bench_function("probemap", |b| {
        let mut base = ProbeMap::with_capacity(SIZE * 2);
        for &k in &keys {
            base.insert((k, k));
        }
        b.iter(|| {
            let mut m = base.clone_table();
            for (&add, &remove) in extra_keys.iter().zip(keys.iter()) {
                m.insert((add, add));
                black_box(m.remove(&remove));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_fxhash", |b| {
        let mut base = HbFxMap::with_capacity_and_hasher(SIZE * 2, FxBuildHasher);
        for &k in &keys {
            base.insert(k, k);
        }
        b.iter(|| {
            let mut m = base.clone();
            for (&add, &remove) in extra_keys.iter().zip(keys.iter()) {
                m.insert(add, add);
                black_box(m.remove(&remove));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_foldhash", |b| {
        let mut base = HbFoldMap::with_capacity(SIZE * 2);
        for &k in &keys {
            base.insert(k, k);
        }
        b.iter(|| {
            let mut m = base.clone();
            for (&add, &remove) in extra_keys.iter().zip(keys.iter()) {
                m.insert(add, add);
                black_box(m.remove(&remove));
            }
            black_box(&mut m);
        });
    });

    g.finish();
}

fn bench_iter(c: &mut Criterion) {
    let keys = random_keys(SIZE);

    let mut probe = ProbeMap::with_capacity(SIZE);
    let mut hb_fx = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldMap::with_capacity(SIZE);
    for &k in &keys {
        probe.insert((k, k));
        hb_fx.insert(k, k);
        hb_fold.insert(k, k);
    }

    let mut g = c.benchmark_group("iter");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            for entry in probe.iter() {
                black_box(entry);
            }
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            for entry in hb_fx.iter() {
                black_box(entry);
            }
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            for entry in hb_fold.iter() {
                black_box(entry);
            }
        });
    });

    g.finish();
}

fn bench_clone(c: &mut Criterion) {
    let keys = random_keys(SIZE);

    let mut probe = ProbeMap::with_capacity(SIZE);
    let mut hb_fx = HbFxMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldMap::with_capacity(SIZE);
    for &k in &keys {
        probe.insert((k, k));
        hb_fx.insert(k, k);
        hb_fold.insert(k, k);
    }

    let mut g = c.benchmark_group("clone");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            black_box(probe.clone_table());
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            black_box(hb_fx.clone());
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            black_box(hb_fold.clone());
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Byte-key benchmarks — exercises Hasher::write(bytes) path
// probemap: AutoHasher -> buthash::low_level_hash
// hb_fxhash: FxHasher -> rotate-xor-multiply per 8-byte chunk
// hb_foldhash: foldhash default
// ---------------------------------------------------------------------------

type ProbeByteMap = SwissTable<PairExtract<Vec<u8>, usize>>;
type HbFxByteMap = HbMap<Vec<u8>, usize, FxBuildHasher>;
type HbFoldByteMap = HbMap<Vec<u8>, usize>;

/// Generate `n` random byte keys of 16 bytes each, skipping `skip` keys first.
fn random_byte_keys(n: usize, skip: usize) -> Vec<Vec<u8>> {
    let mut rng = RandomKeys::new();
    // Each key consumes 2 values from the RNG
    for _ in 0..skip * 2 {
        rng.next();
    }
    (0..n)
        .map(|_| {
            let a = rng.next().unwrap().to_ne_bytes();
            let b = rng.next().unwrap().to_ne_bytes();
            let mut key = Vec::with_capacity(16);
            key.extend_from_slice(&a);
            key.extend_from_slice(&b);
            key
        })
        .collect()
}

fn bench_insert_bytes(c: &mut Criterion) {
    let keys = random_byte_keys(SIZE, 0);
    let mut g = c.benchmark_group("insert_bytes");

    g.bench_function("probemap", |b| {
        let mut m = ProbeByteMap::with_capacity(SIZE);
        b.iter(|| {
            m.clear();
            for (i, k) in keys.iter().enumerate() {
                m.insert((k.clone(), i));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_fxhash", |b| {
        let mut m = HbFxByteMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
        b.iter(|| {
            m.clear();
            for (i, k) in keys.iter().enumerate() {
                m.insert(k.clone(), i);
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_foldhash", |b| {
        let mut m = HbFoldByteMap::with_capacity(SIZE);
        b.iter(|| {
            m.clear();
            for (i, k) in keys.iter().enumerate() {
                m.insert(k.clone(), i);
            }
            black_box(&mut m);
        });
    });

    g.finish();
}

fn bench_lookup_bytes(c: &mut Criterion) {
    let keys = random_byte_keys(SIZE, 0);

    let mut probe = ProbeByteMap::with_capacity(SIZE);
    let mut hb_fx = HbFxByteMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldByteMap::with_capacity(SIZE);
    for (i, k) in keys.iter().enumerate() {
        probe.insert((k.clone(), i));
        hb_fx.insert(k.clone(), i);
        hb_fold.insert(k.clone(), i);
    }

    let mut g = c.benchmark_group("lookup_bytes");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            for k in &keys {
                black_box(probe.get(k));
            }
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            for k in &keys {
                black_box(hb_fx.get(k));
            }
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            for k in &keys {
                black_box(hb_fold.get(k));
            }
        });
    });

    g.finish();
}

fn bench_lookup_bytes_fail(c: &mut Criterion) {
    let keys = random_byte_keys(SIZE, 0);
    let miss_keys = random_byte_keys(SIZE, SIZE); // skip past inserted keys

    let mut probe = ProbeByteMap::with_capacity(SIZE);
    let mut hb_fx = HbFxByteMap::with_capacity_and_hasher(SIZE, FxBuildHasher);
    let mut hb_fold = HbFoldByteMap::with_capacity(SIZE);
    for (i, k) in keys.iter().enumerate() {
        probe.insert((k.clone(), i));
        hb_fx.insert(k.clone(), i);
        hb_fold.insert(k.clone(), i);
    }

    let mut g = c.benchmark_group("lookup_bytes_fail");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            for k in &miss_keys {
                black_box(probe.get(k));
            }
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            for k in &miss_keys {
                black_box(hb_fx.get(k));
            }
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            for k in &miss_keys {
                black_box(hb_fold.get(k));
            }
        });
    });

    g.finish();
}

fn bench_grow_insert_bytes(c: &mut Criterion) {
    let keys = random_byte_keys(SIZE, 0);
    let mut g = c.benchmark_group("grow_insert_bytes");

    g.bench_function("probemap", |b| {
        b.iter(|| {
            let mut m = ProbeByteMap::new();
            for (i, k) in keys.iter().enumerate() {
                m.insert((k.clone(), i));
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_fxhash", |b| {
        b.iter(|| {
            let mut m = HbFxByteMap::with_hasher(FxBuildHasher);
            for (i, k) in keys.iter().enumerate() {
                m.insert(k.clone(), i);
            }
            black_box(&mut m);
        });
    });

    g.bench_function("hb_foldhash", |b| {
        b.iter(|| {
            let mut m = HbFoldByteMap::new();
            for (i, k) in keys.iter().enumerate() {
                m.insert(k.clone(), i);
            }
            black_box(&mut m);
        });
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_insert_serial,
    bench_grow_insert,
    bench_lookup,
    bench_lookup_serial,
    bench_lookup_fail,
    bench_insert_erase,
    bench_iter,
    bench_clone,
    bench_insert_bytes,
    bench_lookup_bytes,
    bench_lookup_bytes_fail,
    bench_grow_insert_bytes,
);
criterion_main!(benches);
