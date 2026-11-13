use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use probemap::{PairExtract, SwissTable};

const N: u64 = 14_000;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Clone, Copy, Default)]
struct ClusterBuildHasher<const CLUSTERS: u64>;

struct ClusterHasher<const CLUSTERS: u64>(u64);

impl<const CLUSTERS: u64> BuildHasher for ClusterBuildHasher<CLUSTERS> {
    type Hasher = ClusterHasher<CLUSTERS>;

    fn build_hasher(&self) -> Self::Hasher {
        ClusterHasher(0)
    }
}

impl<const CLUSTERS: u64> Hasher for ClusterHasher<CLUSTERS> {
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("cluster hasher only hashes u64 keys")
    }

    fn write_u64(&mut self, k: u64) {
        self.0 = k;
    }

    fn finish(&self) -> u64 {
        let tag = mix(self.0) & 0xFE00_0000_0000_0000;
        let start = mix(self.0 % CLUSTERS) & 0x01FF_FFFF_FFFF_FFFF;
        tag | start
    }
}

fn shuffled(keys: impl Iterator<Item = u64>) -> Vec<u64> {
    let mut v: Vec<u64> = keys.collect();
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    for i in (1..v.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.swap(i, (x % (i as u64 + 1)) as usize);
    }
    v
}

fn timed<M>(
    b: &mut criterion::Bencher<'_>,
    m: &mut M,
    prepare: impl Fn(&mut M),
    work: impl Fn(&mut M),
) {
    b.iter_custom(|iters| {
        let mut total = Duration::ZERO;
        for _ in 0..iters {
            prepare(m);
            let start = Instant::now();
            work(m);
            total += start.elapsed();
            black_box(&mut *m);
        }
        total
    });
}

fn bench_cluster<const CLUSTERS: u64>(c: &mut Criterion) {
    let per_cluster = N / CLUSTERS;
    let hits = shuffled(0..N);
    let misses = shuffled(N..2 * N);
    let hasher = ClusterBuildHasher::<CLUSTERS>;

    let mut probe =
        SwissTable::<PairExtract<u64, u64>, _>::with_capacity_and_hasher(N as usize, hasher);
    let mut hb = hashbrown::HashMap::<u64, u64, _>::with_capacity_and_hasher(N as usize, hasher);

    let mut g = c.benchmark_group(format!("probe_c{per_cluster}/insert"));
    g.bench_function("probemap", |b| {
        timed(
            b,
            &mut probe,
            |m| m.clear(),
            |m| {
                for &k in &hits {
                    m.insert((k, k));
                }
            },
        );
    });
    g.bench_function("hashbrown", |b| {
        timed(
            b,
            &mut hb,
            |m| m.clear(),
            |m| {
                for &k in &hits {
                    m.insert(k, k);
                }
            },
        );
    });
    g.finish();

    for &k in &hits {
        probe.insert((k, k));
        hb.insert(k, k);
    }
    for (name, keys) in [("lookup_hit", &hits), ("lookup_miss", &misses)] {
        let mut g = c.benchmark_group(format!("probe_c{per_cluster}/{name}"));
        g.bench_function("probemap", |b| {
            b.iter(|| {
                for k in keys {
                    black_box(probe.get(k));
                }
            });
        });
        g.bench_function("hashbrown", |b| {
            b.iter(|| {
                for k in keys {
                    black_box(hb.get(k));
                }
            });
        });
        g.finish();
    }
}

fn benches(c: &mut Criterion) {
    bench_cluster::<{ N }>(c);
    bench_cluster::<{ N / 8 }>(c);
    bench_cluster::<{ N / 32 }>(c);
    bench_cluster::<{ N / 128 }>(c);
}

criterion_group! {
    name = probing;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    targets = benches
}
criterion_main!(probing);
