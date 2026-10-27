use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::mem;
use std::ops::Index;

use crate::alloc::{Allocator, Global};
use crate::{DefaultHashBuilder, Equivalent, PairExtract, SwissTable, TryReserveError};

type Table<K, V, S, A> = SwissTable<PairExtract<K, V>, S, A>;

pub struct HashMap<K: Hash + Eq, V, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    table: Table<K, V, S, A>,
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    pub fn new() -> Self {
        Self {
            table: SwissTable::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: SwissTable::with_capacity(capacity),
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> HashMap<K, V, S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            table: SwissTable::with_hasher(hash_builder),
        }
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            table: SwissTable::with_capacity_and_hasher(capacity, hash_builder),
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher + Default, A: Allocator> HashMap<K, V, S, A> {
    pub fn new_in(alloc: A) -> Self {
        Self {
            table: SwissTable::new_in(alloc),
        }
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            table: SwissTable::with_capacity_in(capacity, alloc),
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> HashMap<K, V, S, A> {
    pub fn with_hasher_in(hash_builder: S, alloc: A) -> Self {
        Self {
            table: SwissTable::with_hasher_and_alloc(hash_builder, alloc),
        }
    }

    pub fn hasher(&self) -> &S {
        self.table.hasher()
    }

    pub fn allocator(&self) -> &A {
        self.table.allocator()
    }

    pub fn capacity(&self) -> usize {
        self.table.len + self.table.growth_left
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn clear(&mut self) {
        self.table.clear();
    }

    pub fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional);
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.table.try_reserve(additional)
    }

    pub fn shrink_to_fit(&mut self) {
        self.table.shrink_to_fit();
    }

    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.table.replace((k, v)).map(|(_, v)| v)
    }

    #[inline]
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.table.get(k).map(|(_, v)| v)
    }

    #[inline]
    pub fn get_key_value<Q>(&self, k: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.table.get(k).map(|(k, v)| (k, v))
    }

    #[inline]
    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.table.get_mut(k).map(|(_, v)| v)
    }

    #[inline]
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.table.contains(k)
    }

    #[inline]
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.table.take(k).map(|(_, v)| v)
    }

    #[inline]
    pub fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.table.take(k)
    }

    pub fn retain(&mut self, mut f: impl FnMut(&K, &mut V) -> bool) {
        self.table.retain(|(k, v)| f(k, v));
    }

    pub fn drain(&mut self) -> crate::Drain<'_, PairExtract<K, V>, S, A> {
        self.table.drain()
    }

    pub fn extract_if<F>(&mut self, mut pred: F) -> impl FusedIterator<Item = (K, V)>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.table.extract_if(move |(k, v)| pred(k, v))
    }

    pub fn iter(&self) -> Iter<'_, K, V, S, A> {
        Iter(self.table.iter())
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, S, A> {
        IterMut(self.table.iter_mut())
    }

    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> + FusedIterator {
        self.table.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + FusedIterator {
        self.table.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> + FusedIterator {
        self.table.iter_mut().map(|(_, v)| v)
    }

    pub fn into_keys(self) -> impl ExactSizeIterator<Item = K> + FusedIterator {
        self.table.into_iter().map(|(k, _)| k)
    }

    pub fn into_values(self) -> impl ExactSizeIterator<Item = V> + FusedIterator {
        self.table.into_iter().map(|(_, v)| v)
    }

    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S, A> {
        self.table.reserve(1);
        let hash = self.table.hash_builder.hash_one(&key);
        match self.table.find_hashed(hash, &key) {
            Some(idx) => Entry::Occupied(OccupiedEntry {
                table: &mut self.table,
                idx,
            }),
            None => Entry::Vacant(VacantEntry {
                table: &mut self.table,
                hash,
                key,
            }),
        }
    }
}

pub enum Entry<'a, K: Hash + Eq, V, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    Occupied(OccupiedEntry<'a, K, V, S, A>),
    Vacant(VacantEntry<'a, K, V, S, A>),
}

pub struct OccupiedEntry<
    'a,
    K: Hash + Eq,
    V,
    S: BuildHasher = DefaultHashBuilder,
    A: Allocator = Global,
> {
    table: &'a mut Table<K, V, S, A>,
    idx: usize,
}

pub struct VacantEntry<
    'a,
    K: Hash + Eq,
    V,
    S: BuildHasher = DefaultHashBuilder,
    A: Allocator = Global,
> {
    table: &'a mut Table<K, V, S, A>,
    hash: u64,
    key: K,
}

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> Entry<'a, K, V, S, A> {
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    #[inline]
    pub fn or_insert_with(self, default: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }

    #[inline]
    pub fn or_insert_with_key(self, default: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let value = default(&e.key);
                e.insert(value)
            }
        }
    }

    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    #[inline]
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Entry::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }

    #[inline]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => e.key(),
            Entry::Vacant(e) => e.key(),
        }
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> OccupiedEntry<'a, K, V, S, A> {
    #[inline]
    fn pair(&self) -> &(K, V) {
        // SAFETY: idx is a FULL slot found by entry(), the table is borrowed for 'a.
        unsafe { &*self.table.slot_mut(self.idx).cast::<(K, V)>() }
    }

    #[inline]
    pub fn key(&self) -> &K {
        &self.pair().0
    }

    #[inline]
    pub fn get(&self) -> &V {
        &self.pair().1
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: idx is a FULL slot, &mut self gives exclusive access.
        unsafe { &mut (*self.table.slot_mut(self.idx).cast::<(K, V)>()).1 }
    }

    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: idx is a FULL slot, the table is mutably borrowed for 'a.
        unsafe { &mut (*self.table.slot_mut(self.idx).cast::<(K, V)>()).1 }
    }

    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        // SAFETY: idx is a FULL slot, take_at marks it DELETED.
        unsafe { self.table.take_at(self.idx) }
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> VacantEntry<'a, K, V, S, A> {
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let pair = self.table.insert_new_hashed(self.hash, (self.key, value));
        // SAFETY: insert_new_hashed returns a pointer to the value it just wrote.
        unsafe { &mut (*pair).1 }
    }
}

pub struct Iter<'a, K: Hash + Eq, V, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global>(
    crate::Iter<'a, PairExtract<K, V>, S, A>,
);

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> Iterator for Iter<'a, K, V, S, A> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> ExactSizeIterator for Iter<'_, K, V, S, A> {}
impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> FusedIterator for Iter<'_, K, V, S, A> {}

pub struct IterMut<'a, K: Hash + Eq, V, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global>(
    crate::IterMut<'a, PairExtract<K, V>, S, A>,
);

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> Iterator for IterMut<'a, K, V, S, A> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (&*k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> ExactSizeIterator for IterMut<'_, K, V, S, A> {}
impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> FusedIterator for IterMut<'_, K, V, S, A> {}

impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> IntoIterator for HashMap<K, V, S, A> {
    type Item = (K, V);
    type IntoIter = crate::IntoIter<PairExtract<K, V>, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.table.into_iter()
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> IntoIterator for &'a HashMap<K, V, S, A> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher, A: Allocator> IntoIterator
    for &'a mut HashMap<K, V, S, A>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher, A: Allocator> Extend<(K, V)> for HashMap<K, V, S, A> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        self.table.extend(iter);
    }
}

impl<'a, K, V, S, A> Extend<(&'a K, &'a V)> for HashMap<K, V, S, A>
where
    K: Hash + Eq + Copy,
    V: Copy,
    S: BuildHasher,
    A: Allocator,
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        self.table.extend(iter.into_iter().map(|(k, v)| (*k, *v)));
    }
}

impl<K: Hash + Eq, V, S: BuildHasher + Default, A: Allocator + Default> FromIterator<(K, V)>
    for HashMap<K, V, S, A>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self {
            table: iter.into_iter().collect(),
        }
    }
}

impl<K: Hash + Eq, V, const N: usize> From<[(K, V); N]> for HashMap<K, V> {
    fn from(arr: [(K, V); N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher + Default, A: Allocator + Default> Default
    for HashMap<K, V, S, A>
{
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl<K, V, S, A> Clone for HashMap<K, V, S, A>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    A: Allocator + Clone,
{
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
        }
    }
}

impl<K, V, S, A> PartialEq for HashMap<K, V, S, A>
where
    K: Hash + Eq,
    V: PartialEq,
    S: BuildHasher,
    A: Allocator,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<K: Hash + Eq, V: Eq, S: BuildHasher, A: Allocator> Eq for HashMap<K, V, S, A> {}

impl<K, V, S, A> fmt::Debug for HashMap<K, V, S, A>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
    A: Allocator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, Q, V, S, A> Index<&Q> for HashMap<K, V, S, A>
where
    K: Hash + Eq,
    Q: Hash + Equivalent<K> + ?Sized,
    S: BuildHasher,
    A: Allocator,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("key not found")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behaves_like_std() {
        let mut ours: HashMap<u32, String> = HashMap::new();
        let mut std_map = std::collections::HashMap::new();
        let mut x: u64 = 0xDEAD_BEEF;
        for op in 0..50_000u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let k = (x % 3000) as u32;
            match x % 7 {
                0..=2 => assert_eq!(
                    ours.insert(k, op.to_string()),
                    std_map.insert(k, op.to_string()),
                    "insert {k}"
                ),
                3 => assert_eq!(ours.remove(&k), std_map.remove(&k), "remove {k}"),
                4 => {
                    *ours.entry(k).or_default() += "e";
                    *std_map.entry(k).or_default() += "e";
                }
                5 => {
                    if let Entry::Occupied(e) = ours.entry(k) {
                        e.remove();
                    }
                    if let std::collections::hash_map::Entry::Occupied(e) = std_map.entry(k) {
                        e.remove();
                    }
                }
                _ => assert_eq!(ours.get(&k), std_map.get(&k), "get {k}"),
            }
        }
        assert_eq!(ours.len(), std_map.len());
        assert_eq!(ours.iter().len(), std_map.len());
        for (k, v) in &std_map {
            assert_eq!(&ours[k], v);
        }
    }

    #[test]
    fn std_like_api() {
        let mut m = HashMap::from([(1, "a"), (2, "b")]);
        assert_eq!(m.insert(1, "c"), Some("a"));
        assert!(m.contains_key(&2));
        assert_eq!(m.get_key_value(&2), Some((&2, &"b")));
        m.entry(3).and_modify(|v| *v = "x").or_insert("z");
        m.entry(3).and_modify(|v| *v = "x").or_insert("z");
        assert_eq!(m[&3], "x");
        assert_eq!(
            *m.entry(4)
                .or_insert_with_key(|k| if *k == 4 { "four" } else { "?" }),
            "four"
        );

        let mut keys: Vec<_> = m.keys().copied().collect();
        keys.sort_unstable();
        assert_eq!(keys, [1, 2, 3, 4]);
        for v in m.values_mut() {
            *v = "v";
        }
        assert!(m.values().all(|v| *v == "v"));

        m.retain(|k, _| k % 2 == 0);
        assert_eq!(m.len(), 2);
        let mut removed: Vec<_> = m.extract_if(|k, _| *k == 2).collect();
        removed.sort_unstable();
        assert_eq!(removed, [(2, "v")]);
        assert_eq!(m.remove_entry(&4), Some((4, "v")));
        assert!(m.is_empty());

        let a: HashMap<_, _> = (0..10).map(|i| (i, i * 2)).collect();
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(format!("{:?}", HashMap::from([(1, 2)])), "{1: 2}");
        assert!(a.capacity() >= 10);
        let mut vals: Vec<_> = a.into_values().collect();
        vals.sort_unstable();
        assert_eq!(vals, (0..10).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HashMap<String, Vec<u8>>>();
        assert_send_sync::<crate::HashSet<String>>();
        assert_send_sync::<SwissTable<PairExtract<u64, String>>>();
        assert_send_sync::<crate::Iter<'static, PairExtract<u64, String>>>();
        assert_send_sync::<crate::IterMut<'static, PairExtract<u64, String>>>();
        assert_send_sync::<crate::IntoIter<PairExtract<u64, String>>>();

        let map: HashMap<u32, String> = (0..100).map(|i| (i, i.to_string())).collect();
        let shared = std::sync::Arc::new(map);
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let m = std::sync::Arc::clone(&shared);
                std::thread::spawn(move || (0..100).filter(|i| m[i] == i.to_string()).count() + t)
            })
            .collect();
        let total: usize = handles.into_iter().map(|h| h.join().expect("thread")).sum();
        assert_eq!(total, 400 + 6);
    }
}
