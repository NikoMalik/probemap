use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;

use crate::alloc::{Allocator, Global};
use crate::{DefaultHashBuilder, Equivalent, SelfExtract, SwissTable, TryReserveError};

pub struct HashSet<T: Hash + Eq, S: BuildHasher = DefaultHashBuilder, A: Allocator = Global> {
    table: SwissTable<SelfExtract<T>, S, A>,
}

impl<T: Hash + Eq> HashSet<T> {
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

impl<T: Hash + Eq, S: BuildHasher> HashSet<T, S> {
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

impl<T: Hash + Eq, S: BuildHasher + Default, A: Allocator> HashSet<T, S, A> {
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

impl<T: Hash + Eq, S: BuildHasher, A: Allocator> HashSet<T, S, A> {
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
    pub fn insert(&mut self, value: T) -> bool {
        let before = self.table.len();
        self.table.get_or_insert(value);
        self.table.len() != before
    }

    #[inline]
    pub fn replace(&mut self, value: T) -> Option<T> {
        self.table.replace(value)
    }

    #[inline]
    pub fn get_or_insert(&mut self, value: T) -> &T {
        self.table.get_or_insert(value)
    }

    #[inline]
    pub fn get_or_insert_with<Q>(&mut self, value: &Q, f: impl FnOnce(&Q) -> T) -> &T
    where
        Q: Hash + Equivalent<T> + ?Sized,
    {
        self.table.get_or_insert_with(value, || f(value))
    }

    #[inline]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: Hash + Equivalent<T> + ?Sized,
    {
        self.table.contains(value)
    }

    #[inline]
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: Hash + Equivalent<T> + ?Sized,
    {
        self.table.get(value)
    }

    #[inline]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: Hash + Equivalent<T> + ?Sized,
    {
        self.table.remove(value)
    }

    #[inline]
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: Hash + Equivalent<T> + ?Sized,
    {
        self.table.take(value)
    }

    pub fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
        self.table.retain(|v| f(v));
    }

    pub fn drain(&mut self) -> crate::Drain<'_, SelfExtract<T>, S, A> {
        self.table.drain()
    }

    pub fn extract_if<F>(&mut self, mut pred: F) -> impl FusedIterator<Item = T>
    where
        F: FnMut(&T) -> bool,
    {
        self.table.extract_if(move |v| pred(v))
    }

    pub fn iter(&self) -> crate::Iter<'_, SelfExtract<T>, S, A> {
        self.table.iter()
    }

    pub fn difference<'a>(&'a self, other: &'a Self) -> impl FusedIterator<Item = &'a T> {
        self.iter().filter(move |v| !other.contains(*v))
    }

    pub fn symmetric_difference<'a>(&'a self, other: &'a Self) -> impl FusedIterator<Item = &'a T> {
        self.difference(other).chain(other.difference(self))
    }

    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl FusedIterator<Item = &'a T> {
        let (small, big) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        small.iter().filter(move |v| big.contains(*v))
    }

    pub fn union<'a>(&'a self, other: &'a Self) -> impl FusedIterator<Item = &'a T> {
        let (small, big) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        big.iter().chain(small.difference(big))
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).next().is_none()
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.iter().all(|v| other.contains(v))
    }

    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }
}

impl<T: Hash + Eq, S: BuildHasher, A: Allocator> IntoIterator for HashSet<T, S, A> {
    type Item = T;
    type IntoIter = crate::IntoIter<SelfExtract<T>, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.table.into_iter()
    }
}

impl<'a, T: Hash + Eq, S: BuildHasher, A: Allocator> IntoIterator for &'a HashSet<T, S, A> {
    type Item = &'a T;
    type IntoIter = crate::Iter<'a, SelfExtract<T>, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Hash + Eq, S: BuildHasher, A: Allocator> Extend<T> for HashSet<T, S, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
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

impl<'a, T: Hash + Eq + Copy + 'a, S: BuildHasher, A: Allocator> Extend<&'a T>
    for HashSet<T, S, A>
{
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<T: Hash + Eq, S: BuildHasher + Default, A: Allocator + Default> FromIterator<T>
    for HashSet<T, S, A>
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new_in(A::default());
        set.extend(iter);
        set
    }
}

impl<T: Hash + Eq, const N: usize> From<[T; N]> for HashSet<T> {
    fn from(arr: [T; N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T: Hash + Eq, S: BuildHasher + Default, A: Allocator + Default> Default for HashSet<T, S, A> {
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher + Clone, A: Allocator + Clone> Clone
    for HashSet<T, S, A>
{
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
        }
    }
}

impl<T: Hash + Eq, S: BuildHasher, A: Allocator> PartialEq for HashSet<T, S, A> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.is_subset(other)
    }
}

impl<T: Hash + Eq, S: BuildHasher, A: Allocator> Eq for HashSet<T, S, A> {}

impl<T: Hash + Eq + fmt::Debug, S: BuildHasher, A: Allocator> fmt::Debug for HashSet<T, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behaves_like_std() {
        let mut ours: HashSet<u32> = HashSet::new();
        let mut std_set = std::collections::HashSet::new();
        let mut x: u64 = 0xC0FF_EE11;
        for _ in 0..50_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let k = (x % 3000) as u32;
            match x % 4 {
                0 | 1 => assert_eq!(ours.insert(k), std_set.insert(k), "insert {k}"),
                2 => assert_eq!(ours.remove(&k), std_set.remove(&k), "remove {k}"),
                _ => assert_eq!(ours.contains(&k), std_set.contains(&k), "contains {k}"),
            }
        }
        assert_eq!(ours.len(), std_set.len());
        assert!(std_set.iter().all(|k| ours.contains(k)));
    }

    #[test]
    fn set_ops() {
        let a = HashSet::from([1, 2, 3, 4]);
        let b = HashSet::from([3, 4, 5]);
        let sorted = |it: &mut dyn Iterator<Item = &i32>| {
            let mut v: Vec<i32> = it.copied().collect();
            v.sort_unstable();
            v
        };
        assert_eq!(sorted(&mut a.difference(&b)), [1, 2]);
        assert_eq!(sorted(&mut a.symmetric_difference(&b)), [1, 2, 5]);
        assert_eq!(sorted(&mut a.intersection(&b)), [3, 4]);
        assert_eq!(sorted(&mut a.union(&b)), [1, 2, 3, 4, 5]);
        assert!(!a.is_disjoint(&b));
        assert!(HashSet::from([3, 4]).is_subset(&a));
        assert!(a.is_superset(&HashSet::from([1])));
        assert_eq!(a, HashSet::from([4, 3, 2, 1]));
        assert_ne!(a, b);

        let mut s: HashSet<String> = HashSet::new();
        assert!(s.insert("x".to_owned()));
        assert!(!s.insert("x".to_owned()));
        assert_eq!(s.get_or_insert_with("y", |v| v.to_owned()), "y");
        assert!(s.contains("y"));
        assert_eq!(s.take("x"), Some("x".to_owned()));
        assert_eq!(format!("{s:?}"), r#"{"y"}"#);
    }
}
