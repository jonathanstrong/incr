//! Simple, fast and self-contained structs for tracking whether a newly observed
//! value (`u64`) is greater than the largest previously observed value.
//!
//! Use cases include message sequence numbers, timestamps, and other situations
//! that present a need to quickly assess whether incoming data is "new", i.e. its
//! numbering is larger than any previous value.
//!
//! All of the structs include an `is_new` function that returns `true` if the
//! passed value is a new maximum, while simultaneously storing the new value to
//! check against future values.
//!
//! Two of the `is_new` implementations (`Incr` and `Map`) require an `&mut self`
//! signature, while `RcIncr` and `AtomicIncr` and `AtomicMap` require only `&self`
//! due to `RcIncr`'s interior mutability and `AtomicIncr`'s thread safe syncrhonization.
//!
//! The cost of checking a new value is minimal: 0-2ns for the single-threaded
//! implementations, and ~5-10ns for `AtomicIncr`, except in cases of pathological
//! contention. In a worst-case, nightmare scenrio benchmark for the `AtomicIncr`,
//! it's possible to induce delays of hundreds of nanoseconds. A more realistic
//! case of 24 threads contending to increment the atomic but yielding each iteration
//! resulted in checks in the ~5-10ns range.
//!
//! Enabling the "nightly" feature (on by default) allows the use of `AtomicU64`
//! as the backing storage for `AtomicIncr` (vs. `AtomicUsize` otherwise). Also,
//! nightly is required to run the benchmarks.
//!
#![cfg_attr(feature = "nightly", feature(integer_atomics, test))]

#[cfg(all(test, feature = "nightly"))]
extern crate test;

use std::collections::HashMap;
use std::hash::Hash;
use std::cmp;
use std::rc::Rc;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::borrow::Borrow;
#[cfg(feature = "nightly")]
use std::sync::atomic::AtomicU64;
#[cfg(not(feature = "nightly"))]
use std::sync::atomic::AtomicUsize;

#[cfg(feature = "nightly")]
type Atomic = AtomicU64;
#[cfg(not(feature = "nightly"))]
type Atomic = AtomicUsize;

/// A self-contained struct for quickly checking whether a newly observed value
/// is greater than any previously observed value.
///
/// # Examples
///
/// ```
/// use incr::Incr;
/// let mut last = Incr::default();
/// assert_eq!(last.is_new(1), true);
/// assert_eq!(last.is_new(1), false);
/// assert_eq!(last.is_new(2), true);
/// assert_eq!(last.get(), 2);
/// ```
///
#[derive(Default, Clone, PartialEq, PartialOrd, Eq)]
pub struct Incr(u64);

/// A map interface allowing fast checks of whether a newly observed value
/// is greater than any previously observed value for a given key.
///
/// The inner map is a `HashMap<K, u64>`. The signature of `Map`'s `is_new` is
/// unique from the others in also requiring a key.
///
/// # Examples
///
/// ```
/// use incr::Map;
/// let mut last: Map<&'static str> = Default::default();
/// assert_eq!(last.is_new("a", 1), true);
/// assert_eq!(last.is_new("b", 1), true);
/// assert_eq!(last.is_new("a", 1), false);
/// assert_eq!(last.is_new("b", 3), true);
/// assert_eq!(last.is_new("c", 1), true);
/// assert_eq!(last.is_new("c", 1), false);
/// assert_eq!(last.get(&"b"), 3);
/// assert_eq!(last.get(&"not a key"), 0);
/// ```
///
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Map<K: Eq + Hash>(HashMap<K, u64>);

/// The `Rc<Cell<_>>` backed storage of `RcIncr` provides flexibility in situations
/// where the counter must be shared among several disparate objects/contexts while
/// retaining consistentcy between all the references of the count. `RcIncr` is not
/// threadsafe, and even in single-threaded code `Rc<Cell<T>>` has some tricky edge
/// cases, for instance if a `Cell<T>` is used as the key to a hash map and the
/// interior value mutated (fair warning).
///
/// # Examples
///
/// ```
/// use incr::RcIncr;
/// let mut last = RcIncr::default();
/// assert_eq!(last.is_new(1), true);
/// let mut xs = Vec::new();
/// for i in 2..5 {
///     xs.push(last.clone());
///     xs.last().unwrap().is_new(i);
/// }
/// assert_eq!(last.get(), 4);
/// for x in &xs {
///     assert_eq!(x.get(), 4);
/// }
/// ```
///
#[derive(Default, Clone, PartialEq, PartialOrd, Eq)]
pub struct RcIncr(Rc<Cell<u64>>);

/// `AtomicIncr` is a threadsafe, yet very fast counter, utilizing compare
/// and swap instructions to provide speed and safety in the same package.
/// There are some cases where 5ns matters. But in many, many other
/// situations, it's a perfectly good decision to just use the `AtomicIncr`,
/// knowing it can handle anything, and move on to other problems.
///
/// # Examples
///
/// ```
/// # #![cfg_attr(feature = "nightly", feature(integer_atomics))]
/// # use std::sync::atomic::*;
/// # use std::sync::Arc;
/// # use std::thread;
/// # #[cfg(feature = "nightly")]
/// # use std::sync::atomic::AtomicU64;
/// # #[cfg(not(feature = "nightly"))]
/// # use std::sync::atomic::AtomicUsize;
/// use incr::AtomicIncr;
///
/// #[cfg(feature = "nightly")]
/// type Atomic = AtomicU64;
/// #[cfg(not(feature = "nightly"))]
/// type Atomic = AtomicUsize;
///
/// let stop = Arc::new(AtomicBool::new(false));
/// let last: AtomicIncr = Default::default();
/// let mut threads = Vec::new();
/// for _ in 0..5 {
///     let val: Arc<Atomic> = last.clone().into_inner();
///     let stop = Arc::clone(&stop);
///     threads.push(thread::spawn(move || {
///         loop {
///             val.fetch_add(1, Ordering::Relaxed);
///             thread::yield_now();
///             if stop.load(Ordering::Relaxed) { break }
///         }
///     }));
/// }
///
/// let mut i = 1;
///
/// for _ in 0..100 {
///     i = match last.is_new(i) {
///         true => i + 1,
///         false => i.max(last.get()),
///     };
/// }
/// stop.store(true, Ordering::SeqCst);
/// ```
///
#[derive(Default, Clone)]
pub struct AtomicIncr(Arc<Atomic>);

/// Like `Map`, `AtomicMap` provides simple, fast sequence checking by key, but with
/// the thread-safe backing storage of `AtomicIncr`.
///
/// # Tradeoffs
///
/// `AtomicMap` is not a concurrent hashmap. Importantly **key insertion is not
/// synchronized**. The intended use case is to initialize the map fully on program start,
/// inserting whatever keys will be used throughout its life, and cloning this master
/// instance to be used by any threads tracking sequences by those keys.
///
/// A fully synchronized map was not chosen for performance reasons. If keys
/// are not fully known when threads are launched, the best options include:
///
/// - wrap a normal `Map` in an `Arc<Mutex<Map>>` or `Arc<RwLock<Map>>`,
/// - utilize a third-party crate providing a concurrent hashmap implementation
///   (with `Incr` values).
///
/// For a given (already inserted) key, any `clone()`d `AtomicMap` will use/have a
/// value at that key that *is* synchronized across threads (the inner value is
/// an `Arc<AtomicU64>`).
///
/// # Examples
///
/// ```
/// use incr::AtomicMap;
///
/// let mut last: AtomicMap<&'static str> = Default::default();
///
/// assert_eq!(last.insert("a", 1), true);
/// assert_eq!(last.is_new("missing_key", 1), false); // note difference from `Map`
/// assert_eq!(last.insert("b", 1), true);
/// assert_eq!(last.is_new("a", 1), false);
/// assert_eq!(last.is_new("b", 3), true);
/// assert_eq!(last.is_new_or_insert("c", 1), true);
/// assert_eq!(last.is_new("c", 1), false);
/// assert_eq!(last.get("b"), 3);
/// assert_eq!(last.get("not a key"), 0);
/// ```
///
#[derive(Default, Clone)]
pub struct AtomicMap<K: Eq + Hash>(HashMap<K, AtomicIncr>);

impl Incr {
    /// Returns `true` if `val` is greater than the highest previously observed
    /// value. If `val` is a new maximum, it is stored in `self` for checks against
    /// future values subsequent calls `Self::get(&self)` will return `val` until a
    /// new max is observed.
    ///
    pub fn is_new(&mut self, val: u64) -> bool {
        if val > self.0 {
            self.0 = val;
            true
        } else {
            false
        }
    }

    /// Returns the current maximum.
    pub fn get(&self) -> u64 { self.0 }
}

impl<K> Map<K>
    where K: Eq + Hash
{
    pub fn is_new(&mut self, k: K, val: u64) -> bool {
        let prev = self.0.entry(k).or_insert(0);
        if val > *prev {
            *prev = val;
            true
        } else {
            false
        }
    }

    /// Returns the current maximum.
    pub fn get(&self, k: &K) -> u64 {
        self.0.get(k).cloned().unwrap_or(0)
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.0.contains_key(k)
    }

    pub fn len(&self) -> usize { self.0.len() }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

impl RcIncr {
    /// Returns `true` if `val` is greater than the highest previously observed
    /// value. If `val` is a new maximum, it is stored in `self` for checks against
    /// future values subsequent calls `Self::get(&self)` will return `val` until a
    /// new max is observed.
    ///
    pub fn is_new(&self, val: u64) -> bool {
        if val > self.get() {
            self.0.set(val);
            true
        } else {
            false
        }
    }

    /// Returns the current maximum.
    pub fn get(&self) -> u64 { self.0.get() }
}

impl AtomicIncr {
    /// Returns `true` if `val` is greater than the highest previously observed
    /// value. If `val` is a new maximum, it is stored in `self` for checks against
    /// future values subsequent calls `Self::get(&self)` will return `val` until a
    /// new max is observed.
    ///
    pub fn is_new(&self, val: u64) -> bool {
        let mut gt = false;

        #[cfg(not(feature = "nightly"))]
        let val = val as usize;

        loop {
            let prev = self.0.load(Ordering::Acquire);
            if val > prev {
                if let Ok(_) = self.0.compare_exchange(prev, val, Ordering::AcqRel, Ordering::Acquire) {
                    gt = true;
                    break
                }
            } else {
                break
            }
        }
        gt
    }

    /// Returns the current maximum.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Acquire) as u64
    }

    /// Consumes the outer struct, returning the inner `Arc<Atomic>`.
    pub fn into_inner(self) -> Arc<Atomic> {
        self.0
    }
}

impl<K> AtomicMap<K>
    where K: Eq + Hash
{
    /// Returns `true` if `key` exists and `val` is greater than the largest
    /// previously observed value (for `key`). Returns `false` if `key` does
    /// not exist in the inner map. See `AtomicMap::check_or_insert` for a function
    /// that behaves similarly to `Map::is_new`.
    ///
    /// # Tradeoffs
    ///
    /// This function has a different signature and works differently than
    /// `Map::is_new`.
    ///
    /// Specifically, `Map::is_new`:
    ///
    /// - takes `&mut self`
    /// - consumes `key`
    /// - inserts `val` at `key` if `key` was not already present in the map.
    ///
    /// By contrast, `AtomicIncr`:
    ///
    /// - takes `&self`
    /// - borrows `&key`
    /// - does not insert `val` on a key miss, instead "silently" returning `false`
    ///
    /// This design was chosen for several reasons, including:
    ///
    /// - key insertions are not synchronized across threads. Instead, the map is
    ///   expected to have been initialized on program start, and a key miss is most
    ///   likely an error
    /// - A `&self` signature provides more flexibility, and is possible, unlike with
    ///   `Map`, because the `AtomicIncr::is_new` function takes `&self`
    ///
    /// The `AtomicMap::check_or_insert` function provides insert-on-key-miss
    /// functionality if desired.
    ///
    /// Possibly, it would be less confusing if this function returned `Option<bool>`,
    /// where a key miss would return `None`. Feedback on this issue would be
    /// appreciated.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate incr;
    /// use incr::AtomicMap;
    /// # fn main() {
    /// let mut last: AtomicMap<&'static str> = Default::default();
    ///
    /// assert_eq!(last.is_new("a", 1), false);
    /// assert_eq!(last.contains_key("a"), false);
    /// assert_eq!(last.is_new_or_insert("a", 1), true);
    /// assert_eq!(last.get("a"), 1);
    /// assert_eq!(last.insert("b", 1), true);
    /// assert_eq!(last.is_new("b", 2), true);
    /// assert_eq!(last.is_new("b", 2), false);
    /// # }
    /// ```
    ///
    pub fn is_new<Q>(&self, key: &Q, val: u64) -> bool
        where K: Borrow<Q>,
              Q: ?Sized + Hash + Eq
    {
        self.0.get(key)
            .map(move |x| x.is_new(val))
            .unwrap_or(false)
    }

    /// Like `is_new`, but inserts `val` at `key` if the inner map did not
    /// previously contain `key`.
    ///
    /// This may be renamed to `check_or_insert` in the future.
    pub fn is_new_or_insert(&mut self, key: K, val: u64) -> bool {
        self.0.entry(key)
            .or_insert_with(Default::default)
            .is_new(val)
    }

    /// An alias for, and Works identically to, `is_new_or_insert`. It's
    /// not possible, using the public api, to decrease the value at a given
    /// key, so calling this with a `val` lower than the current value
    /// would simply return `false` and leave the higher value in the map
    /// unchanged.
    pub fn insert(&mut self, key: K, val: u64) -> bool {
        self.is_new_or_insert(key, val)
    }

    /// Returns the highest observed value at `key`, or, if `key` does not exist,
    /// returns `0`.
    pub fn get<Q>(&self, key: &Q) -> u64
        where K: Borrow<Q>,
              Q: ?Sized + Hash + Eq
    {
        self.0.get(key)
            .map(|x| x.get())
            .unwrap_or(0)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
        where K: Borrow<Q>,
              Q: ?Sized + Hash + Eq
    {
        self.0.contains_key(key)
    }

    pub fn len(&self) -> usize { self.0.len() }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

impl PartialEq for AtomicIncr {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl PartialOrd for AtomicIncr {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.get().cmp(&other.get()))
    }
}

impl Eq for AtomicIncr {}

impl From<u64> for Incr {
    fn from(val: u64) -> Self {
        Incr(val)
    }
}

impl From<u64> for RcIncr {
    fn from(val: u64) -> Self {
        RcIncr(Rc::new(Cell::new(val)))
    }
}

impl From<u64> for AtomicIncr {
    fn from(val: u64) -> Self {
        AtomicIncr(Arc::new(Atomic::new(val)))
    }
}

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;
    #[cfg(feature = "nightly")]
    use test::Bencher;

    #[test]
    fn atomic_map_key_ergonomics() {
        let mut last: AtomicMap<String> = Default::default();
        let a = String::from("a");
        last.insert(a.clone(), 10);
        assert_eq!(last.get(&a), 10);

        let mut last: AtomicMap<&'static str> = Default::default();
        last.insert("a", 11);
        assert_eq!(last.get("a"), 11);

        let mut last: AtomicMap<u64> = Default::default();
        last.insert(1, 12);
        assert_eq!(last.get(&1), 12);
    }

    macro_rules! stairway_to_heaven {
        ($f:ident, $t:ident) => {
            #[test]
            fn $f() {
                let mut last: $t = Default::default();
                for i in 1..1_000_000u64 {
                    assert!(last.is_new(i), "i = {}", i);
                }
            }
        }
    }

    stairway_to_heaven!(all_true_to_one_million_incr, Incr);
    stairway_to_heaven!(all_true_to_one_million_rc_incr, RcIncr);
    stairway_to_heaven!(all_true_to_one_million_atomic_incr, AtomicIncr);

    macro_rules! purgatory {
        ($f:ident, $t:ident) => {
            #[test]
            fn $f() {
                let mut last: $t = Default::default();
                for _ in 1..1_000_000u64 {
                    assert!(!last.is_new(0), "i = {}", 0);
                }
            }
        }
    }

    purgatory!(never_true_one_million_times_incr, Incr);
    purgatory!(never_true_one_million_times_rc_incr, RcIncr);
    purgatory!(never_true_one_million_times_atomic_incr, AtomicIncr);

    macro_rules! stairway_to_heaven_bench {
        ($f:ident, $t:ident) => {
            #[cfg(feature = "nightly")]
            #[bench]
            fn $f(b: &mut Bencher) {
                let mut last: $t = Default::default();
                let mut i = 1;
                b.iter(|| {
                    i += 1;
                    last.is_new(i - 1)
                })
            }
        }
    }

    stairway_to_heaven_bench!(always_increasing_bench_incr, Incr);
    stairway_to_heaven_bench!(always_increasing_bench_rc_incr, RcIncr);
    stairway_to_heaven_bench!(always_increasing_bench_atomic_incr, AtomicIncr);

    macro_rules! purgatory_bench {
        ($f:ident, $t:ident) => {
            #[cfg(feature = "nightly")]
            #[bench]
            fn $f(b: &mut Bencher) {
                let mut last: $t = Default::default();
                b.iter(|| {
                    last.is_new(0)
                })
            }
        }
    }

    purgatory_bench!(never_incr_bench_incr, Incr);
    purgatory_bench!(never_incr_bench_rc_incr, RcIncr);
    purgatory_bench!(never_incr_bench_atomic_incr, AtomicIncr);

    #[cfg(feature = "nightly")]
    #[bench]
    fn atomic_incr_nightmare_scenario_except_threads_yield_each_iteration(b: &mut Bencher) {
        let n = 24;
        let stop = Arc::new(AtomicBool::new(false));
        let last: AtomicIncr = Default::default();
        let mut threads = Vec::new();
        for _ in 0..n {
            let val = last.clone().into_inner();
            let stop = Arc::clone(&stop);
            threads.push(thread::spawn(move || {
                loop {
                    val.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                    if stop.load(Ordering::Relaxed) { break }
                }
            }));
        }

        let mut i = 1;

        b.iter(|| {
            let is_new = last.is_new(i);
            i = match is_new {
                true => i + 1,
                false => i.max(last.get()),
            };
            is_new
        });
        stop.store(true, Ordering::SeqCst);
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn im_your_worst_nightmare(b: &mut Bencher) {
        let n = 24;
        let stop = Arc::new(AtomicBool::new(false));
        let last: AtomicIncr = Default::default();
        let mut threads = Vec::new();
        for _ in 0..n {
            let val = last.clone().into_inner();
            let stop = Arc::clone(&stop);
            threads.push(thread::spawn(move || {
                loop {
                    val.fetch_add(1, Ordering::Relaxed);
                    if stop.load(Ordering::Relaxed) { break }
                }
            }));
        }

        let mut i = 1;

        b.iter(|| {
            let is_new = last.is_new(i);
            i = match is_new {
                true => i + 1,
                false => i.max(last.get()),
            };
            is_new
        });
        stop.store(true, Ordering::SeqCst);
    }
}
