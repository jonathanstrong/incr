# incr

Simple, fast and self-contained structs for tracking whether a newly observed
value (`u64`) is greater than the largest previously observed value.

Use cases include message sequence numbers, timestamps, and other situations
that present a need to quickly assess whether incoming data is "new", i.e. its
numbering is larger than any previous value.

All of the structs include an `is_new` function that returns `true` if the
passed value is a new maximum, while simultaneously storing the new value to
check against future values.

Two of the `is_new` implementations (`Incr` and `Map`) require an `&mut self`
signature, while `RcIncr` and `AtomicIncr` require only `&self` due to `RcIncr`'s
interior mutability and `AtomicIncr`'s thread safe syncrhonization.

The cost of checking a new value is minimal: 0-2ns for the single-threaded
implementations, and ~5-10ns for `AtomicIncr`, except in cases of pathological
contention. In a worst-case, nightmare scenrio benchmark for the `AtomicIncr`,
it's possible to induce delays of hundreds of nanoseconds. A more realistic
case of 24 threads contending to increment the atomic but yielding each iteration
resulted in checks in the ~5-10ns range.

Enabling the "nightly" feature (on by default) allows the use of `AtomicU64`
as the backing storage for `AtomicIncr` (vs. `AtomicUsize` otherwise). Also,
nightly is required to run the benchmarks.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
incr = "0.1"
```

## Examples

Simple usage:

```rust
use incr::Incr;
let mut last = Incr::default();
assert_eq!(last.is_new(1), true);
assert_eq!(last.is_new(1), false);
assert_eq!(last.is_new(2), true);
assert_eq!(last.get(), 2);
```

`AtomicIncr` offers a threadsafe implementation:

```rust
#![cfg_attr(feature = "nightly", feature(integer_atomics))]
use std::sync::atomic::*;
use std::sync::Arc;
use std::thread;
#[cfg(feature = "nightly")]
use std::sync::atomic::AtomicU64;
#[cfg(not(feature = "nightly"))]
use std::sync::atomic::AtomicUsize;
use incr::AtomicIncr;

#[cfg(feature = "nightly")]
type Atomic = AtomicU64;
#[cfg(not(feature = "nightly"))]
type Atomic = AtomicUsize;

let stop = Arc::new(AtomicBool::new(false));
let last: AtomicIncr = Default::default();
let mut threads = Vec::new();
for _ in 0..5 {
    let val: Arc<Atomic> = last.clone().into_inner();
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

for _ in 0..100 {
    i = match last.is_new(i) {
        true => i + 1,
        false => i.max(last.get()),
    };
}
stop.store(true, Ordering::SeqCst);
```


