/* The runtime a ρ kernel is linked with: a pool of threads that a sweep is
 * split across. One cell is one expression, so a sweep has no order and any
 * part of it can run on any thread; the only decision is how many parts.
 *
 * The pool is made on the first sweep, with RHO_THREADS threads when that is
 * set, else the count the kernel was compiled with, else one per online CPU.
 * Workers spin briefly for the next sweep and then sleep, so a kernel that is
 * called in a tight loop pays no wake-up and one that is idle costs nothing.
 * A sweep shorter than the grain — what the kernel was compiled with, 16384
 * cells unless said otherwise, or RHO_GRAIN — runs on the caller alone.
 */
#if defined(_WIN32)

typedef void (*rho_part_fn)(long lo, long hi, long part);
static long rt_parts = 1;
void rho_rt_run(rho_part_fn fn, long n, long threads, long grain) {
    (void)threads;
    (void)grain;
    rt_parts = 1;
    if (n > 0) fn(0, n, 0);
}
long rho_rt_parts(void) { return rt_parts; }

#else

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

typedef void (*rho_part_fn)(long lo, long hi, long part);

#define RT_MAX_THREADS 256
#define RT_SPINS 50000

static long rt_threads;          /* 0 until the count is settled */
static long rt_grain = 16384;
static pthread_t rt_workers[RT_MAX_THREADS];
static long rt_made;             /* workers running */
static atomic_int rt_quit;       /* set when the kernel is being unloaded */
static pthread_mutex_t rt_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t rt_cv = PTHREAD_COND_INITIALIZER;
/* The sweep being run, published as one word so that a worker's view of it
 * is consistent: generation << 32 | parts << 16 | next part to take. A
 * worker that read an older generation finds its tag gone and takes nothing;
 * one that read this generation cannot take a part past this generation's
 * count. What the word does not carry is written before it is published and
 * not touched again until every part of it has finished. */
static atomic_ulong rt_word;
static atomic_long rt_gen;       /* what sleeping workers wait on */
static atomic_long rt_remaining; /* parts of the current sweep not yet done */
static rho_part_fn rt_fn;
static long rt_n, rt_parts, rt_chunk;

static void rt_take_parts(unsigned long gen) {
    for (;;) {
        unsigned long w = atomic_load_explicit(&rt_word, memory_order_acquire);
        if ((w >> 32) != gen) return;
        unsigned long parts = (w >> 16) & 0xffff;
        unsigned long next = w & 0xffff;
        if (next >= parts) return;
        if (!atomic_compare_exchange_weak_explicit(&rt_word, &w, w + 1,
                                                   memory_order_acq_rel,
                                                   memory_order_acquire))
            continue;
        long p = (long)next;
        long lo = p * rt_chunk;
        long hi = lo + rt_chunk;
        if (hi > rt_n) hi = rt_n;
        if (lo < hi) rt_fn(lo, hi, p);
        atomic_fetch_sub_explicit(&rt_remaining, 1, memory_order_acq_rel);
    }
}

static void *rt_worker(void *arg) {
    (void)arg;
    long seen = 0;
    for (;;) {
        long g;
        int spins = 0;
        while ((g = atomic_load_explicit(&rt_gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&rt_quit, memory_order_acquire)) return 0;
            if (++spins < RT_SPINS) {
                sched_yield();
                continue;
            }
            pthread_mutex_lock(&rt_mu);
            while (atomic_load_explicit(&rt_gen, memory_order_acquire) == seen &&
                   !atomic_load_explicit(&rt_quit, memory_order_acquire))
                pthread_cond_wait(&rt_cv, &rt_mu);
            pthread_mutex_unlock(&rt_mu);
        }
        seen = g;
        rt_take_parts((unsigned long)g);
    }
    return 0;
}

/* The kernel is being unloaded (or the process is ending): no worker may
 * outlive its code. Wake them all, tell them to leave, and wait. */
__attribute__((destructor)) static void rt_stop(void) {
    if (rt_made == 0) return;
    pthread_mutex_lock(&rt_mu);
    atomic_store_explicit(&rt_quit, 1, memory_order_release);
    pthread_cond_broadcast(&rt_cv);
    pthread_mutex_unlock(&rt_mu);
    for (long i = 0; i < rt_made; i++) pthread_join(rt_workers[i], 0);
    rt_made = 0;
}

static long rt_env(const char *name, long fallback) {
    const char *s = getenv(name);
    if (!s || !*s) return fallback;
    long v = strtol(s, 0, 10);
    return v > 0 ? v : fallback;
}

/* Settle how many threads to use; the workers are made when first needed. */
static void rt_settle(long compiled, long grain) {
    long n = rt_env("RHO_THREADS", compiled);
    if (n <= 0) n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) n = 1;
    if (n > RT_MAX_THREADS) n = RT_MAX_THREADS;
    rt_grain = rt_env("RHO_GRAIN", grain > 0 ? grain : rt_grain);
    rt_threads = n;
}

static void rt_start(void) {
    for (long i = 1; i < rt_threads; i++) {
        if (pthread_create(&rt_workers[rt_made], 0, rt_worker, 0) != 0) break;
        rt_made++;
    }
    rt_threads = rt_made + 1; /* as many as could be made */
}

/* Run fn over [0, n) in parts, on the pool and the calling thread. Returns
 * when every part has finished. */
void rho_rt_run(rho_part_fn fn, long n, long threads, long grain) {
    if (n <= 0) {
        rt_parts = 0;
        return;
    }
    if (rt_threads == 0) rt_settle(threads, grain);
    if (rt_threads == 1 || n < rt_grain) {
        rt_parts = 1;
        fn(0, n, 0);
        return;
    }
    if (rt_made == 0) rt_start();
    if (rt_threads == 1) {
        rt_parts = 1;
        fn(0, n, 0);
        return;
    }
    /* Parts of a whole number of 64-cell lines: a boundary that is a
     * multiple of the vector width keeps most of every part in its body. */
    long chunk = (n + rt_threads - 1) / rt_threads;
    chunk = (chunk + 63) & ~63L;
    long parts = (n + chunk - 1) / chunk;
    rt_fn = fn;
    rt_n = n;
    rt_chunk = chunk;
    rt_parts = parts;
    unsigned long gen = (unsigned long)atomic_load_explicit(&rt_gen, memory_order_relaxed) + 1;
    atomic_store_explicit(&rt_remaining, parts, memory_order_release);
    atomic_store_explicit(&rt_word, (gen << 32) | ((unsigned long)parts << 16),
                          memory_order_release);
    pthread_mutex_lock(&rt_mu);
    atomic_store_explicit(&rt_gen, (long)gen, memory_order_release);
    pthread_cond_broadcast(&rt_cv);
    pthread_mutex_unlock(&rt_mu);
    rt_take_parts(gen);
    while (atomic_load_explicit(&rt_remaining, memory_order_acquire) != 0)
        sched_yield();
}

/* How many parts the last run was split into: the partial results a sweep
 * left behind are indexed by part. */
long rho_rt_parts(void) { return rt_parts; }

#endif
