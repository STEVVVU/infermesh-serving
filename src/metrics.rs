use std::fmt::Write as _;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

const LATENCY_BUCKETS_US: [u64; 10] = [
    100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 100_000, 1_000_000,
];

#[derive(Debug)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub request_errors_total: AtomicU64,
    pub admission_rejected_total: AtomicU64,
    pub deadline_exceeded_total: AtomicU64,
    pub batches_total: AtomicU64,
    pub batch_items_total: AtomicU64,
    pub safe_retries_total: AtomicU64,
    pub worker_failures_total: AtomicU64,
    pub cache_hits_total: AtomicU64,
    pub cache_misses_total: AtomicU64,
    pub in_flight: AtomicI64,
    latency_count: AtomicU64,
    latency_sum_us: AtomicU64,
    latency_buckets: [AtomicU64; LATENCY_BUCKETS_US.len()],
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            request_errors_total: AtomicU64::new(0),
            admission_rejected_total: AtomicU64::new(0),
            deadline_exceeded_total: AtomicU64::new(0),
            batches_total: AtomicU64::new(0),
            batch_items_total: AtomicU64::new(0),
            safe_retries_total: AtomicU64::new(0),
            worker_failures_total: AtomicU64::new(0),
            cache_hits_total: AtomicU64::new(0),
            cache_misses_total: AtomicU64::new(0),
            in_flight: AtomicI64::new(0),
            latency_count: AtomicU64::new(0),
            latency_sum_us: AtomicU64::new(0),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl Metrics {
    pub fn observe_latency(&self, latency_us: u64) {
        self.latency_count.fetch_add(1, Ordering::Relaxed);
        self.latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);
        for (boundary, bucket) in LATENCY_BUCKETS_US.iter().zip(&self.latency_buckets) {
            if latency_us <= *boundary {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[must_use]
    pub fn render_prometheus(&self, live_workers: usize) -> String {
        let counter = |name: &str, value: &AtomicU64| {
            format!(
                "# TYPE {name} counter\n{name} {}\n",
                value.load(Ordering::Relaxed)
            )
        };
        let mut output = String::new();
        output.push_str(&counter("infermesh_requests_total", &self.requests_total));
        output.push_str(&counter(
            "infermesh_request_errors_total",
            &self.request_errors_total,
        ));
        output.push_str(&counter(
            "infermesh_admission_rejected_total",
            &self.admission_rejected_total,
        ));
        output.push_str(&counter(
            "infermesh_deadline_exceeded_total",
            &self.deadline_exceeded_total,
        ));
        output.push_str(&counter("infermesh_batches_total", &self.batches_total));
        output.push_str(&counter(
            "infermesh_batch_items_total",
            &self.batch_items_total,
        ));
        output.push_str(&counter(
            "infermesh_safe_retries_total",
            &self.safe_retries_total,
        ));
        output.push_str(&counter(
            "infermesh_worker_failures_total",
            &self.worker_failures_total,
        ));
        output.push_str(&counter(
            "infermesh_kv_cache_hits_total",
            &self.cache_hits_total,
        ));
        output.push_str(&counter(
            "infermesh_kv_cache_misses_total",
            &self.cache_misses_total,
        ));
        output.push_str("# TYPE infermesh_in_flight gauge\n");
        let _ = writeln!(
            output,
            "infermesh_in_flight {}",
            self.in_flight.load(Ordering::Relaxed)
        );
        output.push_str("# TYPE infermesh_live_workers gauge\n");
        let _ = writeln!(output, "infermesh_live_workers {live_workers}");
        output.push_str("# TYPE infermesh_request_latency_us histogram\n");
        for (boundary, bucket) in LATENCY_BUCKETS_US.iter().zip(&self.latency_buckets) {
            let _ = writeln!(
                output,
                "infermesh_request_latency_us_bucket{{le=\"{boundary}\"}} {}",
                bucket.load(Ordering::Relaxed)
            );
        }
        let _ = writeln!(
            output,
            "infermesh_request_latency_us_bucket{{le=\"+Inf\"}} {}",
            self.latency_count.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "infermesh_request_latency_us_sum {}",
            self.latency_sum_us.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "infermesh_request_latency_us_count {}",
            self.latency_count.load(Ordering::Relaxed)
        );
        output
    }
}
