use std::sync::Mutex;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

const LATENCY_BUCKETS_US: [f64; 10] = [
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    25_000.0,
    100_000.0,
    1_000_000.0,
];

/// Control-plane metrics, exported in the Prometheus text exposition format.
///
/// `prometheus-client` owns the encoding and appends the `_total` suffix to
/// counters itself, so counters are registered under their unsuffixed name to
/// keep the exported series names identical to earlier revisions.
///
/// Every metric handle is cheap to clone and internally atomic, so instrumented
/// code paths need no lock. The `Registry` is only locked while rendering.
#[derive(Debug)]
pub struct Metrics {
    pub requests_total: Counter<u64>,
    pub request_errors_total: Counter<u64>,
    pub admission_rejected_total: Counter<u64>,
    pub deadline_exceeded_total: Counter<u64>,
    pub batches_total: Counter<u64>,
    pub batch_items_total: Counter<u64>,
    pub safe_retries_total: Counter<u64>,
    pub worker_failures_total: Counter<u64>,
    pub cache_hits_total: Counter<u64>,
    pub cache_misses_total: Counter<u64>,
    /// Queued requests dropped because the caller stopped awaiting the result.
    ///
    /// Only observable once requests are async tasks rather than blocked
    /// threads, so this series is new in the tokio revision.
    pub client_cancelled_total: Counter<u64>,
    pub in_flight: Gauge<i64>,
    live_workers: Gauge<i64>,
    latency_us: Histogram,
    registry: Mutex<Registry>,
}

/// Registers a default-constructed metric and hands back a handle to it.
///
/// Registering and binding in one step keeps each series named exactly once.
fn registered<M>(registry: &mut Registry, name: &str, help: &str) -> M
where
    M: Clone + Default + prometheus_client::registry::Metric,
{
    let metric = M::default();
    registry.register(name, help, metric.clone());
    metric
}

impl Default for Metrics {
    fn default() -> Self {
        let mut registry = Registry::with_prefix("infermesh");
        let latency_us = Histogram::new(LATENCY_BUCKETS_US);
        registry.register(
            "request_latency_us",
            "End-to-end scheduler latency in microseconds",
            latency_us.clone(),
        );

        Self {
            requests_total: registered(
                &mut registry,
                "requests",
                "Inference requests accepted by the scheduler",
            ),
            request_errors_total: registered(
                &mut registry,
                "request_errors",
                "Inference requests that returned an error",
            ),
            admission_rejected_total: registered(
                &mut registry,
                "admission_rejected",
                "Requests rejected because a model admission queue was full",
            ),
            deadline_exceeded_total: registered(
                &mut registry,
                "deadline_exceeded",
                "Requests whose deadline elapsed while queued or executing",
            ),
            batches_total: registered(
                &mut registry,
                "batches",
                "Dynamic batches dispatched to a worker",
            ),
            batch_items_total: registered(
                &mut registry,
                "batch_items",
                "Requests contained in dispatched dynamic batches",
            ),
            safe_retries_total: registered(
                &mut registry,
                "safe_retries",
                "Retries performed after a proven pre-execution failure",
            ),
            worker_failures_total: registered(
                &mut registry,
                "worker_failures",
                "Engine execution failures reported by workers",
            ),
            cache_hits_total: registered(
                &mut registry,
                "kv_cache_hits",
                "Prefix-cache acquisitions that reused cached tokens",
            ),
            cache_misses_total: registered(
                &mut registry,
                "kv_cache_misses",
                "Prefix-cache acquisitions that reused no cached tokens",
            ),
            client_cancelled_total: registered(
                &mut registry,
                "client_cancelled",
                "Queued requests dropped because the caller stopped awaiting them",
            ),
            in_flight: registered(
                &mut registry,
                "in_flight",
                "Requests currently owned by the scheduler",
            ),
            live_workers: registered(
                &mut registry,
                "live_workers",
                "Workers holding an unexpired heartbeat lease",
            ),
            latency_us,
            registry: Mutex::new(registry),
        }
    }
}

impl Metrics {
    pub fn observe_latency(&self, latency_us: u64) {
        // Latencies far exceed f64's exact-integer range only past ~285,000
        // years, so the cast is lossless for any observable measurement.
        #[allow(clippy::cast_precision_loss)]
        self.latency_us.observe(latency_us as f64);
    }

    /// Renders the Prometheus text exposition for the current metric values.
    ///
    /// `live_workers` is a point-in-time property of the worker registry rather
    /// than an accumulated counter, so the caller supplies it at scrape time.
    #[must_use]
    pub fn render_prometheus(&self, live_workers: usize) -> String {
        self.live_workers
            .set(i64::try_from(live_workers).unwrap_or(i64::MAX));
        let mut output = String::new();
        let registry = self
            .registry
            .lock()
            .expect("metrics registry lock poisoned");
        // Encoding into a String is infallible; the Result exists for the
        // generic `Write` bound.
        let _ = encode(&mut output, &registry);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::Metrics;

    #[test]
    fn exposition_keeps_stable_series_names_and_histogram_buckets() {
        let metrics = Metrics::default();
        metrics.requests_total.inc();
        metrics.batch_items_total.inc_by(3);
        metrics.in_flight.set(2);
        metrics.observe_latency(300);
        let rendered = metrics.render_prometheus(4);

        // Wire-visible series names must not drift; dashboards depend on them.
        assert!(rendered.contains("\ninfermesh_requests_total 1\n"));
        assert!(rendered.contains("\ninfermesh_batch_items_total 3\n"));
        assert!(rendered.contains("\ninfermesh_in_flight 2\n"));
        assert!(rendered.contains("\ninfermesh_live_workers 4\n"));
        assert!(rendered.contains("\ninfermesh_request_latency_us_count 1\n"));
        // 300us falls above the 250us boundary and below the 500us boundary.
        assert!(rendered.contains("infermesh_request_latency_us_bucket{le=\"250.0\"} 0"));
        assert!(rendered.contains("infermesh_request_latency_us_bucket{le=\"500.0\"} 1"));
        // Per OpenMetrics, the TYPE line names the metric *family*, so it
        // carries no `_total` suffix while the sample line does. This differs
        // from the hand-rolled exposition this replaced, which suffixed both.
        assert!(rendered.contains("# TYPE infermesh_requests counter"));
        assert!(rendered.contains("# HELP infermesh_requests "));
    }

    #[test]
    fn counters_start_at_zero_and_are_independently_addressable() {
        let metrics = Metrics::default();
        assert_eq!(metrics.requests_total.get(), 0);
        metrics.admission_rejected_total.inc();
        assert_eq!(metrics.admission_rejected_total.get(), 1);
        assert_eq!(metrics.deadline_exceeded_total.get(), 0);
    }
}
