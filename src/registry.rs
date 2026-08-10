use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::repository::ModelKey;
use crate::{Error, ErrorCode, Result};

pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

#[derive(Debug)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

#[derive(Clone, Debug)]
pub struct WorkerRegistration {
    pub id: String,
    pub capacity: usize,
    pub models: BTreeSet<ModelKey>,
}

#[derive(Clone, Debug)]
struct WorkerState {
    registration: WorkerRegistration,
    in_flight: usize,
    lease_expires_ms: u64,
    cached_prefixes: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerSelection {
    pub id: String,
    pub cache_affinity: bool,
    pub in_flight: usize,
    pub capacity: usize,
}

pub struct WorkerRegistry {
    clock: Arc<dyn Clock>,
    lease_ms: u64,
    workers: Mutex<BTreeMap<String, WorkerState>>,
}

impl WorkerRegistry {
    #[must_use]
    pub fn new(lease_ms: u64) -> Self {
        Self::with_clock(lease_ms, Arc::new(SystemClock))
    }

    #[must_use]
    pub fn with_clock(lease_ms: u64, clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            lease_ms,
            workers: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn register(&self, registration: WorkerRegistration) -> Result<()> {
        if registration.id.is_empty() || registration.capacity == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "worker id and positive capacity are required",
            ));
        }
        let now = self.clock.now_ms();
        self.workers.lock().expect("registry lock poisoned").insert(
            registration.id.clone(),
            WorkerState {
                registration,
                in_flight: 0,
                lease_expires_ms: now.saturating_add(self.lease_ms),
                cached_prefixes: BTreeSet::new(),
            },
        );
        Ok(())
    }

    pub fn heartbeat(
        &self,
        worker_id: &str,
        cached_prefixes: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let now = self.clock.now_ms();
        let mut workers = self.workers.lock().expect("registry lock poisoned");
        let worker = workers.get_mut(worker_id).ok_or_else(|| {
            Error::new(
                ErrorCode::NotFound,
                format!("worker {worker_id} is not registered"),
            )
        })?;
        worker.lease_expires_ms = now.saturating_add(self.lease_ms);
        worker.cached_prefixes = cached_prefixes.into_iter().collect();
        Ok(())
    }

    pub fn reserve(
        &self,
        model: &ModelKey,
        prefix_key: Option<&str>,
        excluded: &BTreeSet<String>,
    ) -> Result<WorkerSelection> {
        let now = self.clock.now_ms();
        let mut workers = self.workers.lock().expect("registry lock poisoned");
        let selected_id = workers
            .iter()
            .filter(|(id, worker)| {
                !excluded.contains(*id)
                    && worker.lease_expires_ms > now
                    && worker.registration.models.contains(model)
                    && worker.in_flight < worker.registration.capacity
            })
            .min_by(|(left_id, left), (right_id, right)| {
                let left_affinity =
                    prefix_key.is_some_and(|key| left.cached_prefixes.contains(key));
                let right_affinity =
                    prefix_key.is_some_and(|key| right.cached_prefixes.contains(key));
                right_affinity
                    .cmp(&left_affinity)
                    .then_with(|| {
                        (left.in_flight * right.registration.capacity)
                            .cmp(&(right.in_flight * left.registration.capacity))
                    })
                    .then_with(|| left_id.cmp(right_id))
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::Unavailable,
                    format!("no capacity for {}:{}", model.name, model.version),
                )
            })?;
        let worker = workers
            .get_mut(&selected_id)
            .expect("selected worker still exists");
        let selection = WorkerSelection {
            id: selected_id,
            cache_affinity: prefix_key.is_some_and(|key| worker.cached_prefixes.contains(key)),
            in_flight: worker.in_flight,
            capacity: worker.registration.capacity,
        };
        worker.in_flight += 1;
        Ok(selection)
    }

    pub fn release(&self, worker_id: &str) {
        if let Some(worker) = self
            .workers
            .lock()
            .expect("registry lock poisoned")
            .get_mut(worker_id)
        {
            worker.in_flight = worker.in_flight.saturating_sub(1);
        }
    }

    pub fn note_cached_prefix(&self, worker_id: &str, prefix_key: String) {
        if let Some(worker) = self
            .workers
            .lock()
            .expect("registry lock poisoned")
            .get_mut(worker_id)
        {
            worker.cached_prefixes.insert(prefix_key);
        }
    }

    #[must_use]
    pub fn live_workers(&self) -> usize {
        let now = self.clock.now_ms();
        self.workers
            .lock()
            .expect("registry lock poisoned")
            .values()
            .filter(|worker| worker.lease_expires_ms > now)
            .count()
    }

    pub fn expire(&self) -> Vec<String> {
        let now = self.clock.now_ms();
        let mut workers = self.workers.lock().expect("registry lock poisoned");
        let expired = workers
            .iter()
            .filter(|(_, worker)| worker.lease_expires_ms <= now)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in &expired {
            workers.remove(id);
        }
        expired
    }
}

#[cfg(test)]
pub mod test_clock {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::Clock;

    #[derive(Debug)]
    pub struct ManualClock(pub AtomicU64);

    impl ManualClock {
        #[must_use]
        pub fn new(now_ms: u64) -> Self {
            Self(AtomicU64::new(now_ms))
        }

        pub fn advance(&self, milliseconds: u64) {
            self.0.fetch_add(milliseconds, Ordering::SeqCst);
        }
    }

    impl Clock for ManualClock {
        fn now_ms(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::test_clock::ManualClock;
    use super::{WorkerRegistration, WorkerRegistry};
    use crate::repository::ModelKey;

    fn model() -> ModelKey {
        ModelKey {
            name: "decoder".to_owned(),
            version: "1".to_owned(),
        }
    }

    #[test]
    fn routes_by_affinity_then_normalized_load() {
        let clock = Arc::new(ManualClock::new(100));
        let registry = WorkerRegistry::with_clock(50, clock);
        for (id, capacity) in [("a", 2), ("b", 8)] {
            registry
                .register(WorkerRegistration {
                    id: id.to_owned(),
                    capacity,
                    models: BTreeSet::from([model()]),
                })
                .unwrap();
        }
        registry.heartbeat("a", ["prefix-1".to_owned()]).unwrap();
        let selected = registry
            .reserve(&model(), Some("prefix-1"), &BTreeSet::new())
            .unwrap();
        assert_eq!(selected.id, "a");
        registry.release("a");

        registry.reserve(&model(), None, &BTreeSet::new()).unwrap();
        let selected = registry.reserve(&model(), None, &BTreeSet::new()).unwrap();
        assert_eq!(selected.id, "b");
    }

    #[test]
    fn lease_expiry_removes_worker_from_routing() {
        let clock = Arc::new(ManualClock::new(100));
        let registry = WorkerRegistry::with_clock(10, Arc::clone(&clock) as Arc<dyn super::Clock>);
        registry
            .register(WorkerRegistration {
                id: "lost".to_owned(),
                capacity: 1,
                models: BTreeSet::from([model()]),
            })
            .unwrap();
        clock.advance(10);
        assert!(registry.reserve(&model(), None, &BTreeSet::new()).is_err());
        assert_eq!(registry.expire(), vec!["lost"]);
    }
}
