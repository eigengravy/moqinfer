use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::error::{enter_runtime, moq_err};
use crate::track::{MoqTrackConsumer, MoqTrackProducer};

/// Producer side of a MoQ broadcast — create tracks.
#[pyclass(module = "moq_py")]
pub struct MoqBroadcastProducer {
    pub(crate) inner: Arc<Mutex<moq_lite::BroadcastProducer>>,
}

#[pymethods]
impl MoqBroadcastProducer {
    /// Create a named track within this broadcast.
    fn create_track(&self, name: &str) -> PyResult<MoqTrackProducer> {
        let _rt = enter_runtime();
        let track = moq_lite::Track::new(name);
        let mut guard = self.inner.lock().unwrap();
        let producer = guard.create_track(track).map_err(moq_err)?;
        Ok(MoqTrackProducer {
            inner: Arc::new(Mutex::new(producer)),
        })
    }

    /// Create a dynamic track handler that receives track requests from subscribers.
    fn dynamic(&self) -> MoqBroadcastDynamic {
        let _rt = enter_runtime();
        let guard = self.inner.lock().unwrap();
        let dynamic = guard.dynamic();
        MoqBroadcastDynamic {
            inner: Arc::new(tokio::sync::Mutex::new(dynamic)),
        }
    }

    /// Create a consumer for this broadcast.
    fn consume(&self) -> MoqBroadcastConsumer {
        let _rt = enter_runtime();
        let guard = self.inner.lock().unwrap();
        MoqBroadcastConsumer {
            inner: guard.consume(),
        }
    }

    /// Close the broadcast with a cancellation error, cascading to all tracks.
    fn close(&self) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard.close(moq_lite::Error::Cancel).map_err(moq_err)
    }
}

/// Dynamic track handler — receives track subscription requests from consumers.
#[pyclass(module = "moq_py")]
pub struct MoqBroadcastDynamic {
    pub(crate) inner: Arc<tokio::sync::Mutex<moq_lite::BroadcastDynamic>>,
}

#[pymethods]
impl MoqBroadcastDynamic {
    /// Wait for a track subscription request, returning MoqTrackProducer or None.
    fn requested_track<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dynamic = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = dynamic.lock().await;
            let result = guard.requested_track().await.map_err(moq_err)?;
            Ok(result.map(|t| MoqTrackProducer {
                inner: Arc::new(std::sync::Mutex::new(t)),
            }))
        })
    }

    /// Create a consumer for the underlying broadcast.
    fn consume(&self) -> MoqBroadcastConsumer {
        let _rt = enter_runtime();
        let guard = self.inner.blocking_lock();
        MoqBroadcastConsumer {
            inner: guard.consume(),
        }
    }
}

/// Consumer side of a MoQ broadcast — subscribe to tracks.
#[pyclass(module = "moq_py")]
pub struct MoqBroadcastConsumer {
    pub(crate) inner: moq_lite::BroadcastConsumer,
}

#[pymethods]
impl MoqBroadcastConsumer {
    /// Subscribe to a track by name, returning a MoqTrackConsumer.
    fn subscribe_track(&self, name: &str) -> PyResult<MoqTrackConsumer> {
        let _rt = enter_runtime();
        let track = moq_lite::Track::new(name);
        let consumer = self.inner.subscribe_track(&track).map_err(moq_err)?;
        Ok(MoqTrackConsumer {
            inner: Arc::new(tokio::sync::Mutex::new(consumer)),
        })
    }
}
