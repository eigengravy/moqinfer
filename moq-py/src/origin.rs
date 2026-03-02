use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::broadcast::{MoqBroadcastConsumer, MoqBroadcastProducer};
use crate::error::enter_runtime;

/// Factory for creating MoQ origins.
#[pyclass(module = "moq_py")]
pub struct MoqOrigin;

#[pymethods]
impl MoqOrigin {
    /// Create a new origin, returning the producer side.
    #[staticmethod]
    fn produce() -> MoqOriginProducer {
        let _rt = enter_runtime();
        MoqOriginProducer {
            inner: moq_lite::OriginProducer::new(),
        }
    }
}

/// Producer side of a MoQ origin — create broadcasts and derive consumers.
#[pyclass(module = "moq_py")]
pub struct MoqOriginProducer {
    pub(crate) inner: moq_lite::OriginProducer,
}

#[pymethods]
impl MoqOriginProducer {
    /// Create a broadcast at the given path.
    ///
    /// Returns a MoqBroadcastProducer, or raises ValueError if the path is not allowed.
    fn create_broadcast(&self, path: &str) -> PyResult<MoqBroadcastProducer> {
        let _rt = enter_runtime();
        self.inner.create_broadcast(path).map_or_else(
            || {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "failed to create broadcast (path not allowed or duplicate)",
                ))
            },
            |bp| {
                Ok(MoqBroadcastProducer {
                    inner: Arc::new(Mutex::new(bp)),
                })
            },
        )
    }

    /// Create a consumer for this origin (used to subscribe to all broadcasts).
    fn consume(&self) -> MoqOriginConsumer {
        let _rt = enter_runtime();
        MoqOriginConsumer {
            inner: Arc::new(tokio::sync::Mutex::new(self.inner.consume())),
        }
    }
}

/// Consumer side of a MoQ origin — watch for broadcast announcements.
#[pyclass(module = "moq_py")]
pub struct MoqOriginConsumer {
    pub(crate) inner: Arc<tokio::sync::Mutex<moq_lite::OriginConsumer>>,
}

#[pymethods]
impl MoqOriginConsumer {
    /// Wait for the next broadcast announcement.
    ///
    /// Returns a tuple (path: str, broadcast: Optional[MoqBroadcastConsumer])
    /// where broadcast is Some for new announcements, None for unannouncements.
    /// Returns None when the origin is closed.
    fn announced<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let consumer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = consumer.lock().await;
            match guard.announced().await {
                Some((path, Some(broadcast))) => Ok(Some((
                    path.as_str().to_string(),
                    Some(MoqBroadcastConsumer { inner: broadcast }),
                ))),
                Some((path, None)) => Ok(Some((
                    path.as_str().to_string(),
                    Option::<MoqBroadcastConsumer>::None,
                ))),
                None => Ok(None),
            }
        })
    }
}
