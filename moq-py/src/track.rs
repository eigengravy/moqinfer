use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::error::{enter_runtime, moq_err};
use crate::group::{MoqGroupConsumer, MoqGroupProducer};

/// Producer side of a MoQ track — create groups and write frames.
#[pyclass(module = "moq_py")]
pub struct MoqTrackProducer {
    pub(crate) inner: Arc<Mutex<moq_lite::TrackProducer>>,
}

#[pymethods]
impl MoqTrackProducer {
    /// Create a new group with an auto-incremented sequence number.
    fn append_group(&self) -> PyResult<MoqGroupProducer> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        let group = guard.append_group().map_err(moq_err)?;
        Ok(MoqGroupProducer {
            inner: Arc::new(Mutex::new(group)),
        })
    }

    /// Convenience: write a single frame in its own group.
    fn write_frame(&self, data: &[u8]) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard
            .write_frame(bytes::Bytes::copy_from_slice(data))
            .map_err(moq_err)
    }

    /// Mark the track as finished (no more groups will be created).
    fn finish(&self) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard.finish().map_err(moq_err)
    }

    /// Close the track with a cancellation error.
    fn close(&self) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard.close(moq_lite::Error::Cancel).map_err(moq_err)
    }

    /// The track name.
    #[getter]
    fn name(&self) -> String {
        let guard = self.inner.lock().unwrap();
        guard.info.name.clone()
    }
}

/// Consumer side of a MoQ track — receive groups.
#[pyclass(module = "moq_py")]
pub struct MoqTrackConsumer {
    pub(crate) inner: Arc<tokio::sync::Mutex<moq_lite::TrackConsumer>>,
}

#[pymethods]
impl MoqTrackConsumer {
    /// Wait for the next group, returning MoqGroupConsumer or None if the track is finished.
    fn next_group<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let consumer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = consumer.lock().await;
            let result = guard.next_group().await.map_err(moq_err)?;
            Ok(result.map(|g| MoqGroupConsumer {
                inner: Arc::new(tokio::sync::Mutex::new(g)),
            }))
        })
    }
}
