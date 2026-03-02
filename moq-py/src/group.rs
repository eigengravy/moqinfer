use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::{enter_runtime, moq_err};

/// Producer side of a MoQ group — write frames into it.
#[pyclass(module = "moq_py")]
pub struct MoqGroupProducer {
    pub(crate) inner: Arc<Mutex<moq_lite::GroupProducer>>,
}

#[pymethods]
impl MoqGroupProducer {
    /// Write a single frame (bytes) into this group.
    fn write_frame(&self, data: &[u8]) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard
            .write_frame(bytes::Bytes::copy_from_slice(data))
            .map_err(moq_err)
    }

    /// Mark the group as cleanly finished (no more frames).
    fn finish(&self) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard.finish().map_err(moq_err)
    }

    /// Abort the group with a cancellation error.
    fn close(&self) -> PyResult<()> {
        let _rt = enter_runtime();
        let mut guard = self.inner.lock().unwrap();
        guard.close(moq_lite::Error::Cancel).map_err(moq_err)
    }
}

/// Consumer side of a MoQ group — read frames from it.
#[pyclass(module = "moq_py")]
pub struct MoqGroupConsumer {
    pub(crate) inner: Arc<tokio::sync::Mutex<moq_lite::GroupConsumer>>,
}

#[pymethods]
impl MoqGroupConsumer {
    /// Read the next frame, returning bytes or None if the group is finished.
    fn read_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let consumer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = consumer.lock().await;
            let result = guard.read_frame().await.map_err(moq_err)?;
            Python::with_gil(|py| -> PyResult<PyObject> {
                match result {
                    Some(data) => Ok(PyBytes::new(py, &data).into_any().unbind()),
                    None => Ok(py.None()),
                }
            })
        })
    }
}
