use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Convert a moq_lite::Error into a PyErr.
pub fn moq_err(e: moq_lite::Error) -> PyErr {
    match e {
        moq_lite::Error::Transport
        | moq_lite::Error::Closed
        | moq_lite::Error::Cancel
        | moq_lite::Error::Dropped => PyConnectionError::new_err(e.to_string()),

        moq_lite::Error::NotFound
        | moq_lite::Error::Version
        | moq_lite::Error::InvalidRole
        | moq_lite::Error::Duplicate => PyValueError::new_err(e.to_string()),

        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Convert an anyhow::Error into a PyErr.
pub fn anyhow_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Enter the tokio runtime context for sync methods that
/// internally spawn tokio tasks (e.g. moq-lite cleanup tasks).
pub fn enter_runtime() -> tokio::runtime::EnterGuard<'static> {
    pyo3_async_runtimes::tokio::get_runtime().enter()
}
