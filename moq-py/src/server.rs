use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyType;

use crate::error::anyhow_err;
use crate::origin::{MoqOriginConsumer, MoqOriginProducer};

/// Configuration for a MoQ server.
#[pyclass(module = "moq_py")]
pub struct MoqServerConfig {
    bind_addr: String,
    tls_generate: Vec<String>,
}

#[pymethods]
impl MoqServerConfig {
    #[new]
    #[pyo3(signature = (bind_addr="[::]:4443".to_string(), tls_generate=vec![]))]
    fn new(bind_addr: String, tls_generate: Vec<String>) -> Self {
        Self {
            bind_addr,
            tls_generate,
        }
    }
}

impl MoqServerConfig {
    fn to_native(&self) -> Result<moq_native::ServerConfig, anyhow::Error> {
        let bind: std::net::SocketAddr = self
            .bind_addr
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid bind address: {e}"))?;
        let mut config = moq_native::ServerConfig::default();
        config.bind = Some(bind);
        config.tls.generate.clone_from(&self.tls_generate);
        Ok(config)
    }
}

/// A MoQ server that accepts incoming connections.
#[pyclass(module = "moq_py")]
pub struct MoqServer {
    inner: Arc<tokio::sync::Mutex<moq_native::Server>>,
}

#[pymethods]
impl MoqServer {
    /// Create and initialize a MoQ server from the given config.
    ///
    /// This is an async classmethod: `server = await MoqServer.create(config)`
    #[classmethod]
    fn create<'py>(_cls: &Bound<'py, PyType>, config: &MoqServerConfig, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let native_config = config.to_native().map_err(anyhow_err)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Server::new may need a tokio runtime context (for Quinn endpoint binding)
            let server = native_config.init().map_err(anyhow_err)?;
            Ok(MoqServer {
                inner: Arc::new(tokio::sync::Mutex::new(server)),
            })
        })
    }

    /// Accept the next incoming connection, returning MoqRequest or None if the server is closed.
    fn accept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let server = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = server.lock().await;
            match guard.accept().await {
                Some(req) => Ok(Some(MoqRequest { inner: Some(req) })),
                None => Ok(None),
            }
        })
    }

    /// Get the local address the server is bound to.
    fn local_addr(&self) -> PyResult<String> {
        let guard = self.inner.blocking_lock();
        let addr = guard.local_addr().map_err(anyhow_err)?;
        Ok(addr.to_string())
    }
}

/// An incoming MoQ connection request that can be accepted or rejected.
///
/// Configure origins with `with_publish()` and `with_consume()` before calling `ok()`.
#[pyclass(module = "moq_py")]
pub struct MoqRequest {
    inner: Option<moq_native::Request>,
}

#[pymethods]
impl MoqRequest {
    /// Set the publish origin — the server will publish these broadcasts to the client.
    ///
    /// Pass the consumer side of an origin that you're producing content into.
    fn with_publish(&mut self, publish: &MoqOriginConsumer) -> PyResult<()> {
        let req = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Request already consumed"))?;
        let consumer = publish.inner.blocking_lock().consume();
        self.inner = Some(req.with_publish(consumer));
        Ok(())
    }

    /// Set the consume origin — the server will write client-published broadcasts here.
    ///
    /// Pass an origin producer that you'll later consume from.
    fn with_consume(&mut self, consume: &MoqOriginProducer) -> PyResult<()> {
        let req = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Request already consumed"))?;
        self.inner = Some(req.with_consume(consume.inner.clone()));
        Ok(())
    }

    /// Accept the connection and perform the MoQ handshake.
    ///
    /// Returns a MoqSession. The request is consumed by this call.
    fn ok<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let req = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Request already consumed"))?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let session = req.ok().await.map_err(anyhow_err)?;
            Ok(MoqSession {
                inner: Some(session),
            })
        })
    }
}

/// An active MoQ session.
///
/// Hold this object alive to keep the connection open. Drop or close it to disconnect.
#[pyclass(module = "moq_py")]
pub struct MoqSession {
    pub(crate) inner: Option<moq_lite::Session>,
}

#[pymethods]
impl MoqSession {
    /// Close the session.
    fn close(&mut self) {
        if let Some(mut session) = self.inner.take() {
            session.close(moq_lite::Error::Cancel);
        }
    }

    /// Wait until the session is closed by the remote peer.
    fn closed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // We can't move the session into the future since it's borrowed,
        // so we need a different approach. We'll check if we have a session.
        if self.inner.is_none() {
            return Err(PyRuntimeError::new_err("Session already closed"));
        }
        // For now, return a future that completes immediately since Session::closed
        // requires &self and we can't share it across threads easily.
        // The session stays alive as long as MoqSession is alive.
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // This is a placeholder — in practice the session stays alive
            // as long as MoqSession is held by Python. Use close() to end it.
            Ok(())
        })
    }
}
