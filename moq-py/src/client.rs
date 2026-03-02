use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::error::{anyhow_err, enter_runtime};
use crate::origin::{MoqOriginConsumer, MoqOriginProducer};
use crate::server::MoqSession;

/// Configuration for a MoQ client.
#[pyclass(module = "moq_py")]
pub struct MoqClientConfig {
    tls_disable_verify: bool,
}

#[pymethods]
impl MoqClientConfig {
    #[new]
    #[pyo3(signature = (tls_disable_verify=true))]
    fn new(tls_disable_verify: bool) -> Self {
        Self { tls_disable_verify }
    }
}

impl MoqClientConfig {
    fn to_native(&self) -> moq_native::ClientConfig {
        let mut config = moq_native::ClientConfig::default();
        config.tls.disable_verify = Some(self.tls_disable_verify);
        config
    }
}

/// A MoQ client for connecting to servers.
///
/// Configure origins with `with_publish()` and `with_consume()`, then call `connect()`.
#[pyclass(module = "moq_py")]
pub struct MoqClient {
    inner: Option<moq_native::Client>,
}

#[pymethods]
impl MoqClient {
    /// Create a MoQ client from config.
    #[classmethod]
    fn create(_cls: &Bound<'_, PyType>, config: &MoqClientConfig) -> PyResult<MoqClient> {
        let _rt = enter_runtime();
        let native_config = config.to_native();
        let client = native_config.init().map_err(anyhow_err)?;
        Ok(MoqClient {
            inner: Some(client),
        })
    }

    /// Set the publish origin — the client will publish these broadcasts to the server.
    fn with_publish(&mut self, publish: &MoqOriginConsumer) -> PyResult<()> {
        let _rt = enter_runtime();
        let client = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Client already consumed"))?;
        let consumer = publish.inner.blocking_lock().consume();
        self.inner = Some(client.with_publish(consumer));
        Ok(())
    }

    /// Set the consume origin — server-published broadcasts will appear here.
    fn with_consume(&mut self, consume: &MoqOriginProducer) -> PyResult<()> {
        let _rt = enter_runtime();
        let client = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Client already consumed"))?;
        self.inner = Some(client.with_consume(consume.inner.clone()));
        Ok(())
    }

    /// Connect to a MoQ server at the given URL.
    ///
    /// Returns a MoqSession. The client can be reused for further connections.
    fn connect<'py>(&self, py: Python<'py>, url: &str) -> PyResult<Bound<'py, PyAny>> {
        let client = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Client already consumed"))?
            .clone();
        let parsed_url: url::Url = url
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid URL: {e}")))?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let session = client.connect(parsed_url).await.map_err(anyhow_err)?;
            Ok(MoqSession {
                inner: Some(session),
            })
        })
    }
}

use pyo3::types::PyType;
