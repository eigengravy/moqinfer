mod broadcast;
mod client;
mod error;
mod group;
mod origin;
mod server;
mod track;

use pyo3::prelude::*;

/// MoQ (Media over QUIC) Python bindings.
///
/// Provides native MoQ transport capabilities via PyO3 bindings
/// to moq-lite and moq-native Rust crates.
#[pymodule]
fn moq_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Server types
    m.add_class::<server::MoqServerConfig>()?;
    m.add_class::<server::MoqServer>()?;
    m.add_class::<server::MoqRequest>()?;
    m.add_class::<server::MoqSession>()?;

    // Client types
    m.add_class::<client::MoqClientConfig>()?;
    m.add_class::<client::MoqClient>()?;

    // Origin types
    m.add_class::<origin::MoqOrigin>()?;
    m.add_class::<origin::MoqOriginProducer>()?;
    m.add_class::<origin::MoqOriginConsumer>()?;

    // Broadcast types
    m.add_class::<broadcast::MoqBroadcastProducer>()?;
    m.add_class::<broadcast::MoqBroadcastDynamic>()?;
    m.add_class::<broadcast::MoqBroadcastConsumer>()?;

    // Track types
    m.add_class::<track::MoqTrackProducer>()?;
    m.add_class::<track::MoqTrackConsumer>()?;

    // Group types
    m.add_class::<group::MoqGroupProducer>()?;
    m.add_class::<group::MoqGroupConsumer>()?;

    Ok(())
}
