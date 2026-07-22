#![cfg_attr(feature = "nightly-simd", feature(portable_simd))]
#![recursion_limit = "256"]

mod bot;
mod cli;
mod configuration;
mod image;
mod ocr_space;

use anyhow::Context as _;
use anyhow::Result;
use configuration::app::{Dial9TelemetryConfig, load_config};
use dial9_tokio_telemetry::tracing_layer::Dial9TokioLayer;
use dial9_tokio_telemetry::{
    TracedRuntime,
    telemetry::{RotatingWriter, TelemetryGuard},
};
use std::{path::PathBuf, time::Duration};
use tracing_subscriber::{EnvFilter, filter::Targets, fmt, prelude::*};

fn main() -> Result<()> {
    let _tracing_guard = init_tracing();
    if cli::handle_config_free_cli()? {
        return Ok(());
    }
    let config = load_config().context("loading app config for runtime initialization")?;
    let (runtime, telemetry_guard) = build_traced_runtime(&config.telemetry.dial9)?;
    let dial9_shutdown_timeout =
        Duration::from_secs(config.telemetry.dial9.shutdown_timeout_seconds);

    let result = runtime.block_on(async {
        if cli::handle_standalone_cli(&config).await? {
            return Ok(());
        }

        bot::runtime::run_bot(config).await
    });
    drop(runtime);

    if let Err(source) = telemetry_guard.graceful_shutdown(dial9_shutdown_timeout) {
        tracing::warn!(
            event = "telemetry.dial9_shutdown_failed",
            ?source,
            timeout_ms = dial9_shutdown_timeout.as_millis(),
            "failed to gracefully flush Dial9 telemetry"
        );
    }

    result
}

fn init_tracing() -> tracing_appender::non_blocking::WorkerGuard {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("discord_sightline=info,info"));
    let (writer, guard) = tracing_appender::non_blocking(std::io::stdout());
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(writer).json())
        .with(Dial9TokioLayer::new().with_filter(
            Targets::new().with_target("discord_sightline::bot::worker", tracing::Level::INFO),
        ))
        .init();
    guard
}

fn build_traced_runtime(
    config: &Dial9TelemetryConfig,
) -> Result<(tokio::runtime::Runtime, TelemetryGuard)> {
    let mut tokio_builder = tokio::runtime::Builder::new_multi_thread();
    tokio_builder.enable_all();
    if !config.enabled {
        return TracedRuntime::build_disabled(tokio_builder)
            .context("building Tokio runtime with Dial9 disabled");
    }

    let runtime_name = config.runtime_name.clone();
    let trace_dir = PathBuf::from(&config.trace_dir);
    std::fs::create_dir_all(&trace_dir)
        .with_context(|| format!("creating Dial9 trace directory {}", trace_dir.display()))?;
    let trace_path = trace_dir.join("trace.bin");
    let writer = RotatingWriter::builder()
        .base_path(trace_path.clone())
        .max_total_size(config.max_disk_usage_mb.saturating_mul(1024 * 1024))
        .rotation_period(Duration::from_secs(config.rotation_seconds))
        .build()
        .with_context(|| format!("building Dial9 trace writer at {}", trace_path.display()))?;
    TracedRuntime::builder()
        .with_trace_path(trace_path)
        .with_runtime_name(runtime_name)
        .with_task_tracking(true)
        .build_and_start(tokio_builder, writer)
        .context("building Tokio runtime with Dial9 telemetry")
}
