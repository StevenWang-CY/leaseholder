//! `lh-gateway --config <file>`: serve until SIGTERM or SIGINT, then drain in-flight requests and exit 0.
//! A configuration error exits 2; a listener that cannot bind exits 1.

use std::future::Future;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use lh_gateway::config::{LogFormat, LogLevel};
use lh_gateway::{Config, LogSink, Server};
use tokio::signal::unix::{SignalKind, signal};
use tracing_subscriber::filter::LevelFilter;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Leaseholder gateway: the serving harness in front of vLLM"
)]
struct Args {
    /// The configuration file (TOML).
    #[arg(long)]
    config: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = match Config::load(&args.config) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("lh-gateway: {err}");
            return ExitCode::from(2);
        }
    };
    init_logging(config.log_format, config.log_level);
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            tracing::error!("cannot start the runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async {
        // Signal handlers go in before anything binds, so an early SIGTERM still drains.
        let shutdown = match shutdown_signal() {
            Ok(shutdown) => shutdown,
            Err(err) => {
                tracing::error!("cannot install signal handlers: {err}");
                return ExitCode::FAILURE;
            }
        };
        match Server::bind(config, Arc::new(LogSink)).await {
            Ok(server) => {
                server.run(shutdown).await;
                ExitCode::SUCCESS
            }
            Err(err) => {
                tracing::error!("cannot listen: {err}");
                ExitCode::FAILURE
            }
        }
    })
}

/// Completes on the first SIGTERM or SIGINT.
fn shutdown_signal() -> std::io::Result<impl Future<Output = ()>> {
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    Ok(async move {
        tokio::select! {
            _ = terminate.recv() => {}
            _ = interrupt.recv() => {}
        }
    })
}

fn init_logging(format: LogFormat, level: LogLevel) {
    let level = match level {
        LogLevel::Error => LevelFilter::ERROR,
        LogLevel::Warn => LevelFilter::WARN,
        LogLevel::Info => LevelFilter::INFO,
        LogLevel::Debug => LevelFilter::DEBUG,
        LogLevel::Trace => LevelFilter::TRACE,
    };
    let builder = tracing_subscriber::fmt().with_max_level(level);
    match format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().flatten_event(true).init(),
    }
}
