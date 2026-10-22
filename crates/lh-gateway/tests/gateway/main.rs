//! The gateway between a test client and an in-process mock engine, over loopback TCP and a Unix socket.

mod admission;
mod common;
mod events;
mod proxy;
mod streams;
