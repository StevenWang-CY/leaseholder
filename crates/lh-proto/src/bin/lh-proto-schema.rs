//! Writes the JSON Schemas of the wire types for the Python components:
//! `cargo run -p lh-proto --bin lh-proto-schema [DIR]` (default: this crate's `schema/` directory).

use std::path::PathBuf;
use std::{env, fs, io};

fn main() -> io::Result<()> {
    let dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schema"));
    fs::create_dir_all(&dir)?;
    for (name, schema) in lh_proto::schemas() {
        let path = dir.join(format!("{name}.schema.json"));
        fs::write(&path, lh_proto::schema_json(&schema))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
