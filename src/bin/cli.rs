#[macro_use]
extern crate serde;

use clap::Parser;
use core::error::Error;
use deus_ex_diorama::*;
use tracing::debug;

#[derive(Parser, Serialize, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// HF repo
    #[arg(long)]
    hf_repo: String,

    /// HF model file
    #[arg(long)]
    hf_file: String,

    /// Number of GPU layers
    #[arg(long, default_value_t = 99)]
    ngl: i32,

    /// Number of batches for warming up the models
    #[arg(short, long, default_value_t = 2048)]
    batch_size: i32,

    /// Prompt
    #[arg(short, long, default_value_t = String::from("You are a helpful assistant."))]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // install global collector configured based on RUST_LOG env var.
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    // Parse arguments
    let args = Args::parse();
    debug!(name: "args", args = serde_json::to_string(&args)?);

    test_llama(&args.hf_repo, &args.hf_file, &args.prompt, args.batch_size).await?;

    Ok(())
}
