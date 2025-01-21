extern crate deus_ex_diorama;

use clap::Parser;
use core::error::Error;
use deus_ex_diorama::*;
use llamacpp_sys::*;
use serde::Serialize;
use std::ffi::CString;
use tracing::{debug, instrument};

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

    /// Number of tokens to predict
    #[arg(short, long, default_value_t = 32)]
    n_predict: i32,

    /// Prompt
    #[arg(short, long, default_value_t = String::from("You are a helpful assistant."))]
    prompt: String,
}

#[instrument]
async fn test_llama(args: &Args) -> Result<(), Box<dyn Error>> {
    let hf_model_path = hf_load_file(&args.hf_repo, &args.hf_file).await?;
    debug!(name: "hf_model_path", hf_model_path = hf_model_path.to_str().unwrap());
    let model_path = CString::new(hf_model_path.to_str().unwrap())?;
    let prompt = CString::new(args.prompt.clone())?;
    unsafe {
        ggml_backend_load_all();
        let mut model_params = llama_model_default_params();
        model_params.n_gpu_layers = args.ngl as i32;
        let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
        llama_model_free(model);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // install global collector configured based on RUST_LOG env var.
    tracing_subscriber::fmt::init();

    // Parse arguments
    let args = Args::parse();
    debug!(name: "args", args = serde_json::to_string(&args)?);

    let _ = test_llama(&args).await?;

    Ok(())
}
