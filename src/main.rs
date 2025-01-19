extern crate deus_ex_diorama;

use clap::Parser;
use core::error::Error;
use deus_ex_diorama::*;
use llamacpp_sys::*;
use std::ffi::CString;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model path
    #[arg(short, long)]
    model_path: String,

    /// Number of GPU layers
    #[arg(short, long, default_value_t = 128)]
    ngl: usize,

    /// Prompt
    #[arg(short, long)]
    prompt: String,
}

fn test_llama() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let model_path = CString::new(args.model_path)?;
    unsafe {
        ggml_backend_load_all();
        let model_params = llama_model_default_params();
        let _ = llama_load_model_from_file(model_path.as_ptr(), model_params);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let res = add(1, 2);
    println!("{}", res);
    let _ = test_llama();
    Ok(())
}
