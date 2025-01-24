use core::error::Error;
use libllama_sys::*;
use std::ffi::CString;
use std::path::PathBuf;
use tracing::{debug, instrument};

#[instrument]
pub async fn hf_load_file(repo: &str, file: &str) -> Result<PathBuf, Box<dyn Error>> {
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model(repo.to_string());
    let file_name = repo.get(file).await?;

    Ok(file_name)
}

#[instrument]
pub async fn test_llama(
    hf_repo: &str,
    hf_file: &str,
    prompt: &str,
    ngl: i32,
) -> Result<(), Box<dyn Error>> {
    let hf_model_path = hf_load_file(hf_repo, hf_file).await?;
    debug!(name: "hf_model_path", hf_model_path = hf_model_path.to_str().unwrap());
    let model_path = CString::new(hf_model_path.to_str().unwrap())?;
    let _prompt = CString::new(prompt)?;
    unsafe {
        ggml_backend_load_all();
        let mut model_params = llama_model_default_params();
        model_params.n_gpu_layers = ngl;
        let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
        llama_model_free(model);
    }
    Ok(())
}
