use core::error::Error;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::mpsc;
use tracing::{debug, info, instrument};
pub mod llama;
use llama::*;

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
    batch_size: i32,
) -> Result<(), Box<dyn Error>> {
    let model_path = hf_load_file(hf_repo, hf_file).await?;
    let model_path = model_path.to_str().unwrap().to_owned();
    debug!(%model_path);

    let (tx, rx) = mpsc::channel::<(Vec<LlamaChatMessage>, mpsc::Sender<LlamaGenerated>)>();

    // Copy prompt string
    let prompt = prompt.to_owned();

    // Loop
    let lib_handle = tokio::task::spawn_blocking(|| {
        // init backend
        let _backend = LlamaBackend::default();

        // Load model
        let model = LlamaModel::new(model_path, None)?;

        model.run(rx)
    });

    // Client
    {
        // Use IIFE to take ownership of tx so it will be dropped automatically after done sending
        let tx = (move || tx)();
        let (gen_tx, gen_rx) = mpsc::channel::<LlamaGenerated>();
        let mut messages: Vec<LlamaChatMessage> = Vec::new();
        messages.push(LlamaChatMessage {
            role: c"user".into(),
            content: CString::new(prompt.clone()).unwrap(),
        });
        let _ = tx.send((messages.clone(), gen_tx));
        let mut res = Vec::new();
        let mut n_recv = 0;
        let buf_size = 32;
        while let Ok(Ok(s)) = gen_rx.recv() {
            res.push(s);
            n_recv += 1;
            if n_recv % buf_size == 0 {
                let buf = res[(n_recv - buf_size)..n_recv].join("");
                tokio::task::spawn(async move {
                    info!(buffer = %buf);
                });
            }
        }
        {
            let buf = res[(n_recv - n_recv % buf_size)..n_recv].join("");
            tokio::task::spawn(async move {
                info!(buffer = %buf);
            });
        }
        messages.push(LlamaChatMessage {
            role: c"assistant".into(),
            content: CString::new(res.join("")).unwrap(),
        });

        let fmt_msg = format!("{messages:?}");
        info!(messages = %fmt_msg, %n_recv);

        // Drop tx after done sending (no need if tx is moved to inner scope)
        //drop(tx);
    }

    // Wait for tasks
    let _ = lib_handle.await;

    // After returning, future holding loop is dropped, the loop is stopped
    Ok(())
}
