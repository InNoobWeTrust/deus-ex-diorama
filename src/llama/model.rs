use tracing::{error, info};

use super::*;
use core::ops::Deref;
use std::ffi::CString;
use std::sync::mpsc;
use std::time::Instant;

#[derive(Debug)]
pub struct LlamaModel(*mut llama_model);

impl Deref for LlamaModel {
    type Target = *mut llama_model;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            llama_model_free(self.0);
        }
    }
}

unsafe impl Sync for LlamaModel {}
unsafe impl Send for LlamaModel {}

impl LlamaModel {
    pub fn new(
        model_path: String,
        model_params: Option<llama_model_params>,
    ) -> Result<Self, String> {
        let model_path = CString::new(model_path).unwrap();
        let m_params: llama_model_params =
            model_params.unwrap_or_else(|| unsafe { llama_model_default_params() });
        let model = unsafe { llama_load_model_from_file(model_path.as_ptr(), m_params) };
        if model.is_null() {
            let err_msg = format!("failed to load model: {model_path:?}");
            error!(err_msg);
            return Err(err_msg);
        }
        Ok(Self(model))
    }

    /// loop to process prompts
    pub fn run(
        &self,
        rx: mpsc::Receiver<(Vec<LlamaChatMessage>, mpsc::Sender<LlamaGenerated>)>,
    ) -> Result<(), String> {
        let vocab = LlamaVocab::from(self);
        // Chat template to apply to messages
        let chat_template = LlamaChatTemplate::from(self);
        let template_str = chat_template.get_chat_template().unwrap_or("".to_string());
        info!(chat_template = %template_str);

        // Create context
        let ctx = LlamaContext::from_model(self, None)?;
        // Create sampler
        let smpl = LlamaSampler::new(None, None)?;

        // Warm up context
        //ctx.warm_up(&vocab);

        'PROMPT: while let Ok((messages, gen_tx)) = rx.recv() {
            ctx.reset();
            smpl.reset();

            let prompt = chat_template.apply(&messages);
            if prompt.is_err() {
                let n = prompt.err().unwrap();
                let _ = gen_tx.send(Err(format!("Failed to apply chat template, size: {n}")));
                continue;
            }
            let prompt = prompt.unwrap();
            let fmt_prompt = format!("{prompt:?}");
            info!(%fmt_prompt);
            // Tokenize and check
            let is_first = ctx.get_nctx_used() == 0;
            let tokens = vocab.tokenize(&prompt, is_first, true);
            if tokens.is_err() {
                // Failed to tokenize, send error and continue with next prompt
                let _ = gen_tx.send(Err(format!("Failed to tokenize:\n{prompt:?}")));
                continue;
            }
            let mut tokens = tokens.unwrap();

            let t_start = Instant::now();
            // Computation scope
            let mut batch =
                unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };

            // If model has encoder then encode first
            if ctx.has_encoder() {
                let enc_res = unsafe { llama_encode(*ctx, batch) } == 0;

                if !enc_res {
                    let _ = gen_tx.send(Err("Failed to encode prompt".to_string()));
                    continue;
                }

                let mut decoder_start_token_id = vocab.decoder_start_token();
                batch = unsafe { llama_batch_get_one(&mut decoder_start_token_id, 1) };
            }

            let chunks = LlamaGeneratedChunks {
                ctx: &ctx,
                vocab: &vocab,
                smpl: &smpl,
                batch,
                n_pos: 0,
                is_err: false,
            };
            let mut n_decode = 0;
            for chunk in chunks {
                if chunk.is_err() {
                    let err = chunk.err().unwrap();
                    error!(err);
                    let _ = gen_tx.send(Err(err));
                    continue 'PROMPT;
                }

                let _ = gen_tx.send(Ok(chunk.unwrap()));
                n_decode += 1;
            }
            let elapsed = humantime::format_duration(t_start.elapsed());
            let speed = n_decode as f64 / t_start.elapsed().as_secs_f64();
            info!("decoded {n_decode} tokens in {elapsed}, speed: {speed} TOPS");
        }

        Ok(())
    }
}
