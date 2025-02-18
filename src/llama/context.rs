use tracing::{error, info, warn};

use super::*;
use core::ops::Deref;
use std::time::Instant;

#[derive(Debug)]
pub struct LlamaContext {
    has_encoder: bool,
    has_decoder: bool,
    ctx: *mut llama_context,
}

impl Deref for LlamaContext {
    type Target = *mut llama_context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.ctx);
        }
    }
}

unsafe impl Sync for LlamaContext {}
unsafe impl Send for LlamaContext {}

impl From<&LlamaModel> for LlamaContext {
    fn from(model: &LlamaModel) -> Self {
        let mut c_params = unsafe { llama_context_default_params() };
        c_params.n_ctx = unsafe { llama_model_n_ctx_train(**model) } as u32;
        let ctx = unsafe { llama_init_from_model(**model, c_params) };
        assert!(!ctx.is_null());
        let has_encoder = unsafe { llama_model_has_encoder(**model) };
        let has_decoder = unsafe { llama_model_has_decoder(**model) };

        Self {
            has_encoder,
            has_decoder,
            ctx,
        }
    }
}

impl LlamaContext {
    pub fn from_model(
        model: &LlamaModel,
        context_params: Option<llama_context_params>,
    ) -> Result<Self, String> {
        let c_params = context_params.unwrap_or_else(|| {
            let mut p = unsafe { llama_context_default_params() };
            p.n_ctx = unsafe { llama_model_n_ctx_train(**model) } as u32;
            p
        });
        let ctx = unsafe { llama_init_from_model(**model, c_params) };
        if ctx.is_null() {
            let err_msg = "failed to create context";
            error!(err_msg);
            return Err(err_msg.into());
        }

        let has_encoder = unsafe { llama_model_has_encoder(**model) };
        let has_decoder = unsafe { llama_model_has_decoder(**model) };

        Ok(Self {
            has_encoder,
            has_decoder,
            ctx,
        })
    }

    pub fn has_encoder(&self) -> bool {
        self.has_encoder
    }

    pub fn has_decoder(&self) -> bool {
        self.has_decoder
    }

    /// Get n_batch
    pub fn get_nbatch(&self) -> u32 {
        unsafe { llama_n_batch(**self) }
    }

    /// Get n_ctx size
    pub fn get_nctx(&self) -> u32 {
        unsafe { llama_n_ctx(**self) }
    }

    /// Get n_ctx_used
    pub fn get_nctx_used(&self) -> i32 {
        unsafe { llama_get_kv_cache_used_cells(**self) }
    }

    /// Check if tokens is exceeding context size
    pub fn is_exceeding_context_size(&self, n_tokens: u32) -> bool {
        let n_ctx = self.get_nctx();
        let n_ctx_used = self.get_nctx_used();
        let res = n_ctx_used + n_tokens as i32 > n_ctx as i32;

        if res {
            warn!("{n_tokens} exceeding context size ({n_ctx_used}/{n_ctx})");
        }

        res
    }

    /// Reset context
    pub fn reset(&self) {
        unsafe {
            llama_kv_cache_clear(**self);
            llama_synchronize(**self);
            llama_perf_context_reset(**self);
        }
    }

    /// Warmup without actual run
    pub fn warm_up(&self, vocab: &LlamaVocab) {
        info!("warming up model");
        let t_start = Instant::now();
        let mut tmp: Vec<llama_token> = Vec::new();
        let bos = vocab.bos();
        let eos = vocab.eos();

        // some models (e.g. T5) don't have a BOS token
        if bos != LLAMA_TOKEN_NULL {
            tmp.push(bos);
        }
        if eos != LLAMA_TOKEN_NULL {
            tmp.push(eos);
        }
        if tmp.is_empty() {
            tmp.push(0);
        }

        unsafe {
            if self.has_encoder {
                llama_encode(
                    **self,
                    llama_batch_get_one(tmp.as_mut_ptr(), tmp.len() as i32),
                );
                let decoder_start_token_id = vocab.decoder_start_token();
                tmp.clear();
                tmp.push(decoder_start_token_id);
            }
            if self.has_decoder {
                llama_decode(
                    **self,
                    llama_batch_get_one(
                        tmp.as_mut_ptr(),
                        std::cmp::min(tmp.len() as i32, self.get_nbatch() as i32),
                    ),
                );
            }
        }
        self.reset();
        let elapsed = humantime::format_duration(t_start.elapsed());
        info!("warm up done, took {elapsed}");
    }
}
