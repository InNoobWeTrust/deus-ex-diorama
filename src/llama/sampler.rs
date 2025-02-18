use tracing::error;

use super::*;
use core::ops::Deref;

#[derive(Debug)]
pub struct LlamaSampler(*mut llama_sampler);

impl Deref for LlamaSampler {
    type Target = *mut llama_sampler;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_sampler_free(self.0);
        }
    }
}

unsafe impl Sync for LlamaSampler {}
unsafe impl Send for LlamaSampler {}

impl Default for LlamaSampler {
    fn default() -> Self {
        let s_params = unsafe { llama_sampler_chain_default_params() };
        let smpl = unsafe { llama_sampler_chain_init(s_params) };
        // Just panic if null
        assert!(!smpl.is_null());
        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_greedy()) };

        Self(smpl)
    }
}

impl LlamaSampler {
    pub fn new(
        sampler_params: Option<llama_sampler_chain_params>,
        temperature: Option<f32>,
    ) -> Result<Self, String> {
        let s_params =
            sampler_params.unwrap_or_else(|| unsafe { llama_sampler_chain_default_params() });
        let smpl = unsafe { llama_sampler_chain_init(s_params) };
        if smpl.is_null() {
            let err_msg = "failed to create sampler";
            error!(err_msg);
            return Err(err_msg.into());
        }

        // Default temperature is 0.8
        let temperature = temperature.unwrap_or(0.8);
        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05, 1)) };
        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature)) };
        unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_dist(u32::MAX)) };

        Ok(Self(smpl))
    }

    pub fn sample(&self, ctx: &LlamaContext, idx: i32) -> llama_token {
        unsafe { llama_sampler_sample(**self, **ctx, idx) }
    }

    pub fn reset(&self) {
        unsafe { llama_sampler_reset(**self) };
    }
}
