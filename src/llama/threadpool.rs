use super::*;
use core::ops::Deref;

#[derive(Debug)]
pub struct LlamaThreadpool {
    dev_type: ggml_backend_dev_type,
    threadpool: *mut ggml_threadpool,
}

impl Deref for LlamaThreadpool {
    type Target = *mut ggml_threadpool;

    fn deref(&self) -> &Self::Target {
        &self.threadpool
    }
}

impl Drop for LlamaThreadpool {
    fn drop(&mut self) {
        unsafe {
            libllama_free_threadpool(self.dev_type, self.threadpool);
        }
    }
}

impl LlamaThreadpool {
    pub fn init_from(dev_type: ggml_backend_dev_type, t_params: ggml_threadpool_params) -> Self {
        let threadpool = unsafe { libllama_init_threadpool(dev_type, t_params) };

        Self {
            dev_type,
            threadpool,
        }
    }

    pub fn attach(&self, ctx: &LlamaContext) {
        unsafe { llama_attach_threadpool(**ctx, self.threadpool, std::ptr::null_mut()) };
    }
}
