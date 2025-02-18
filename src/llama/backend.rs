use super::*;

#[derive(Debug)]
pub struct LlamaBackend {}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        unsafe {
            llama_backend_free();
        }
    }
}

impl Default for LlamaBackend {
    fn default() -> Self {
        // init backend
        unsafe {
            ggml_backend_load_all();
            llama_backend_init();
            llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
            llama_log_set(Some(nolog_callback), std::ptr::null_mut());
        }

        Self {}
    }
}
