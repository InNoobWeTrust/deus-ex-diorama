use core::error::Error;
use std::path::PathBuf;
use tracing::{debug, instrument};

#[instrument]
pub async fn hf_load_file(repo: &str, file: &str) -> Result<PathBuf, Box<dyn Error>> {
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model(repo.to_string());
    let file_name = repo.get(file).await?;

    Ok(file_name)
}

pub mod llama {
    use core::error::Error;
    use libllama_sys::*;
    use std::ffi::CString;
    use std::ops::Deref;
    use std::time::Instant;
    use tracing::{debug, error, instrument};

    pub const LLAMA_TOKEN_NULL: llama_token = -1;

    #[derive(Debug)]
    pub struct LlamaModel(*mut llama_model);

    impl LlamaModel {
        #[instrument]
        pub fn new(
            model_path: String,
            model_params: Option<llama_model_params>,
        ) -> Result<Self, Box<dyn Error>> {
            let model_path = CString::new(model_path).unwrap();
            let m_params: llama_model_params =
                model_params.unwrap_or_else(|| unsafe { llama_model_default_params() });
            let model: *mut llama_model;
            unsafe {
                model = llama_load_model_from_file(model_path.as_ptr(), m_params);
                if model.is_null() {
                    let err_msg = "failed to load model";
                    error!(err_msg);
                    return Err(err_msg.into());
                }
            }
            Ok(Self(model))
        }

        /// Warmup model without actual run
        pub fn warm_up(&self, ctx: &LlamaContext, n_batch: i32) {
            debug!("warming up model");
            let t_start = Instant::now();
            unsafe {
                let mut tmp: Vec<llama_token> = Vec::new();
                let vocab = llama_model_get_vocab(**self);
                let bos = llama_vocab_bos(vocab);
                let eos = llama_vocab_eos(vocab);

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

                if llama_model_has_encoder(**self) {
                    llama_encode(
                        **ctx,
                        llama_batch_get_one(tmp.as_mut_ptr(), tmp.len() as i32),
                    );
                    let mut decoder_start_token_id = llama_model_decoder_start_token(**self);
                    if decoder_start_token_id == LLAMA_TOKEN_NULL {
                        decoder_start_token_id = bos;
                    }
                    tmp.clear();
                    tmp.push(decoder_start_token_id);
                }
                if llama_model_has_decoder(**self) {
                    llama_decode(
                        **ctx,
                        llama_batch_get_one(
                            tmp.as_mut_ptr(),
                            std::cmp::min(tmp.len() as i32, n_batch),
                        ),
                    );
                }
                llama_kv_cache_clear(**ctx);
                llama_synchronize(**ctx);
                llama_perf_context_reset(**ctx);
            }
            let elapsed = humantime::format_duration(t_start.elapsed());
            debug!("warmup done, took {elapsed}");
        }

        /// Run model in async loop waiting for prompt
        #[instrument]
        pub async fn async_loop(
            &self,
            ctx: &LlamaContext,
            smpl: &LlamaSampler,
            rx: &mut tokio::sync::mpsc::Receiver<String>,
            tx: tokio::sync::mpsc::Sender<String>,
        ) -> Result<(), &'static str> {
            loop {
                match rx.recv().await {
                    Some(orig_prompt) => {
                        let prompt = std::ffi::CString::new(orig_prompt.clone()).expect("cstring from prompt");
                        // Computation scope
                        let mut n_decode = 0;
                        let t_start = Instant::now();
                        unsafe {
                            tx.send("Hello".into()).await;
                            tx.send(", ".into()).await;
                            tx.send("bug".into()).await;
                            tx.send("!".into()).await;
                            tx.send(orig_prompt.into()).await;
                        }
                        let elapsed = humantime::format_duration(t_start.elapsed());
                        let speed = n_decode as f64 / t_start.elapsed().as_secs_f64();
                        debug!("decoded {n_decode} tokens in {elapsed}, speed: {speed} t/s");
                    }
                    _ => break,
                }
            }

            Ok(())
        }
    }

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

    #[derive(Debug)]
    pub struct LlamaContext(*mut llama_context);

    impl LlamaContext {
        #[instrument]
        pub fn new(
            model: &LlamaModel,
            context_params: Option<llama_context_params>,
        ) -> Result<Self, Box<dyn Error>> {
            let ctx: *mut llama_context;
            let c_params =
                context_params.unwrap_or_else(|| unsafe { llama_context_default_params() });
            unsafe {
                ctx = llama_init_from_model(**model, c_params);
                if ctx.is_null() {
                    let err_msg = "failed to create context";
                    error!(err_msg);
                    return Err(err_msg.into());
                }
            }
            Ok(Self(ctx))
        }
    }

    impl Deref for LlamaContext {
        type Target = *mut llama_context;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Drop for LlamaContext {
        fn drop(&mut self) {
            unsafe {
                llama_free(self.0);
            }
        }
    }

    unsafe impl Sync for LlamaContext {}
    unsafe impl Send for LlamaContext {}

    #[derive(Debug)]
    pub struct LlamaSampler(*mut llama_sampler);

    impl LlamaSampler {
        #[instrument]
        pub fn new(
            sampler_params: Option<llama_sampler_chain_params>,
        ) -> Result<Self, Box<dyn Error>> {
            let smpl: *mut llama_sampler;
            let s_params =
                sampler_params.unwrap_or_else(|| unsafe { llama_sampler_chain_default_params() });
            unsafe {
                smpl = llama_sampler_chain_init(s_params);
                if smpl.is_null() {
                    let err_msg = "failed to create sampler";
                    error!(err_msg);
                    return Err(err_msg.into());
                }
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            }
            Ok(Self(smpl))
        }
    }

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

    /// init backend
    pub fn init() {
        unsafe {
            llama_backend_init();
            llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        }
    }
}

#[instrument]
pub async fn test_llama(
    hf_repo: &str,
    hf_file: &str,
    prompt: &str,
    batch_size: i32,
) -> Result<(), Box<dyn Error>> {
    let hf_model_path = hf_load_file(hf_repo, hf_file).await?;
    let hf_model_path = hf_model_path.to_str().unwrap().to_owned();
    debug!(name: "hf_model_path", %hf_model_path);

    // init backend
    llama::init();

    // Load model, create context and sampler
    let model = llama::LlamaModel::new(hf_model_path, None)?;
    let ctx = llama::LlamaContext::new(&model, None)?;
    let smpl = llama::LlamaSampler::new(None)?;

    //model.warm_up(&ctx, batch_size);

    let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::channel::<String>(2);
    let (gen_tx, mut gen_rx) = tokio::sync::mpsc::channel::<String>(2048);

    // Loop
    let loop_task = model.async_loop(&ctx, &smpl, &mut prompt_rx, gen_tx);

    // Copy prompt string
    let prompt = prompt.to_owned();
    let mut tasks = tokio::task::JoinSet::new();

    // Writer
    tasks.spawn(async move {
        prompt_tx.send(prompt.clone()).await;
        // Sleep 2 seconds
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        prompt_tx.send(prompt.clone()).await;
        // Sleep 2 seconds
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        prompt_tx.send(prompt.clone()).await;
    });
    // Reader
    tasks.spawn(async move {
        loop {
            match gen_rx.recv().await {
                Some(data) => debug!(%data),
                _ => break,
            }
        }
    });

    // Wait for loop to exit first
    loop_task.await?;
    // Wait for read write tasks to cleanup
    tasks.join_all().await;

    // After returning, future holding loop is dropped, the loop is stopped
    Ok(())
}
