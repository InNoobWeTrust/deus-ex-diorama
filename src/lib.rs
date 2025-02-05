use core::error::Error;
use std::path::PathBuf;
use tracing::{debug, info, instrument};

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
    use tracing::{debug, error, info, instrument};

    pub const LLAMA_TOKEN_NULL: llama_token = -1;

    //======================================================= LlamaModel wrapper

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

        /// Get vocab of the model
        pub fn get_vocab(&self) -> *const llama_vocab {
            unsafe { llama_model_get_vocab(**self) }
        }

        /// Tokenize the text using model's vocab
        pub fn tokenize(
            &self,
            txt: &str,
            add_special: bool,
            parse_special: bool,
        ) -> Result<Vec<llama_token>, Box<dyn Error>> {
            let upper_limit = txt.len() + 2 * (if add_special { 1 } else { 0 });
            let txt_cstr = CString::new(txt).expect("cstring from txt");
            let mut result: Vec<llama_token> = Vec::with_capacity(upper_limit);
            let n_tokens = unsafe {
                llama_tokenize(
                    self.get_vocab(),
                    txt_cstr.as_ptr(),
                    txt.len() as i32,
                    result.as_mut_ptr(),
                    upper_limit as i32,
                    add_special,
                    parse_special,
                )
            };
            if n_tokens < 0 {
                // Resize vector to fix number of tokens
                result.clear();
                result.resize(-n_tokens as usize, llama_token::default());
                // Tokenize again
                let check = unsafe {
                    llama_tokenize(
                        self.get_vocab(),
                        txt_cstr.as_ptr(),
                        txt.len() as i32,
                        result.as_mut_ptr(),
                        -n_tokens,
                        add_special,
                        parse_special,
                    )
                };
                if check != -n_tokens {
                    let fmt_n_tokens = -n_tokens;
                    return Err(format!(
                        "Failed to tokenize: check ({check}) != n_tokens ({fmt_n_tokens})"
                    )
                    .into());
                }
            }

            Ok(result)
        }

        /// Get chat template string from model file
        pub fn get_chat_template(&self) -> Option<String> {
            let ptr_tmpl;
            unsafe {
                ptr_tmpl = llama_model_chat_template(**self);
            }
            if ptr_tmpl.is_null() {
                return None;
            }

            unsafe { Some(ptr_tmpl.as_ref().unwrap().to_string()) }
        }

        /// Warmup model without actual run
        pub fn warm_up(&self, ctx: &LlamaContext, n_batch: i32) {
            info!("warming up model");
            let t_start = Instant::now();
            unsafe {
                let mut tmp: Vec<llama_token> = Vec::new();
                let vocab = self.get_vocab();
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
            err_tx: tokio::sync::mpsc::Sender<String>,
        ) -> Result<(), &'static str> {
            while let Some(orig_prompt) = rx.recv().await {
                let prompt = CString::new(orig_prompt.clone()).expect("cstring from prompt");
                // Tokenize and check
                let tokens = self.tokenize(&orig_prompt, true, true);
                if tokens.is_err() || tokens.unwrap().len() as u32 > ctx.get_nctx() - 4 {
                    // Failed to tokenize, send error and continue with next prompt
                    err_tx
                        .send(format!("Failed to tokenize:\n{orig_prompt}").into())
                        .await;
                    continue;
                }
                // Computation scope
                let mut n_decode = 0;
                let t_start = Instant::now();
                unsafe {
                    let _ = tx.send("Hello".into()).await;
                    let _ = tx.send(", ".into()).await;
                    let _ = tx.send("bug".into()).await;
                    let _ = tx.send("!".into()).await;
                    let _ = tx.send(orig_prompt.into()).await;
                }
                let elapsed = humantime::format_duration(t_start.elapsed());
                let speed = n_decode as f64 / t_start.elapsed().as_secs_f64();
                info!("decoded {n_decode} tokens in {elapsed}, speed: {speed} TOPS");
            }

            Ok(())
        }
    }

    //===================================================== LlamaContext wrapper

    #[derive(Debug)]
    pub struct LlamaContext(*mut llama_context);

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

    impl LlamaContext {
        #[instrument]
        pub fn from_model(
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

        /// Get n_ctx size
        pub fn get_nctx(&self) -> u32 {
            unsafe { llama_n_ctx(**self) }
        }

        pub fn attach_cpu_threadpool(
            &self,
            tpp: ggml_threadpool_params,
            tpp_batch: ggml_threadpool_params,
        ) -> Result<(), Box<dyn Error>> {
            unsafe {
                let res = libllama_init_attach_cpu_threadpool(**self, tpp, tpp_batch);

                if res != 0 {
                    return Err("Failed to init and attach cpu threadpool".into());
                }
            }

            Ok(())
        }
    }

    //===================================================== LlamaSampler wrapper

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

    //==================================================== Wrap common functions

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
    let ctx = llama::LlamaContext::from_model(&model, None)?;
    let smpl = llama::LlamaSampler::new(None)?;

    //model.warm_up(&ctx, batch_size);

    let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::channel::<String>(2);
    let (gen_tx, mut gen_rx) = tokio::sync::mpsc::channel::<String>(2048);
    let (err_tx, mut err_rx) = tokio::sync::mpsc::channel::<String>(2048);

    // Loop
    let loop_task = model.async_loop(&ctx, &smpl, &mut prompt_rx, gen_tx, err_tx);

    // Copy prompt string
    let prompt = prompt.to_owned();
    let mut tasks = tokio::task::JoinSet::new();

    // Writer
    tasks.spawn(async move {
        let _ = prompt_tx.send(prompt.clone()).await;
        // Sleep 2 seconds
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        let _ = prompt_tx.send(prompt.clone()).await;
        // Sleep 2 seconds
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        let _ = prompt_tx.send(prompt.clone()).await;
    });
    // Reader
    tasks.spawn(async move {
        while let Some(data) = gen_rx.recv().await {
            info!(%data, "data received");
        }
    });
    // Err Reader
    tasks.spawn(async move {
        while let Some(data) = err_rx.recv().await {
            info!(%data, "error received");
        }
    });

    // Wait for loop to exit first
    loop_task.await?;
    // Wait for read write tasks to cleanup
    tasks.join_all().await;

    // After returning, future holding loop is dropped, the loop is stopped
    Ok(())
}
