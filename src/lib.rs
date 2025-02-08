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

    //===================================================== LlamaBackend wrapper

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
                llama_backend_init();
                llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
            }

            Self {}
        }
    }

    //================================================== LlamaThreadpool wrapper

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
        pub fn init_from(
            dev_type: ggml_backend_dev_type,
            t_params: ggml_threadpool_params,
        ) -> Self {
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
            let model = unsafe { llama_load_model_from_file(model_path.as_ptr(), m_params) };
            if model.is_null() {
                let err_msg = "failed to load model";
                error!(err_msg);
                return Err(err_msg.into());
            }
            Ok(Self(model))
        }

        /// Get chat template string from model file
        pub fn get_chat_template(&self) -> Option<String> {
            let ptr_tmpl = unsafe { llama_model_chat_template(**self) };
            if ptr_tmpl.is_null() {
                return None;
            }

            unsafe { Some(ptr_tmpl.as_ref().unwrap().to_string()) }
        }
    }

    //====================================================== LlamaVocab wrrapper

    #[derive(Debug)]
    pub struct LlamaVocab {
        decoder_start_token: llama_token,
        vocab: *const llama_vocab,
    }

    impl Deref for LlamaVocab {
        type Target = *const llama_vocab;

        fn deref(&self) -> &Self::Target {
            &self.vocab
        }
    }

    unsafe impl Sync for LlamaVocab {}
    unsafe impl Send for LlamaVocab {}

    impl From<&LlamaModel> for LlamaVocab {
        fn from(model: &LlamaModel) -> Self {
            let vocab = unsafe { llama_model_get_vocab(**model) };
            // Get start token from model or from embedded vocab
            let mut decoder_start_token = unsafe { llama_model_decoder_start_token(**model) };
            if decoder_start_token == LLAMA_TOKEN_NULL {
                decoder_start_token = unsafe { llama_vocab_bos(vocab) };
            }

            Self {
                decoder_start_token,
                vocab,
            }
        }
    }

    impl LlamaVocab {
        pub fn bos(&self) -> llama_token {
            unsafe { llama_vocab_bos(**self) }
        }

        pub fn eos(&self) -> llama_token {
            unsafe { llama_vocab_eos(**self) }
        }

        pub fn eot(&self) -> llama_token {
            unsafe { llama_vocab_eot(**self) }
        }

        pub fn nl(&self) -> llama_token {
            unsafe { llama_vocab_nl(**self) }
        }

        pub fn cls(&self) -> llama_token {
            unsafe { llama_vocab_cls(**self) }
        }

        pub fn pad(&self) -> llama_token {
            unsafe { llama_vocab_pad(**self) }
        }

        pub fn sep(&self) -> llama_token {
            unsafe { llama_vocab_sep(**self) }
        }

        pub fn decoder_start_token(&self) -> llama_token {
            self.decoder_start_token
        }

        pub fn is_eog(&self, token: llama_token) -> bool {
            unsafe { llama_vocab_is_eog(**self, token) }
        }

        /// Tokenize the text using model's vocab
        pub fn tokenize(
            &self,
            txt: &str,
            add_special: bool,
            parse_special: bool,
        ) -> Result<Vec<llama_token>, String> {
            let upper_limit = if add_special {
                txt.len() + 2
            } else {
                txt.len()
            };
            let txt_cstr = CString::new(txt).expect("cstring from txt");
            let mut result: Vec<llama_token> = vec![0; upper_limit];
            let n_tokens = unsafe {
                llama_tokenize(
                    **self,
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
                        **self,
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
                    ));
                }
            }

            debug!("Tokenized {n_tokens} tokens: {result:?}");

            Ok(result)
        }
    }

    //===================================================== LlamaContext wrapper

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
            let c_params = unsafe { llama_context_default_params() };
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
        #[instrument]
        pub fn from_model(
            model: &LlamaModel,
            context_params: Option<llama_context_params>,
        ) -> Result<Self, Box<dyn Error>> {
            let c_params =
                context_params.unwrap_or_else(|| unsafe { llama_context_default_params() });
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

        /// Get n_ctx size
        pub fn get_nctx(&self) -> u32 {
            unsafe { llama_n_ctx(**self) }
        }

        /// Warmup without actual run
        pub fn warm_up(&self, vocab: &LlamaVocab, n_batch: i32) {
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
                            std::cmp::min(tmp.len() as i32, n_batch),
                        ),
                    );
                }
                llama_kv_cache_clear(**self);
                llama_synchronize(**self);
                llama_perf_context_reset(**self);
            }
            let elapsed = humantime::format_duration(t_start.elapsed());
            info!("warm up done, took {elapsed}");
        }

        /// Run in async loop waiting for prompt
        /// TODO: fix decode logic following llama.cpp, and apply chat template using minijinja
        #[instrument]
        pub async fn async_loop(
            &self,
            vocab: &LlamaVocab,
            smpl: &LlamaSampler,
            rx: &mut tokio::sync::mpsc::Receiver<String>,
            tx: tokio::sync::mpsc::Sender<String>,
            err_tx: tokio::sync::mpsc::Sender<String>,
        ) {
            while let Some(prompt) = rx.recv().await {
                // Tokenize and check
                let tokens = vocab.tokenize(&prompt, true, true);
                if tokens.is_err() {
                    // Failed to tokenize, send error and continue with next prompt
                    let _ = err_tx.send(format!("Failed to tokenize:\n{prompt}")).await;
                    continue;
                }
                let mut tokens = tokens.unwrap();
                if tokens.len() as u32 > self.get_nctx() - 4 {
                    // Failed to tokenize, send error and continue with next prompt
                    let _ = err_tx
                        .send(format!("Tokenize exceed context:\n{prompt}"))
                        .await;
                    continue;
                }

                // Add decoder start token
                tokens.push(vocab.decoder_start_token);

                // Computation scope
                let mut n_decode = 0;
                let mut batch =
                    unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
                let mut new_token = llama_token::default();
                let t_start = Instant::now();
                while !vocab.is_eog(new_token) {
                    let decode_result = unsafe { llama_decode(**self, batch) };
                    if decode_result != 0 {
                        let err_msg = format!("Failed to decode batch for:\n{prompt}");
                        error!(err_msg);
                        let _ = err_tx.send(err_msg).await;
                        break;
                    }

                    // sample next token
                    new_token = unsafe { llama_sampler_sample(**smpl, **self, -1) };

                    let mut buf = vec![0i8; 128];
                    let n = unsafe {
                        llama_token_to_piece(**vocab, new_token, buf.as_mut_ptr(), 127, 0, true)
                    };
                    if n < 0 {
                        let err_msg = format!("Failed to convert token to piece for:\n{prompt}");
                        error!(err_msg);
                        let _ = err_tx.send(err_msg).await;
                        break;
                    }
                    buf.resize(n as usize, 0);

                    match String::from_utf8(buf.iter().map(|&c| c as u8).collect()) {
                        Ok(s) => {
                            let _ = tx.send(s).await;
                        }
                        Err(e) => {
                            let err_msg =
                                format!("Failed to convert piece to string:\n{e:?}");
                            //error!(err_msg);
                            let _ = err_tx.send(err_msg).await;
                            //break;
                        }
                    }

                    // Prepare new batch with the sampled token
                    batch = unsafe { llama_batch_get_one(vec![new_token].as_mut_ptr(), 1) };
                    n_decode += 1;
                }
                let elapsed = humantime::format_duration(t_start.elapsed());
                let speed = n_decode as f64 / t_start.elapsed().as_secs_f64();
                info!("decoded {n_decode} tokens in {elapsed}, speed: {speed} TOPS");
            }
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
        #[instrument]
        pub fn new(
            sampler_params: Option<llama_sampler_chain_params>,
        ) -> Result<Self, Box<dyn Error>> {
            let s_params =
                sampler_params.unwrap_or_else(|| unsafe { llama_sampler_chain_default_params() });
            let smpl = unsafe { llama_sampler_chain_init(s_params) };
            if smpl.is_null() {
                let err_msg = "failed to create sampler";
                error!(err_msg);
                return Err(err_msg.into());
            }
            unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_greedy()) };
            Ok(Self(smpl))
        }

        pub fn reset(&self) {
            unsafe { llama_sampler_reset(**self) };
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
    let _backend = llama::LlamaBackend::default();
    // Load model and vocab, can be reused
    let model = llama::LlamaModel::new(hf_model_path, None)?;
    let vocab = llama::LlamaVocab::from(&model);
    // Create context and sampler for single run
    let ctx = llama::LlamaContext::from_model(&model, None)?;
    let smpl = llama::LlamaSampler::new(None)?;

    let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::channel::<String>(2);
    let (gen_tx, mut gen_rx) = tokio::sync::mpsc::channel::<String>(2048);
    let (err_tx, mut err_rx) = tokio::sync::mpsc::channel::<String>(2048);

    // Copy prompt string
    let prompt = prompt.to_owned();
    let mut tasks = tokio::task::JoinSet::new();

    // Warn up before looping
    //ctx.warm_up(&vocab, batch_size);

    // Loop
    tasks.spawn(async move {
        let _ = ctx
            .async_loop(&vocab, &smpl, &mut prompt_rx, gen_tx, err_tx)
            .await;
    });
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
        let mut s = "".to_string();
        while let Some(data) = gen_rx.recv().await {
            //debug!(%data, "data received");
            s += &data;
        }
        info!("Generated:\n{s}");
    });
    // Err Reader
    tasks.spawn(async move {
        while let Some(data) = err_rx.recv().await {
            debug!(%data, "error received");
        }
    });

    // Wait for tasks
    tasks.join_all().await;

    // After returning, future holding loop is dropped, the loop is stopped
    Ok(())
}
