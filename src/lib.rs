use core::error::Error;
use std::ffi::CString;
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
    use libllama_sys::*;
    use std::ffi::CString;
    use std::ops::Deref;
    use std::time::Instant;
    use tracing::instrument;
    use tracing::{error, info};

    pub const LLAMA_TOKEN_NULL: llama_token = -1;

    //================================================ Alias for generated chunk

    pub type LlamaGenerated = Result<Option<String>, String>;

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
                ggml_backend_load_all();
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
    }

    //================================================ LlamaChatMessage wrrapper

    #[derive(Debug, Clone)]
    pub struct LlamaChatMessage {
        pub role: CString,
        pub content: CString,
    }

    unsafe impl Sync for LlamaChatMessage {}
    unsafe impl Send for LlamaChatMessage {}

    impl LlamaChatMessage {
        pub fn raw(&self) -> llama_chat_message {
            llama_chat_message {
                role: self.role.as_ptr(),
                content: self.content.as_ptr(),
            }
        }
    }

    //=============================================== LlamaChatTemplate wrrapper

    #[derive(Debug)]
    pub struct LlamaChatTemplate(*const std::os::raw::c_char);

    impl Deref for LlamaChatTemplate {
        type Target = *const std::os::raw::c_char;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    unsafe impl Sync for LlamaChatTemplate {}
    unsafe impl Send for LlamaChatTemplate {}

    impl From<&LlamaModel> for LlamaChatTemplate {
        fn from(model: &LlamaModel) -> Self {
            Self(unsafe { llama_model_chat_template(**model, std::ptr::null()) })
        }
    }

    impl LlamaChatTemplate {
        /// Get chat template string from model file
        pub fn get_chat_template(&self) -> Option<String> {
            if self.0.is_null() {
                return None;
            }

            let s = unsafe { std::ffi::CStr::from_ptr(self.0) };

            match s.to_string_lossy() {
                std::borrow::Cow::Borrowed(s) => Some(s.to_string()),
                std::borrow::Cow::Owned(s) => Some(s),
            }
        }

        /// Apply chat template
        pub fn apply(&self, messages: &[LlamaChatMessage], n_ctx: u32) -> CString {
            let buf = unsafe { CString::from_vec_unchecked(vec![0; n_ctx as usize]) };
            let buf_ptr = buf.into_raw();
            let raw_messages = messages.iter().map(|m| m.raw()).collect::<Vec<_>>();
            let _ = unsafe {
                llama_chat_apply_template(
                    self.0,
                    raw_messages.as_ptr(),
                    raw_messages.len(),
                    true,
                    buf_ptr,
                    n_ctx as i32,
                )
            };
            unsafe { CString::from_raw(buf_ptr) }
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
            txt: &CString,
            add_special: bool,
            parse_special: bool,
        ) -> Result<Vec<llama_token>, String> {
            let mut result: Vec<llama_token> = Vec::new();
            let n_tokens = unsafe {
                llama_tokenize(
                    **self,
                    txt.as_ptr(),
                    txt.as_bytes().len() as i32,
                    core::ptr::null_mut(),
                    0i32,
                    add_special,
                    parse_special,
                )
            };
            let fmt_n_tokens = -n_tokens;

            // Resize vector to fit number of tokens
            result.clear();
            result.resize(-n_tokens as usize, llama_token::default());
            // Tokenize again
            let check = unsafe {
                llama_tokenize(
                    **self,
                    txt.as_ptr(),
                    txt.as_bytes().len() as i32,
                    result.as_mut_ptr(),
                    -n_tokens,
                    add_special,
                    parse_special,
                )
            };
            if check < 0 {
                return Err(format!(
                    "Failed to tokenize: check ({check}) != n_tokens ({fmt_n_tokens})"
                ));
            }

            info!("Tokenized {fmt_n_tokens} tokens: {result:?}");

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
        ) -> Result<Self, String> {
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
            chat_template: &LlamaChatTemplate,
            vocab: &LlamaVocab,
            smpl: &LlamaSampler,
            rx: &mut tokio::sync::mpsc::Receiver<Vec<LlamaChatMessage>>,
            tx: tokio::sync::mpsc::Sender<LlamaGenerated>,
        ) {
            while let Some(messages) = rx.recv().await {
                let prompt = chat_template.apply(&messages, self.get_nctx());
                let fmt_prompt = format!("{prompt:?}");
                info!(%fmt_prompt);
                // Tokenize and check
                let tokens = vocab.tokenize(&prompt, true, true);
                if tokens.is_err() {
                    // Failed to tokenize, send error and continue with next prompt
                    let _ = tx
                        .send(Err(format!("Failed to tokenize:\n{prompt:?}")))
                        .await;
                    continue;
                }
                let mut tokens = tokens.unwrap();
                if tokens.len() as u32 > self.get_nctx() - 4 {
                    // Failed to tokenize, send error and continue with next prompt
                    let _ = tx
                        .send(Err(format!("Tokenize exceed context:\n{prompt:?}")))
                        .await;
                    continue;
                }

                // Add decoder start token
                if self.has_decoder {
                    tokens.push(vocab.decoder_start_token);
                }

                // Computation scope
                let mut n_decode = 0usize;
                let mut batch =
                    unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
                let mut new_token_id;
                let t_start = Instant::now();
                loop {
                    let decode_result = unsafe { llama_decode(**self, batch) };
                    if decode_result != 0 {
                        let err_msg = format!("Failed to decode batch for:\n{prompt:?}");
                        error!(%n_decode, err_msg);
                        let _ = tx.send(Err(err_msg)).await;
                        break;
                    }

                    // sample next token
                    new_token_id = unsafe { llama_sampler_sample(**smpl, **self, -1) };
                    if vocab.is_eog(new_token_id) {
                        break;
                    }

                    let buf = unsafe { CString::from_vec_unchecked(vec![0u8; 256]) };
                    let buf_ptr = buf.into_raw();
                    let n = unsafe {
                        llama_token_to_piece(**vocab, new_token_id, buf_ptr, 256i32, 0, true)
                    };
                    if n < 0 {
                        let err_msg = format!("Failed to convert token to piece for:\n{prompt:?}");
                        error!(err_msg);
                        let _ = tx.send(Err(err_msg)).await;
                        break;
                    }

                    let s = unsafe { CString::from_raw(buf_ptr) };
                    let s = match s.to_string_lossy() {
                        std::borrow::Cow::Borrowed(s) => s.to_string(),
                        std::borrow::Cow::Owned(s) => s,
                    };
                    let _ = tx.send(Ok(Some(s))).await;

                    // Prepare new batch with the sampled token
                    let mut tmp_tokens = vec![new_token_id];
                    batch = unsafe { llama_batch_get_one(tmp_tokens.as_mut_ptr(), 1) };
                    n_decode += 1;
                }
                // Signal end of generation
                let _ = tx.send(Ok(None)).await;
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

            //unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_greedy()) };

            let temperature = temperature.unwrap_or(0.8);
            unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05, 1)) };
            unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature)) };
            unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_dist(u32::MAX)) };

            Ok(Self(smpl))
        }

        pub fn reset(&self) {
            unsafe { llama_sampler_reset(**self) };
        }
    }
}

pub async fn spawn_model_default(
    model_path: String,
    mut prompt_rx: tokio::sync::mpsc::Receiver<Vec<llama::LlamaChatMessage>>,
    gen_tx: tokio::sync::mpsc::Sender<llama::LlamaGenerated>,
) -> Result<(), String> {
    // init backend
    let _backend = llama::LlamaBackend::default();

    // Load model and vocab, can be reused
    let model = llama::LlamaModel::new(model_path, None)?;
    let vocab = llama::LlamaVocab::from(&model);
    // Create context and sampler for single run
    let ctx = llama::LlamaContext::from_model(&model, None)?;
    let smpl = llama::LlamaSampler::new(None, None)?;
    // Chat template to apply to messages
    let chat_template = llama::LlamaChatTemplate::from(&model);
    let template_str = chat_template.get_chat_template().unwrap();
    info!(chat_template = %template_str);

    // Warn up before looping
    //ctx.warm_up(&vocab, batch_size);

    let _ = ctx
        .async_loop(&chat_template, &vocab, &smpl, &mut prompt_rx, gen_tx)
        .await;

    Ok(())
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

    let (prompt_tx, prompt_rx) = tokio::sync::mpsc::channel::<Vec<llama::LlamaChatMessage>>(2);
    let (gen_tx, mut gen_rx) = tokio::sync::mpsc::channel::<llama::LlamaGenerated>(2048);

    // Copy prompt string
    let prompt = prompt.to_owned();

    // Loop
    let lib_handle: tokio::task::JoinHandle<Result<(), String>> =
        tokio::spawn(spawn_model_default(hf_model_path, prompt_rx, gen_tx));
    // Client
    let client_handle: tokio::task::JoinHandle<Result<(), String>> = tokio::spawn(async move {
        let mut messages: Vec<llama::LlamaChatMessage> = Vec::new();
        messages.push(llama::LlamaChatMessage {
            role: CString::new("system").unwrap(),
            content: CString::new("You are a helpful assistant.").unwrap(),
        });
        messages.push(llama::LlamaChatMessage {
            role: CString::new("user").unwrap(),
            content: CString::new(prompt.clone()).unwrap(),
        });
        let _ = prompt_tx.send(messages.clone()).await;
        let mut res = "".to_string();
        while let Ok(Some(s)) = gen_rx.recv().await.unwrap() {
            res += &s;
        }
        info!(%res);
        messages.push(llama::LlamaChatMessage {
            role: CString::new("assistant").unwrap(),
            content: CString::new(res).unwrap(),
        });
        // Sleep 2 seconds
        //tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        //messages.push(llama::LlamaChatMessage {
        //    role: CString::new("user").unwrap(),
        //    content: CString::new("Can you tell me more about that?").unwrap(),
        //});
        //let _ = prompt_tx.send(messages.clone()).await;
        //let mut res = "".to_string();
        //while let Ok(Some(s)) = gen_rx.recv().await.unwrap() {
        //    res += &s;
        //}
        //info!(%res);
        //messages.push(llama::LlamaChatMessage {
        //    role: CString::new("assistant").unwrap(),
        //    content: CString::new(res).unwrap(),
        //});

        let fmt_msg = format!("{messages:?}");
        info!(messages = %fmt_msg);

        Ok(())
    });

    // Wait for tasks
    let (_, _) = tokio::join!(lib_handle, client_handle);

    // After returning, future holding loop is dropped, the loop is stopped
    Ok(())
}
