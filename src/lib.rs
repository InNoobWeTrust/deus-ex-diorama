use core::error::Error;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::mpsc;
use tracing::{debug, info, instrument};

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
    use std::sync::mpsc;
    use std::time::Instant;
    use tracing::{error, info, warn};

    //================================================== Redefine some defaults

    pub const LLAMA_TOKEN_NULL: llama_token = -1;

    unsafe extern "C" fn nolog_callback(_level: ggml_log_level, _text: *const ::std::os::raw::c_char, _user_data: *mut ::std::os::raw::c_void) {}

    //================================================ Alias for generated chunk

    pub type LlamaGenerated = Result<String, String>;

    #[derive(Debug, Clone, Copy)]
    pub struct LlamaGeneratedChunks<'a> {
        ctx: &'a LlamaContext,
        vocab: &'a LlamaVocab,
        smpl: &'a LlamaSampler,
        batch: llama_batch,
        n_pos: usize,
        is_err: bool,
    }

    impl Iterator for LlamaGeneratedChunks<'_> {
        type Item = LlamaGenerated;

        fn next(&mut self) -> Option<Self::Item> {
            if self.is_err {
                return None;
            }

            self.ctx
                .is_exceeding_context_size(self.batch.n_tokens as u32);

            let decode_result = unsafe { llama_decode(**self.ctx, self.batch) } == 0;
            if !decode_result {
                self.is_err = true;
                let err_msg = format!("Failed to decode batch: {:?}", self.batch);
                return Some(Err(err_msg));
            }

            self.n_pos += self.batch.n_tokens as usize;

            // sample next token
            let mut new_token_id = self.smpl.sample(self.ctx, -1);

            if self.vocab.is_eog(new_token_id) {
                // End of generation
                return None;
            }

            let n = unsafe {
                -llama_token_to_piece(**self.vocab, new_token_id, std::ptr::null_mut(), 0, 0, true)
            };
            let buf = unsafe { CString::from_vec_unchecked(vec![0u8; n as usize]) };
            let buf_ptr = buf.into_raw();
            let n =
                unsafe { llama_token_to_piece(**self.vocab, new_token_id, buf_ptr, n, 0, true) };

            if n < 0 {
                self.is_err = true;
                let err_msg = format!("Failed to convert token to piece: {new_token_id:?}");
                return Some(Err(err_msg));
            }

            let s = unsafe { CString::from_raw(buf_ptr) };
            let s = s
                .into_string()
                .or_else(|e| Ok(format!("{:?}", e.into_cstring())));

            // Prepare new batch with the sampled token
            self.batch = unsafe { llama_batch_get_one(&mut new_token_id, 1) };

            Some(s)
        }
    }

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
                llama_log_set(Some(nolog_callback), std::ptr::null_mut());
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
            let template_str = chat_template.get_chat_template().unwrap();
            info!(chat_template = %template_str);

            // Create context
            let ctx = LlamaContext::from_model(self, None)?;
            // Create sampler
            let smpl = LlamaSampler::new(None, None)?;

            // Warm up context
            //ctx.warm_up(&vocab);

            while let Ok((messages, gen_tx)) = rx.recv() {
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
                if ctx.has_encoder {
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
                let mut is_err = false;
                let mut n_decode = 0;
                for chunk in chunks {
                    if chunk.is_err() {
                        let err = chunk.err().unwrap();
                        error!(err);
                        let _ = gen_tx.send(Err(err));
                        is_err = true;
                        break;
                    }

                    let _ = gen_tx.send(Ok(chunk.unwrap()));
                    n_decode += 1;
                }
                if is_err {
                    continue;
                }
                let elapsed = humantime::format_duration(t_start.elapsed());
                let speed = n_decode as f64 / t_start.elapsed().as_secs_f64();
                info!("decoded {n_decode} tokens in {elapsed}, speed: {speed} TOPS");
            }

            Ok(())
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
        pub fn apply(&self, messages: &[LlamaChatMessage]) -> Result<CString, i32> {
            let raw_messages = messages.iter().map(|m| m.raw()).collect::<Vec<_>>();
            let n = unsafe {
                llama_chat_apply_template(
                    self.0,
                    raw_messages.as_ptr(),
                    raw_messages.len(),
                    true,
                    std::ptr::null_mut(),
                    0,
                )
            };
            let buf = unsafe { CString::from_vec_unchecked(vec![0u8; n as usize]) };
            let buf_ptr = buf.into_raw();
            let n = unsafe {
                llama_chat_apply_template(
                    self.0,
                    raw_messages.as_ptr(),
                    raw_messages.len(),
                    true,
                    buf_ptr,
                    n,
                )
            };

            if n < 0 {
                error!("Failed to apply template, size: {n}");
                return Err(n);
            }

            Ok(unsafe { CString::from_raw(buf_ptr) })
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
            let n_tokens = unsafe {
                -llama_tokenize(
                    **self,
                    txt.as_ptr(),
                    txt.as_bytes().len() as i32,
                    core::ptr::null_mut(),
                    0i32,
                    add_special,
                    parse_special,
                )
            };

            // Create vec of tokens with corresponding size
            let mut result = vec![llama_token::default(); n_tokens as usize];
            // Tokenize again
            let check = unsafe {
                llama_tokenize(
                    **self,
                    txt.as_ptr(),
                    txt.as_bytes().len() as i32,
                    result.as_mut_ptr(),
                    n_tokens,
                    add_special,
                    parse_special,
                )
            };
            if check < 0 {
                return Err(format!(
                    "Failed to tokenize: check ({check}) != n_tokens ({n_tokens})"
                ));
            }

            info!("Tokenized {n_tokens} tokens: {result:?}");

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

    let (tx, rx) = mpsc::channel::<(
        Vec<llama::LlamaChatMessage>,
        mpsc::Sender<llama::LlamaGenerated>,
    )>();

    // Copy prompt string
    let prompt = prompt.to_owned();

    // Loop
    let lib_handle = tokio::task::spawn_blocking(|| {
        // init backend
        let _backend = llama::LlamaBackend::default();

        // Load model
        let model = llama::LlamaModel::new(model_path, None)?;

        model.run(rx)
    });

    // Client
    let (gen_tx, gen_rx) = mpsc::channel::<llama::LlamaGenerated>();
    let mut messages: Vec<llama::LlamaChatMessage> = Vec::new();
    messages.push(llama::LlamaChatMessage {
        role: CString::new("user").unwrap(),
        content: CString::new(prompt.clone()).unwrap(),
    });
    let _ = tx.send((messages.clone(), gen_tx));
    let mut res = "".to_string();
    let mut n_recv = 0;
    while let Ok(Ok(s)) = gen_rx.recv() {
        res += &s;
        n_recv += 1;
    }
    messages.push(llama::LlamaChatMessage {
        role: CString::new("assistant").unwrap(),
        content: CString::new(res).unwrap(),
    });

    let fmt_msg = format!("{messages:?}");
    info!(messages = %fmt_msg, %n_recv);

    // Drop tx after done sending
    drop(tx);

    // Wait for tasks
    let _ = lib_handle.await;

    // After returning, future holding loop is dropped, the loop is stopped
    Ok(())
}
