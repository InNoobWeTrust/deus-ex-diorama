use tracing::info;

use super::*;
use core::ops::Deref;
use std::ffi::CString;

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
