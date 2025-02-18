use super::*;
use std::ffi::CString;

pub type LlamaGenerated = Result<String, String>;

#[derive(Debug, Clone, Copy)]
pub struct LlamaGeneratedChunks<'a> {
    pub ctx: &'a LlamaContext,
    pub vocab: &'a LlamaVocab,
    pub smpl: &'a LlamaSampler,
    pub batch: llama_batch,
    pub n_pos: usize,
    pub is_err: bool,
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
        let n = unsafe { llama_token_to_piece(**self.vocab, new_token_id, buf_ptr, n, 0, true) };

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
