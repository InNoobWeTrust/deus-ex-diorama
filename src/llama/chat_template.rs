use tracing::error;

use super::*;
use core::ops::Deref;
use std::ffi::CString;

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
