use super::*;
use std::ffi::CString;

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
