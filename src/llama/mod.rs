use libllama_sys::*;

//================================================== Redefine some defaults

pub const LLAMA_TOKEN_NULL: llama_token = -1;

unsafe extern "C" fn nolog_callback(
    _level: ggml_log_level,
    _text: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
}

//================================================ Alias for generated chunk
mod generated;
pub use generated::*;

//===================================================== LlamaBackend wrapper
mod backend;
pub use backend::*;

//================================================== LlamaThreadpool wrapper
mod threadpool;
pub use threadpool::*;

//======================================================= LlamaModel wrapper
mod model;
pub use model::*;

//================================================ LlamaChatMessage wrrapper
mod chat_message;
pub use chat_message::*;

//=============================================== LlamaChatTemplate wrrapper
mod chat_template;
pub use chat_template::*;

//====================================================== LlamaVocab wrrapper
mod vocab;
pub use vocab::*;

//===================================================== LlamaContext wrapper
mod context;
pub use context::*;

//===================================================== LlamaSampler wrapper
mod sampler;
pub use sampler::*;
