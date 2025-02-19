use libllama_sys::*;

//======================================================= Redefine some defaults
pub const LLAMA_TOKEN_NULL: llama_token = -1;

//====================================================================== Logging
mod log;
pub use log::*;

//==================================================== Alias for generated chunk
mod generated;
pub use generated::*;

//========================================================= LlamaBackend wrapper
mod backend;
pub use backend::*;

//====================================================== LlamaThreadpool wrapper
mod threadpool;
pub use threadpool::*;

//=========================================================== LlamaModel wrapper
mod model;
pub use model::*;

//==================================================== LlamaChatMessage wrrapper
mod chat_message;
pub use chat_message::*;

//=================================================== LlamaChatTemplate wrrapper
mod chat_template;
pub use chat_template::*;

//========================================================== LlamaVocab wrrapper
mod vocab;
use tracing::{debug, error, info, warn};
pub use vocab::*;

//========================================================= LlamaContext wrapper
mod context;
pub use context::*;

//========================================================= LlamaSampler wrapper
mod sampler;
pub use sampler::*;
