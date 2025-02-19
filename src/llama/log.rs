use super::*;
use std::ffi::CStr;

pub unsafe extern "C" fn nolog_callback(
    level: ggml_log_level,
    text: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
    let msg = unsafe { CStr::from_ptr(text) };
    let msg = msg
        .to_str()
        .map(|s| s.to_owned())
        .unwrap_or(format!("{msg:?}"));
    match level {
        GGML_LOG_LEVEL_DEBUG => debug!(target: "llama.cpp", msg),
        GGML_LOG_LEVEL_INFO => info!(target: "llama.cpp", msg),
        GGML_LOG_LEVEL_WARN => warn!(target: "llama.cpp", msg),
        GGML_LOG_LEVEL_ERROR => error!(target: "llama.cpp", msg),
        _ => {}
    }
}
