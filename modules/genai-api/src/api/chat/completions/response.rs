pub mod chunk;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Response {
    id: String,
    choices: Vec<Choice>,
    created: u64,
    model: String,
    service_tier: Option<String>,
    #[doc = r#"This fingerprint represents the backend configuration that the model runs with.

Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism."#]
    system_fingerprint: String,
    #[doc = r"The object type, which is always `chat.completion`."]
    object: String,
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Choice {
    #[doc = r"The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, `content_filter` if content was omitted due to a flag from our content filters, `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function."]
    finish_reason: String,
    index: usize,
    message: choice::Message,
    logprobs: choice::LogProbs,
}

pub mod choice {
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub struct Message {
        content: Option<String>,
        refusal: Option<String>,
        tool_calls: Vec<message::ToolCall>,
    }

    pub mod message {
        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub struct ToolCall {
            id: String,
            r#type: String,
            function: tool_call::Function,
        }

        pub mod tool_call {
            #[derive(Debug, Serialize, Deserialize)]
            #[serde(rename_all = "snake_case")]
            pub struct Function {
                name: String,
                arguments: String,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub struct LogProbs {
        content: Vec<logprobs::Content>,
        refusal: Vec<logprobs::Refusal>,
    }

    pub mod logprobs {
        use serde_json::Number;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub struct Content {
            token: String,
            #[doc = r"The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely."]
            logprob: Number,
            #[doc = r"A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token."]
            bytes: Vec<u8>,
            top_logprobs: Vec<content::TopLogprob>,
        }

        pub mod content {
            use serde_json::Number;

            #[derive(Debug, Serialize, Deserialize)]
            #[serde(rename_all = "snake_case")]
            pub struct TopLogprob {
                token: String,
                lobprob: Number,
                bytes: Vec<u8>,
            }
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub struct Refusal {
            token: String,
            logprob: Number,
            bytes: Vec<u8>,
            top_logprobs: Vec<refusal::TopLogprob>,
        }

        pub mod refusal {
            use serde_json::Number;

            #[derive(Debug, Serialize, Deserialize)]
            #[serde(rename_all = "snake_case")]
            pub struct TopLogprob {
                token: String,
                lobprob: Number,
                bytes: Vec<u8>,
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    completion_tokens: usize,
    total_tokens: usize,
    completion_tokens_details: usage::CompletionTokensDetails,
    prompt_tokens_details: usage::PromptTokensDetails,
}

pub mod usage {
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub struct CompletionTokensDetails {
        accepted_prediction_tokens: usize,
        audio_tokens: usize,
        reasoning_tokens: usize,
        rejected_prediction_tokens: usize,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub struct PromptTokensDetails {
        audio_tokens: usize,
        cached_tokens: usize,
    }
}
