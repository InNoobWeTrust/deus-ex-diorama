use std::str::FromStr;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChunkResponse {
    id: String,
    choices: Vec<Choice>,
    created: u64,
    model: String,
    service_tier: Option<String>,
    system_fingerprint: String,
    #[doc = r"The object type, which is always `chat.completion.chunk`."]
    object: String,
    usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Choice {
    delta: choice::Delta,
    logprobs: Option<choice::Logprobs>,
    finish_reason: Option<String>,
    index: usize,
}

pub mod choice {
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub struct Delta {}

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub struct Logprobs {
        content: Vec<logprobs::Content>,
        refusal: Vec<logprobs::Refusal>,
    }

    pub mod logprobs {
        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub struct Content {
            token: String,
            logprob: serde_json::Number,
            bytes: Vec<u8>,
            top_logprobs: Vec<content::TopLogprobs>,
        }

        pub mod content {
            #[derive(Debug, Serialize, Deserialize)]
            #[serde(rename_all = "snake_case")]
            pub struct TopLogprobs {
                token: String,
                logprob: serde_json::Number,
                bytes: Vec<u8>,
            }
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub struct Refusal {
            token: String,
            logprob: serde_json::Number,
            bytes: Vec<u8>,
            top_logprobs: Vec<refusal::TopLogprobs>,
        }

        pub mod refusal {
            #[derive(Debug, Serialize, Deserialize)]
            #[serde(rename_all = "snake_case")]
            pub struct TopLogprobs {
                token: String,
                logprob: serde_json::Number,
                bytes: Vec<u8>,
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    completion_tokens: usize,
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ParseChunkResponseError {
    reason: String,
    is_done: bool,
}

impl FromStr for ChunkResponse {
    type Err = ParseChunkResponseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if !s.starts_with("data: ") {
            return Err(Self::Err{
                reason: "Invalid chunk".into(),
                is_done: false,
            });
        }

        if s.ends_with("[DONE]") {
            return Err(Self::Err{
                reason: "Done".into(),
                is_done: true,
            })
        }

        let typed_chunk = serde_json::from_str::<Self>(&s[6..]);

        match typed_chunk {
            Ok(typed_chunk) => Ok(typed_chunk),
            Err(e) => Err(Self::Err{reason: e.to_string(), is_done: false}),
        }
    }
}
