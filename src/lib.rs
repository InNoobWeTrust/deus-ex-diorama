use core::error::Error;
use std::path::PathBuf;
use serde_derive::{Deserialize, Serialize};

pub async fn hf_load_file(repo: &str, file: &str) -> Result<PathBuf, Box<dyn Error>> {
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model(repo.to_string());
    let file_name = repo.get(file).await?;

    Ok(file_name)
}

pub mod gen_ai {
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Provider {
        OpenAi,
        Anthropic,
        Groq,
        SelfHosted,
        Local,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ChatRole {
        System,
        User,
        Developer,
        Assistant,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum StrOrArr {
        Str(String),
        Arr(Vec<String>),
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub struct Message {
        role: ChatRole,
        content: StrOrArr,
        name: Option<String>,
    }
}
