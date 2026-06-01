use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use inquire::Select;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    name = "ai",
    version,
    about = "Generate shell commands from natural-language prompts"
)]
struct Args {
    /// Natural-language command request.
    #[arg(required = true, trailing_var_arg = true)]
    prompt: Vec<String>,

    /// Print the command only, without explanations or selection UI.
    #[arg(long)]
    plain: bool,

    /// Dump AI request and response details to stderr.
    #[arg(long)]
    debug: bool,
}

#[derive(Debug, Deserialize)]
struct Config {
    api_key: String,
    model: String,
    #[serde(default = "default_base_url")]
    base_url: String,
    #[serde(default)]
    temperature: Option<f32>,
}

#[derive(Debug)]
struct ShellContext {
    shell: String,
    os: String,
    cwd: PathBuf,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ModelPlan {
    Command {
        command: String,
        explanation: String,
        #[serde(default)]
        dry_runs: Vec<String>,
    },
    Clarification {
        question: String,
        options: Vec<String>,
    },
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = Config::load()?;
    let client = LlmClient::new(config, args.debug)?;
    let shell_context = ShellContext::detect()?;
    let mut user_prompt = args.prompt.join(" ");

    for _ in 0..3 {
        let plan = client.plan_command(&shell_context, &user_prompt).await?;
        match plan {
            ModelPlan::Command {
                command,
                explanation,
                dry_runs,
            } => {
                if args.plain {
                    println!("{command}");
                    return Ok(());
                }
                present_command(&command, &explanation, &dry_runs);
                return Ok(());
            }
            ModelPlan::Clarification { question, options } => {
                if options.is_empty() {
                    bail!("model asked for clarification without providing options");
                }
                let answer = Select::new(&question, options)
                    .with_help_message("Use arrow keys and Enter to choose")
                    .prompt()
                    .context("failed to read clarification response")?;
                user_prompt = format!("{user_prompt}\nClarification: {question}\nAnswer: {answer}");
            }
        }
    }

    bail!("too many clarification rounds without a command");
}

struct LlmClient {
    http: reqwest::Client,
    config: Config,
    debug: bool,
}

impl LlmClient {
    fn new(config: Config, debug: bool) -> Result<Self> {
        let mut headers = HeaderMap::new();
        let auth_value = format!("Bearer {}", config.api_key);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).context("invalid API key header value")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            http,
            config,
            debug,
        })
    }

    async fn plan_command(&self, context: &ShellContext, prompt: &str) -> Result<ModelPlan> {
        let request = ChatRequest {
            model: self.config.model.clone(),
            temperature: self.config.temperature,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt(context),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
        };

        self.debug_request(&request)?;

        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let response = self
            .http
            .post(url)
            .json(&request)
            .send()
            .await
            .context("failed to call LLM API")?
            .error_for_status()
            .context("LLM API returned an error")?
            .json::<ChatResponse>()
            .await
            .context("failed to parse LLM API response")?;

        self.debug_response(&response)?;

        let content = response
            .choices
            .first()
            .map(|choice| choice.message.content.trim())
            .ok_or_else(|| anyhow!("LLM API response did not include any choices"))?;

        parse_model_plan(content)
    }

    fn debug_request(&self, request: &ChatRequest) -> Result<()> {
        if self.debug {
            eprintln!("--- ai-shell debug: request ---");
            eprintln!("{}", serde_json::to_string_pretty(request)?);
        }
        Ok(())
    }

    fn debug_response(&self, response: &ChatResponse) -> Result<()> {
        if self.debug {
            eprintln!("--- ai-shell debug: response ---");
            eprintln!("{}", serde_json::to_string_pretty(response)?);
        }
        Ok(())
    }
}

impl Config {
    fn load() -> Result<Self> {
        let path = config_path()?;
        if !path.exists() {
            write_config_template(&path)?;
            bail!(
                "created config template at {}. Fill in api_key and model, then run ai again",
                path.display()
            );
        }

        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read config at {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
    }
}

impl ShellContext {
    fn detect() -> Result<Self> {
        Ok(Self {
            shell: current_shell(),
            os: os_context(),
            cwd: env::current_dir().context("failed to detect current directory")?,
        })
    }
}

fn config_path() -> Result<PathBuf> {
    if let Some(config_home) = env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(config_home).join("ai-shell/config.toml"));
    }

    let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
    Ok(Path::new(&home).join(".config/ai-shell/config.toml"))
}

fn write_config_template(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }

    fs::write(
        path,
        r#"# ai-shell configuration
# OpenAI-compatible API endpoint. Override for local or third-party providers.
base_url = "https://api.openai.com/v1"

api_key = "replace-me"
model = "gpt-4.1-mini"
temperature = 0.2
"#,
    )
    .with_context(|| format!("failed to create config template at {}", path.display()))
}

fn current_shell() -> String {
    env::var("SHELL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn os_context() -> String {
    Command::new("uname")
        .arg("-a")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| env::consts::OS.to_string())
}

fn system_prompt(context: &ShellContext) -> String {
    format!(
        r#"You generate shell commands for a user.

Context:
- shell: {shell}
- os: {os}
- cwd: {cwd}

Return only JSON. Do not wrap it in Markdown.

When the request is specific enough, return:
{{
  "type": "command",
  "command": "single shell command",
  "explanation": "brief explanation",
  "dry_runs": ["optional safer preview commands"]
}}

When the request is too ambiguous or risky to answer safely, return:
{{
  "type": "clarification",
  "question": "short question",
  "options": ["option A", "option B"]
}}

Rules:
- The command must be a single command line for the user's shell.
- Prefer safe commands and include dry-run variants where practical.
- Do not use destructive flags unless the user explicitly asks for them.
- Ask a clarification question if required paths, targets, or intent are missing."#,
        shell = context.shell,
        os = context.os,
        cwd = context.cwd.display()
    )
}

fn parse_model_plan(content: &str) -> Result<ModelPlan> {
    let json = content
        .strip_prefix("```json")
        .and_then(|value| value.strip_suffix("```"))
        .or_else(|| {
            content
                .strip_prefix("```")
                .and_then(|value| value.strip_suffix("```"))
        })
        .unwrap_or(content)
        .trim();

    serde_json::from_str(json).context("model response was not valid plan JSON")
}

fn present_command(command: &str, explanation: &str, dry_runs: &[String]) {
    println!("{command}");
    println!();
    println!("{explanation}");

    if !dry_runs.is_empty() {
        println!();
        println!("Dry-run variants:");
        for dry_run in dry_runs {
            println!("- {dry_run}");
        }
    }

    if let Some(copy_command) = clipboard_command(command) {
        println!();
        println!("Copy command:");
        println!("{copy_command}");
    }
}

fn clipboard_command(command: &str) -> Option<String> {
    let escaped = command.replace('\'', "'\\''");
    if command_exists("pbcopy") {
        return Some(format!("printf '%s' '{escaped}' | pbcopy"));
    }
    if command_exists("wl-copy") {
        return Some(format!("printf '%s' '{escaped}' | wl-copy"));
    }
    if command_exists("xclip") {
        return Some(format!(
            "printf '%s' '{escaped}' | xclip -selection clipboard"
        ));
    }
    None
}

fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_command_plan() {
        let plan = parse_model_plan(
            r#"{
                "type": "command",
                "command": "ls -la",
                "explanation": "Lists files.",
                "dry_runs": ["ls"]
            }"#,
        )
        .unwrap();

        assert_eq!(
            plan,
            ModelPlan::Command {
                command: "ls -la".to_string(),
                explanation: "Lists files.".to_string(),
                dry_runs: vec!["ls".to_string()],
            }
        );
    }

    #[test]
    fn parses_markdown_wrapped_plan() {
        let plan = parse_model_plan(
            r#"```json
{"type":"clarification","question":"Which branch?","options":["main","current"]}
```"#,
        )
        .unwrap();

        assert_eq!(
            plan,
            ModelPlan::Clarification {
                question: "Which branch?".to_string(),
                options: vec!["main".to_string(), "current".to_string()],
            }
        );
    }

    #[test]
    fn config_default_base_url_is_openai() {
        let config: Config = toml::from_str(
            r#"
api_key = "test"
model = "gpt-4.1-mini"
"#,
        )
        .unwrap();

        assert_eq!(config.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn writes_config_template() {
        let path =
            env::temp_dir().join(format!("ai-shell-test-{}-config.toml", std::process::id()));
        let _ = fs::remove_file(&path);

        write_config_template(&path).unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("api_key = \"replace-me\""));
        assert!(raw.contains("model = \"gpt-4.1-mini\""));

        fs::remove_file(path).unwrap();
    }
}
