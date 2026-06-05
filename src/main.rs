use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use inquire::{Select, Text};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

const CUSTOM_ANSWER_OPTION: &str = "Write your own";

#[derive(Parser, Debug)]
#[command(
    name = "ai",
    version,
    about = "Generate shell commands from natural-language prompts"
)]
struct Args {
    /// Natural-language command request.
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    /// Print the command only, without explanations or selection UI.
    #[arg(long)]
    plain: bool,

    /// Dump AI request and response details to stderr.
    #[arg(long)]
    debug: bool,

    /// Attach `ls -la` output for PATH as prompt context. Defaults to the current directory.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = ".")]
    ls: Vec<PathBuf>,

    /// Print recent history prompts for shell completions.
    #[arg(long, hide = true)]
    history_completions: bool,
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
    listings: Vec<DirectoryListing>,
}

#[derive(Debug)]
struct DirectoryListing {
    path: PathBuf,
    output: String,
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

struct HistoryDb {
    connection: Connection,
}

fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.history_completions {
        let history = HistoryDb::open()?;
        history.print_completion_prompts()?;
        return Ok(());
    }

    if args.prompt.is_empty() {
        bail!("missing prompt");
    }

    let config = Config::load()?;
    let history = HistoryDb::open()?;
    let client = LlmClient::new(config, history, args.debug)?;
    let shell_context = ShellContext::detect(&args.ls)?;
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
                let revision = handle_command_session(&command, &explanation, &dry_runs)?;
                if let Some(revision) = revision {
                    user_prompt = format!(
                        "{user_prompt}\nPrevious command: {command}\nRevision request: {revision}"
                    );
                    continue;
                }
                return Ok(());
            }
            ModelPlan::Clarification { question, options } => {
                let answer = ask_free_form_question(&question, &options)?;
                user_prompt = format!("{user_prompt}\nClarification: {question}\nAnswer: {answer}");
            }
        }
    }

    bail!("too many clarification rounds without a command");
}

struct LlmClient {
    http: reqwest::Client,
    config: Config,
    history: HistoryDb,
    debug: bool,
}

impl LlmClient {
    fn new(config: Config, history: HistoryDb, debug: bool) -> Result<Self> {
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
            history,
            debug,
        })
    }

    async fn plan_command(&self, context: &ShellContext, prompt: &str) -> Result<ModelPlan> {
        let system = system_prompt(context);
        let request = ChatRequest {
            model: self.config.model.clone(),
            temperature: self.config.temperature,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system.clone(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
        };

        if let Some(response) = self.history.lookup(&self.config.model, &system, prompt)? {
            self.debug_cache_hit(&request, &response)?;
            return plan_from_response(&response);
        }

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
        self.history
            .store(&self.config.model, &system, prompt, &request, &response)?;

        plan_from_response(&response)
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

    fn debug_cache_hit(&self, request: &ChatRequest, response: &ChatResponse) -> Result<()> {
        if self.debug {
            eprintln!("--- ai-shell debug: cache hit ---");
            eprintln!("request:");
            eprintln!("{}", serde_json::to_string_pretty(request)?);
            eprintln!("cached response:");
            eprintln!("{}", serde_json::to_string_pretty(response)?);
        }
        Ok(())
    }
}

impl HistoryDb {
    fn open() -> Result<Self> {
        Self::open_at(history_path()?)
    }

    fn open_at(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create history directory {}", parent.display())
            })?;
        }

        let connection = Connection::open(&path)
            .with_context(|| format!("failed to open history database {}", path.display()))?;
        let db = Self { connection };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.connection.execute_batch(
            r#"
CREATE TABLE IF NOT EXISTS llm_exchanges (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    use_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(model, system_prompt, user_prompt)
);
"#,
        )?;
        Ok(())
    }

    fn lookup(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<Option<ChatResponse>> {
        let response_json = self
            .connection
            .query_row(
                r#"
SELECT response_json
FROM llm_exchanges
WHERE model = ?1 AND system_prompt = ?2 AND user_prompt = ?3
"#,
                params![model, system_prompt, user_prompt],
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        let Some(response_json) = response_json else {
            return Ok(None);
        };

        self.connection.execute(
            r#"
UPDATE llm_exchanges
SET last_used_at = CURRENT_TIMESTAMP,
    use_count = use_count + 1
WHERE model = ?1 AND system_prompt = ?2 AND user_prompt = ?3
"#,
            params![model, system_prompt, user_prompt],
        )?;

        serde_json::from_str(&response_json)
            .context("cached LLM response was not valid response JSON")
            .map(Some)
    }

    fn store(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        request: &ChatRequest,
        response: &ChatResponse,
    ) -> Result<()> {
        self.connection.execute(
            r#"
INSERT INTO llm_exchanges (
    model,
    system_prompt,
    user_prompt,
    request_json,
    response_json
) VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT(model, system_prompt, user_prompt) DO UPDATE SET
    request_json = excluded.request_json,
    response_json = excluded.response_json,
    last_used_at = CURRENT_TIMESTAMP
"#,
            params![
                model,
                system_prompt,
                user_prompt,
                serde_json::to_string(request)?,
                serde_json::to_string(response)?,
            ],
        )?;
        Ok(())
    }

    fn print_completion_prompts(&self) -> Result<()> {
        for prompt in self.completion_prompts()? {
            println!("{prompt}");
        }

        Ok(())
    }

    fn completion_prompts(&self) -> Result<Vec<String>> {
        let mut statement = self.connection.prepare(
            r#"
SELECT user_prompt
FROM llm_exchanges
GROUP BY user_prompt
ORDER BY MAX(last_used_at) DESC, MAX(id) DESC
LIMIT 50
"#,
        )?;

        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        let mut prompts = Vec::new();
        for row in rows {
            let prompt = row?;
            if let Some(first_line) = prompt.lines().next()
                && !first_line.trim().is_empty()
            {
                prompts.push(first_line.to_string());
            }
        }

        Ok(prompts)
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
    fn detect(listing_paths: &[PathBuf]) -> Result<Self> {
        Ok(Self {
            shell: current_shell(),
            os: os_context(),
            cwd: env::current_dir().context("failed to detect current directory")?,
            listings: collect_directory_listings(listing_paths)?,
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

fn history_path() -> Result<PathBuf> {
    if let Some(data_home) = env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(data_home).join("ai-shell/history.sqlite"));
    }

    let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
    Ok(Path::new(&home).join(".local/share/ai-shell/history.sqlite"))
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

fn collect_directory_listings(paths: &[PathBuf]) -> Result<Vec<DirectoryListing>> {
    paths
        .iter()
        .map(|path| {
            if !path.is_dir() {
                bail!("--ls path is not a directory: {}", path.display());
            }

            let output = Command::new("ls")
                .arg("-la")
                .arg(path)
                .output()
                .with_context(|| format!("failed to run ls -la {}", path.display()))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                bail!("ls -la {} failed: {}", path.display(), stderr.trim());
            }

            Ok(DirectoryListing {
                path: path.clone(),
                output: String::from_utf8_lossy(&output.stdout)
                    .trim_end()
                    .to_string(),
            })
        })
        .collect()
}

fn system_prompt(context: &ShellContext) -> String {
    let mut prompt = format!(
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
- Always add flags to print progress when working with normally-silent tools.
- Ask a clarification question if required paths, targets, or intent are missing."#,
        shell = context.shell,
        os = context.os,
        cwd = context.cwd.display()
    );

    if !context.listings.is_empty() {
        prompt.push_str("\n\nDirectory listings:");
        for listing in &context.listings {
            prompt.push_str(&format!(
                "\n\n`ls -la {}`:\n```text\n{}\n```",
                listing.path.display(),
                listing.output
            ));
        }
    }

    prompt
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

fn plan_from_response(response: &ChatResponse) -> Result<ModelPlan> {
    let content = response
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .ok_or_else(|| anyhow!("LLM API response did not include any choices"))?;

    parse_model_plan(content)
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
}

fn ask_free_form_question(question: &str, options: &[String]) -> Result<String> {
    let mut choices = options.to_vec();
    choices.push(CUSTOM_ANSWER_OPTION.to_string());

    let selected = Select::new(question, choices)
        .with_help_message("Use arrow keys and Enter to choose")
        .prompt()
        .context("failed to read clarification response")?;

    let answer = if selected == CUSTOM_ANSWER_OPTION {
        Text::new("Your answer:")
            .prompt()
            .context("failed to read custom clarification response")?
    } else {
        selected
    };

    let answer = answer.trim();
    if answer.is_empty() {
        bail!("clarification answer cannot be empty");
    }

    Ok(answer.to_string())
}

fn handle_command_session(
    command: &str,
    explanation: &str,
    dry_runs: &[String],
) -> Result<Option<String>> {
    present_command(command, explanation, dry_runs);

    let action = Select::new("What would you like to do?", command_actions(dry_runs))
        .with_help_message("Use arrow keys and Enter to choose")
        .prompt()
        .context("failed to read command action")?;

    match action {
        CommandAction::Run => {
            run_shell_command(command)?;
            Ok(None)
        }
        CommandAction::Edit => {
            let edited = Text::new("Edit command:")
                .with_initial_value(command)
                .prompt()
                .context("failed to read edited command")?;
            run_shell_command(&edited)?;
            Ok(None)
        }
        CommandAction::RunDryRun => {
            let dry_run = Select::new("Which dry-run command?", dry_runs.to_vec())
                .with_help_message("Use arrow keys and Enter to choose")
                .prompt()
                .context("failed to read dry-run selection")?;
            run_shell_command(&dry_run)?;
            Ok(None)
        }
        CommandAction::RequestRevision => {
            let revision = Text::new("What should change?")
                .prompt()
                .context("failed to read revision request")?;
            Ok(Some(revision))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CommandAction {
    Run,
    Edit,
    RunDryRun,
    RequestRevision,
}

impl std::fmt::Display for CommandAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Run => write!(formatter, "Run"),
            Self::Edit => write!(formatter, "Edit"),
            Self::RunDryRun => write!(formatter, "Run Dry-Run"),
            Self::RequestRevision => write!(formatter, "Request Revision"),
        }
    }
}

fn command_actions(dry_runs: &[String]) -> Vec<CommandAction> {
    let mut actions = vec![CommandAction::Run, CommandAction::Edit];
    if !dry_runs.is_empty() {
        actions.push(CommandAction::RunDryRun);
    }
    actions.push(CommandAction::RequestRevision);
    actions
}

fn run_shell_command(command: &str) -> Result<()> {
    let shell = current_shell();
    let status = Command::new(shell)
        .arg("-c")
        .arg(command)
        .status()
        .with_context(|| format!("failed to run command: {command}"))?;

    if !status.success() {
        bail!("command exited with status {status}");
    }

    Ok(())
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

    #[test]
    fn stores_and_replays_history_response() {
        let path = temp_history_path("stores-and-replays");
        let _ = fs::remove_file(&path);
        let history = HistoryDb::open_at(path.clone()).unwrap();
        let request = test_request("show files");
        let response = test_response(
            r#"{"type":"command","command":"ls","explanation":"Lists files.","dry_runs":[]}"#,
        );

        assert!(
            history
                .lookup("model", "system", "show files")
                .unwrap()
                .is_none()
        );
        history
            .store("model", "system", "show files", &request, &response)
            .unwrap();

        let cached = history
            .lookup("model", "system", "show files")
            .unwrap()
            .unwrap();
        assert_eq!(
            plan_from_response(&cached).unwrap(),
            ModelPlan::Command {
                command: "ls".to_string(),
                explanation: "Lists files.".to_string(),
                dry_runs: Vec::new(),
            }
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn cached_clarification_response_replays_as_question() {
        let response = test_response(
            r#"{"type":"clarification","question":"Which branch?","options":["main","current"]}"#,
        );

        assert_eq!(
            plan_from_response(&response).unwrap(),
            ModelPlan::Clarification {
                question: "Which branch?".to_string(),
                options: vec!["main".to_string(), "current".to_string()],
            }
        );
    }

    #[test]
    fn history_completion_prompts_are_newest_first_unique_first_lines() {
        let path = temp_history_path("completion-prompts");
        let _ = fs::remove_file(&path);
        let history = HistoryDb::open_at(path.clone()).unwrap();
        let response = test_response(
            r#"{"type":"command","command":"ls","explanation":"Lists files.","dry_runs":[]}"#,
        );

        history
            .store(
                "model",
                "system",
                "list files",
                &test_request("list files"),
                &response,
            )
            .unwrap();
        history
            .store(
                "model",
                "system",
                "remove cache\nClarification: Which cache?\nAnswer: npm",
                &test_request("remove cache"),
                &response,
            )
            .unwrap();

        let prompts = history.completion_prompts().unwrap();
        assert_eq!(
            prompts,
            vec!["remove cache".to_string(), "list files".to_string()]
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn dry_run_action_only_appears_when_available() {
        assert_eq!(
            command_actions(&[]),
            vec![
                CommandAction::Run,
                CommandAction::Edit,
                CommandAction::RequestRevision,
            ]
        );

        assert_eq!(
            command_actions(&["ls --dry-run".to_string()]),
            vec![
                CommandAction::Run,
                CommandAction::Edit,
                CommandAction::RunDryRun,
                CommandAction::RequestRevision,
            ]
        );
    }

    #[test]
    fn system_prompt_includes_directory_listing_context() {
        let context = ShellContext {
            shell: "/bin/sh".to_string(),
            os: "test-os".to_string(),
            cwd: PathBuf::from("/tmp/example"),
            listings: vec![DirectoryListing {
                path: PathBuf::from("."),
                output: "total 0\n-rw-r--r-- file.txt".to_string(),
            }],
        };

        let prompt = system_prompt(&context);
        assert!(prompt.contains("Directory listings:"));
        assert!(prompt.contains("`ls -la .`:"));
        assert!(prompt.contains("-rw-r--r-- file.txt"));
    }

    #[test]
    fn directory_listing_errors_for_non_directory() {
        let path = env::temp_dir().join(format!("ai-shell-file-{}", std::process::id()));
        fs::write(&path, "not a directory").unwrap();

        let error = collect_directory_listings(std::slice::from_ref(&path)).unwrap_err();
        assert!(error.to_string().contains("--ls path is not a directory"));

        fs::remove_file(path).unwrap();
    }

    fn temp_history_path(name: &str) -> PathBuf {
        env::temp_dir().join(format!("ai-shell-{name}-{}.sqlite", std::process::id()))
    }

    fn test_request(prompt: &str) -> ChatRequest {
        ChatRequest {
            model: "model".to_string(),
            temperature: None,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "system".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
        }
    }

    fn test_response(content: &str) -> ChatResponse {
        ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: content.to_string(),
                },
            }],
        }
    }
}
