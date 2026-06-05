# ai-shell

`ai-shell` provides an `ai` command that turns natural-language requests into shell commands. It sends your prompt plus local context to an OpenAI-compatible chat API, then presents the generated command with a short explanation, dry-run alternatives where practical, and an interactive action menu.

## Features

- Generates one shell command from a natural-language prompt.
- Sends shell, `uname -a`, and current working directory context to the model.
- Supports OpenAI-compatible APIs through a TOML config file.
- Replays cached LLM responses from SQLite history to save tokens.
- Replays cached clarification questions interactively.
- Offers `Run`, `Edit`, and `Request Revision` actions before execution.
- Provides Fish shell completions backed by recent prompt history.
- Includes `--debug` to inspect AI requests, responses, and cache hits.

## Installation

Build locally with Cargo:

```sh
cargo build --release
install -Dm755 target/release/ai ~/.local/bin/ai
```

On Arch Linux, build and install the in-place package:

```sh
cd packaging/arch
makepkg -Cfi
```

The package installs the binary to `/usr/bin/ai` and Fish completions to `/usr/share/fish/vendor_completions.d/ai.fish`.

## Configuration

On first run, `ai` creates a config template at:

```text
$XDG_CONFIG_HOME/ai-shell/config.toml
```

or:

```text
~/.config/ai-shell/config.toml
```

Example:

```toml
base_url = "https://api.openai.com/v1"
api_key = "replace-me"
model = "gpt-4.1-mini"
temperature = 0.2
```

## Usage

```sh
ai list the largest files under this directory
ai --plain show listening TCP ports
ai --debug remove old build artifacts
ai --ls -- summarize this directory
ai --ls --ls /var/log -- find large logs
```

Normal mode shows the generated command and explanation, then asks whether to run it, edit it first, or request a revision.

Use `--ls` to attach `ls -la` output as extra model context. On its own it lists the current directory; with a path it lists that directory. The flag can be repeated. When using bare `--ls`, add `--` before the prompt so the first prompt word is not parsed as a directory path.

## History

History is stored in SQLite at:

```text
$XDG_DATA_HOME/ai-shell/history.sqlite
```

or:

```text
~/.local/share/ai-shell/history.sqlite
```

The database stores full request and response JSON. Cached responses are keyed by model, prompt, and system context, so commands generated in one working directory are not silently replayed in another.

## Development

```sh
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
```
