# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Rust binary crate named `ai-shell`.

- `Cargo.toml` defines package metadata, the Rust edition, and dependencies.
- `src/main.rs` contains the application entry point.
- Add integration tests under `tests/` when behavior is exercised through the compiled binary or public APIs.
- Add reusable modules under `src/` and declare them from `main.rs` with `mod module_name;`.

Keep generated build output in `target/`; it should not be committed.

## Build, Test, and Development Commands

- `cargo run` builds and runs the local binary.
- `cargo build` compiles the project in debug mode.
- `cargo build --release` creates an optimized release build in `target/release/`.
- `cargo test` runs unit and integration tests.
- `cargo fmt` formats Rust code using `rustfmt`.
- `cargo clippy --all-targets --all-features` runs Clippy lints across the crate.

Run `cargo fmt` and `cargo test` before opening a pull request.

## Coding Style & Naming Conventions

Use standard Rust formatting with four-space indentation, as enforced by `rustfmt`. Follow Rust naming conventions:

- `snake_case` for functions, variables, modules, and test names.
- `PascalCase` for structs, enums, traits, and type aliases.
- `SCREAMING_SNAKE_CASE` for constants and statics.

Prefer small functions with explicit error handling. When dependencies are added, keep them minimal and document why they are needed in the relevant change.

## Testing Guidelines

Place unit tests next to the code they cover inside `#[cfg(test)] mod tests`. Use `tests/` for integration tests that should exercise the crate from the outside.

Name tests after the behavior being verified, for example `parses_valid_command` or `returns_error_for_empty_input`. Add tests for new parsing, command execution, configuration, and error-handling behavior.

Run the full suite with:

```sh
cargo test
```

## Commit & Pull Request Guidelines

The current history only contains an initial commit, so no strict project-specific convention exists yet. Use short, imperative commit messages, for example `Add command parser` or `Handle empty input`.

Pull requests should include a concise summary, test results, and any relevant issue links. For user-facing shell behavior, include sample commands and output when helpful. Keep PRs focused; separate unrelated refactors from feature or bug-fix changes.

## Security & Configuration Tips

Be careful when adding code that executes shell commands or reads environment variables. Validate inputs, avoid logging secrets, and prefer explicit allowlists for command behavior where practical.
