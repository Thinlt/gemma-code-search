---
name: Bug report
about: Something is broken — a crash, a wrong result, or an install failure
title: ""
labels: bug
assignees: ""
---

<!--
Please search existing issues first: https://github.com/Thinlt/gemma-code-search/issues
Do not report security vulnerabilities here — see SECURITY.md.
-->

## What happened

A clear description of the bug.

## Steps to reproduce

1.
2.
3.

Command(s) you ran:

```bash
# e.g. code-search "rate limiting" --top-k 6
```

## Expected behaviour

What you expected to happen instead.

## Actual output

Paste the relevant output. Add `--json` if it helps. Redact paths or code you cannot share.

<details>
<summary>Output</summary>

```text

```

</details>

## Environment

| | |
|---|---|
| OS / version | e.g. macOS 15.2 (Apple Silicon), Ubuntu 24.04 |
| Python | `python3 --version` |
| Install method | `./install.sh` / `--skills-global` / `--project <repo>` / uv run from clone |
| Backend | torch (MPS / CUDA / CPU) / ONNX int8 / ANE |
| Agent (if any) | Claude Code / Cursor / Antigravity / opencode / Gemini CLI / plain CLI |
| Indexed repo size | rough file count, e.g. ~2k files |

## Additional context

Logs from the daemon (`uv run --project scripts python scripts/cli.py daemon --status`), screenshots,
or anything else that helps.
