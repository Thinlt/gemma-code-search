# Security Policy

## Supported versions

Gemma Code Search is pre-1.0 and ships from the `main` branch. Security fixes are applied to
`main` only; there are no maintained release branches yet.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| older snapshots / forks | ❌ |

## Reporting a vulnerability

**Please do not report security vulnerabilities in public issues, discussions or pull requests.**

Use GitHub's private vulnerability reporting: go to the
[Security tab](https://github.com/Thinlt/gemma-code-search/security/advisories/new) of this
repository and open a draft advisory. If you cannot use that form, contact the maintainer
[@Thinlt](https://github.com/Thinlt) to arrange a private channel.

Please include, as far as you can:

- A description of the issue and its impact
- Steps to reproduce, or a minimal proof of concept
- The affected component (`scripts/cli.py`, `scripts/index.py`, `scripts/repo_ignore.py`, `scripts/daemon.py`, the shim, `install.sh`, …)
- Your operating system, Python version and backend (torch / ONNX / ANE)

## What to expect

- **Acknowledgement** within 5 working days.
- An assessment and a proposed remediation timeline after triage.
- Credit in the advisory once a fix is released, unless you prefer to stay anonymous.

We ask that you give us a reasonable amount of time to release a fix before any public
disclosure.

## Scope

This project runs **locally** — there is no hosted service. The most relevant risks are therefore:

- **Path handling** — the indexer and daemon read files from `--code-dir`; unsafe path resolution
  or symlink handling that escapes the intended repository root is in scope.
- **Local IPC** — the daemon listens on a Unix socket (or TCP localhost when
  `EMBEDDINGGEMMA_TCP=1`). Anything that lets another local user send crafted requests, read
  another user's index data, or hijack the socket is in scope.
- **Installer / shim scripts** — `install.sh` and `shim/*.sh` write into `$HOME/.local/bin` and
  agent skill directories; command injection or unsafe use of environment variables is in scope.
- **Dependencies** — vulnerabilities in upstream packages should be reported upstream, but do tell
  us if this project pins or invokes them unsafely.

## Out of scope

- Model output quality — irrelevant, misleading or "hallucinated" search results are correctness
  bugs, not security issues. Report them as regular issues.
- Attacks that require an already-compromised local account with read access to the project files.
- Resource exhaustion from indexing a repository you chose to index.

## Safe harbor

We consider security research conducted in good faith under this policy to be authorized. We will
not pursue or support legal action against researchers who follow it, and we will work with you to
understand and resolve the issue.
