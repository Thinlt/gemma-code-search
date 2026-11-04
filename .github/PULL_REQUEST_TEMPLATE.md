## Summary

<!-- What does this pull request change, and why? -->

## Related issue

<!-- "Closes #123" if applicable -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation
- [ ] Refactor / performance
- [ ] Regression fix

## How was it tested?

| | |
|---|---|
| OS / hardware | |
| Backend tested | torch (MPS / CUDA / CPU) / ONNX / ANE |
| Repository used for testing | rough size, e.g. ~2k files |
| Commands run | |

```text
# paste the relevant output
```

## Checklist

- [ ] I read [DESIGN.md](../DESIGN.md) and this change does not silently reverse a settled decision
      (or the reversal was discussed in an issue first).
- [ ] `uvx ruff check scripts` is clean.
- [ ] I exercised the change end-to-end on a real repository.
- [ ] `SKILL.md` is updated if the agent-facing behaviour changed.
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`.
- [ ] A new dependency was added to **both** `scripts/pyproject.toml` and `scripts/requirements.txt`,
      and the lockfile was refreshed.
- [ ] No model weights, indexes, or virtual environments are committed.

## Notes for reviewers

<!-- Anything you are unsure about, trade-offs, follow-up work. -->
