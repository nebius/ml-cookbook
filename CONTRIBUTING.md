# Contributing

Thanks for adding a recipe! To keep the cookbook consistent as more people
contribute, every pull request runs a small set of automated quality checks.
You can — and should — run them locally before pushing.

## One-time setup

Requires Python 3.12+.

```bash
pip install pre-commit
pre-commit install      # run the checks automatically on every `git commit`
pre-commit autoupdate   # pin the hook versions to their latest releases
```

## Run all checks manually

```bash
pre-commit run --all-files
```

Most issues are auto-fixed in place (formatting, import order, trailing
whitespace). Re-stage the changes and commit again.

## What runs

| Area | Tool | What it checks |
| --- | --- | --- |
| Python + notebooks | [Ruff](https://github.com/astral-sh/ruff) | Linting and formatting for `.py` and `.ipynb` |
| Python + notebooks | [Bandit](https://github.com/PyCQA/bandit) (via [nbQA](https://github.com/nbQA-dev/nbQA) on notebooks) | Common security issues |
| Shell | [ShellCheck](https://github.com/koalaman/shellcheck) | Bugs and pitfalls in `bash`/`sh` scripts |

Tool settings live in `pyproject.toml` (Ruff, Bandit) and `.shellcheckrc`
(ShellCheck). Prefer adjusting rules there over scattering inline ignores.

## CI

The same checks run on every pull request via
`.github/workflows/code-quality.yml`. PRs need to be green to merge — turn on
branch protection for `main` and mark the `code-quality` checks as required so
this is enforced automatically.
