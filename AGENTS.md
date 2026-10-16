# functional-relational-python

A Python library, published to PyPI as `functional-relational-python`. Python 3.14.

## Layout

A uv workspace with two members:

| Package | Import name | Published | Contents |
| --- | --- | --- | --- |
| `packages/functional-relational-python` | `functional_relational_python` | Yes | The library, and the user CLI `frp` (`cli.py`) |
| `packages/frp-dev` | `frp_dev` | No | The dev CLI `frp-dev`: `setup` and `teardown` |

The library never imports `frp_dev`; `tach` enforces this.

## Setup

A fresh clone or worktree needs `uv` and [proto](https://moonrepo.dev/docs/proto/install). Then:

```
uv run frp-dev setup      # proto install, uv sync, lefthook install, symlink docs/
uv run frp-dev teardown   # removes the docs/ symlink before deleting a worktree
```

Orca runs both automatically (`orca.yaml`). `setup` symlinks the docs folder in as
`docs/` (gitignored). It reads the docs folder from `$FRP_DOCS_SOURCE`, falling back to
`~/Git/functional-relational-python-docs/functional-relational-python-docs`.

## Python

Follow the conventions in `CODE-CONVENTIONS.md`, referenced by ID (e.g. TE-9c).

- **No `tests/` folder.** Tests live next to the code they test, as `test_<module>.py`. They are excluded from the published distribution (`MANIFEST.in`), which `moon run :smoke` checks.
- **Never use primitives as function parameters, fields or return types.** Wrap them in a Pydantic `RootModel`. Enforced by `moon run :noprim`. The only exemption is `bool` options on Typer commands.
- **Return errors as values** with [safe-result](https://github.com/overflowy/safe-result). The CLI turns an error into a message and a non-zero exit.

## CLIs

- Built with Typer. Every CLI takes `--quiet`, which sets the log level.
- Secrets come from environment variables or `~/.config/<app>/`. Developer config lives in `~/.config/<app>/`. Project config is found by searching from `$PWD` upward, stopping at the directory containing `.git`.

## Moon

Always run tasks through moon, never the tool directly: `moon run :test`, not `pytest`. Moon runs task dependencies and caches results.

| Task | Does |
| --- | --- |
| `moon run :phase-1` | Fast checks for pre-commit: fixes lint and formatting, then typecheck, tach, noprim, actionlint |
| `moon run :full` | Every check, without fixing |
| `moon ci` | What CI runs: every task with `runInCI` (i.e. `full` and its dependencies) |
| `moon run :test` | pytest (parallel, random order) |
| `moon run :lint` / `:lint-fix` | ruff check |
| `moon run :format-check` / `:format` | ruff format |
| `moon run :typecheck` | pyrefly |
| `moon run :tach` | tach check + tach check-external |
| `moon run :noprim` | noprim |
| `moon run :actionlint` | actionlint over `.github/workflows/` |
| `moon run :smoke` | builds the library's sdist and wheel, checks no tests ship, installs it into a clean venv |

A task's `inputs` must include every file it reads. Otherwise moon replays a stale cached result.

## Tooling

- Tool settings live in each tool's own file (`ruff.toml`, `pyrefly.toml`, `tach.toml`, `pytest.toml`, `noprim.toml`, `releaserc.toml`), not in `pyproject.toml`.
- Commits are checked by lefthook pre-commit hooks (`lefthook.yml`), which run `moon run :phase-1`.
- Renovate opens and auto-merges dependency updates on weekday mornings.

## Workflow

- Tickets are in Linear (team MB, project functional-relational-python), managed with `mw`.
- We use [conventional commits](https://www.conventionalcommits.org/).

## Releasing

The version is not written down anywhere. `setuptools_scm` derives it from the git tag,
and python-semantic-release derives the tag from commit messages. PRs are squash-merged,
so **the PR title is the release note** and must be a conventional commit:

| Prefix | Effect below 1.0.0 |
| --- | --- |
| `fix: …` | patch |
| `feat: …` | minor |
| `feat!: …` or a `BREAKING CHANGE:` footer | minor, until 1.0.0 (`major_on_zero = false`) |
| `chore: …`, `docs: …`, `refactor: …`, `test: …`, `ci: …` | no release |

Merging to `main` runs `ci`; on success, `release` tags, creates the GitHub Release,
builds the library, and publishes to PyPI via trusted publishing (no API token).
To cut a release from an unchanged `main`, dispatch `release` manually.

Publishing needs a PyPI trusted publisher, and dependency updates need the Renovate
GitHub app. Both are one-time steps only a logged-in human can do:
`bash scripts/wizard.sh` walks through them.
