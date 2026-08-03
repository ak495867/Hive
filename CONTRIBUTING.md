# Contributing to Hive

Thanks for taking a look at Hive. This started as a research project, not a production trading system, so contributions here are welcome but scoped a bit differently than a typical OSS repo. Read this before opening a PR or issue.

---

## What This Project Is (and Isn't)

Hive is a research demonstration of zero-shot generalisation from a narrow volatility-only training curriculum to a broad, unseen asset universe. It is **not** a production trading system, and PRs that push it in that direction (live broker integration, order execution, account management, etc.) are out of scope unless discussed first in an issue.

If you're not sure whether something fits, open an issue before writing code.

---

## Ways to Contribute

**Welcome:**
- Bug fixes in `backtest.py`, `paper.py`, or `hive.py`
- Additional walk-forward or out-of-sample validation methodology
- Alternative evaluation metrics or diagnostic plots
- Improvements to transaction cost / slippage modeling realism
- Documentation fixes and clarity improvements
- Reproducibility fixes (dependency pinning, environment issues, seed handling)

**Open an issue first:**
- New asset universes or data sources
- Architecture changes to the model itself
- Changes to the training curriculum (the six volatility tickers)
- New features that materially change scope (live trading, broker APIs, etc.)

**Please don't:**
- Submit PRs that remove or weaken the disclaimers, risk warnings, or the "not investment advice" framing
- Submit PRs that hardcode API keys, tokens, or personal credentials
- Submit large binary files (data dumps, alternate model checkpoints) without discussion — open an issue first

---

## Reporting Bugs

Open an issue with:
1. What you ran (`backtest.py` / `paper.py` / other)
2. What you expected to happen
3. What actually happened (include the full error/traceback if there is one)
4. Your environment: Python version, OS, and whether you installed from `requirements.txt` as-is or modified it

If it's a numerical/results discrepancy rather than a crash, include the exact ticker universe and date range you ran, since results are sensitive to both.

---

## Submitting a Pull Request

1. Fork the repo and create a branch off `main`
2. Keep PRs focused — one fix or feature per PR, not a bundle of unrelated changes
3. If you're changing behavior (not just fixing an obvious bug), explain the reasoning in the PR description, not just the diff
4. Re-run `backtest.py` and `paper.py` before submitting and include the resulting metrics in the PR description if your change could plausibly affect them
5. Do not commit `.pt` files, logs, or generated CSVs/plots as part of a code change — those should only change via an explicit re-run/update PR, clearly labeled as such

---

## Code Style

- Keep it readable over clever — this is a research repo people will read to understand the methodology, not just run as a black box
- Comment non-obvious modeling decisions (why a parameter is set where it is, why a feature is included), since the "why" is often more valuable here than the "what"
- No hardcoded local file paths, API keys, or credentials — use environment variables or config if something needs to be configurable

---

## Questions

If something is unclear about the methodology, the results, or why a design decision was made, open an issue rather than guessing — happy to explain the reasoning.
