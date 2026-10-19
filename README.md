# Anime Sentiment Analyzer (MLOps + DevOps)

An end-to-end NLP app that predicts sentiment of anime reviews.

## Tech Stack
- FastAPI, Docker, GitHub Actions, DVC, MLflow, Scikit-Learn.

## Git Flow
`feature/*` -> `dev` -> `main`

## CI/CD Pipelines
- CI: Runs Ruff (linting) and Pytest (testing) on PRs.
- CD: Builds and pushes Docker image to GHCR on merges.