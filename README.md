# Anime Sentiment Analyzer (MLOps & DevOps Capstone)

An end-to-end NLP application that predicts the sentiment of anime reviews. This project demonstrates a complete DevOps and MLOps lifecycle, from data versioning to automated deployment.

##  Tech Stack
- **App**: FastAPI, HTML/JS
- **Machine Learning**: Scikit-learn (TF-IDF + Logistic Regression)
- **MLOps**: MLflow (Experiment Tracking), DVC (Data & Model Versioning)
- **DevOps**: Docker, GitHub Actions (CI/CD), GitHub Container Registry (GHCR)

##  Git Branching Strategy
We strictly follow a collaborative branching model:
`feature/*` ➔ `dev` ➔ `staging` ➔ `main`

- **Feature Branches**: Developers create features here.
- **Dev Branch**: Integration branch for team members. 
- **Staging Branch**: Pre-production testing. Merging here creates a staging environment image.
- **Main Branch**: Production environment. 

##  CI/CD Pipeline Flow (Commit ➔ Prod)
Our automation is driven by GitHub Actions. Here is how the code moves:

1. **Commit & Push (PR to `dev` or `main`)**:
   - Triggers the **CI Pipeline** (`.github/workflows/ci.yml`).
   - Runs `Ruff` for linting/static analysis.
   - Runs `Pytest` for unit tests.
   - Builds the Docker image to ensure no build errors. 
   - *Rule: A PR cannot be merged if the CI fails.*

2. **Merge to `dev`**:
   - Triggers the **CD Pipeline** (`cd.yml`).
   - Builds a Docker image tagged `dev` and pushes it to GHCR.

3. **Merge to `staging`**:
   - Triggers the **CD Pipeline**.
   - Uses GitHub Secrets to generate a staging-specific `.env` file.
   - Builds a Docker image tagged `staging` and pushes it to GHCR.

4. **Merge to `main` (Production)**:
   - Triggers the **CD Pipeline**.
   - Uses production GitHub Secrets to generate the production `.env` file.
   - Builds the final production Docker image tagged `main` and pushes it to GHCR.

##  Local Development Setup
1. Clone the repo: `git clone ...`
2. Setup environment: `python -m venv .venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Pull the dataset and model via DVC: `dvc pull`
5. Run the app: `uvicorn app.main:app --reload`
6. View API docs: `http://127.0.0.1:8000/docs`