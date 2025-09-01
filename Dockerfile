# Use slim Python base image
FROM python:3.11-bookworm

# Install build deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc nano \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_PATH="/opt/venvs" \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set workdir
WORKDIR /eqphases

RUN touch README.md

# Copy dependency files first (for caching)
COPY pyproject.toml ./

# Copy source code
COPY ./seistools ./seistools

# Install deps (system-wide since venv is disabled)
RUN poetry install --no-interaction --no-ansi


