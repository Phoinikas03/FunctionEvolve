#!/usr/bin/env bash

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export SEARCH_LLM_CONFIG="llm_config.yaml"
export VERIFY_LLM_CONFIG="llm_config.yaml"
export OPENAI_API_KEY="1"

SPLIT="${SPLIT:-bio_pop_growth,matsci,chem_react}"
exec ./run.sh full -s "$SPLIT" "$@"
