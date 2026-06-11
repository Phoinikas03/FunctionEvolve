#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh [MODE] [options] -- [extra main.py args]

Modes:
  full
      Full FunctionEvolve: LLM Generator, LLM Selector, AST Mutator,
      LLM Mutator, and the structure-aware coefficient optimizer.

  degen_generator
      w/o Generator. Replaces the LLM Generator with MockGenerator:
      empty domain knowledge, no preprocessing proposals, and fixed
      fallback seed expressions.

  degen_selector1
      w/o Selector. Replaces the LLM Selector with MockSelector:
      rank-based Boltzmann parent sampling over numerical fitness,
      with no LLM call during parent selection.

  degen_selector2
      Selector without explicit AST metadata. Keeps LLM parent selection,
      but strips AST-derived structural fields such as parameter count,
      tree depth, and operator counts from the selector prompt.

  degen_mutator1
      w/o LLM Mutator. Disables LLM-generated local refinements and keeps
      only programmatic AST-rule mutation.

  degen_mutator2
      w/o AST Mutator. Disables programmatic AST-rule mutation and keeps
      only LLM-generated ADD/SUBST edits with the standard structural
      parent context.

  degen_mutator3
      w/o AST Mutator plus no-AST mutator prompt. Uses only
      LLM-generated ADD/SUBST edits and removes the annotated AST block
      and AST-local mutation guidance from the mutator prompt.

  degen_mutator3_selector2
      Removes explicit AST signals from both LLM interfaces used here:
      selector2 hides AST-derived selector metadata, and mutator3 uses the
      no-AST LLM-only mutator variant.

  degen_all
      w/o All from the ablation table: combines w/o Generator,
      w/o Selector, w/o AST Mutator, and w/o Structure-aware Optimizer.
      This is close to a parallel multi-offspring LLM-SR style run.

  degen_all_mutator3
      Same combined ablation as degen_all, but uses mutator3 instead of
      mutator2, so the LLM mutator also loses AST-local prompt context.

  only_structure
      Rule-only + Structure-aware Optimizer. Disables Generator,
      LLM Selector, and LLM Mutator, while retaining programmatic
      AST-rule mutation and the structure-aware coefficient optimizer.

  lbfgs
      w/o Structure-aware Optimizer. Keeps the symbolic search stack but
      replaces the structure-aware coefficient optimizer with L-BFGS-B.

  refine_full
      Full search with final output refinement and resume.
      This is a refinement preset rather than a paper ablation setting.

Dataset selection:
  -d, --dataset DATASET       llm-srbench | llm-srbench-noise1pct |
                              llm-srbench-noise5pct | aifeynman
                              default: llm-srbench
  -s, --split SPLITS          Comma-separated split list. If omitted, main.py
                              runs all splits in the selected dataset.
  -e, --equations EQUATIONS   Comma-separated concrete equation names.
                              Combined with --split by union in main.py.
  --list                      List selected equations and exit.

LLM config:
  -c, --config PATH           Search LLM config.
                              default: $SEARCH_LLM_CONFIG, then $LLM_CONFIG,
                              then llm_config.yaml
  --verify-config PATH        Optional verifier LLM config. If omitted,
                              main.py falls back to --llm-config.

Runtime:
  -j, --equation-workers N    Active equations. Defaults: full/refine_full=10,
                              other modes=30.
  --global-workers N          Forward to main.py.
  --tag TAG                   Run tag. default: MODE.
                              Logs are written under logs/DATASET/TAG/MODEL/.
  --resume                    Resume checkpoints.
  --data-dir PATH             Optional dataset directory override.
  --dry-run                   Print command without executing.
  -h, --help                  Show this help.

Examples:
  ./run.sh full -d aifeynman
  ./run.sh full -d aifeynman -s feynmanequations
  ./run.sh full -d llm-srbench-noise5pct -s bio_pop_growth,matsci
  ./run.sh only_structure -d llm-srbench -e PO36,PO37 --resume
  ./run.sh full -d aifeynman -- --max-steps 10 --quiet
EOF
}

MODE="full"
if [[ $# -gt 0 && "${1:-}" != -* && "${1:-}" != "--" ]]; then
    MODE="$1"
    shift
fi

DATASET="${DATASET:-llm-srbench}"
SPLIT="${SPLIT:-}"
EQUATIONS="${EQUATIONS:-}"
SEARCH_CONFIG="${SEARCH_LLM_CONFIG:-${LLM_CONFIG:-llm_config.yaml}}"
VERIFY_CONFIG="${VERIFY_LLM_CONFIG:-}"
EQUATION_WORKERS="${EQUATION_WORKERS:-}"
GLOBAL_WORKERS="${GLOBAL_WORKERS:-}"
RUN_TAG="${RUN_TAG:-}"
DATA_DIR="${DATA_DIR:-}"
RESUME=0
LIST_EQUATIONS=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            DATASET="${2:?Missing value for $1}"
            shift 2
            ;;
        --dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        -s|--split)
            SPLIT="${2:?Missing value for $1}"
            shift 2
            ;;
        --split=*)
            SPLIT="${1#*=}"
            shift
            ;;
        -e|--equations|--equation)
            EQUATIONS="${2:?Missing value for $1}"
            shift 2
            ;;
        --equations=*|--equation=*)
            EQUATIONS="${1#*=}"
            shift
            ;;
        -c|--config|--llm-config)
            SEARCH_CONFIG="${2:?Missing value for $1}"
            shift 2
            ;;
        --config=*|--llm-config=*)
            SEARCH_CONFIG="${1#*=}"
            shift
            ;;
        --verify-config|--verify-llm-config)
            VERIFY_CONFIG="${2:?Missing value for $1}"
            shift 2
            ;;
        --verify-config=*|--verify-llm-config=*)
            VERIFY_CONFIG="${1#*=}"
            shift
            ;;
        -j|--equation-workers)
            EQUATION_WORKERS="${2:?Missing value for $1}"
            shift 2
            ;;
        --equation-workers=*)
            EQUATION_WORKERS="${1#*=}"
            shift
            ;;
        --global-workers|--global-eval-workers)
            GLOBAL_WORKERS="${2:?Missing value for $1}"
            shift 2
            ;;
        --global-workers=*|--global-eval-workers=*)
            GLOBAL_WORKERS="${1#*=}"
            shift
            ;;
        --tag|--run-tag)
            RUN_TAG="${2:?Missing value for $1}"
            shift 2
            ;;
        --tag=*|--run-tag=*)
            RUN_TAG="${1#*=}"
            shift
            ;;
        --data-dir)
            DATA_DIR="${2:?Missing value for $1}"
            shift 2
            ;;
        --data-dir=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --list|--list-equations)
            LIST_EQUATIONS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown run.sh argument: $1" >&2
            echo "Use -- to pass arbitrary arguments to main.py." >&2
            exit 2
            ;;
    esac
done

MODE_ARGS=()
OPTIMIZER="Structure"
SELECTOR_CONTEXT_SIZE="${SELECTOR_CONTEXT_SIZE:-}"

case "$MODE" in
    full)
        : "${EQUATION_WORKERS:=10}"
        : "${SELECTOR_CONTEXT_SIZE:=200}"
        ;;
    degen_generator)
        MODE_ARGS+=(--degenerated-generator)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_selector1)
        MODE_ARGS+=(--degenerated-selector1)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_selector2)
        MODE_ARGS+=(--degenerated-selector2)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_mutator1)
        MODE_ARGS+=(--degenerated-mutator1)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_mutator2)
        MODE_ARGS+=(--degenerated-mutator2)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_mutator3)
        MODE_ARGS+=(--degenerated-mutator3)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_mutator3_selector2)
        MODE_ARGS+=(--degenerated-mutator3 --degenerated-selector2)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_all)
        MODE_ARGS+=(--degenerated-generator --degenerated-selector1 --degenerated-mutator2)
        OPTIMIZER="L-BFGS-B"
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    degen_all_mutator3)
        MODE_ARGS+=(--degenerated-generator --degenerated-selector1 --degenerated-mutator3)
        OPTIMIZER="L-BFGS-B"
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    only_structure)
        MODE_ARGS+=(--degenerated-generator --degenerated-selector1 --degenerated-mutator1)
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    lbfgs)
        OPTIMIZER="L-BFGS-B"
        : "${EQUATION_WORKERS:=30}"
        : "${SELECTOR_CONTEXT_SIZE:=2000}"
        ;;
    refine_full)
        MODE_ARGS+=(--refine-output)
        RESUME=1
        : "${EQUATION_WORKERS:=10}"
        : "${SELECTOR_CONTEXT_SIZE:=200}"
        ;;
    *)
        echo "Unknown MODE=$MODE" >&2
        usage >&2
        exit 2
        ;;
esac

RUN_TAG="${RUN_TAG:-${MODE}}"
if [[ "$RUN_TAG" == */* || "$RUN_TAG" == *\\* || "$RUN_TAG" == "." || "$RUN_TAG" == ".." ]]; then
    echo "Invalid run tag '$RUN_TAG': use a single path component. Dataset is already the outer log layer." >&2
    exit 2
fi

ulimit -n 500000 || true

# Gated HuggingFace datasets should use the official endpoint for token auth.
# Set USE_HF_MIRROR=1 explicitly for public, non-gated downloads.
if [[ "${USE_HF_MIRROR:-0}" == "1" ]]; then
    export HF_ENDPOINT=https://hf-mirror.com
else
    unset HF_ENDPOINT || true
fi

CMD=(
    python main.py
    --llm-config "$SEARCH_CONFIG"
    --run-tag "$RUN_TAG"
    --dataset "$DATASET"
    --candidate-num 5
    --equation-workers "$EQUATION_WORKERS"
    --max-steps 30
    --n-seeds 20
    --selector-context-size "$SELECTOR_CONTEXT_SIZE"
    --optimizer "$OPTIMIZER"
    --max-mature-nodes 50
    --overfit-min-depth 10
)

if [[ -n "$VERIFY_CONFIG" ]]; then
    CMD+=(--verify-llm-config "$VERIFY_CONFIG")
fi
if [[ -n "$SPLIT" ]]; then
    CMD+=(--split "$SPLIT")
fi
if [[ -n "$EQUATIONS" ]]; then
    CMD+=(--equations "$EQUATIONS")
fi
if [[ -n "$GLOBAL_WORKERS" ]]; then
    CMD+=(--global-workers "$GLOBAL_WORKERS")
fi
if [[ -n "$DATA_DIR" ]]; then
    CMD+=(--data-dir "$DATA_DIR")
fi
if [[ "$RESUME" == "1" ]]; then
    CMD+=(--resume)
fi
if [[ "$LIST_EQUATIONS" == "1" ]]; then
    CMD+=(--list-equations)
fi

CMD+=("${MODE_ARGS[@]}")
CMD+=("${EXTRA_ARGS[@]}")

echo "============================================================"
echo "Mode:          $MODE"
echo "Dataset:       $DATASET"
echo "Split:         ${SPLIT:-<all>}"
echo "Equations:     ${EQUATIONS:-<none>}"
echo "Search config: $SEARCH_CONFIG"
echo "Verify config: ${VERIFY_CONFIG:-<main.py fallback>}"
echo "Run tag:       $RUN_TAG"
echo "Eq workers:    $EQUATION_WORKERS"
echo "============================================================"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

exec "${CMD[@]}"
