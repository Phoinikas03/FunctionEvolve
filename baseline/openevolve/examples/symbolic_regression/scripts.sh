#!/bin/bash
# Global concurrency limit across all splits: at most max_workers jobs at any time
# (e.g. max_workers=4 can run 2 BPG + 2 CRK simultaneously).

set -m

max_workers=4

# Optional list of specific test cases to run (comma-separated, no spaces)
# If empty, all test cases will be run
# Example: EQUATIONS="MatSci25,MatSci28,BPG0,CRK5"
EQUATIONS="${EQUATIONS:-MatSci25,MatSci28}"

usage() {
    echo "Usage: $0 [--max-workers N]" >&2
    echo "  --max-workers N   Max concurrent test cases (default: 4). Splits are not serialized." >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-workers)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --max-workers requires a value" >&2
                usage
                exit 1
            fi
            max_workers="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$max_workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-workers must be a positive integer, got: $max_workers" >&2
    exit 1
fi

wait_for_slot() {
    local limit=$1
    while (( $(jobs -r | wc -l) >= limit )); do
        wait -n 2>/dev/null || sleep 0.2
    done
}

# Define the number of problems for each split
declare -A split_counts=(
    ["bio_pop_growth"]=24
    ["chem_react"]=36
    ["matsci"]=29
    ["phys_osc"]=44
)

declare -A split_problem_dir_prefixes=(
    ["bio_pop_growth"]="BPG"
    ["chem_react"]="CRK"
    ["matsci"]="MatSci"
    ["phys_osc"]="PO"
)

# Fixed order (associative array iteration order is undefined in bash)
split_order=(bio_pop_growth chem_react matsci phys_osc)

base_problems_dir="./problems"

# Function to check if a problem is in the EQUATIONS list
should_run_problem() {
    local problem_id="$1"
    if [[ -z "$EQUATIONS" ]]; then
        return 0  # Run all if EQUATIONS is empty
    fi
    # Check if problem_id is in the comma-separated EQUATIONS list
    [[ ",$EQUATIONS," == *",$problem_id,"* ]]
}

echo "Starting all experiments (max_workers=$max_workers, global pool across splits)..."
if [[ -n "$EQUATIONS" ]]; then
    echo "  Running only equations: $EQUATIONS"
else
    echo "  Running all equations"
fi

for split_name in "${split_order[@]}"; do
    count=${split_counts[$split_name]}
    problem_dir_prefix=${split_problem_dir_prefixes[$split_name]}

    if [ -z "$problem_dir_prefix" ] && [ "${split_problem_dir_prefixes[$split_name]+_}" != "_" ]; then
        :
    elif [ -z "$problem_dir_prefix" ]; then
        echo ""
        echo "Warning: No problem directory prefix defined for split '$split_name' in 'split_problem_dir_prefixes'. Skipping this split."
        continue
    fi

    echo ""
    echo "----------------------------------------------------"
    echo "Queueing Split: $split_name"
    echo "Number of problems: $count"
    echo "Problem directory prefix: '$problem_dir_prefix'"
    echo "Expected problem path structure: $base_problems_dir/$split_name/${problem_dir_prefix}[ID]/"
    echo "----------------------------------------------------"

    for ((i = 0; i < count; i++)); do
        problem_id="$problem_dir_prefix$i"

        if ! should_run_problem "$problem_id"; then
            continue
        fi

        problem_dir="$base_problems_dir/$split_name/$problem_id"

        initial_program_path="$problem_dir/initial_program.py"
        evaluator_path="$problem_dir/evaluator.py"
        config_path="$problem_dir/config.yaml"

        if [[ ! -f "$initial_program_path" ]]; then
            echo "  [$split_name Problem $i] SKIPPING: Initial program not found at $initial_program_path"
            continue
        fi
        if [[ ! -f "$evaluator_path" ]]; then
            echo "  [$split_name Problem $i] SKIPPING: Evaluator not found at $evaluator_path"
            continue
        fi
        if [[ ! -f "$config_path" ]]; then
            echo "  [$split_name Problem $i] SKIPPING: Config file not found at $config_path"
            continue
        fi

        wait_for_slot "$max_workers"

        echo "  Launching $split_name - Problem $i ($initial_program_path)"
        python ../../openevolve-run.py "$initial_program_path" "$evaluator_path" --config "$config_path" --iterations 200 &
    done
done

echo ""
echo "Waiting for all background processes to complete (up to $max_workers were concurrent)..."
wait
echo ""
echo "All experiments have completed."
