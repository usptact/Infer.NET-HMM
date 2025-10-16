#!/bin/bash
# Script to run Factorial HMM inference multiple times and select the best run

if [ $# -lt 3 ]; then
    echo "Usage: ./run_multiple.sh <csv_file> <num_chains> <states_per_chain...> [num_runs]"
    echo "Example: ./run_multiple.sh data.csv 2 2 2 10"
    exit 1
fi

DATAFILE=$1
NUM_CHAINS=$2
shift 2
STATES="$@"
NUM_RUNS=${!#}  # Last argument

# Check if last arg is a number (num_runs), otherwise default to 10
if [[ $NUM_RUNS =~ ^[0-9]+$ ]] && [ $NUM_RUNS -gt 3 ]; then
    # Remove num_runs from STATES
    STATES=${@:1:$#-1}
else
    NUM_RUNS=10
fi

echo "======================================"
echo "Factorial HMM: Multiple Run Analysis"
echo "======================================"
echo "Data file: $DATAFILE"
echo "Number of chains: $NUM_CHAINS"
echo "States per chain: $STATES"
echo "Number of runs: $NUM_RUNS"
echo ""

# Create output directory
OUTDIR="multi_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

best_evidence=-999999
best_run=0

echo "Running inference..."
echo ""

for i in $(seq 1 $NUM_RUNS); do
    printf "Run %2d/%2d: " $i $NUM_RUNS
    
    # Run inference and capture output
    output=$(dotnet run --project FactorialHiddenMarkovModel $DATAFILE $NUM_CHAINS $STATES 2>&1)
    
    # Extract model evidence
    evidence=$(echo "$output" | grep "Model Evidence" | sed 's/.*: //')
    
    if [ -z "$evidence" ]; then
        echo "FAILED (no evidence)"
        continue
    fi
    
    printf "Evidence = %8.2f" $evidence
    
    # Save full output
    echo "$output" > "$OUTDIR/run_${i}.txt"
    
    # Save results file
    if [ -f "$(basename $DATAFILE .csv).factorial.csv" ]; then
        cp "$(basename $DATAFILE .csv).factorial.csv" "$OUTDIR/run_${i}.factorial.csv"
    fi
    
    # Track best (note: higher evidence is better, but values are negative)
    # So we want the "maximum" (least negative)
    is_better=$(echo "$evidence > $best_evidence" | bc -l 2>/dev/null || echo "0")
    
    if [ "$is_better" = "1" ]; then
        best_evidence=$evidence
        best_run=$i
        printf " ← NEW BEST!"
    fi
    
    echo ""
done

echo ""
echo "======================================"
echo "RESULTS"
echo "======================================"
echo "Best run: #$best_run"
echo "Best evidence: $best_evidence"
echo ""
echo "Output directory: $OUTDIR/"
echo "Best results: $OUTDIR/run_${best_run}.txt"
echo "            : $OUTDIR/run_${best_run}.factorial.csv"
echo ""

# Copy best results to main directory
if [ -f "$OUTDIR/run_${best_run}.factorial.csv" ]; then
    cp "$OUTDIR/run_${best_run}.factorial.csv" "$(basename $DATAFILE .csv).factorial_best.csv"
    echo "Best inferred states saved to: $(basename $DATAFILE .csv).factorial_best.csv"
fi

# Show top 5 runs
echo ""
echo "Top 5 runs by model evidence:"
echo "Rank | Run | Evidence"
echo "-----|-----|----------"

for file in "$OUTDIR"/run_*.txt; do
    run_num=$(basename "$file" .txt | sed 's/run_//')
    evidence=$(grep "Model Evidence" "$file" | sed 's/.*: //')
    if [ ! -z "$evidence" ]; then
        echo "$run_num $evidence"
    fi
done | sort -k2 -rn | head -5 | nl -w1 -s' | ' | awk '{printf "%5s | %4s | %10s\n", $1, $2, $3}'

echo ""
echo "All results saved in: $OUTDIR/"

