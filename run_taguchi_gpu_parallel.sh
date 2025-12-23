#!/bin/bash
# run_taguchi_gpu_parallel.sh - Run Taguchi experiments in parallel across multiple GPUs

# Check if we're in the right directory
if [ ! -f "build/ga_solver" ]; then
    echo "Error: ga_solver not found in build/"
    echo "Please run this script from the GA_Solver root directory"
    exit 1
fi

# Configuration
NUM_GPUS=8
CLEAN_START=false
if [ "$1" == "--clean" ]; then
    CLEAN_START=true
fi

# Create results directory structure
if [ "$CLEAN_START" = true ]; then
    echo "Cleaning previous results..."
    rm -rf taguchi_results
fi

mkdir -p taguchi_results
mkdir -p taguchi_results/logs

# Datasets to test
DATASETS="bays29,berlin52,eil101,eil51,eil76,fri26,gr17,gr21,gr24,gr48"

# L18 Orthogonal Array Parameters
EXPERIMENTS=(
    "4000 0.2 3 15 2000 0"
    "4000 0.2 5 20 2500 1"
    "4000 0.2 7 30 3000 0"
    "4000 0.3 3 20 3000 1"
    "4000 0.3 5 30 2000 0"
    "4000 0.3 7 15 2500 1"
    "4000 0.4 3 30 2500 0"
    "4000 0.4 5 15 3000 1"
    "4000 0.4 7 20 2000 0"
    "5000 0.2 3 30 3000 1"
    "5000 0.2 5 15 2000 0"
    "5000 0.2 7 20 2500 1"
    "5000 0.3 3 15 2500 0"
    "5000 0.3 5 20 3000 1"
    "5000 0.3 7 30 2000 0"
    "5000 0.4 3 20 2000 1"
    "5000 0.4 5 30 2500 0"
    "5000 0.4 7 15 3000 1"
    "6000 0.2 3 20 2500 0"
    "6000 0.2 5 30 3000 1"
    "6000 0.2 7 15 2000 0"
    "6000 0.3 3 30 2000 1"
    "6000 0.3 5 15 2500 0"
    "6000 0.3 7 20 3000 1"
    "6000 0.4 3 15 3000 0"
    "6000 0.4 5 20 2000 1"
    "6000 0.4 7 30 2500 0"
)

# Number of runs per experiment configuration
RUNS_PER_CONFIG=3

# Results summary file
SUMMARY_FILE="taguchi_results/experiment_summary.csv"
if [ "$CLEAN_START" = true ] || [ ! -f "$SUMMARY_FILE" ]; then
    echo "Experiment,Run,Pop_Size,Mut_Rate,Tour_Size,Div_Perc,Stag_Lim,Cost_Aware,Dataset,Best_Fitness,Solution_Gen,Total_Gens,Total_Time,Time_Per_Gen" > $SUMMARY_FILE
fi

# Function to run a single experiment on a specific GPU
run_experiment_on_gpu() {
    local exp_params=$1
    local exp_num=$2
    local run=$3
    local gpu_id=$4
    
    # Parse experiment parameters
    IFS=' ' read -r pop mut tour div stag cost <<< "$exp_params"
    
    # Set up directories
    exp_dir="taguchi_results/exp_${exp_num}"
    mkdir -p "$exp_dir"
    
    run_dir="$exp_dir/run_${run}"
    if [ -d "$run_dir" ]; then
        rm -rf "$run_dir"
    fi
    mkdir -p "$run_dir"
    
    # Construct command with GPU specification
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id ./build/ga_solver"
    cmd="$cmd -d $DATASETS"
    cmd="$cmd -p $pop"
    cmd="$cmd -m $mut"
    cmd="$cmd -t $tour"
    cmd="$cmd -v $div"
    cmd="$cmd -s $stag"
    cmd="$cmd -o 0"
    cmd="$cmd -g 5000"
    cmd="$cmd -l $run_dir"
    
    if [ "$cost" -eq 1 ]; then
        cmd="$cmd -c"
    fi
    
    # Log file
    log_file="$run_dir/output.txt"
    
    echo "[GPU $gpu_id] Running Exp $exp_num Run $run: Pop=$pop Mut=$mut Tour=$tour"
    echo "[GPU $gpu_id] Command: $cmd"
    
    # Execute
    eval "$cmd" > "$log_file" 2>&1
    
    # Process results
    results_csv="$run_dir/results_log.csv"
    if [ -f "$results_csv" ]; then
        # Skip header and add to summary
        tail -n +2 "$results_csv" | while IFS=',' read -r dataset best_fit sol_gen total_gen total_time time_per_gen mode extra; do
            echo "$exp_num,$run,$pop,$mut,$tour,$div,$stag,$cost,$dataset,$best_fit,$sol_gen,$total_gen,$total_time,$time_per_gen" >> $SUMMARY_FILE
        done
        echo "[GPU $gpu_id] Exp $exp_num Run $run completed successfully"
    else
        echo "[GPU $gpu_id] WARNING: No results for Exp $exp_num Run $run"
    fi
}

# Export function for parallel execution
export -f run_experiment_on_gpu
export DATASETS SUMMARY_FILE

# Create job queue
JOB_QUEUE=""
for exp_idx in "${!EXPERIMENTS[@]}"; do
    exp_num=$((exp_idx + 1))
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        JOB_QUEUE="$JOB_QUEUE${EXPERIMENTS[$exp_idx]}|$exp_num|$run\n"
    done
done

# Total number of jobs
TOTAL_JOBS=$((${#EXPERIMENTS[@]} * RUNS_PER_CONFIG))
echo "Total experiments to run: $TOTAL_JOBS"
echo "Using $NUM_GPUS GPUs in parallel"
echo "=================================================="

# Run jobs in parallel using GNU parallel
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job distribution"
    
    # Create jobs file
    echo -e "$JOB_QUEUE" | grep -v '^$' > taguchi_results/jobs.txt
    
    # Run with GNU parallel
    cat taguchi_results/jobs.txt | parallel -j $NUM_GPUS --colsep '|' \
        --eta --progress \
        run_experiment_on_gpu {1} {2} {3} {%}
        
else
    echo "GNU parallel not found. Using background jobs instead."
    
    # Simple parallel execution with background jobs
    gpu_id=0
    job_count=0
    
    echo -e "$JOB_QUEUE" | grep -v '^$' | while IFS='|' read -r exp_params exp_num run; do
        # Run experiment in background on specific GPU
        run_experiment_on_gpu "$exp_params" "$exp_num" "$run" "$gpu_id" &
        
        # Increment counters
        job_count=$((job_count + 1))
        gpu_id=$(((gpu_id + 1) % NUM_GPUS))
        
        # Wait if we've launched NUM_GPUS jobs
        if [ $((job_count % NUM_GPUS)) -eq 0 ]; then
            wait  # Wait for current batch to complete
            echo "Completed batch: $job_count/$TOTAL_JOBS"
        fi
    done
    
    # Wait for final jobs
    wait
fi

echo "=================================================="
echo "All experiments completed!"
echo "Results saved to: $SUMMARY_FILE"
echo "=================================================="

# Summary statistics
echo ""
echo "Summary Statistics:"
echo "Total experiments run: ${#EXPERIMENTS[@]}"
echo "Total configurations tested: $TOTAL_JOBS"
echo "Results file lines: $(wc -l < $SUMMARY_FILE)"

# Check for any failures
if grep -q "WARNING" taguchi_results/exp_*/run_*/output.txt 2>/dev/null; then
    echo ""
    echo "⚠️  Some experiments had warnings. Check individual log files."
fi