#!/bin/bash
# find_rws_checkpoints.sh - Find available RWS checkpoints with params_*.pkl files

set -euo pipefail

PROJECT_DIR="${1:-/scratch/work/yangw4/ogbench}"
RWS_WEIGHTS_DIR="${PROJECT_DIR}/weights/ReachabilityEstimation"
OUTPUT_FILE="${2:-rws_checkpoints.txt}"

echo "Searching for RWS checkpoints in: ${RWS_WEIGHTS_DIR}"
echo "Output will be saved to: ${OUTPUT_FILE}"
echo "=========================================="

# Check if weights directory exists
if [[ ! -d "${RWS_WEIGHTS_DIR}" ]]; then
    echo "ERROR: RWS weights directory not found: ${RWS_WEIGHTS_DIR}"
    echo "Please ensure RWS models have been trained first."
    exit 1
fi

# Clear output file
> "${OUTPUT_FILE}"

# Function to find checkpoint epochs in a directory
find_epochs() {
    local dir="$1"
    local epochs=""
    
    # Look for params_EPOCH.pkl files
    for ckpt in "${dir}"/params_*.pkl; do
        if [[ -f "${ckpt}" ]]; then
            # Extract epoch number from filename
            basename="${ckpt##*/}"
            if [[ "${basename}" =~ params_([0-9]+)\.pkl ]]; then
                epochs="${epochs} ${BASH_REMATCH[1]}"
            fi
        fi
    done
    
    echo "${epochs}" | xargs -n1 | sort -nu | xargs
}

# Environment list (same as in train_rws_triton_all.sh)
ENVIRONMENTS=(
    "pointmaze-medium-navigate-v0"
    "pointmaze-large-navigate-v0"
    "pointmaze-giant-navigate-v0"
    "pointmaze-teleport-navigate-v0"
    "pointmaze-medium-stitch-v0"
    "pointmaze-large-stitch-v0"
    "pointmaze-giant-stitch-v0"
    "pointmaze-teleport-stitch-v0"
    "antmaze-medium-navigate-v0"
    "antmaze-large-navigate-v0"
    "antmaze-giant-navigate-v0"
    "antmaze-teleport-navigate-v0"
    "antmaze-medium-stitch-v0"
    "antmaze-large-stitch-v0"
    "antmaze-giant-stitch-v0"
    "antmaze-teleport-stitch-v0"
    "antmaze-medium-explore-v0"
    "antmaze-large-explore-v0"
    "antmaze-teleport-explore-v0"
    "humanoidmaze-medium-navigate-v0"
    "humanoidmaze-large-navigate-v0"
    "humanoidmaze-giant-navigate-v0"
    "humanoidmaze-medium-stitch-v0"
    "humanoidmaze-large-stitch-v0"
    "humanoidmaze-giant-stitch-v0"
)

echo "Environment | RWS Directory | Available Epochs | Checkpoint Files | Status" | tee -a "${OUTPUT_FILE}"
echo "------------|---------------|------------------|------------------|--------" | tee -a "${OUTPUT_FILE}"

# Search for checkpoints for each environment
for env in "${ENVIRONMENTS[@]}"; do
    # Expected directory pattern
    rws_dir_pattern="${RWS_WEIGHTS_DIR}/rws_${env}_rws"
    
    found=false
    for rws_dir in ${rws_dir_pattern}*; do
        if [[ -d "${rws_dir}" ]]; then
            # Find seed directories
            for seed_dir in "${rws_dir}"/sd*; do
                if [[ -d "${seed_dir}" ]]; then
                    found=true
                    dir_name="${seed_dir##*/}"
                    relative_path="rws_${env}_rws/${dir_name}"
                    
                    # Find available epochs
                    epochs=$(find_epochs "${seed_dir}")
                    
                    if [[ -n "${epochs}" ]]; then
                        # Get actual filenames (limit to first 3 for display)
                        checkpoint_files=""
                        count=0
                        for epoch in ${epochs}; do
                            if [[ -f "${seed_dir}/params_${epoch}.pkl" ]]; then
                                if [[ ${count} -lt 3 ]]; then
                                    checkpoint_files="${checkpoint_files} params_${epoch}.pkl"
                                    count=$((count + 1))
                                else
                                    checkpoint_files="${checkpoint_files} ..."
                                    break
                                fi
                            fi
                        done
                        status="✓ Ready"
                        echo "${env} | ${relative_path} | ${epochs} | ${checkpoint_files} | ${status}" | tee -a "${OUTPUT_FILE}"
                    else
                        status="⚠ No checkpoints"
                        echo "${env} | ${relative_path} | - | - | ${status}" | tee -a "${OUTPUT_FILE}"
                    fi
                fi
            done
        fi
    done
    
    if [[ "${found}" == "false" ]]; then
        echo "${env} | - | - | - | ✗ Not found" | tee -a "${OUTPUT_FILE}"
    fi
done

echo ""
echo "=========================================="
echo "Summary saved to: ${OUTPUT_FILE}"
echo ""
echo "To use these checkpoints, update the TSV file with:"
echo "  - RWSDir: the relative path from the table above (e.g., rws_antmaze-medium-navigate-v0_rws/sd042_s_12899477.0.20251028_185233)"
echo "  - RWSEpoch: one of the available epochs (typically 400000)"
echo ""
echo "The script expects to find params_RWSEpoch.pkl in the RWSDir."
echo ""
echo "Example TSV line:"
echo "antmaze-medium-navigate-v0	gciql	0	0.003	0.9	rws_antmaze-medium-navigate-v0_rws/sd042_s_12899477.0.20251028_185233	400000	vanilla	-"