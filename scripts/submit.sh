#!/bin/bash
#
# Allocate SLURM nodes with the NIXL container for DeepEP testing.
#
# Usage:
#   Allocate:  bash scripts/submit.sh <portfolio> <num_nodes>
#   Attach:    bash scripts/submit.sh <portfolio> attach <node#>
#
# Examples:
#   bash scripts/submit.sh network 2     # allocate 2 nodes
#   bash scripts/submit.sh network attach 1  # attach to node 1
#   bash scripts/submit.sh network attach 2  # attach to node 2

set -euo pipefail

if [ "$1" == "network" ]; then
    PORTFOLIO=network_research_advdev
elif [ "$1" == "trtllm" ]; then
    PORTFOLIO=coreai_comparch_trtllm
elif [ "$1" == "net" ]; then
    PORTFOLIO=coreai_tritoninference_triton3
else
    echo "Error: Invalid portfolio. Use 'network', 'trtllm', or 'net'"
    exit 1
fi

if [ "$2" == "attach" ]; then
    if [ -z "${3:-}" ]; then
        echo "Need node number as the third argument for attach"
        exit 1
    fi
    NODE_NUM=$3
else
    if [ -z "${2:-}" ]; then
        echo "Need number of nodes as the second argument"
        exit 1
    fi
    NODE_NUM=$2
fi

ASCHWARTZ_DIR=/lustre/fsw/network_research_advdev/aschwartz
DEEPEP_DIR=$ASCHWARTZ_DIR/deepep_pr/DeepEP
PARTITION=batch
TIME=04:00:00
SUBPROJECT=deepep
JOB_NAME=$PORTFOLIO-$SUBPROJECT.dev
C_NAME=$SUBPROJECT.dev_$USER
C_IMAGE=$ASCHWARTZ_DIR/latest.sqsh

SLURM_BIN=/usr/bin/
SLURM_PATHS=""
for i in srun sinfo scontrol squeue; do
    SLURM_PATHS+="$SLURM_BIN/$i,"
done
SLURM_PATHS+=/usr/lib/x86_64-linux-gnu/slurm,
SLURM_PATHS+=/lib/x86_64-linux-gnu/libmunge.so.2,
SLURM_PATHS+=/run/munge

MOUNTS=$SLURM_PATHS
MOUNTS+=,/home/$USER:/workspace/home
MOUNTS+=,$ASCHWARTZ_DIR:/workspace/aschwartz
MOUNTS+=,$DEEPEP_DIR:/workspace/deepep
MOUNTS+=,/etc/slurm
MOUNTS+=,/usr/bin/srun
MOUNTS+=,/usr/bin/sinfo
MOUNTS+=,/usr/bin/scontrol
MOUNTS+=,/usr/bin/squeue
MOUNTS+=,/usr/src/linux-headers-5.15.0-88-generic:/lib/modules/5.15.0-88-generic/build
MOUNTS+=,/lustre/fsw/network_research_advdev/bblack/gdaki/linux-headers-5.15.0-88-generic:/usr/src/linux-headers-5.15.0-88-generic
MOUNTS+=,/lustre/fsw/network_research_advdev/bblack/gdaki/linux-headers-5.15.0-88:/usr/src/linux-headers-5.15.0-88
MOUNTS+=,/lustre/fsw/network_research_advdev/bblack/gdaki/linux-headers-5.15.0-88:/lib/modules/5.15.0-88-generic/linux-headers-5.15.0-88

if [ "$2" != "attach" ]; then
    echo "=== Allocating $NODE_NUM node(s) ==="
    echo "Image: $C_IMAGE"
    echo "DeepEP mounted at: /workspace/deepep"
    echo ""
    salloc -N $NODE_NUM -p $PARTITION -A $PORTFOLIO -t $TIME -J $JOB_NAME \
         --container-name=$C_NAME --container-image=$C_IMAGE \
         --container-mounts=$MOUNTS \
         --ntasks-per-node=1
else
    JOBID=$(squeue -u $USER -n $JOB_NAME --noheader --format="%i" | head -n 1)
    if [ -z "$JOBID" ]; then
        echo "Error: No running job found for user $USER with job name $JOB_NAME"
        squeue -u $USER --format="%i %j %T %M %N"
        exit 1
    fi

    JOB_NODES=$(squeue -j $JOBID --noheader --format="%N")
    HOSTNAMES=$(scontrol show hostnames $JOB_NODES)
    TOTAL_NODES=$(echo "$HOSTNAMES" | wc -l)

    if [ $NODE_NUM -gt $TOTAL_NODES ]; then
        echo "Error: Requested node $NODE_NUM but job only has $TOTAL_NODES nodes"
        echo "$HOSTNAMES" | nl
        exit 1
    fi

    TO_NODE=$(echo "$HOSTNAMES" | sed -n "${NODE_NUM}p")
    echo "Attaching to Job $JOBID, Node $NODE_NUM ($TO_NODE)"
    srun --jobid=$JOBID --overlap --container-name=$C_NAME --container-image=$C_IMAGE \
         --nodelist=$TO_NODE --container-mounts=$MOUNTS --pty bash
fi
