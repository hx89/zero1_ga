#! /bin/bash
set -x

LOG_DIR=${1:-'./'}

NSYS_OUTPUT_FILE="${LOG_DIR}/output-nsys-profile"
XLA_DUMP_DIR="${LOG_DIR}/xla_dump"

rm -rf ${XLA_DUMP_DIR}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export CUDA_DEVICE_MAX_CONNECTIONS=16

export NCCL_NVLS_ENABLE=1
#export NCCL_ALGO=NVLS
export NCCL_CTA_POLICY=1

export XLA_FLAGS="--xla_gpu_enable_nccl_user_buffers=true --xla_dump_hlo_as_text --xla_dump_to=${XLA_DUMP_DIR} --xla_dump_hlo_pass_re=.*"

# mkdir -p /workspace

if [ "$SLURM_PROCID" -eq 0 ]; then
  nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --cuda-graph-trace=node python3 -u zero1_ga_test.py
else
  python3 -u zero1_ga_test.py
fi

set +x