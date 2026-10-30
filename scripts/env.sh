if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "activate the iac conda env first (conda activate iac)" >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONNOUSERSITE=1
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export FLASHINFER_EXTRA_CUDAFLAGS="--compress-mode=default"
