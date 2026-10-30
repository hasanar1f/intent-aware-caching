#!/usr/bin/env bash
set -euo pipefail

SGLANG_TAG="v0.5.19"
SGLANG_COMMIT="0bcd822377da7b5718e674eaf9c870d349424dd1"
PATCH="patches/sglang-${SGLANG_TAG}-iac.patch"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGL_DIR="${ROOT}/third_party/sglang"
cd "${ROOT}"

if [[ ! -d "${SGL_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${SGLANG_TAG}" https://github.com/sgl-project/sglang.git "${SGL_DIR}"
fi

actual="$(git -C "${SGL_DIR}" rev-parse HEAD)"
if [[ "${actual}" != "${SGLANG_COMMIT}" ]]; then
    echo "error: ${SGL_DIR} is at ${actual}, expected ${SGLANG_COMMIT} (${SGLANG_TAG})" >&2
    exit 1
fi

if git -C "${SGL_DIR}" apply --reverse --check "${ROOT}/${PATCH}" 2>/dev/null; then
    echo "IAC patch already applied."
else
    git -C "${SGL_DIR}" apply "${ROOT}/${PATCH}"
    echo "Applied ${PATCH}."
fi

if [[ "${1:-}" == "--no-install" ]]; then
    exit 0
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: activate a conda env first (conda create -n iac python=3.12 && conda activate iac)" >&2
    exit 1
fi

RUST_CHANNEL="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "${SGL_DIR}/rust/rust-toolchain.toml")"
if command -v rustup >/dev/null 2>&1; then
    rustup toolchain install "${RUST_CHANNEL}" --profile minimal
    export RUSTUP_TOOLCHAIN="${RUST_CHANNEL}"
elif ! command -v cargo >/dev/null 2>&1; then
    echo "error: building SGLang needs Rust ${RUST_CHANNEL}; install rustup (https://rustup.rs) and rerun" >&2
    exit 1
fi

python -m pip install --upgrade pip uv
uv pip install --prerelease=allow -e "${SGL_DIR}/python"
uv pip install --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel==0.4.6.post1 --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --force-reinstall sgl-deep-gemm==0.1.7 --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
uv pip install -e "${ROOT}"

"${CONDA_EXE:-conda}" install -y -p "${CONDA_PREFIX}" -c conda-forge "cuda-nvcc=12.9" "cuda-cudart-dev=12.9" libcurand-dev "cuda-version=12.9"
[[ -e "${CONDA_PREFIX}/lib64" ]] || ln -s lib "${CONDA_PREFIX}/lib64"
[[ -e "${CONDA_PREFIX}/lib/stubs" ]] || ln -s ../targets/x86_64-linux/lib/stubs "${CONDA_PREFIX}/lib/stubs"
"${CONDA_EXE:-conda}" env config vars set -p "${CONDA_PREFIX}" CUDA_HOME="${CONDA_PREFIX}" PYTHONNOUSERSITE=1 \
    FLASHINFER_EXTRA_CUDAFLAGS="--compress-mode=default"
echo "Done. Reactivate the env (conda deactivate && conda activate iac) before using SGLang."
