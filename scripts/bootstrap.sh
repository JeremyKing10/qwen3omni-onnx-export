#!/usr/bin/env bash
# 一键准备 Qwen3-Omni ONNX 导出环境（macOS arm64 / Linux x86_64）。
# 已存在的目录和环境不会被重复下载或覆盖。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 $PYTHON_BIN，请先安装 Python 3.11 或用 PYTHON_BIN 指定解释器"
    exit 1
fi

# 1. 固定版本的 Transformers 源码（模型实现所在地，必须可复现）
if [ ! -d transformers-v5.2.0 ]; then
    git clone --depth 1 --branch v5.2.0 \
        https://github.com/huggingface/transformers.git transformers-v5.2.0
fi
git -C transformers-v5.2.0 checkout --detach 7d9754a05193eb79b1d86aa744b622b8068008cd

# 2. Qwen 官方示例仓库（可选，仅参考推理调用方式）
if [ ! -d Qwen3-Omni ]; then
    git clone --depth 1 https://github.com/QwenLM/Qwen3-Omni.git || \
        echo "[WARN] Qwen3-Omni 克隆失败，可稍后手动补拉（不影响导出工具）"
fi

# 3. NVIDIA 参考导出器（可选，仅对照线需要）
if [ ! -d TensorRT-Edge-LLM-v0.10.1 ]; then
    git clone --depth 1 --branch v0.10.1 \
        https://github.com/NVIDIA/TensorRT-Edge-LLM.git TensorRT-Edge-LLM-v0.10.1 || \
        echo "[WARN] TensorRT-Edge-LLM 克隆失败（仅参考线需要，可忽略）"
fi

# 4. 虚拟环境与依赖
if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ./transformers-v5.2.0

# 5. 验证固定源码来源
python - <<'PY'
from qwen3_omni_onnx_cases import assert_transformers_provenance
print("[OK] transformers provenance:", assert_transformers_provenance())
PY

echo
echo "[OK] 环境就绪。下一步："
echo "  source .venv/bin/activate"
echo "  python run_local_thinking_pipeline.py   # 本地 tiny 全链路（无需权重）"
