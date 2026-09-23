# Qwen3-Omni ONNX 导出与编译器验证任务交接说明

## 0. 接管前必读（2026-09-23 修订）

> 接管时先读第 0 节。第 1～16 节是历史决策和实验记录，包括已过期的编译器任务、命令、哈希、节点数与旧验收结论，不作为本轮实测证据。当前操作以 `README.md`、当前 CLI 和重新生成的 schema v2 报告为准；本轮执行结果集中在 `Qwen3_Omni_ONNX_自检验证报告.md` 第 0.1 节。

### 0.1 任务定义（已更新）

- 目标：把 `Qwen3-Omni-30B-A3B-Thinking`（多模态输入 → 文本输出）导出为**只含标准 ONNX 算子**的组件级 ONNX 产品，并给出算子清单与可复现验证。
- **自研编译器已不在交付范围**：只需交付 ONNX 产品 + 算子清单 + 验证证据。
- 代码仓库（已提交 main 分支）：`https://github.com/JeremyKing10/qwen3omni-onnx-export.git`
- 本机工作目录：`/Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work`

### 0.2 当前真实进度

已有实现与历史实验：

1. 固定 Transformers v5.2.0 commit `7d9754a0…`；历史环境为 Python 3.11.9、PyTorch 2.8.0、ONNX 1.22.0、ORT 1.30.0。
2. 四组件接口：`vision_encoder`、`audio_encoder`、`thinker_prefill`、`thinker_decode`。tiny 文本 1 层/2 个 KV，vision 1 层，audio 1 层/16 mel/20 帧；官方 48 层/96 个 KV 尚未真实权重验收，real 默认 audio 为 101 帧。
3. 历史 tiny 三步 Decode 和两组路由输出有通过记录；旧 461 节点 / 41 种算子不作为本轮结果。Decode 的 64 上界为导出约束，不表示全范围已测。
4. 本轮公共模块 `onnx_artifact_utils.py` 与持久 `tests/test_*.py` 覆盖证据、路径与数据交换等边界；实际执行结果见自检报告第 0.1 节：59 项 unittest 通过，tiny 全流程退出码 0，重导出为 467 节点 / 41 种算子。
5. 导出改为“暂存 + 回滚”的事务流程：先在暂存目录导出并通过 Checker，全部成功才整体替换旧产物并失效化旧全局证据；任何异常反向恢复，`--force` 不再会在预检阶段就删除旧 ONNX。
5. 本轮区分官方顶层语义 → Wrapper 与 Wrapper → ORT 两条参考链，避免共享错误位置逻辑。schema v2 要求绑定 ONNX、external data、metadata 与向量；导出时真实源码 bytes 归档仍待实现与验收，当前源码 hash/environment/git 不等于源码归档。旧产物须重导出验收，不能手改 JSON 升级。

**当前定位仍为 tiny 接口验证件**，不能保证官方规模或全部执行路径，不能以有限数值样例宣称数学等价。`tools/` 是打包时快照，不是不可篡改证据；哈希不是签名。lock 当前为 `torch==2.8.0`，不是 macOS wheel URL；环境还需 bootstrap、固定源码及平台依赖恢复。

### 0.3 已知阻塞、未知项与下一步

- 本机 48 GiB 不具备安全加载约 59.08 GiB 权重的条件；不得调低 `--minimum-memory-gib` 到 96 来硬跑，也不要用 offload 绕过保护。
- 128 GiB 导出 / 192 GiB 端到端只是规划门槛；仍需验证可用内存、峰值驻留、磁盘、checkpoint 完整性、PyTorch 设备、ORT provider 与全部内核。
- 条件式一键流程：先恢复固定源码和环境、运行 tiny 回归、准备固定 revision 权重和 processor/config、用 `run_real_thinking_pipeline.py --preflight-only` 做只读预检（不加载 59 GiB 权重、不写产品目录），再运行带 `--run-end-to-end` 的 real runner。精确参数见 `README.md` 第 9 节，不能将这里的计划说成已执行。
- `--device cpu/cuda` 决定 PyTorch 设备，`--provider CPUExecutionProvider/CUDAExecutionProvider` 决定 ORT 后端，必须分别确认。BF16 CPU IOBinding 小图交换成功不代表完整 Qwen 的 BF16 内核已验收。
- `--component` 单独执行只导出该组件；默认 real runner 不跑端到端，不满足最终官方验收。必须重生成并核验组件报告、两条参考链、端到端、算子汇总与 manifest。
- 仅支持 Qwen3-Omni Thinking。DeepSeek/Kimi 不能靠换 `--model-path` 自动导出；多模型复用与适配边界见 `README.md` 第 14 节，不重组当前目录。

验收判据（必须由工具重验并生成当前 `manifest.json`，不可手工设置）：

```text
schema_version                  = 2
passed                          = true
status                          = official-weight-components-validated
official_weights_included       = true
shared_checkpoint_fingerprint   = true
source_equivalence_passed       = true
evidence_sources_match          = true
end_to_end_validation_passed    = true
operator_summary_valid          = true
```

还须核验各报告的产物绑定与完整输出覆盖；哈希不代替可信生成过程。

### 0.4 一键自检（任何机器改代码后必跑）

```bash
set -e
source .venv/bin/activate
python -B -m unittest discover -s tests -v
python run_local_thinking_pipeline.py --offline
for c in rmsnorm moe_block tiny_thinker; do
  python export_onnx.py --case "$c" --output-dir "artifacts/$c" --force
  python validate_onnx.py --case "$c" --model "artifacts/$c/model.onnx"
  python inspect_onnx.py --model "artifacts/$c/model.onnx" --fail-on-custom-domain
done
```

这是验收命令参考，不是本轮执行记录。先保留需要的旧产物，且不要覆盖 real 目录；旧 schema 产物必须重导出。以实际退出码、结构化报告和测试输出验收，不凭筛选出的 `[OK]` 行判断。反向场景由隔离目录中的持久 unittest 覆盖，不修改正式产物来手造通过。

### 0.5 禁止事项

- 不要把当前 tiny ONNX 说成官方权重 ONNX。
- 不要用降低容差、`--atol inf`、绕过哈希等方式把验证“变绿”。
- 不要用磁盘 offload 在本机强行加载 59 GiB 权重。
- 不要导出 `generate()` 或试图把整个 Python 应用塞进一个 ONNX。
- 不要再次重组织工作区文件路径结构（此前做过一次并已按用户要求完整回退；现在所有脚本都在根目录，靠 `WORKSPACE` 定位）。
- 不要提交权重、`.venv`、三个第三方源码克隆到 Git（`.gitignore` 已排除）。

### 0.6 固定版本表

| 项目 | 值 |
|---|---|
| Transformers | v5.2.0，commit `7d9754a05193eb79b1d86aa744b622b8068008cd` |
| Qwen3-Omni 官方仓库 | commit `e4235853125589c789f06a2dd83e9f4126df5e9d` |
| NVIDIA 参考仓库 | v0.10.1，commit `e8b29522938901f6df19ebeedd4b69bc8edbcd97` |
| Thinking checkpoint revision | `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a8…` 完整值 `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b` |
| Python / PyTorch / ONNX / ORT | 3.11.9 / 2.8.0 / 1.22.0 / 1.30.0 |

### 0.7 新机器环境准备（虚拟环境 / 依赖 / 固定源码）

> 第 14 节 Step 3 记录的是 **Mac 本机历史步骤**（含 macOS 绝对路径），新机器请按本节操作。

**推荐：一条命令**（自动完成建 venv、装依赖、拉固定源码、校验来源）：

```bash
git clone https://github.com/JeremyKing10/qwen3omni-onnx-export.git
cd qwen3omni-onnx-export
bash scripts/bootstrap.sh      # 需要 python3.11；可用 PYTHON_BIN=/path/to/python3.11 指定
source .venv/bin/activate
```

`bootstrap.sh` 实际做的事：

```text
1. 检查 python3.11
2. 若缺 transformers-v5.2.0 → 浅克隆 v5.2.0 并 checkout 固定 commit 7d9754a0…
3. 克隆两个可选参考仓库（Qwen3-Omni → e4235853、TensorRT-Edge-LLM → e8b2952）并 checkout 各自固定 commit；失败只告警，不影响导出
4. 创建 .venv → pip install -r requirements.txt → pip install -e ./transformers-v5.2.0
5. 校验 transformers.__file__ 来自固定源码
```

**手动方式**（想自己控制每一步时）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ./transformers-v5.2.0     # 必须 editable 安装固定源码，不能装 PyPI transformers
```

**依赖清单**（`requirements.txt`）：

```text
torch==2.8.0            # Linux+CUDA 请先按 pytorch.org 装 CUDA 版，版本同为 2.8.0
accelerate==1.15.0
safetensors==0.8.0
huggingface_hub>=0.34
onnx==1.22.0
onnxruntime==1.30.0
onnxscript==0.7.2
numpy>=2.0
psutil>=5.9
```

Linux + CUDA 示例（先装 torch 再跑 bootstrap，避免被 CPU 版覆盖）：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
bash scripts/bootstrap.sh
```

**装完必做的两项验证**：

```bash
python -c "from qwen3_omni_onnx_cases import assert_transformers_provenance as f; print(f())"
python -c "import torch, transformers, onnx, onnxruntime as o; print(torch.__version__, transformers.__version__, onnx.__version__, o.__version__)"
```

期望输出：

```text
{'imported_file': '.../transformers-v5.2.0/src/transformers/__init__.py', 'revision': '7d9754a05193eb79b1d86aa744b622b8068008cd'}
2.8.0 5.2.0 1.22.0 1.30.0
```

最后跑一次无权重全链路确认环境可用：

```bash
python run_local_thinking_pipeline.py
```

> 注意：如果 `assert_transformers_provenance()` 报错，说明当前环境的 `transformers` 不是固定源码（例如误装了 PyPI 版），导出工具会在每次导出前拒绝执行，这是刻意设计，不要绕过。

---

## 1. 任务背景

原始任务反馈为：

> 找到模型导出 ONNX 的接口；只要能够得到 ONNX，就可以尝试使用自研编译器进行编译。

最初听到的“onmix”已经确认实际是 **ONNX（Open Neural Network Exchange）**。

目标模型：

- 官方仓库：<https://github.com/QwenLM/Qwen3-Omni>
- 推荐首个检查点：`Qwen/Qwen3-Omni-30B-A3B-Thinking`
- Transformers 模型类型：`qwen3_omni_moe`
- 本地工作目录：`/Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work`

本任务当前处于**源码、Python 环境已准备，微型真实组件导出待实现阶段**。三个源码仓库均已固定，便携 ONNX 环境和基础导出冒烟测试已完成；尚未下载 30B 权重，也未实现 Qwen3-Omni 组件导出脚本。当前唯一执行主线是：先实现本地导出/验证/检查工具，再按“基础层 → 单个真实 MoE Block → 小配置 Thinker Text”完成标准 ONNX 本地闭环。编译器约束与命令并行向负责人收集，但不阻塞本地前三层验证；在信息到齐前不运行编译器。

### 1.1 本次合并后的关键决策

采用“两条路线、一个验收闭环”：

1. **便携标准 ONNX 路线（主线）**：基于 Hugging Face Transformers 的张量级 `forward()`，使用 `torch.onnx.export(..., dynamo=True)` 导出尽可能只包含标准 ONNX 域算子的模型，供自研编译器验证。
2. **NVIDIA 参考路线（对照线）**：固定 `TensorRT-Edge-LLM v0.10.1`，研究其已实现的 Qwen3-Omni 六组件 ONNX 导出器和接口拆分；该路线可快速获得成熟实现参考，但产物可能包含 TensorRT 定向图改写或插件，不能直接假定适合自研编译器。
3. **统一验收闭环**：每个导出组件都必须经过 ONNX Checker、Shape Inference、ONNX Runtime 或等价参考后端加载、PyTorch 数值对比、真实算子统计、自研编译器首个失败点记录。

两个检查点承担不同目标：

| 目标 | 检查点 | 原因 |
|---|---|---|
| 首个便携最小 ONNX | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | 只含 Thinker，边界更简单，适合先验证纯文本 MoE |
| 完整六组件参考导出 | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 含 Thinker、Talker、CodePredictor、Code2Wav，可对照 NVIDIA 已有流程 |

当前本机是 `Darwin arm64`、系统 Python `3.9.6`，可用磁盘约 `759 GiB`。本机适合源码分析、环境脚本开发和小配置/小图验证；NVIDIA 参考工具要求 Python 3.10+ 且依赖 CUDA，完整 30B 导出和编译验证应放在 Linux/NVIDIA 大内存机器上完成。

---

## 2. 已明确的核心目标

任务不是直接导出完整的端到端语音交互流程，也不是把 `generate()` 整体转换为 ONNX，而是：

1. 找到适合 ONNX 导出的张量级 `forward()` 接口。
2. 先导出一个最小、有效、可验证的 ONNX 模型。
3. 从实际 ONNX 图中统计算子，而不是只根据 PyTorch 源码猜测算子。
4. 将 ONNX 交给自研编译器，记录能否导入、推导和编译，以及首个失败位置。
5. 根据编译失败原因逐步扩大或调整导出范围。

推荐总流程：

```text
明确导出对象和接口
→ 准备最小输入
→ 导出 ONNX
→ 校验 ONNX 正确性
→ 统计真实 ONNX 算子
→ 使用自研编译器编译
→ 根据失败算子或限制迭代
```

---

## 3. 为什么不能直接导出完整模型

`Qwen3-Omni` 是多模态 MoE 模型，不是单一的纯文本稠密语言模型。整体结构可以粗略理解为：

```text
Qwen3-Omni
├── Thinker
│   ├── Vision Encoder
│   ├── Audio Encoder
│   └── Text MoE Decoder
├── Talker
│   ├── Text MoE
│   └── Code Predictor
└── Code2Wav
```

完整链路涉及：

- 文本、图片、视频和音频输入；
- 多模态预处理及特征拼接；
- MoE 动态专家路由；
- KV Cache；
- 自回归生成循环；
- Talker 和 Code2Wav 语音输出链路。

一次性导出全部组件，失败后很难判断问题来自模型接口、动态控制流、算子支持、动态 Shape、超大权重还是编译器限制。因此应采用分阶段策略。

---

## 4. 推荐的第一版导出范围

首轮只验证 **Thinker 的纯文本 MoE Decoder 前向图**：

| 项目 | 第一版设置 |
|---|---|
| 检查点 | `Qwen/Qwen3-Omni-30B-A3B-Thinking` |
| 模态 | 纯文本输入、文本输出 |
| Batch | 固定为 `1` |
| 序列长度 | 固定短序列，例如 `16` |
| KV Cache | 禁用，`use_cache=False` |
| Attention | 优先使用 `eager` |
| 动态 Shape | 暂不启用 |
| 导出对象 | `forward()`，不导出 `generate()` |
| 输出 | 第一版只返回 `logits` |
| 精度 | 需结合运行环境和编译器能力确认 BF16/FP16/FP32 |

建议优先使用 Thinking 版本，因为它不包含完整 Talker 和 Code2Wav 语音输出链路，导出边界更简单。

### 为什么不能导出 `generate()`

`generate()` 包含：

- Python 层自回归循环；
- 采样或搜索策略；
- 停止条件；
- KV Cache 管理；
- 非纯张量控制逻辑。

ONNX 首轮应导出单次前向计算图，而不是整段生成流程。

---

## 5. 需要研究的代码仓库

建议本地拉取三个仓库，并分别记录 commit。不要把不同路线的 Python 依赖安装在同一个环境中。

### 5.1 创建工作目录

```bash
mkdir -p /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
```

### 5.2 Qwen3-Omni 官方仓库

用途：查看官方 README、推理示例、Cookbook 和推荐依赖。

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
[ -d Qwen3-Omni/.git ] || \
  git clone --no-checkout https://github.com/QwenLM/Qwen3-Omni.git Qwen3-Omni
git -C Qwen3-Omni fetch --depth 1 origin \
  e4235853125589c789f06a2dd83e9f4126df5e9d
git -C Qwen3-Omni checkout --detach \
  e4235853125589c789f06a2dd83e9f4126df5e9d
```

验证：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work/Qwen3-Omni
git remote -v
git log -1 --oneline
```

当前实验基线固定在 commit：

```text
e4235853125589c789f06a2dd83e9f4126df5e9d
```

本轮实验期间不要执行 `git pull`。如需评估上游更新，应另开目录或分支，并作为新的实验基线重新记录。

### 5.3 Hugging Face Transformers 源码

用途：模型的实际结构、`forward()` 接口和主要算子实现位于 Transformers，而不是只在 Qwen3-Omni 示例仓库中。

此前建议先固定 `v5.2.0` 以便复现：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
git clone --depth 1 --branch v5.2.0 https://github.com/huggingface/transformers.git transformers-v5.2.0
```

重点目录：

```text
/Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work/transformers-v5.2.0/
└── src/transformers/models/qwen3_omni_moe/
```

重点文件：

```text
modeling_qwen3_omni_moe.py
modular_qwen3_omni_moe.py
configuration_qwen3_omni_moe.py
processing_qwen3_omni_moe.py
```

重点搜索的模型类：

```text
Qwen3OmniMoeForConditionalGeneration
Qwen3OmniMoeThinkerForConditionalGeneration
Qwen3OmniMoeThinkerTextModel
Qwen3OmniMoeVisionEncoder
Qwen3OmniMoeAudioEncoder
```

已优先研究并确认：

```text
Qwen3OmniMoeThinkerForConditionalGeneration.forward
Qwen3OmniMoeThinkerTextModel.forward
Qwen3OmniMoeThinkerTextExperts.forward
```

首轮标准 ONNX 组合使用 `Qwen3OmniMoeThinkerTextModel + thinker.lm_head`；完整 Thinker 只作为后续一致性验证入口。

> 注意：`v5.2.0` 是 Qwen README 当前推荐范围的最低基线，不代表 NVIDIA 导出器使用的版本。两条路线必须分别建环境、分别记录依赖。

### 5.4 NVIDIA TensorRT Edge-LLM 参考导出器

用途：研究已经落地的 Qwen3-Omni ONNX 拆分、包装器、图改写和输入输出接口。固定版本：`v0.10.1`，commit 为 `e8b29522938901f6df19ebeedd4b69bc8edbcd97`。

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
git clone --depth 1 --branch v0.10.1 \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git \
  TensorRT-Edge-LLM-v0.10.1
```

其导出 CLI 入口为：

```text
tensorrt-edgellm-export = tensorrt_edgellm.scripts.export:main
```

支持按组件导出：

```text
thinker
talker
code_predictor
visual
audio
code2wav
```

预期六组件目录：

```text
onnx/
├── llm/
│   ├── thinker/
│   ├── talker/
│   └── code_predictor/
├── audio/
│   ├── audio_encoder/
│   └── code2wav/
└── vision/
```

参考导出命令形式：

```bash
tensorrt-edgellm-export \
  --components thinker,talker,code_predictor,visual,audio,code2wav \
  CHECKPOINT_ROOT \
  OUTPUT_ONNX_DIR
```

`v0.10.1` 的 `export` 依赖组固定使用 Python 3.10+、`torch==2.13.0`、`transformers==5.14.1`、`onnx==1.19.0`、`onnxscript==0.7.1`。它依赖 CUDA/NVIDIA 平台，不能直接在当前 Apple Silicon Mac 上完整执行；本机先做源码分析，Linux/NVIDIA 机器再复现导出。

在 NVIDIA 官方支持矩阵匹配的 Linux/CUDA 主机上，独立环境的源码安装入口为：

```bash
cd /path/to/qwen3-omni-onnx-work/TensorRT-Edge-LLM-v0.10.1
python3.11 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -U pip
python -m pip install -e '.[export]'
tensorrt-edgellm-export --help
```

只有当 `nvidia-smi`、CUDA/TensorRT 版本和官方 `v0.10.1` 支持矩阵一致，且 `--help` 正常后，才执行六组件命令。`CHECKPOINT_ROOT` 和 `OUTPUT_ONNX_DIR` 必须替换为目标机真实路径；原始 HF checkpoint 与量化 checkpoint 的适用组件必须以 CLI `--help` 和对应 recipe 为准，不能混用。

源码已确认：NVIDIA 路线不是简单导出原始 Hugging Face 顶层模型，而是构造 TensorRT-Edge-LLM 自己的组件模型、装载或重排权重，再调用 `torch.onnx.export(..., dynamo=True)`。除部分 Audio/Code2Wav 路径外，产物普遍可能包含 `trt` 或 `trt_edgellm` custom-domain 节点，目标是其插件运行时。因此该路线主要作为组件拆分和图改写参考，不能替代标准 ONNX 主线。

使用该路线时必须检查：

- ONNX 节点的 `domain` 是否都是标准域；
- 是否依赖 TensorRT plugin/custom op；
- 是否有 TensorRT 专用量化节点或图改写；
- 自研编译器是否能读取其 external data 和数据类型；
- 与便携标准 ONNX 路线的接口和算子差异。

---

## 6. 导出接口的确定结论

固定 `Transformers v5.2.0` 源码分析得到以下边界：

| 组件 | Transformers 类/组合 | 首轮张量输出 | 主要导出风险 |
|---|---|---|---|
| Thinker Text | `Qwen3OmniMoeThinkerTextModel + thinker.lm_head` | `logits` | MoE 动态命中专家循环、Cache 对象、因果 Mask |
| 完整 Thinker | `Qwen3OmniMoeThinkerForConditionalGeneration` | `logits`、可选 Cache | 多模态条件分支、placeholder 替换、3D RoPE 状态 |
| Audio | `Qwen3OmniMoeAudioEncoder` | 音频 embedding | `.tolist()`、动态 split/pad、Python 循环、变长 attention |
| Vision | `Qwen3OmniMoeVisionEncoder` | pooler/deepstack features | 动态 grid、插值索引、变长 attention、列表输出 |
| Talker | `Qwen3OmniMoeTalkerForConditionalGeneration` | codec logits/hidden/cache | 自回归调度、Cache、MoE |
| CodePredictor | `Qwen3OmniMoeTalkerCodePredictorModel` | 后续 codec logits | 顺序码本预测、循环与 Cache |
| Code2Wav | `Qwen3OmniMoeCode2Wav` | waveform | 变长 codec、卷积/上采样链路 |

顶层 `Qwen3OmniMoeForConditionalGeneration` 主要负责 `generate()` 中的 Thinker、Talker、Code2Wav 编排，不作为首轮 `forward()` 导出对象。

完整 Thinker 前向接口可能涉及：

```text
input_ids
attention_mask
position_ids
past_key_values
input_features
feature_attention_mask
pixel_values
pixel_values_videos
image_grid_thw
video_grid_thw
```

输出通常涉及：

```text
logits
past_key_values
```

第一版纯文本、无 Cache 导出建议包装成最小接口：

### 输入

```text
input_ids
attention_mask
position_ids
```

### 输出

```text
logits
```

应增加一个很薄的 `torch.nn.Module` 包装器。固定版本源码分析后，第一版决定**不直接包装顶层 `Qwen3OmniMoeForConditionalGeneration`，也尽量不走完整 `Qwen3OmniMoeThinkerForConditionalGeneration.forward()`**，而是组合：

```text
Qwen3OmniMoeThinkerTextModel
→ Qwen3OmniMoeThinkerForConditionalGeneration.lm_head
→ logits
```

原因：

- `Qwen3OmniMoeForConditionalGeneration` 主要实现复杂的 `generate()` 编排，不是首轮 ONNX 边界；
- `Qwen3OmniMoeThinkerTextModel` 可直接接收纯文本 `input_ids`，绕过 Audio/Vision placeholder、特征拼接和多模态 RoPE 扫描；
- `Qwen3OmniMoeThinkerTextModel` 只输出 hidden states，因此 wrapper 再显式调用 `thinker.lm_head` 得到 logits；
- wrapper 固定 `use_cache=False`，不传 DeepStack、多模态输入和 `Cache` 对象；
- wrapper 将 Transformers dataclass 输出压平为单个 `logits` 张量。

若该最小组合通过，再测试完整 `model.thinker(...)` 纯文本路径，用于验证宿主层与最小组合的一致性。

---

## 7. ONNX 导出路线

### 路线 A：快速测试 Optimum

先检查 `optimum-cli export onnx` 是否已注册 `qwen3_omni_moe`。

目前对话调研中没有发现 Qwen3-Omni 已获得明确、完整的专用 ONNX 导出支持。普通 Qwen3 支持不代表 Qwen3-Omni 支持。

如果出现以下错误，应直接转手工导出，不必长时间停留：

```text
Unsupported model type: qwen3_omni_moe
```

### 路线 B：使用 PyTorch ONNX 导出器

建议优先研究：

```python
torch.onnx.export(..., dynamo=True)
```

关键配置：

- `dynamo=True`；
- `model.eval()`；
- `use_cache=False`；
- `attn_implementation="eager"`；
- 第一版固定输入 Shape；
- `opset_version` 根据自研编译器支持范围选择，优先确认 17 或 18；
- 大模型使用 ONNX external data；
- 避免 FlashAttention、Triton 和后端融合算子进入导出路径。

建议输出命名：

```text
qwen3_omni_thinker_text_prefill.onnx
qwen3_omni_thinker_text_prefill.onnx.data
```

30B 模型权重大幅超过单个 ONNX 文件常见的 2 GB 限制，因此必须提前确认：

- PyTorch 导出器如何生成 external data；
- 自研编译器是否支持 ONNX external data；
- 编译器能否处理该模型的权重规模和数据类型。

---

## 8. 最大技术风险：MoE 动态路由

Qwen3-Omni 使用 MoE，相关逻辑可能包含：

- `TopK` 专家选择；
- 128 个专家；
- 每个 token 选择 8 个专家；
- `OneHot`；
- `NonZero`；
- `Where`；
- 动态专家循环；
- `index_add_`；
- `ScatterElements` 或 `ScatterND` 类操作。

核心风险是：源码可能根据当前输入的 `TopK` 路由结果，在 Python 中循环处理实际命中的专家。

如果采用传统 tracing，可能只记录示例输入命中的专家分支，导致：

- 表面上成功生成 ONNX；
- 但换一组输入后专家路由不完整；
- ONNX 输出与 PyTorch 不一致。

因此必须注意：

1. 优先尝试 `dynamo=True`。
2. 使用至少两组会产生不同路由结果的输入检查输出一致性。
3. 检查 ONNX 图是否包含完整专家权重和合理的路由结构。
4. 不要以“生成了 `.onnx` 文件”作为正确导出的唯一标准。
5. 如果导出器因数据相关控制流失败，这本身也是重要结论。
6. 必要时需将 MoE 重写为更适合 ONNX 的向量化 `Gather + MatMul + Scatter` 形式，或按编译器要求定义自定义算子。

---

## 9. ONNX 生成后的验证

必须先验证 ONNX，再交给自研编译器。

### 9.1 合法性与 external data 检查

小图可以直接加载：

```python
import onnx

model = onnx.load("qwen3_omni_thinker_text_prefill.onnx")
onnx.checker.check_model(model)
```

对 2 GB 以上且使用 external data 的真实模型，优先让 Checker 按模型路径检查，避免 Python 进程先把全部权重载入内存：

```python
import onnx

onnx.checker.check_model(
    "qwen3_omni_thinker_text_prefill.onnx",
    full_check=True,
)
```

同时校验每个 external data 文件存在、大小非零，并记录 SHA-256。

### 9.2 Shape 推导

使用 ONNX shape inference 检查中间张量的 Shape 是否可推导；记录动态维度或未知维度出现的位置。

### 9.3 ONNX Runtime 加载

确认 ONNX Runtime 能否创建 Session，并记录：

- 不支持的算子；
- 数据类型问题；
- external data 加载问题；
- 内存不足问题。

### 9.4 PyTorch 与 ONNX 数值对比

使用完全相同的输入和固定随机种子，对比 `logits`：

- 最大绝对误差和最大相对误差；
- `numpy.testing.assert_allclose` 是否通过；
- 是否出现 NaN/Inf；
- 多组输入结果是否稳定；
- 不同 MoE 路由输入是否仍然一致。

首轮建议阈值，最终仍需按编译器和 dtype 调整：

```text
FP32: rtol=1e-4, atol=1e-5
FP16/BF16: rtol=1e-2, atol=1e-2
```

记录 ONNX Runtime provider。当前 Mac 小图使用 `CPUExecutionProvider`；完整 30B 图若无法因内存运行 ORT，必须明确标记“仅完成结构验证，数值验证待目标机完成”，不能写成通过。MoE 路由差异应通过捕获各层 router 的 `topk` indices 或其哈希证明，而不是仅凭输入文本不同推断。

---

## 10. 从实际 ONNX 统计算子

不要先根据源码手工确定最终算子清单。PyTorch 模块导出后可能被分解或融合，真实情况应以 ONNX 图为准。

统计示例：

```python
import collections
import onnx

model = onnx.load(
    "qwen3_omni_thinker_text_prefill.onnx",
    load_external_data=False,
)
counter = collections.Counter(
    ((node.domain or "ai.onnx"), node.op_type)
    for node in model.graph.node
)

print("opset imports:")
for item in model.opset_import:
    print(item.domain or "ai.onnx", item.version)

print("operators:")
for (domain, op_type), count in sorted(counter.items()):
    print(f"{domain}::{op_type}: {count}")
```

可能出现但仍需以实际图为准的算子包括：

```text
Gather
MatMul
Add
Mul
Div
Pow
ReduceMean
Sqrt
Reciprocal
Reshape
Transpose
Concat
Slice
Split
Softmax
TopK
OneHot
NonZero
Where
Cast
Expand
ScatterElements
ScatterND
Sin
Cos
```

建议整理为：

| Domain | ONNX 算子 | 数量 | 编译器是否支持 | Shape/类型限制 | 首次失败信息 |
|---|---|---:|---|---|---|
| `ai.onnx` | `MatMul` | 待统计 | 待验证 | 待补充 | |
| `ai.onnx` | `TopK` | 待统计 | 待验证 | 动态路由相关 | |
| `ai.onnx` | `OneHot` | 待统计 | 待验证 | 待补充 | |
| `ai.onnx` | `NonZero` | 待统计 | 待验证 | 可能产生动态 Shape | |
| `ai.onnx` | `ScatterND` | 待统计 | 待验证 | 待补充 | |
| `trt` / `trt_edgellm` | `*` | 待统计 | 默认视为不支持，需确认 | NVIDIA 参考路线专用 | |

---

## 11. 自研编译器验证重点

建议分步骤记录编译器执行情况：

```text
ONNX 读取
→ 权重及 external data 加载
→ Shape 推导
→ 图优化
→ 算子 Lowering
→ 代码生成或编译
```

失败时应区分：

- ONNX 模型本身不合法；
- 编译器不支持某个标准 ONNX 算子；
- 不支持动态 Shape；
- 不支持 BF16/FP16；
- 不支持 external data；
- 不支持超大模型或权重规模；
- 不支持 MoE 动态路由；
- 算子本身支持，但某种属性、轴、数据类型或 Shape 组合不支持。

每轮只记录**第一个根因明确的失败点**，修复或规避后再继续，不要一次罗列大量由前序错误引起的级联报错。

编译器信息目前尚未提供，因此该环节是最终闭环的外部阻塞项。收到信息后，至少固定以下内容：

```text
编译器名称、版本/commit、获取方式
运行平台、驱动和依赖
完整可复制命令与参数
输入 ONNX 和 external data 路径
标准输出/错误日志路径
成功返回码与成功判据
失败阶段和第一条根因错误
```

建议每次运行保存 `command.txt`、`environment.txt`、`stdout.log`、`stderr.log` 和 `result.json`；在得到真实编译器命令前，不创建含虚构参数的 `compiler_test.sh`。

信息到齐后的通用执行外壳如下；`COMPILER_COMMAND` 必须由编译器负责人给出的真实命令填充：

```bash
set -o pipefail
: "${COMPILER_COMMAND:?请设置完整的自研编译器命令}"
mkdir -p compiler-results/run-001
printf '%s\n' "$COMPILER_COMMAND" > compiler-results/run-001/command.txt
python --version > compiler-results/run-001/environment.txt 2>&1
uname -a >> compiler-results/run-001/environment.txt 2>&1
bash -lc "$COMPILER_COMMAND" \
  > >(tee compiler-results/run-001/stdout.log) \
  2> >(tee compiler-results/run-001/stderr.log >&2)
printf '%s\n' "$?" > compiler-results/run-001/exit_code.txt
```

该外壳只负责可复现记录，不猜测任何编译器参数；得到真实命令后再封装成 `compiler_test.sh`。

---

## 12. 推荐分阶段计划

### 阶段 0：明确验收标准与资源边界

在下载权重前向编译器负责人确认：最高 opset、标准/自定义域、external data、FP32/FP16/BF16、动态 Shape、单模型大小、KV Cache 形式以及验收要求。若暂时得不到回复，首轮默认：

```text
ONNX opset 18
标准 ONNX 域优先
FP32 小图验证，真实权重按原始 dtype
Batch=1，Sequence=16，固定 Shape
use_cache=False
attn_implementation="eager"
external data 开启
验收至少覆盖：合法、可加载、数值一致、可导入编译器
```

### 阶段 1：固定源码与环境

1. 固定并记录 `Qwen3-Omni`、`transformers v5.2.0`、`TensorRT-Edge-LLM v0.10.1` commit。
2. 创建 Python 3.11 的便携导出环境；不要使用系统 Python 3.9.6。
3. NVIDIA 参考路线建立独立 Linux/CUDA 环境，严格按其 `export` extra 依赖执行。
4. 保存 `python --version`、`pip freeze`、OS、CPU/GPU、内存、磁盘信息。

阶段出口：形成 `versions.txt` 或等价记录，所有后续日志都能对应到唯一源码与环境。

### 阶段 2：源码和接口静态分析

同时完成：

- 从 Qwen Demo 确认 `processor → model.generate` 应用调用链；
- 从 Transformers 确认 `forward()` 组件边界；
- 从 NVIDIA 导出器确认六组件包装与 ONNX 图改写；
- 列出每个组件的输入、输出、dtype、Shape、动态维和 Python 控制流；
- 形成“源码预测算子清单”，但明确标记为预测，不作为最终算子报告。

阶段出口：确定主线首个导出对象和薄包装器接口。

### 阶段 3：建立导出阶梯，不直接挑战 30B 整图

按以下顺序逐级推进，每一级都先单独导出、完成 Checker/Shape/ORT/数值验证并统计算子；编译器信息到齐后，再按同一顺序补跑编译验证：

1. 单个 RMSNorm/Attention/MLP 小图：验证基础导出和 ONNX Runtime 工具链。
2. 单个真实 MoE Block：集中验证 `TopK`、索引、Scatter、专家路由。
3. 小配置随机权重的 Thinker Text + LM Head：验证完整层间调用而不承担 30B 权重成本。
4. 真实 Thinking checkpoint 的 Thinker 纯文本 prefill：固定 B1/S16，无 Cache。

其中第 1～3 级先排除导出和 ONNX Runtime 基础问题；编译器条件具备后用于排除编译器基础问题。第 4 级才是首个真实权重里程碑。

### 阶段 4：首个真实便携 ONNX

目标图：

```text
qwen3_omni_thinker_text_prefill.onnx
```

约束：

```text
纯文本 Thinker + LM Head
固定 batch 和 sequence length
无 KV Cache
无视觉、音频和视频输入
无 Talker
返回 logits 张量
Attention 使用 eager
```

阶段出口：PyTorch 与 ONNX 多输入数值一致，并证明不同输入触发不同 MoE 路由时图仍然正确。

### 阶段 5：加入 KV Cache

拆成两个图：

```text
thinker_prefill.onnx
thinker_decode.onnx
```

`decode.onnx` 显式输入和输出每层 K/V 张量，不把 Transformers 的 `Cache` Python 对象暴露给 ONNX。先固定最大长度，再评估动态长度。

### 阶段 6：分别加入多模态编码器

分别导出：

```text
vision_encoder.onnx
audio_encoder.onnx
thinker_prefill.onnx
thinker_decode.onnx
```

图片、视频、音频文件读取、特征预处理、位置索引构造和多模态 token 拼接先保留在宿主代码中。对 Audio/Vision 内部的 `.tolist()`、动态 `split`、Python 循环采用固定 Shape、预计算辅助张量或薄包装器隔离。

### 阶段 7：语音输出链路

使用 Instruct checkpoint，依次处理：

```text
talker_prefill.onnx
talker_decode.onnx
code_predictor.onnx
code2wav.onnx
```

将自回归调度、采样、停止条件和流式 chunk 管理保留在宿主程序中。

### 阶段 8：NVIDIA 参考线对照与最终整合

1. 在兼容 Linux/NVIDIA 环境运行 `tensorrt-edgellm-export`。
2. 对六组件产物统计 `op_type + domain`，识别插件和专用图改写。
3. 对比标准 ONNX 主线与 NVIDIA 路线的组件边界、节点数、算子和性能。
4. 选择自研编译器真正要接入的产物；必要时将 NVIDIA 的拆分经验迁移到标准 ONNX 包装器，而不是直接采用其专用节点。

---

## 13. 建议交付物

### 交付物一：ONNX 导出接口调研

应包括：

- 官方仓库是否提供现成导出脚本；
- Optimum 是否识别该模型；
- 实际选择的 Transformers 模型类；
- `forward()` 输入输出张量；
- 导出边界和不包含的功能；
- opset、Shape、dtype、KV Cache 方案；
- 已知 MoE 风险。

### 交付物二：最小 ONNX 模型

建议包括：

```text
export_qwen3_omni_onnx.py
qwen3_omni_thinker_text_prefill.onnx
外部权重文件
固定测试输入或输入生成方式
导出环境及依赖版本
PyTorch/ONNX 数值对比结果
```

### 交付物三：算子及编译器支持报告

建议包括：

```text
ONNX 算子类型与数量
模型输入输出名称、Shape 和 dtype
编译器支持情况
完整编译命令
首个失败位置与日志
问题归类
建议的规避或实现方案
```

---

## 14. 从零开始的逐步操作清单

下面是实际执行顺序。每一步满足“完成标准”后再进入下一步。

### Step 1：建立工作目录并拉取三个仓库（已完成）

```bash
mkdir -p /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work

# 已存在时不重复 clone；无论何时执行都强制回到固定 commit
[ -d Qwen3-Omni/.git ] || \
  git clone --no-checkout https://github.com/QwenLM/Qwen3-Omni.git Qwen3-Omni
git -C Qwen3-Omni fetch --depth 1 origin \
  e4235853125589c789f06a2dd83e9f4126df5e9d
git -C Qwen3-Omni checkout --detach \
  e4235853125589c789f06a2dd83e9f4126df5e9d

[ -d transformers-v5.2.0/.git ] || \
  git clone --depth 1 --branch v5.2.0 \
  https://github.com/huggingface/transformers.git \
  transformers-v5.2.0
git -C transformers-v5.2.0 checkout --detach \
  7d9754a05193eb79b1d86aa744b622b8068008cd

[ -d TensorRT-Edge-LLM-v0.10.1/.git ] || \
  git clone --depth 1 --branch v0.10.1 \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git \
  TensorRT-Edge-LLM-v0.10.1
git -C TensorRT-Edge-LLM-v0.10.1 checkout --detach \
  e8b29522938901f6df19ebeedd4b69bc8edbcd97
```

完成标准：三个目录都存在且 `git status` 正常。注意这里拉的是代码，不是模型权重；暂时不要 Git LFS 克隆模型仓库。

### Step 2：记录源码版本（已完成）

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work

for repo in Qwen3-Omni transformers-v5.2.0 TensorRT-Edge-LLM-v0.10.1; do
  echo "===== $repo ====="
  git -C "$repo" remote get-url origin
  git -C "$repo" rev-parse HEAD
  git -C "$repo" status --short --branch
done
```

当前已确认：

```text
Qwen3-Omni: e4235853125589c789f06a2dd83e9f4126df5e9d
transformers v5.2.0 tag commit: 7d9754a05193eb79b1d86aa744b622b8068008cd
TensorRT-Edge-LLM v0.10.1 tag commit: e8b29522938901f6df19ebeedd4b69bc8edbcd97
```

完成标准：实际输出与上述 commit 一致。若要强制回到本轮基线，先确认工作区没有需要保留的修改，再执行：

```bash
git -C Qwen3-Omni checkout --detach e4235853125589c789f06a2dd83e9f4126df5e9d
git -C transformers-v5.2.0 checkout --detach 7d9754a05193eb79b1d86aa744b622b8068008cd
git -C TensorRT-Edge-LLM-v0.10.1 checkout --detach e8b29522938901f6df19ebeedd4b69bc8edbcd97
```

### Step 3：创建便携导出环境（已完成）

> **本节是 Mac 本机历史记录**（含 `/Users/bojunjin/.workbuddy/...` 等 macOS 绝对路径）。**新机器请直接看第 0.7 节**（`scripts/bootstrap.sh` + `requirements.txt`），不要照抄本节路径。

当前系统 Python 3.9.6 不用作项目环境，且当前机器没有 Conda、Mamba、Homebrew、pyenv 或 uv。已安装独立 Python 3.11.9，可直接创建工作区虚拟环境：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
/Users/bojunjin/.workbuddy/binaries/python/versions/3.11.9/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

如换到其他机器，也可用 Conda/Mamba 创建等价的 Python 3.11 环境。当前 Mac 便携基线采用以下已验证安装命令：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
source .venv/bin/activate
python -m pip install \
  'torch==2.8.0' \
  -e ./transformers-v5.2.0 \
  'accelerate==1.15.0' \
  'onnx==1.22.0' \
  'onnxruntime==1.30.0' \
  'onnxscript==0.7.2' \
  'safetensors==0.8.0'
```

`qwen-omni-utils` 只用于处理真实音频/图像/视频输入，首轮纯文本微型图不需要；进入多模态阶段时再固定版本安装。

> 上面是 Mac 本机历史安装命令，**未包含 `numpy`、`psutil`（`--minimum-memory-gib` 内存门禁依赖它）、`huggingface_hub`**。新机器请以 `requirements.txt` 为准。

Apple Silicon 上不要安装 `flash-attn`。当前便携环境已实际安装并验证：

```text
Python 3.11.9
PyTorch 2.8.0
Transformers 5.2.0（本地 editable 源码）
ONNX 1.22.0
ONNX Runtime 1.30.0
ONNXScript 0.7.2
Accelerate 1.15.0
Safetensors 0.8.0
```

记录命令：

```bash
source /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work/.venv/bin/activate
python --version
python -m pip freeze > portable-export-requirements-lock.txt
python -c "import torch, transformers, onnx, onnxruntime; print(torch.__version__, transformers.__version__, onnx.__version__, onnxruntime.__version__)"
```

已完成最小 `Linear → SiLU → Linear` 的 `torch.onnx.export(dynamo=True, opset_version=18)` 和 `onnx.checker` 冒烟测试。未安装 `torchvision` 时导出器会打印跳过 torchvision 自定义算子注册的警告，不影响当前纯文本小图。

完成标准：核心包可以 import，Transformers 报告 `5.2.0`，最小 dynamo ONNX 导出通过。

### Step 4：完成静态源码定位（已完成）

依次定位：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work

rg -n "class Qwen3OmniMoe.*(ForConditionalGeneration|Encoder|Code2Wav|CodePredictor)" \
  transformers-v5.2.0/src/transformers/models/qwen3_omni_moe

rg -n "def (forward|generate)|torch\.topk|one_hot|nonzero|where|index_add|tolist" \
  transformers-v5.2.0/src/transformers/models/qwen3_omni_moe

rg -n "qwen3_omni_moe|code_predictor|code2wav|tensorrt-edgellm-export" \
  TensorRT-Edge-LLM-v0.10.1
```

重点确认：

- `Qwen3OmniMoeThinkerForConditionalGeneration.forward()` 含 LM Head 并返回 logits；
- `Qwen3OmniMoeThinkerTextModel.forward()` 只返回 hidden states，不含 LM Head；
- MoE 路由是否包含数据相关 Python 循环；
- NVIDIA 如何拆分六组件以及如何重写 MoE/Attention/Cache。

完成标准：输出一张“组件—类—forward 输入—forward 输出—动态逻辑—预计算子”的表。

### Step 5：并行收集编译器约束（不阻塞 Step 6 本地验证）

至少记录：

```text
opset = 17/18/其他
external data = 支持/不支持
支持 dtype = FP32/FP16/BF16/INT8
动态 Shape = 支持/不支持
KV Cache = 是否要求
标准域以外节点 = 是否允许
单图/单权重大小上限
验收 = 可读取/可编译/可运行/数值正确/性能
```

没有反馈时按阶段 0 默认值推进，但不能在最终报告里把默认值写成编译器事实。

### Step 6：建立微型导出与验证闭环（已完成；现由 `run_local_thinking_pipeline.py` 一键执行）

先实现三个本地共享工具：

```text
export_onnx.py             # 调用 torch.onnx.export(dynamo=True)
validate_onnx.py           # checker、shape inference、ORT、数值比较
inspect_onnx.py            # 输入输出、op_type、domain、节点数量、external data
```

调用接口统一设计为：

```bash
source /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work/.venv/bin/activate
# 改代码后先跑持久回归：python -B -m unittest discover -s tests -v
# 导出目录非空时必须带 --force，否则工具拒绝覆盖（防误删）
python export_onnx.py --case rmsnorm --output-dir artifacts/rmsnorm --force
python validate_onnx.py --case rmsnorm --model artifacts/rmsnorm/model.onnx
python inspect_onnx.py --model artifacts/rmsnorm/model.onnx --output artifacts/rmsnorm/operators.json

python export_onnx.py --case moe_block --output-dir artifacts/moe_block --force
python validate_onnx.py --case moe_block --model artifacts/moe_block/model.onnx
python inspect_onnx.py --model artifacts/moe_block/model.onnx --output artifacts/moe_block/operators.json

python export_onnx.py --case tiny_thinker --output-dir artifacts/tiny_thinker --force
python validate_onnx.py --case tiny_thinker --model artifacts/tiny_thinker/model.onnx
python inspect_onnx.py --model artifacts/tiny_thinker/model.onnx --output artifacts/tiny_thinker/operators.json
```

以上脚本是当前下一项编码工作；完成前这些命令代表必须实现的 CLI 契约。按“基础层 → 单个真实 MoE Block → 小配置 Thinker Text”的顺序测试。每次都产生：ONNX、固定输入、PyTorch 输出摘要、ONNX 输出摘要和算子 JSON。

编译器信息到齐后才创建 `compiler_test.sh`，再为已经验证的 ONNX 补跑编译器日志。完成标准分两级：

1. **本地闭环完成**：至少一个真实 Qwen3-Omni MoE 小图通过导出、Checker、Shape、ORT、数值对比和算子统计。
2. **全闭环完成**：在本地闭环基础上，再通过自研编译器既定验收阶段，或记录第一个根因明确的失败点。

### Step 7：下载真实 Thinking 权重并导出首个真实图

只有在确认存储、内存/显存、访问权限和 external data 后才下载。2026-09-15 查询到的模型 revision 为：

```text
Thinking: 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b
Instruct: 26291f793822fb6be9555850f06dfe95f2d7e695
```

使用当前 `huggingface_hub` 的 `hf` CLI 固定 revision 下载；把目标目录替换为大容量磁盘的真实路径：

```bash
source /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work/.venv/bin/activate
hf download \
  Qwen/Qwen3-Omni-30B-A3B-Thinking \
  --revision 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
  --local-dir /目标磁盘/Qwen3-Omni-30B-A3B-Thinking
```

下载后保存 `config.json`、tokenizer/processor 配置以及所有权重分片的文件列表、大小和 SHA-256。导出首个真实图：固定 B1/S16、纯文本、`use_cache=False`、`attn_implementation="eager"`、只返回 logits。不要调用 `generate()`。

完成标准：至少两组输入的 PyTorch/ONNX 结果一致，且两组输入的 MoE 路由不完全相同。

### Step 8：提取真实算子并测试自研编译器

算子报告必须同时统计：

```text
node.domain
node.op_type
节点数量
输入输出 dtype/Shape
动态维
external data 文件及大小
```

按“读取 → 权重加载 → Shape 推导 → 图优化 → Lowering → 代码生成/编译”逐级运行编译器，只保留每轮第一个根因明确的失败点。

### Step 9：扩展到 KV、多模态和语音

严格按第 12 节阶段 5～8 执行：Thinker prefill/decode → Vision → Audio → Talker → CodePredictor → Code2Wav → NVIDIA 六组件结果对照。

---

## 15. 进入真实 30B 权重和编译器阶段前必须确认的问题

以下问题与 Step 6 本地微型图编码并行收集，不阻塞基础层、MoE Block 和小配置 Thinker 的本地导出验证；但在下载 30B 权重或运行自研编译器前必须确认：

1. 首轮是否只要求“获得一个 Qwen3-Omni 相关的 ONNX 并验证可编译”，还是必须覆盖完整多模态模型？
2. 是否接受先导出 Thinker 的纯文本子图？
3. 自研编译器最高支持哪个 ONNX opset？
4. 是否支持 ONNX external data？
5. 支持哪些数据类型：FP32、FP16、BF16、INT8？
6. 是否支持动态 Shape？如果不支持，目标 batch 和 sequence length 是多少？
7. 是否要求 KV Cache？首轮可以禁用吗？
8. 编译环境能够承受多大的模型及权重文件？
9. 是否已有普通 Qwen3、MoE、`TopK` 或 Scatter 类算子的编译经验？
10. 最终验收标准是“能导入”“能编译”“能运行”，还是还要求数值正确性和性能数据？

---

## 16. 历史状态与注意事项（截至 2026-09-22）

> 本节保留旧实验与修复记录，不声明本轮通过。下方旧 hash、461/41 和旧 e2e 不对应 schema v2；当前边界与操作看第 0 节和 `README.md`。

### 历史已完成

- 已确认“onmix”实际指 ONNX。
- 已明确总体任务流程和“标准 ONNX 主线 + NVIDIA 参考线”的双路线。
- 已识别 Qwen3-Omni 的六个部署组件和首轮最小导出范围。
- 已确认核心模型实现位于 Transformers，不在 Qwen3-Omni 示例仓库。
- 已确认 Transformers `v5.2.0` 包含 Thinker、Talker、CodePredictor、AudioEncoder、VisionEncoder、Code2Wav 实现。
- 已确认 TensorRT Edge-LLM `v0.10.1` 提供 `tensorrt-edgellm-export` 和 Qwen3-Omni 六组件导出路线。
- 已识别 MoE 动态路由是标准 ONNX 导出的最关键风险。
- 已确认本地 `Qwen3-Omni` 仓库存在、工作区干净：commit `e4235853125589c789f06a2dd83e9f4126df5e9d`。
- 已确认本机环境：Darwin arm64、系统 Python 3.9.6、可用磁盘约 759 GiB。
- 已拉取 `transformers-v5.2.0` 和 `TensorRT-Edge-LLM-v0.10.1`，均处于固定 tag 对应的 detached HEAD，符合只读研究基线。
- 已安装 Python 3.11.9 并创建工作区 `.venv`，核心 ONNX 导出依赖可正常 import。
- 已生成 `portable-export-requirements-lock.txt`（43 行，SHA-256：`e2eaaa325754cb063dc743c9f27124de47afdb72505a9bf12f6321e5e5563a61`）。
- 已完成最小 dynamo ONNX 导出及 ONNX Checker 冒烟测试。
- 已确认首轮真实模型包装边界为 Thinker Text Model + Thinker LM Head。
- 已确认 NVIDIA 参考产物普遍包含 TensorRT 自定义域，主要用作拆分和改写参考。

### 尚未完成（本轮仍适用）

- **官方 30B 四组件与 real 三步 Decode 尚未验收**：本机 48 GiB 内存不足是已知阻塞，但不是唯一未知项；迁移大内存 Linux 仍需实际验证内核、dtype、峰值资源和官方语义，见 `README.md` 第 9 节。
- 未提供“单个 ONNX 文件”形态（当前产品是四组件 + 宿主调度；如有硬性单文件需求需新增导出模式）。
- 语音输出组件（Talker / CodePredictor / Code2Wav）未实现（Thinking checkpoint 本身无语音链路）。
- 未运行 Optimum 支持性快速测试；该测试不是主线阻塞项。
- 自研编译器相关约束已不在本任务交付范围（如后续需要，再做算子支持矩阵映射即可）。

### 2026-09-22 追加完成

- 历史日志记录四组件 tiny 导出和 Wrapper → ORT 通过；不等于本轮已完成独立官方顶层参考验收。
- 历史 tiny Prefill/Decode 每步实际为 2 个 K/V，支持动态 past-sequence 与三步续接；此前将官方配置的 96 个误写成实测数，现更正。
- 早期三级回归（rmsnorm / moe_block / tiny_thinker）并入回归基线。
- 旧产品含 manifest、端到端报告、算子汇总、向量和 `tools/` 快照；不符合本轮 schema v2 的产物需要重导出，不能只更新证据字段。
- 当时记录修复了 real 导出前证据删除、fp16 接力 dtype、tiny 覆盖 real 报告、source-dir config 指纹等问题；不据此宣称已发现并修好全部缺陷。
- 历史仓库创建与提交记录保留：`https://github.com/JeremyKing10/qwen3omni-onnx-export.git`，本轮不自动提交。
- 历史统计保留：461 节点、41 种标准算子、0 自定义 domain；不是当前代码重导出的结果。

### 2026-09-22 第二轮修复（全量代码审查 + 全 md 命令核对）

**代码（本机已实测验证）**

- `validate_onnx.py`：`--case` 传错**不再**覆盖该模型原有的 `validation.json`（此前会毁掉有效证据，已实测复现）；只有 ONNX 哈希不一致才失效化旧报告。
- `qwen3_omni_thinking_components.py`：
  - `cu_seqlens` 强制 int32（此前 `.cumsum(0)` 把显式 int32 提升成 int64，导出图实测为 INT64）；
  - Decode `position_ids` 改为 float32，与 Prefill（官方 `get_rope_index`）一致（此前两图一个 FLOAT 一个 INT64）；
  - `interface["chunk_count"]` 由实际分块数推导（此前恒写 2）；
  - real 容量守卫改用与实际 prompt 一致的 `audio_feature_length`，阈值由 `V+A+2` 修正为 `V+A+5`（tiny 实测：旧阈值 8 会放行 8/9/11 这些必然失败的序列长度，真实最小是 12）；
  - `build_real_thinking_component()` 增加 `assert_transformers_provenance()`。
- `run_local_thinking_pipeline.py`：新增 `--force` 透传（此前产品目录里若有 real 端到端报告，tiny 流程会中断且无法从一键脚本打开开关）。
- `export_thinking_onnx.py`：证据失效化挪到 `--force` 检查**之后**且对任意组件生效（实测：不带 `--force` 时退出 1 且证据仍在；带 `--force` 时才失效化）。

**代码（real 路径，本机装不下 59 GiB，仅静态审查）**

- 显存门禁：索引缺 `total_size` 时改为按磁盘分片实际大小估算（此前算出 0 → **静默跳过门禁**）；不再强依赖 `model.safetensors.index.json`；校验 `--device cuda:N` 设备号。
- 历史调整 `--minimum-memory-gib` 默认保护线；设备默认值以各 CLI 当前代码为准，迁移命令显式指定 PyTorch `--device` 和 ORT `--provider`，二者不能混用。
- 历史普通 NumPy 喂 BF16 失败只说明当时数据交换路径有问题，不能推导 CPU 不支持全部 BF16。当前 CPU IOBinding Cast/Identity 与 NPZ 位保持测试成功；完整 Qwen 图仍依赖实际内核/环境，不保证改 CUDA 即可。
- 当时增加 real 验收不满足时非零退出；最终以 schema v2 的实际绑定与报告覆盖为准。

**文档（历史修订记录）**

- 曾修正 case 说明、图打印示例、失效链接、force 参数与旧节点/hash 记录。不能声称全部文档命令已实跑；本机不能执行官方 30B，本文的迁移命令仅作参考。

### 特别注意

- 不要意外下载几十 GiB 权重或在本机绕过内存保护。
- 不导出 `generate()`；生成 ONNX 文件不代表语义、路由或内核验收通过。
- 算子清单以当前 ONNX 实际统计为准；旧数字不能当本轮结果。
- 当前公共产物模块和持久 unittest 均在根目录结构下维护；不重组目录。
- 验证逐组件实际 dtype：Vision 索引 `int32`，Text 位置 mask `bool`，Audio `cu_seqlens int32`，real 浮点张量并非全部 `float32`。

---

## 17. 一句话交接结论

已有 Qwen3-Omni Thinking 四组件工具和 tiny 历史验证；本轮修复必须通过持久 unittest、重新导出与 schema v2 双参考链验收，结果看自检报告第 0.1 节。官方 30B 在本机始终未验收，大内存仅满足部分前提；按 `README.md` 第 9 节预检并执行条件式流程，不承诺无需改代码或一键必成。只有实际报告及其当前产物绑定全部满足，才允许产生 `official-weight-components-validated`。
