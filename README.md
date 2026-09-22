# Qwen3-Omni ONNX 导出工具

把 [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)（Hugging Face `qwen3_omni_moe` 架构）的多模态 MoE 模型，导出为**只含标准 ONNX 算子**、可被 ONNX Runtime 直接加载验证的组件级 ONNX 产品，并附带完整的自检证据链（哈希绑定、数值对比、MoE 路由覆盖、external data 校验、算子清单）。

- 目标模型：`Qwen/Qwen3-Omni-30B-A3B-Thinking`（多模态输入 → 文本输出，无语音输出）
- 固定模型实现：Hugging Face `transformers` **v5.2.0**（commit `7d9754a05193eb79b1d86aa744b622b8068008cd`）
- 已验证环境：Python 3.11.9 / PyTorch 2.8.0 / ONNX 1.22.0 / ONNX Runtime 1.30.0 / onnxscript 0.7.2（macOS arm64 实测通过）

---

## 1. 当前状态速览

| 能力 | 状态 |
|---|---|
| 四组件正式导出接口（Vision / Audio / Thinker Prefill / Thinker Decode） | ✅ 已实现并验证（tiny 随机权重） |
| 显式 KV Cache（Prefill 输出 96 个 K/V → Decode 输入输出） | ✅ 已实现并验证 |
| Decode 动态 past-sequence 轴（自回归续接，KV 逐轮回灌） | ✅ 已实现并验证（3 步续接） |
| MoE 动态路由（TopK→Gather→BatchedMatMul→ReduceSum） | ✅ 标准 ONNX，两种路由均数值一致 |
| 端到端张量接力验证（Vision+Audio → Prefill → 3×Decode） | ✅ 已验证 |
| 官方 30B 权重导出 | ⏳ 代码就绪，需 ≥128 GiB 内存的 Linux 机器（本机 48 GiB 不够，59.08 GiB 权重装不下） |
| 语音输出（Talker / CodePredictor / Code2Wav） | ❌ 不在范围（Thinking checkpoint 本身无语音链路；如需请改用 Instruct） |

一句话：**导出方法和验证体系已在真实 Qwen3-Omni 类 + 缩小配置上全链路跑通；剩下唯一一步是在大内存 Linux 上把官方权重灌进同一套流程。**

### 1.1 `tiny` 是什么，与官方模型的差距

`tiny` = **用官方 Qwen3-Omni 的类，配一套很小的尺寸，再用随机权重初始化**。它和官方模型是**同一套代码、同一套算法、同一套算子路径**，唯一区别是**规模与权重值**。

| 维度 | tiny（本机产物） | 官方 30B-A3B-Thinking | 差距 |
|---|---:|---:|---|
| 文本层数 | 1 | 48 | 48× |
| hidden_size | 8 | 2048 | 256× |
| 注意力头 / KV 头 | 1 / 1 | 32 / 4 | — |
| MoE 专家数 / top-k | 4 / 2 | 128 / 8 | 32× / 4× |
| 词表 | 32 | 152064 | 4752× |
| Vision 深度 | 2 | 27 | 13.5× |
| Audio 层数 / mel | 2 / 20 | 32 / 128 | — |
| 参数量 | 几万级 | 约 300 亿（A3B 激活） | ~10⁶ 倍 |
| 权重来源 | **随机初始化** | 官方训练权重 | **本质区别** |
| 单文件大小 | 约 0.3 MB | 预计 59 GB+ | — |

**验证过的是什么**：同一输入分别跑 PyTorch 与 ONNX Runtime，输出在浮点误差内一致（实测 1e-8 ~ 1e-11），且 MoE 两种路由都一致、未知 Shape 为 0。这证明**“PyTorch → ONNX 转换”这一步数学等价**。

**没有验证的是什么**：模型输出是否有语义。随机权重不携带任何知识，tiny 产物**不能**用来做真实推理，只能用于验证导出接口、图结构、算子与验证链路。

类比：现在造好了编译器并用 hello-world 证明编译器正确；最终目标是用它编译出那个 300 亿参数的正式程序——缺的就是把官方权重灌进去。

## 2. 目录结构

```text
.
├── README.md                        # 本文件
├── requirements.txt                 # 新机器安装用的可移植依赖清单（跨平台）
├── portable-export-requirements-lock.txt  # 本机已验证环境的完整 pip freeze 快照（43 行，仅作环境证据，非跨平台安装用）
├── scripts/bootstrap.sh             # 一键拉取固定源码 + 建环境
├── .gitignore                       # 排除大目录/权重/生成产物
├── Qwen3_Omni_ONNX_导出工具说明.md   # 详细设计说明（上一版文档）
├── Qwen3-Omni_ONNX_任务交接说明.md   # 任务背景与决策记录
├── qwen3_omni_onnx_cases.py         # 库：tiny 回归模型/路由捕获/哈希工具
├── qwen3_omni_thinking_components.py# 库：四组件 Wrapper 与官方 MRoPE 输入构造
├── export_onnx.py                   # CLI：早期三级 tiny 回归导出
├── validate_onnx.py                 # CLI：单 ONNX 严格验证（Checker/Shape/ORT/数值）
├── inspect_onnx.py                  # CLI：算子/domain/external data 报告
├── export_thinking_onnx.py          # CLI：四组件正式导出（tiny/real 两种模式）
├── validate_thinking_pipeline.py    # CLI：端到端张量接力验证（含 3 步 Decode）
├── aggregate_operators.py           # CLI：算子汇总（CSV + summary.json）
├── build_thinking_package.py        # CLI：整理最终产品目录 + manifest
├── run_local_thinking_pipeline.py   # CLI：一键本地全流程（tiny）
├── run_real_thinking_pipeline.py    # CLI：一键官方权重全流程（Linux 大内存）
├── artifacts/                       # 早期 tiny ONNX + 官方模型元数据（gitignore）
└── Qwen3-Omni-30B-A3B-Thinking-ONNX/# 本地流水线生成的产品包（gitignore，可重建）
```

> `Qwen3-Omni/`、`transformers-v5.2.0/`、`TensorRT-Edge-LLM-v0.10.1/` 三个第三方克隆不入库，用 `scripts/bootstrap.sh` 重新拉取固定版本。

## 3. 环境准备

```bash
# 全新机器（macOS arm64 或 Linux x86_64，需要 Python 3.11 和 git）
bash scripts/bootstrap.sh
source .venv/bin/activate
```

脚本做四件事：拉取并固定 `transformers v5.2.0`（commit `7d9754a0…`）、可选拉取两个参考仓库、创建 `.venv` 并按 `requirements.txt` 装依赖、最后校验 `transformers.__file__` 确实来自固定源码目录（防止误用 PyPI wheel，源头不可复现）。

Linux + CUDA 的 torch 安装请参考 pytorch.org 选择对应 wheel 后再跑 bootstrap（或先装 torch 再 `pip install -r requirements.txt`）。

### 3.1 `requirements.txt` 与 `portable-export-requirements-lock.txt` 的区别

| 文件 | 是什么 | 怎么用 |
|---|---|---|
| `requirements.txt` | **可移植安装清单**：只锁关键包版本（torch / onnx / onnxruntime / onnxscript / accelerate / safetensors …），不绑死平台 | 新机器一律用它：`pip install -r requirements.txt`；Linux + CUDA 先按 pytorch.org 装好 torch 再装 |
| `portable-export-requirements-lock.txt` | **本机已验证环境的完整快照**：`pip freeze` 的全部 43 行，包含所有传递依赖（certifi、click、filelock、flatbuffers……），其中 torch 是 **macOS arm64 wheel** | **只作环境证据/审计**，不要拿到 Linux 上 `pip install -r`（平台 wheel 不匹配会直接失败）。它被 `build_thinking_package.py` 复制进产品包 `tools/`，作为“这个 ONNX 是在什么环境里产出的”的不可篡改证据 |

简单记：**装环境用 `requirements.txt`；留证据用 lock 文件。**

## 4. 快速开始（无需下载任何权重，约 1 分钟）

```bash
source .venv/bin/activate
python run_local_thinking_pipeline.py
```

一条命令完成：导出 4 个组件 ONNX → 逐组件严格验证 → 算子检查 → 端到端张量接力（3 步 Decode）→ 算子汇总 → 生成产品目录 `Qwen3-Omni-30B-A3B-Thinking-ONNX/`。

产物（tiny 随机权重，接口与官方版完全一致）：

```text
Qwen3-Omni-30B-A3B-Thinking-ONNX/
├── manifest.json              # 产品自述：状态、接口、哈希、验证结论
├── config.json / generation_config.json
├── processor/                 # tokenizer / audio / image processor 配置
├── onnx/{vision_encoder,audio_encoder,thinker_prefill,thinker_decode}/
│   ├── model.onnx
│   ├── model.onnx.data
│   ├── export_metadata.json
│   └── validation.json
├── validation/end_to_end.json # 端到端证据
├── operators/*.json + all_operators.csv + summary.json
├── test_data/<组件>/          # 固定测试输入与 PyTorch 参考输出
└── tools/                     # 本工具的随包快照 + 依赖锁
```

## 5. 命令参考

**哪个 ONNX 由哪条命令产出**（全仓库一共 7 个 `model.onnx`）：

**A. `artifacts/` 下的三个（早期三级 tiny 回归，`export_onnx.py`）**

```bash
python export_onnx.py --case rmsnorm      --output-dir artifacts/rmsnorm      --force
python export_onnx.py --case moe_block    --output-dir artifacts/moe_block    --force
python export_onnx.py --case tiny_thinker --output-dir artifacts/tiny_thinker --force
```

**B. `Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/` 下的四个（`export_thinking_onnx.py`）**

```bash
python export_thinking_onnx.py --mode tiny --component vision_encoder --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force
python export_thinking_onnx.py --mode tiny --component audio_encoder  --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force
python export_thinking_onnx.py --mode tiny --component thinker_prefill --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force
python export_thinking_onnx.py --mode tiny --component thinker_decode  --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force
```

逐条对应：

- `onnx/vision_encoder/model.onnx` ← 第一条（`--component vision_encoder`）
- `onnx/audio_encoder/model.onnx` ← 第二条（`--component audio_encoder`）
- `onnx/thinker_prefill/model.onnx` ← 第三条（`--component thinker_prefill`）
- `onnx/thinker_decode/model.onnx` ← 第四条（`--component thinker_decode`）

> 四个组件也可以一条命令全出：`--component all`（见 5.1）。
> 想一步到位连导出+验证+端到端+算子汇总+打包全跑完，用 `python run_local_thinking_pipeline.py`（第 4 节）。

### 5.1 四组件导出

```bash
# tiny 模式（随机小权重，接口与官方一致）——上面 B 组四条命令的合并版
python export_thinking_onnx.py --mode tiny --component all --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force

# real 模式（官方权重；仅在大内存 Linux 执行，见第 8 节）
python export_thinking_onnx.py --mode real --component all --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking --dtype float16 --device cuda --minimum-memory-gib 128 --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force
```

`--component` 可选 `vision_encoder | audio_encoder | thinker_prefill | thinker_decode | all`。

### 5.2 单模型严格验证

```bash
python validate_onnx.py --case thinker_prefill --model Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_prefill/model.onnx
```

说明：

- `--case` **可省略**——不传时工具直接读取同目录 `export_metadata.json` 来判断这是哪个模型。
- 传了 `--case` 会做一次**交叉校验**：参数值必须与该 ONNX 的导出元数据一致，不一致直接报错（防止拿错文件还以为验证通过了）。
- 合法取值共 7 个：
  - 早期三级回归：`rmsnorm` / `moe_block` / `tiny_thinker`
  - 四组件：`vision_encoder` / `audio_encoder` / `thinker_prefill` / `thinker_decode`

```bash
# 不传 --case（推荐，最不容易出错）
python validate_onnx.py --model Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_decode/model.onnx

# artifacts 下的早期回归模型同样可以不传 --case（目录里也有 export_metadata.json）
python validate_onnx.py --case moe_block --model artifacts/moe_block/model.onnx
```

### 5.3 算子 / 结构检查

```bash
python inspect_onnx.py --model Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_decode/model.onnx --output Qwen3-Omni-30B-A3B-Thinking-ONNX/operators/thinker_decode.json --fail-on-custom-domain
```

### 5.4 端到端接力验证

```bash
python validate_thinking_pipeline.py --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX
```

real 模式追加 `--mode real --model-path ... --provider CUDAExecutionProvider`；覆盖不同模式的旧报告需显式 `--force`（防止 tiny 证据覆盖 real 证据）。一键脚本同理：产品目录里若已有 real 报告，`python run_local_thinking_pipeline.py` 会被拒绝，需 `python run_local_thinking_pipeline.py --force`。

### 5.5 算子汇总与打包

```bash
python aggregate_operators.py --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX
python build_thinking_package.py --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --offline
```

### 5.6 早期三级 tiny 回归（RMSNorm / MoE Block / Tiny Thinker）

`artifacts/` 下的三个 ONNX 都由 `export_onnx.py` 产出，一条命令一个（**必须带 `--force`**，否则目录非空会拒绝覆盖）：

```bash
python export_onnx.py --case rmsnorm      --output-dir artifacts/rmsnorm      --force
python export_onnx.py --case moe_block    --output-dir artifacts/moe_block    --force
python export_onnx.py --case tiny_thinker --output-dir artifacts/tiny_thinker --force
```

验证与算子检查（`--case` 可省略，工具会读同目录 `export_metadata.json` 自动判断；下面显式写 `--case` 是为了顺带做一次交叉校验）：

```bash
python validate_onnx.py --case rmsnorm      --model artifacts/rmsnorm/model.onnx
python validate_onnx.py --case moe_block    --model artifacts/moe_block/model.onnx
python validate_onnx.py --case tiny_thinker --model artifacts/tiny_thinker/model.onnx

python inspect_onnx.py --model artifacts/rmsnorm/model.onnx      --fail-on-custom-domain
python inspect_onnx.py --model artifacts/moe_block/model.onnx    --fail-on-custom-domain
python inspect_onnx.py --model artifacts/tiny_thinker/model.onnx --fail-on-custom-domain
```

> 输出目录被限制为 `artifacts/` 的直接子目录（`export_onnx.py` 强制校验），所以 `--output-dir` 不能写到别处。

## 6. 四组件 ONNX 接口

| 组件 | 主要输入 | 主要输出 | 说明 |
|---|---|---|---|
| `vision_encoder` | `pixel_values [N,3·t·h·w]`、`position_indices/weights`、`rotary_cos/sin`、`attention_mask` | `vision_embeddings [Σtokens,H]`、`deepstack_visual_0..2` | grid 相关 Python 预处理（位置索引/变长拼接）由宿主完成，ONNX 内是纯张量计算 |
| `audio_encoder` | `padded_features [chunks,16,mel]`、`valid_indices`、`cu_seqlens` | `audio_embeddings [Σtokens,H]` | 变长分块/padding/mask 构造由宿主完成 |
| `thinker_prefill` | `input_ids`、`attention_mask`、`position_ids[3,B,S]`、`cache_position`、`multimodal_embeddings [B,S,H]`、`multimodal_mask`、`visual_position_mask`、`deepstack_visual_*` | `logits [B,S,V]` + `present_key/value_0..47`（共 96 个 Cache 张量） | 多模态 embedding 按位合并；MRoPE position IDs 用官方 `get_rope_index` 生成 |
| `thinker_decode` | `input_ids [B,1]`、`attention_mask [B,past+1]`、`position_ids`、`cache_position`、`past_key/value_0..47` | `logits [B,1,V]` + `present_key/value_0..47` | **past-sequence 为动态轴**；自回归时把本步 96 个输出直接回灌为下步输入 |

宿主程序负责：tokenizer 与媒体预处理、多模态占位与位置计算、自回归循环/采样/停止条件、prefill→decode 的 KV 交接。这些是 Python 控制流，不适合也不应该塞进单个 ONNX。

**dtype 契约（已在导出时固定，宿主必须按此喂数据）**：`input_ids` / `attention_mask` / `cache_position` / `valid_indices` 为 `int64`，`cu_seqlens` 为 `int32`，`position_ids` 在 Prefill 与 Decode 两个图里**都是 `float32`**（来源为官方 `get_rope_index`），其余张量为 `float32`。

### 6.1 三部分权重、四张 ONNX 图

官方 checkpoint 里这三部分权重**命名空间互相独立**，可以分别导出：

| 权重命名空间 | 对应 ONNX | 说明 |
|---|---|---|
| `thinker.visual.*` | `vision_encoder/model.onnx` | 视觉编码器（ViT + DeepStack merger） |
| `thinker.audio_tower.*` | `audio_encoder/model.onnx` | 音频编码器（Conv + Transformer + projection） |
| `thinker.model.*` + `thinker.lm_head.*` | `thinker_prefill/model.onnx` **和** `thinker_decode/model.onnx` | **同一份文本权重，导出成两张图**（prefill 与 decode 只是输入输出形态不同） |

因此是 **3 组权重 → 4 张 ONNX 图**。

每个组件都有自己的**独立算子报告**，不需要跑完四个才有清单：

```text
operators/vision.json           vision_encoder 的算子清单
operators/audio.json            audio_encoder 的算子清单
operators/thinker_prefill.json  prefill 图的算子清单
operators/thinker_decode.json   decode 图的算子清单
operators/all_operators.csv     汇总（带 Component 列，可按组件筛选）
operators/summary.json          各组件节点数 + 全局唯一算子
```

也可以只导出其中一部分（`--component vision_encoder`）。

⚠️ **当前限制**：`load_real_thinking_model()` 会**整体加载** checkpoint（约 59 GiB），即使只导 `vision_encoder` 也要先把全量权重装进内存——这正是本机 48 GiB 无法执行的原因。若要在小内存机器上只导视觉/音频，需要“按 key 选择性加载”的增强（见第 8 节局限性）。

### 6.2 tiny 尺寸定义在哪些代码里

| 配置 | 文件 | 函数 | 行 |
|---|---|---|---|
| 文本（tiny） | `qwen3_omni_onnx_cases.py` | `make_tiny_text_config()` | 101 |
| 视觉（tiny） | `qwen3_omni_thinking_components.py` | `make_tiny_vision_config()` | 268 |
| 音频（tiny） | `qwen3_omni_thinking_components.py` | `make_tiny_audio_config()` | 304 |
| 组合（tiny） | `qwen3_omni_thinking_components.py` | `make_tiny_thinker_config()` | 287 |
| 官方真实尺寸 | `artifacts/real_thinking_metadata/config.json` | — | — |

## 7. 如何自证导出正确（自检体系）

每个 ONNX 必须连续通过以下检查，任何一步失败即退出非零：

1. **来源固定**：导出前校验 `transformers.__file__` 来自固定源码目录且 commit 为 `7d9754a0…`；real 模式对 checkpoint 的 config/index/每个权重分片做 SHA-256 指纹并写入元数据。
2. **ONNX Checker**：`onnx.checker.check_model(..., full_check=True)`。
3. **严格 Shape Inference**：`strict_mode=True`，并统计未知维度（当前全部为 0）。
4. **ONNX Runtime 加载与执行**：仅用标准 `CPUExecutionProvider`（tiny 验证），证明不依赖任何自定义运行时。
5. **数值一致性**：同一输入分别跑 PyTorch 与 ORT，比对名称/Shape/dtype/NaN/Inf，`rtol=1e-4, atol=1e-5`。
6. **MoE 路由覆盖**：每个模型固定两组输入，捕获每层 Top-K expert indices 并记录 SHA-256；两组路由必须不同，且两组输出都必须与 PyTorch 一致——证明没有把第一次导出的专家静态固化进图。
7. **结构纯净**：`inspect_onnx.py --fail-on-custom-domain` 递归统计子图/函数节点，要求无 `trt`/`trt_edgellm` 等自定义 domain；external data 校验存在性、非空、offset/length 越界和 SHA-256。
8. **哈希链**：`export_metadata.json` 记录模型/输入/参考输出的 SHA-256；`validate_onnx.py` 重算比对；`aggregate_operators.py` 与 `build_thinking_package.py` 再次交叉比对当前 ONNX 哈希，文件被替换即报错。
9. **端到端接力**：Vision/Audio ONNX 输出拼进 Prefill 输入，Prefill 的 96 个 KV 逐个回灌 Decode，连续 3 步自回归，每步与 PyTorch 双缓存独立推进的结果逐张量对比。

当前实测（tiny，`rtol=1e-4/atol=1e-5` 全部通过）：

| ONNX | 节点数 | 最大绝对误差 |
|---|---:|---:|
| `rmsnorm` | 7 | 2.4e-7 |
| `moe_block` | 33 | 1.5e-11 |
| `tiny_thinker` | 142 | 2.2e-8 |
| `vision_encoder` | 78 | 5.6e-9（含单图/多图/视频 grid 三种 profile） |
| `audio_encoder` | 65 | 1.2e-9（多 chunk + 尾 chunk） |
| `thinker_prefill` | 160 | logits 4.5e-8 |
| `thinker_decode`（动态 past 12/14） | 158 | logits 2.2e-8 |

四组件合计 **461 节点、41 种标准算子、0 个自定义 domain**（清单见产品包 `operators/all_operators.csv`）。

关键算子（编译器/部署方最该关注的）：`MatMul(25) Mul(61) Transpose(44) Unsqueeze(40) Reshape(37) Add(39) Gather(14) Gemm(18) Softmax(6) LayerNormalization(7) Conv(4) Erf(8) TopK(2) GatherND(9) ScatterND(5) ScatterElements(4) NonZero(2) Where(3) Slice(19) Concat(13) ReduceMean(10) ReduceSum(6) Sin/Cos(2+2) Range(1) Shape(2) Expand(4)` 等。

### 7.1 怎么查看导出的 ONNX

#### `.onnx` 和 `.onnx.data` 分别是什么

| 文件 | 内容 | 说明 |
|---|---|---|
| `model.onnx` | **计算图**：节点、连接关系、输入输出 Shape/dtype、算子属性 | protobuf 二进制，用文本编辑器打开必然是乱码，**不要编辑** |
| `model.onnx.data` | **权重**（initializer） | ONNX 单文件 protobuf 上限 2 GB，大模型权重必须外置；`model.onnx` 里记录每个张量在 data 文件中的 offset/length |

**铁律**：两个文件必须**同目录、成对**存放/拷贝/传输，缺一个或改名就打不开。

#### 方式一：Netron 可视化（最直观，推荐）

网页版，无需安装：

1. 打开 <https://netron.app>
2. 把 `model.onnx` **拖进页面**
3. 即可看到：整图拓扑、每个节点的算子与属性、输入输出 Shape/dtype、点击节点看权重信息

> 想让 Netron 显示权重数值，把同目录的 `model.onnx.data` 也放在本地即可（网页版读取的是你本地文件，不会上传模型）。

本地安装（可选）：

```bash
pip install netron && netron artifacts/rmsnorm/model.onnx   # 浏览器打开
brew install --cask netron                                  # macOS App
```

#### 方式二：本仓库自带的 `inspect_onnx.py`（出 JSON 报告）

```bash
python inspect_onnx.py --model artifacts/rmsnorm/model.onnx
```

输出 ir_version、opset、输入输出名/Shape/dtype、节点数、每种 `domain::OpType` 计数、自定义 domain、external data 校验，并写入 `operators.json`。

#### 方式三：ONNX 官方 API 打印可读计算图

```bash
python -c "
import onnx
m = onnx.load('artifacts/rmsnorm/model.onnx')
print(onnx.printer.to_text(m.graph))   # 旧版用 onnx.helper.printable_graph(m.graph)
"
```

实测输出（RMSNorm 的真实计算图，onnx 1.22）：

```text
main_graph (float[1,4,8] hidden_states) => (float[1,4,8] normalized_hidden_states)
   <float[8] weight =  {1,1,1,1,1,1,1,1}, int64[1] val_3 =  {-1}, float val_0 =  {2}, float val_4 =  {1e-06}, ...>
{
   [node_pow_1] pow_1 = Pow (hidden_states, val_0)
   [node_mean] mean = ReduceMean <noop_with_empty_axes: int = 0, keepdims: int = 1> (pow_1, val_3)
   [node_add] add = Add (mean, val_4)
   [node_Sqrt_5] val_5 = Sqrt (add)
   [node_rsqrt] rsqrt = Reciprocal (val_5)
   [node_mul] mul = Mul (hidden_states, rsqrt)
   [node_mul_1] normalized_hidden_states = Mul (weight, mul)
}
```

看单个节点：

```bash
python -c "
import onnx
m = onnx.load('artifacts/rmsnorm/model.onnx', load_external_data=False)
for i, n in enumerate(m.graph.node[:5]):
    print(f'{i}: {n.op_type:12s} inputs={list(n.input)} outputs={list(n.output)}')
print('inputs :', [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in m.graph.input])
print('outputs:', [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in m.graph.output])
"
```

> `load_external_data=False` 表示**只读图结构、不读权重**，所以几十 GB 的真实模型也能秒开查看（`inspect_onnx.py` 内部就是这么做的）。

## 8. 局限性（务必阅读）

1. **当前交付的是 tiny 接口验证件**：结构、接口、算子与官方一致，但权重是随机小配置（1 层/hidden 8/4 专家）。官方 30B 版需按第 9 节执行，完成后 `manifest.json` 的 `status` 才会变为 `official-weight-components-validated`。
2. **`generate()` 不导出**：自回归循环、采样、停止条件属于 Python 控制流，保留在宿主程序。
3. **Prefill/Vision/Audio 为固定 Shape**（tiny 档 S=12、vision grid 1×4×4；real 档默认 S=32、vision grid 1×4×4、audio 201 帧，均可在 `build_real_thinking_component` 调整）；只有 Decode 的 past-sequence 是动态轴。其他动态需求需改 profile 并重跑验证。
4. **Attention 用 eager、MoE 用 `batched_mm`**：均为 Transformers v5.2.0 官方可切换实现，非自造算法；但与 FlashAttention/SDPA 存在浮点顺序差异，属正常数值误差。
5. **媒体质量未评测**：端到端验证证明“张量接力正确”，不是语音/视觉感知质量基准。
6. **官方权重本机不可导出**：59.08 GiB 权重 > 48 GiB 内存，已由 `--minimum-memory-gib` 门禁强制拦截，不会假成功。
7. **Thinking 无语音输出**；需要语音请改用 Instruct checkpoint 并新增 Talker/CodePredictor/Code2Wav 三个组件（接口调研结论见 `Qwen3-Omni_ONNX_任务交接说明.md` 第 12 节与本文件第 12 节）。

## 9. 后续步骤：在大内存 Linux 上导出官方权重

### 9.1 资源要求

```text
Linux x86_64，≥128 GiB RAM（导出四组件；端到端验证建议 ≥192 GiB）
磁盘 ≥200 GiB（权重 59.08 GiB + ONNX external data + 中间产物）
GPU 可选但推荐（`--device` 默认是 `cpu`，用 GPU 必须显式传 `--device cuda`；此时脚本会按权重总量 1.2× 检查可用显存）
精度：`--dtype float16` 或 `bfloat16`；**用 CPU 做 ONNX Runtime 验证时必须选 `float16`**——bf16 图在 `CPUExecutionProvider` 上跑不起来（ORT 会拒绝喂数），只有 CUDA EP 才可能支持。选错时 `validate_thinking_pipeline.py` 会明确报错
`--minimum-memory-gib` 默认 128（real 模式物理内存保护线，低于该值直接拒绝执行）
```

### 9.2 步骤

```bash
git clone https://github.com/<你>/qwen3omni-onnx-export.git
cd qwen3omni-onnx-export
bash scripts/bootstrap.sh
source .venv/bin/activate

# 下载固定 revision 的官方权重（绝不入库，.gitignore 已排除）
hf download Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --revision 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
    --local-dir /data/Qwen3-Omni-30B-A3B-Thinking

# 一键导出 + 组件验证 + 算子汇总 + 打包（默认不跑端到端，避免同时驻留两份模型）
python run_real_thinking_pipeline.py \
    --model-path /data/Qwen3-Omni-30B-A3B-Thinking \
    --dtype float16 --device cuda \
    --minimum-memory-gib 128

# 内存充足时追加官方权重端到端验证（三步 Decode）
python run_real_thinking_pipeline.py \
  --model-path /data/Qwen3-Omni-30B-A3B-Thinking \
  --run-end-to-end --minimum-memory-gib 128
```

⚠️ 上面第一条命令（默认不跑端到端）结束时 `build_thinking_package.py` 会以**非 0 退出**并打印 `status=unverified-real-artifacts`——这是预期行为，因为第 9.3 节的验收判据要求端到端通过。要拿到 `official-weight-components-validated`，必须显式加 `--run-end-to-end`。

### 9.3 交付验收判据

打开 `manifest.json`，只有同时满足以下条件才视为官方权重产品完成：

```text
status = "official-weight-components-validated"
official_weights_included = true          # 四组件均为 real profile
shared_checkpoint_fingerprint = true      # 四组件来自同一 checkpoint 指纹
source_equivalence_passed = true          # Wrapper 与固定 Transformers 原始 forward 等价
end_to_end_validation_passed = true       # real 端到端 3 步 Decode 通过且绑定当前模型哈希
```

任一条件不满足时 status 会是 `unverified-real-artifacts` 或 `tiny-interface-validation-only`，此时不得对外宣称是“官方权重 ONNX”。

### 9.4 完整命令速查（照抄即可，注意两种内存档位）

**档位 A：内存 ≥128 GiB —— 一条命令得到最终产品**

`run_real_thinking_pipeline.py` 内部已经串好了全部步骤（导出 → 逐组件 validate → 逐组件 inspect → 算子汇总 → 打包），**不需要再手动补任何命令**：

```bash
source .venv/bin/activate
MODEL=/data/Qwen3-Omni-30B-A3B-Thinking
OUT=Qwen3-Omni-30B-A3B-Thinking-ONNX

# 有 GPU
python run_real_thinking_pipeline.py \
  --model-path "$MODEL" --dtype float16 --device cuda \
  --minimum-memory-gib 128

# 无 GPU
# 注意：无 GPU 时必须用 float16。bfloat16 图无法被 ORT 的 CPUExecutionProvider 执行
# （实测喂 float32 与 ml_dtypes.bfloat16 都会被拒绝），工具会直接报错而不是给出假的通过。
python run_real_thinking_pipeline.py \
  --model-path "$MODEL" --dtype float16 --device cpu \
  --minimum-memory-gib 128

# 内存 ≥192 GiB 时，追加官方权重端到端验证（三步 Decode）
python run_real_thinking_pipeline.py \
  --model-path "$MODEL" --run-end-to-end --minimum-memory-gib 128
```

**档位 B：内存 96–128 GiB —— 逐个组件导出（必须补收尾命令）**

⚠️ 逐个导出只做“导出”这一步，**不会自动生成 manifest / 端到端 / 算子汇总**。必须再跑下面第二段收尾命令，否则拿不到完整产品包。

```bash
source .venv/bin/activate
MODEL=/data/Qwen3-Omni-30B-A3B-Thinking
OUT=Qwen3-Omni-30B-A3B-Thinking-ONNX

# ① 逐个导出（四个进程，各自重新加载一次权重，慢但峰值低）
for c in vision_encoder audio_encoder thinker_prefill thinker_decode; do
  python export_thinking_onnx.py --mode real --component "$c" \
    --model-path "$MODEL" --dtype float16 --device cuda \
    --minimum-memory-gib 96 --package-dir "$OUT" --force
done

# ② 收尾：逐组件验证 + 算子检查（缺一不可）
for c in vision_encoder audio_encoder thinker_prefill thinker_decode; do
  python validate_onnx.py --model "$OUT/onnx/$c/model.onnx"
  python inspect_onnx.py --model "$OUT/onnx/$c/model.onnx" \
    --output "$OUT/operators/${c%_encoder}.json" --fail-on-custom-domain
done

# ③ 收尾：端到端接力（可选，内存 ≥192 GiB 时跑）+ 算子汇总 + 打包
python validate_thinking_pipeline.py --mode real --model-path "$MODEL" \
  --package-dir "$OUT" --force
python aggregate_operators.py --package-dir "$OUT"
python build_thinking_package.py --package-dir "$OUT" \
  --source-dir "$MODEL" --offline
```

完成后检查：

```bash
grep -E '"status"|official_weights_included|shared_checkpoint_fingerprint|source_equivalence_passed|end_to_end_validation_passed' \
  "$OUT/manifest.json"
```

## 10. 版本管理（GitHub）

`.gitignore` 已排除：三个第三方源码克隆、`.venv/`、全部生成产物（`artifacts/*`、产品包）、任何权重文件。仓库本身只有工具代码 + 文档 + 参考元数据，体积极小。

```bash
git init
git add .
git commit -m "qwen3omni onnx export tool: 4-component pipeline + verification"
git branch -M main
git remote add origin git@github.com:<你>/qwen3omni-onnx-export.git
git push -u origin main
```

日常“勤 push、可回退”：

```bash
git add -A && git commit -m "..." && git push
git log --oneline          # 找回退点
git revert <commit>        # 安全回退某次改动
git checkout <commit> -- <file>   # 恢复单个文件
```

## 11. 常见问题

- **`ModuleNotFoundError`**：所有命令必须在仓库根目录执行（脚本按根目录定位产品包和 artifacts）。
- **`transformers provenance` 报错**：当前 Python 环境的 transformers 不是固定源码，重新执行 `bash scripts/bootstrap.sh` 或 `pip install -e ./transformers-v5.2.0`。
- **`--force` 被拒绝**：导出目录必须在 `artifacts/` 的一级子目录内（防误删工作区）。
- **tiny 验证想覆盖 real 报告**：加 `--force`（工具会显式提示）。
- **换模型/换 Shape**：改 `qwen3_omni_thinking_components.py` 中的 profile 后必须重跑全部验证，哈希链会自动暴露未重验的产物。

## 12. 设计决策与实现选择

以下内容来自早期设计文档的精华，已并入本 README（`Qwen3_Omni_ONNX_导出工具说明.md` 仅保留为历史指针）。

### 12.1 为什么不用原始 eager Experts 循环

Qwen3-Omni 原始 MoE 专家计算会根据输入的路由结果执行 Python 循环：

```text
one_hot → nonzero → for expert_idx → where → index_add_
```

这含数据相关 Python 控制流，传统 tracing 极易只固化示例输入命中的专家，导致“导出成功但换输入就错”。

Transformers v5.2.0 自带等价的可切换实现 `experts_implementation="batched_mm"`：

```text
TopK indices → Gather 对应专家权重 → batched MatMul → 路由权重 → reshape + ReduceSum
```

本工具使用官方提供的这一实现，**不是自行改写模型数学定义**；这样得到的图只含标准 ONNX 算子，且路由随输入动态变化（已用两组不同路由验证）。

### 12.2 为什么 KV Cache 必须显式展平

Transformers 的 `DynamicCache` 是 Python 对象，不能作为 ONNX 公共接口。因此：

- `thinker_prefill` 输出 **96 个显式张量**（`present_key/value_0..47`）
- `thinker_decode` 输入/输出同样 96 个（`past_*` / `present_*`）
- 自回归时把本步 96 个输出直接喂回下一步的 96 个输入
- past-sequence 是唯一动态轴，其余维度固定

### 12.3 为什么要拆成四个组件而不是一个图

Qwen3-Omni 完整链路含 tokenizer、媒体文件读取、grid/分块预处理、自回归循环、采样停止条件、可选模态分支，这些都是 Python 控制流。ONNX 适合表达“一次张量前向”，不适合表达完整应用。因此产品是**四个神经网络组件 + 宿主调度程序**，而不是单个 ONNX。

### 12.4 早期三级回归模型的定位

`rmsnorm`（7 节点）/ `moe_block`（33）/ `tiny_thinker`（142）是**单元测试级**产物：

```text
rmsnorm      → 验证导出环境、Checker、ORT、归一化算子
moe_block    → 专项验证 MoE 动态路由（最关键风险）
tiny_thinker → 验证 Attention+RoPE+Mask+MoE+LM Head 组合
```

它们与四组件是**回归基线关系**：改动正式导出代码后先跑这三个，能快速定位是环境问题还是组件问题。

## 13. 参考文档

- `Qwen3_Omni_ONNX_自检验证报告.md`：**凭什么说这套工具有效**——完整核实方案、实际执行输出、反向测试与证据链（想手动复核看这份）
- `Qwen3-Omni_ONNX_任务交接说明.md`：任务背景、路线决策（标准 ONNX 主线 + NVIDIA 参考线）、接管指南与最新状态（交给另一台机器的 AI 时先读它的第 0 节）
- `artifacts/real_thinking_metadata/resource_assessment.json`：官方权重规模与本机资源评估证据
- `Qwen3_Omni_ONNX_导出工具说明.md`：历史设计说明，内容已并入本 README（保留文件仅为兼容旧链接）
