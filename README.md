# Qwen3-Omni ONNX 导出工具

面向 [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)（Hugging Face `qwen3_omni_moe` 架构）的组件级 ONNX 导出与验收工具。目标产物只含标准 ONNX 算子，并记录数值对比、有限 MoE 路由样例、external data 校验、算子清单和哈希绑定；实际支持范围须经当前环境和产物验证。

- 目标模型：`Qwen/Qwen3-Omni-30B-A3B-Thinking`（多模态输入 → 文本输出，无语音输出）
- 固定模型实现：Hugging Face `transformers` **v5.2.0**（commit `7d9754a05193eb79b1d86aa744b622b8068008cd`）
- 已验证环境：Python 3.11.9 / PyTorch 2.8.0 / ONNX 1.22.0 / ONNX Runtime 1.30.0 / onnxscript 0.7.2（macOS arm64 实测通过）

---

## 1. 当前状态速览

| 能力 | 状态与边界 |
|---|---|
| 四组件接口（Vision / Audio / Thinker Prefill / Thinker Decode） | 已实现；tiny 随机权重历史流程已运行，本轮须重导出生成 schema v2 证据 |
| 显式 KV Cache | tiny 文本 1 层，Prefill 输出 2 个 K/V；官方 48 层对应 96 个，尚未真实权重验收 |
| Decode 动态 past-sequence 与三步续接 | 有限 tiny profile 回归；64 是导出约束上界，不是全范围实测承诺 |
| MoE 动态路由 | 采用标准张量路径并检查多组路由，不代表所有专家组合均已覆盖 |
| 两条数值参考链 | 官方顶层语义 → Wrapper；Wrapper → ORT，必须分别验收，不能共用错误位置计算冒充正确 |
| 官方 30B 权重导出 | 本机 48 GiB 未加载/验收；大内存流程仍有内核、峰值资源、精度与语义待验证项 |
| 语音输出（Talker / CodePredictor / Code2Wav） | 不在范围；换 Instruct 仍须新增组件适配 |

**当前不能交付“已验收的官方 30B ONNX”。** 历史 tiny 结果只能支持对应样例的结论；本轮修复与实测记录分开列在 `Qwen3_Omni_ONNX_自检验证报告.md`，不能把旧报告自动升级为新证据。

### 1.1 `tiny` 是什么，与官方模型的差距

`tiny` 使用官方 Qwen3-Omni 类、小配置和随机初始化权重。它可用于验证接口及部分执行路径，但小尺寸、不同路由和导出特化可能改变图结构，不能声称与官方规模覆盖完全相同的算子路径。

| 维度 | tiny 当前配置 | 官方 30B-A3B-Thinking |
|---|---:|---:|
| 文本层数 / KV 张量数 | 1 / 2 | 48 / 96 |
| hidden_size | 8 | 2048 |
| 注意力头 / KV 头 | 1 / 1 | 32 / 4 |
| MoE 专家数 / top-k | 4 / 2 | 128 / 8 |
| 词表 | 32 | 152064 |
| Vision 深度 | 1 | 27 |
| Audio 层数 / mel bins | 1 / 16 | 32 / 128 |
| Audio 默认输入帧数 | 20 | real profile 为 101 |
| 权重来源 | 随机初始化 | 官方训练权重，约 59.08 GiB |

**数值验证的含义**：在记录的环境、输入、Shape、dtype 与容差内，官方参考、Wrapper 与 ORT 的输出达到指定一致性标准。有限样例通过不是数学等价证明，也不证明所有输入、专家路由或长序列均正确。

**尚未验证**：官方权重完整四组件导出和 real 端到端验收、目标设备全部内核支持、真实媒体质量和生成效果。tiny 不携带训练知识，不能用于有意义的模型推理。

## 2. 目录结构

```text
.
├── README.md                        # 本文件
├── requirements.txt                 # 新机器安装用的可移植依赖清单（跨平台）
├── portable-export-requirements-lock.txt  # 历史依赖快照，torch==2.8.0；不是平台或真实性保证
├── scripts/bootstrap.sh             # 一键拉取固定源码 + 建环境
├── .gitignore                       # 排除大目录/权重/生成产物
├── Qwen3_Omni_ONNX_导出工具说明.md   # 历史设计指针与当前边界
├── Qwen3-Omni_ONNX_任务交接说明.md   # 任务背景、历史记录与接管速览
├── Qwen3_Omni_ONNX_自检验证报告.md   # 历史输出与本轮实测分开记录
├── onnx_artifact_utils.py           # 库：公共产物校验、数据交换、源码与证据绑定
├── tests/test_*.py                  # 持久 unittest 回归入口
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
├── onnx_artifact_utils.py           # 库：安全路径、完整产物指纹、BF16 数据通路与严格判据（与模型无关）
├── tests/                           # 持久 unittest：公共证据 / 语义 / 端到端 / 打包验收
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
| `portable-export-requirements-lock.txt` | 历史环境依赖快照；当前 torch 条目为 `torch==2.8.0`，不是 macOS wheel URL | 可用于审计与环境恢复参考，但不能单凭此文件保证跨平台安装成功或证明导出环境。Linux/CUDA 仍需选择匹配的 wheel、驱动及 ORT 安装 |

简单记：**环境安装从 `requirements.txt` 与 `scripts/bootstrap.sh` 开始，lock 辅助记录版本，不是环境或真实性保证。** `tools/` 是打包时复制的快照；导出时真实源码 bytes 归档仍待实现与验收，当前记录的源码 hash/environment/git 不能恢复源码，也不代表已经存在 `source_snapshot/` 归档目录。恢复环境仍需 bootstrap、固定 Transformers 源码和平台依赖。所有这些哈希只能防止意外混用，不是签名或不可篡改证明。

## 4. 快速开始（tiny 无需官方权重）

```bash
source .venv/bin/activate
python -B -m unittest discover -s tests -v
python run_local_thinking_pipeline.py --offline
```

`--offline` 表示非权重资源只走本地文件或已有缓存；允许联网时可省略。runner 在导出**之前**检查这些资源，并会在写入前拒绝覆盖 real 产物（确需覆盖才加 `--force`）。随后编排：四组件导出 → 逐组件验证 → 算子检查 → 三步 Decode 端到端 → 算子汇总 → 打包。实际结果见自检报告第 0.1 节。

产物使用 tiny 随机权重；组件划分沿用目标模型，但层数、张量数量和尺寸不同：

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
> 想一步到位连导出+验证+端到端+算子汇总+打包全跑完，用 `python run_local_thinking_pipeline.py --offline`（第 4 节）。

### 5.1 四组件导出

```bash
# tiny 模式（随机小权重，组件划分相同但尺寸不同）——B 组四条命令的合并版
python export_thinking_onnx.py --mode tiny --component all --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force

# real 模式（官方权重；仅在大内存 Linux 执行，见第 8 节）
python export_thinking_onnx.py --mode real --component all --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking --dtype float16 --device cuda --minimum-memory-gib 128 --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --force
```

`--component` 可选 `vision_encoder | audio_encoder | thinker_prefill | thinker_decode | all`。

**real 模式注意**：`--device cuda/cpu` 决定 PyTorch 设备，`--provider` 属于 ONNX Runtime、由验证阶段使用，两者独立；上面 real 示例的 `--minimum-memory-gib 128` 是**导出**门槛，启用端到端验证需 192（见第 9 节）。

### 5.1.1 导出事务、回滚与并发

导出不再“边检查边删除”，而是：

1. 纯只读预检：目标目录、祖先路径、符号链接、权限与 `--force` 授权；
2. 在暂存目录导出全部组件并通过 ONNX Checker；
3. 全部成功后，把旧组件与旧全局证据移入备份，再整体安装新产物；
4. 任何异常反向恢复旧组件与旧证据；回滚本身失败时保留备份目录并报错，不静默清理。

并发与残留：

- 同一产品目录的导出用 `.export.lock` 互斥；另一个进程正在导出时会直接报错而不是等待。
- 若发现 `.transaction-*` 残留目录，说明上一次事务未正常结束：工具**不会自动删除**未知目录，请先检查备份内容再手工处理。
- 正常异常可回滚；SIGKILL、断电等不保证完整性，这也是失败时保留备份的原因。

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

**报告文件与严格性**（避免“为什么 validation.json 没更新”的困惑）：

| 运行方式 | 写入文件 | 是否可作为正式验收证据 |
|---|---|---|
| 默认（`rtol≤1e-4`、`atol≤1e-5`、完成 Shape Inference） | `validation.json` | ✅ 严格通过 |
| `--skip-shape-inference`，或容差宽于上述基线 | `validation.diagnostic.json` | ❌ 仅诊断，打包会拒绝 |
| 校验过程异常（provider 不可用、输入不合法等） | `validation.failure.json` | ❌ 失败记录 |

- 参数错误（`--case` 传错、provider 不存在、`--atol inf`）会在**写任何报告之前**失败，原有 `validation.json` 保持原样。
- 只有身份（模型、external、metadata、向量）与判据都通过的 `validation.json` 才会被打包采信；诊断报告不能顶替。

### 5.3 算子 / 结构检查

```bash
python inspect_onnx.py --model Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_decode/model.onnx --output Qwen3-Omni-30B-A3B-Thinking-ONNX/operators/thinker_decode.json --fail-on-custom-domain
```

### 5.4 端到端接力验证

```bash
python validate_thinking_pipeline.py --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX
```

real 模式须显式提供 `--mode real --model-path ... --device cpu/cuda --provider ...` 及相应资源设置；端到端验证还需 `--minimum-memory-gib ≥192`，精确流程见第 9 节。

- `--force` 只表示“允许覆盖模式不同的旧端到端报告”，**不是证据升级手段**；新成功后以更新时间戳覆盖旧失败语义，不会删旧文件。
- 执行失败写 `validation/end_to_end.failure.json`，旧报告保留；若失败时间戳不早于旧成功报告，旧成功不再被采信。
- 不要用 tiny 重建命令覆盖已有 real 产品；先保留所需产物并选用独立输出目录。

### 5.5 算子汇总与打包

```bash
python aggregate_operators.py --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX
python build_thinking_package.py --package-dir Qwen3-Omni-30B-A3B-Thinking-ONNX --offline
```

`--source-dir <checkpoint 根目录>` 用于从本地 checkpoint 取 `config.json`、`generation_config.json` 与 processor/tokenizer 文件；显式指定时必须本地**整套齐全**，缺一即失败，不会与 HF 混用。`--offline` 表示只使用本地文件或已有缓存；允许联网时可省略这两个参数的离线限制。

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

设文本层数为 `L`，DeepStack 输出数由配置决定（tiny 为 1，官方配置为 3）。

| 组件 | 主要输入 | 主要输出 | 说明 |
|---|---|---|---|
| `vision_encoder` | `pixel_values [N,3·t·h·w]`、`position_indices/weights`、`rotary_cos/sin`、`attention_mask` | `vision_embeddings [Σtokens,H]`、`deepstack_visual_*` | grid 对应的位置索引/插值和 packed mask 由宿主准备 |
| `audio_encoder` | `padded_features [chunks,mel,chunk_frames]`、`valid_indices`、`cu_seqlens` | `audio_embeddings [Σtokens,H]` | 分块与 padding 在宿主；ONNX 内由 `cu_seqlens` 构造 attention mask，不另收 mask 输入 |
| `thinker_prefill` | `input_ids`、`attention_mask`、`position_ids [3,B,S]`、`cache_position`、`multimodal_embeddings [B,S,H]`、两个位置 mask、`deepstack_visual_*` | `logits [B,S,V]` + `present_key/value_0..L-1` | tiny 2 个 KV；官方配置 96 个。MRoPE 与官方顶层独立参考比对 |
| `thinker_decode` | `input_ids [B,1]`、`attention_mask [B,past+1]`、`position_ids`、`cache_position`、`past_key/value_0..L-1` | `logits [B,1,V]` + `present_key/value_0..L-1` | 每步回灌 2L 个 KV；past-sequence 动态，不能据此无限延长 |

宿主负责 tokenizer、媒体预处理、多模态占位、位置计算、KV 交接、自回归/采样/停止条件。`cache_position` 是物理 Cache 索引；MRoPE 位置需遵循官方语义，不能把两者无条件混用。

**dtype 必须逐组件、逐张量读取当前 ONNX/ORT 接口，不能统一转换成 float32：**

| 张量 | 当前输入构造约定 |
|---|---|
| Vision `position_indices` | `int32` |
| Vision `attention_mask` | 浮点加性 mask，跟随该输入 profile；不是文本 int64 mask |
| Text `input_ids` / `attention_mask` / `cache_position` | `int64` |
| Text `multimodal_mask` / `visual_position_mask` | `bool` |
| Audio `valid_indices` / `cu_seqlens` | `int64` / `int32` |
| Text `position_ids` | 当前 profile 为 `float32`；具体以导出接口为准，不按模型权重精度强制转换 |
| 像素、音频特征、embedding、logits、KV、插值权重与 rotary 张量 | tiny 多为 `float32`；real 可能为 FP16/BF16，辅助张量仍可能保留 FP32，须逐项核对 |

BF16 通过位保持 NPZ 与 ORT IOBinding 交换，不能伪装成 float32 喂入。CPU IOBinding Cast/Identity 小图已验证数据交换可行；完整 Qwen 图是否执行取决于该 provider 的全部算子/dtype 内核，不能断言 CPU 必然不支持或 CUDA 必然支持。

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

| 配置 | 文件 | 定位 |
|---|---|---|
| 文本（tiny） | `qwen3_omni_onnx_cases.py` | `make_tiny_text_config()` |
| 视觉（tiny） | `qwen3_omni_thinking_components.py` | `make_tiny_vision_config()` |
| 音频（tiny） | `qwen3_omni_thinking_components.py` | `make_tiny_audio_config()` |
| 组合（tiny） | `qwen3_omni_thinking_components.py` | `make_tiny_thinker_config()` |
| 官方真实尺寸 | `artifacts/real_thinking_metadata/config.json` | 下载的固定 revision 配置；加载时须与 checkpoint 对应 |

## 7. 如何验证导出（自检体系与证据边界）

验收要求如下；是否通过须看当前产物的实际报告，不能仅依据此清单：

1. **来源与身份**：固定 Transformers 路径/commit；real checkpoint 的 config/index/分片纳入指纹。导出时真实源码 bytes 归档及 metadata 绑定仍待实现与验收，不能把当前 hash 记录或打包时源码冒充导出源码归档。
2. **官方语义参考**：使用固定官方顶层 forward 独立检查位置、模态拼接、DeepStack 和 Cache；它与 Wrapper → ORT 是两条检查链，不能让双方共用有缺陷的位置计算后自行证明正确。
3. **Checker / Shape**：执行 full check、strict shape inference，并记录未知维度与动态维；历史“未知为 0”不是所有 profile 的保证。
4. **Wrapper → ORT**：记录实际 provider，逐个校验输出名称、Shape、dtype、有限值与容差；tiny 默认 `rtol=1e-4, atol=1e-5`，其它精度以明确的验收设置为准。
5. **MoE 路由样例**：捕获各层 Top-K indices；路由样例不同且输出通过，提供有限动态路由回归，不证明全部专家组合。
6. **结构与 external data**：递归统计图/子图/函数 domain；检查 external data 路径、范围与内容哈希。标准域仍需目标运行时支持相应 dtype/内核。
7. **schema v2 绑定**：组件、算子与端到端报告绑定 ONNX、external data、metadata 和测试向量；汇总与打包重验绑定及输出覆盖。旧 schema 必须重导出验收，不能手改 JSON 升级。
8. **端到端**：Vision/Audio → Prefill → 三步 Decode；tiny 回灌 2 个 KV，官方配置为 96 个且尚未 real 验收；官方参考、Wrapper 与 ORT 缓存分别推进。
9. **持久回归**：`python -B -m unittest discover -s tests -v`。公共校验实现在 `onnx_artifact_utils.py`，测试在 `tests/test_*.py`；本轮结果见自检报告第 0.1 节。

哈希是防止意外改动/混用的机制，不是签名或真实性自证；有人同时重写全部证据时，仍需外部可信来源约束。

**以下为 2026-09-22 历史 tiny 数字，保留作对照，不是本轮结果**（旧容差 `rtol=1e-4/atol=1e-5`）：

| ONNX | 节点数 | 最大绝对误差 |
|---|---:|---:|
| `rmsnorm` | 7 | 2.4e-7 |
| `moe_block` | 33 | 1.5e-11 |
| `tiny_thinker` | 142 | 2.2e-8 |
| `vision_encoder` | 78 | 5.6e-9（含单图/多图/视频 grid 三种 profile） |
| `audio_encoder` | 65 | 1.2e-9（多 chunk + 尾 chunk） |
| `thinker_prefill` | 160 | logits 4.5e-8 |
| `thinker_decode`（动态 past 12/14） | 158 | logits 2.2e-8 |

四组件历史合计 **461 节点、41 种标准算子、0 个自定义 domain**；本轮（schema v2，2026-09-23）重导出为 **467 节点、41 种标准算子、0 个自定义 domain**（清单见产品包 `operators/all_operators.csv`）。差异来自本轮语义与接口修复，旧数字不再代表当前产物。

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

1. **当前为 tiny 接口验证件**：1 层文本、hidden 8、4 专家随机权重，不保证官方规模执行路径与图结构。官方 30B 尚未在本机验收。
2. **`generate()` 不导出**：自回归、采样和停止条件留在宿主。
3. **固定与动态 Shape**：tiny Prefill 默认 S=12、vision grid 1×4×4、audio 20 帧；real 默认 S=32、vision grid 1×4×4、audio 101 帧。Decode 的 past 长度是动态轴，但 64 是导出约束而非全范围实测；改变 profile 须重导出和验证。
4. **Attention / MoE 实现选择**：使用 eager 和官方 `batched_mm`；与其它实现的浮点顺序差异仍须按实际容差检查，不能自动判为可接受。
5. **媒体质量未评测**：有限张量接力测试不是真实音视频理解或生成质量基准。
6. **资源门槛不是保证**：48 GiB 本机不具备安全加载约 59.08 GiB 权重条件；大内存目标机仍可能遇到峰值驻留、磁盘、内核、精度及导出问题。不能靠调低门槛强跑。
7. **Thinking 无语音输出**：Instruct 需要新增 Talker/CodePredictor/Code2Wav 适配，不能只换 checkpoint。
8. **BF16 支持是分层问题**：NPZ 位保持和 CPU IOBinding Cast/Identity 小图已测试成功；完整 Qwen 图的 CPU/CUDA 内核与环境仍待实测。

## 9. 后续步骤：在大内存 Linux 上导出官方权重

### 9.1 资源与环境前提

- 规划 Linux x86_64、导出至少 128 GiB RAM、端到端至少 192 GiB；这些是保护/规划门槛，**不保证**实际峰值可承受。物理内存与当前可用内存都要检查，不教调低到 96 GiB 强跑。
- 磁盘应容纳原始约 59.08 GiB 权重、两份文本 ONNX 权重、编码器产物、临时文件和报告；至少 200 GiB 仅作初始预算，应按实际剩余空间留余量。
- PyTorch `--device cpu/cuda` 与 ORT `--provider CPUExecutionProvider/CUDAExecutionProvider` 是两个不同设置。CUDA 需要相应 torch、ORT GPU 安装和驱动，不能仅凭有 GPU 或 CUDA 导出就断言 ORT 在 GPU 上运行。
- `--dtype float16/bfloat16` 需匹配目标 provider 的图内核。BF16 数据交换已修复，CPU 小图可行；完整 Qwen BF16/FP16 图的兼容性和误差仍待实测。
- 固定源码、环境、checkpoint/config/index/全部分片和 tokenizer/processor 文件必须齐全。预检可提前阻止部分错误，但不加载完整模型执行，不能证明正式导出一定成功。

### 9.2 步骤

```bash
git clone https://github.com/JeremyKing10/qwen3omni-onnx-export.git
cd qwen3omni-onnx-export
bash scripts/bootstrap.sh
source .venv/bin/activate

# 下载固定 revision 的官方权重（绝不入库，.gitignore 已排除）
hf download Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --revision 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
    --local-dir /data/Qwen3-Omni-30B-A3B-Thinking

MODEL=/data/Qwen3-Omni-30B-A3B-Thinking
OUT=Qwen3-Omni-30B-A3B-Thinking-ONNX-real

# ① 只做前置检查：不加载 59 GiB 权重，也不写产品目录
python run_real_thinking_pipeline.py \
  --model-path "$MODEL" --package-dir "$OUT" \
  --device cpu --provider CPUExecutionProvider \
  --minimum-memory-gib 192 --run-end-to-end --preflight-only

# ② 预检通过后执行完整流程；此处显式选择 PyTorch CPU 与 ORT CPU
python run_real_thinking_pipeline.py \
  --model-path "$MODEL" --package-dir "$OUT" \
  --dtype float16 --device cpu --provider CPUExecutionProvider \
  --minimum-memory-gib 192 --run-end-to-end
```

`--preflight-only` 检查 checkpoint 索引、内存门槛、磁盘余量、device/provider 可用性和非权重资源；它不验证目标 provider 是否支持该图的全部算子内核。

以上是目标机迁移命令，不是本机已执行结果，也不保证 FP16 CPU 内核全部可用。选择 GPU 时应同时按实际需求配置 `--device cuda` 与 `--provider CUDAExecutionProvider`，并确认相应 torch、ORT GPU、驱动及显存。默认不带 `--run-end-to-end` 的流程缺少最终验收，组件通过也应保持 `unverified-real-artifacts` 并非零退出。

### 9.3 交付验收判据

必须由工具重验当前产物并生成 `manifest.json`，不能手工设置字段。官方验收同时要求：

```text
schema_version = 2
passed = true
status = "official-weight-components-validated"
official_weights_included = true          # 四组件均为真实权重 profile
shared_checkpoint_fingerprint = true      # 同一 checkpoint 身份
source_equivalence_passed = true          # 有限向量的独立官方顶层参考通过
evidence_sources_match = true             # 导出源码/环境证据核对通过
end_to_end_validation_passed = true        # real 三步 Decode 与全部产物绑定通过
operator_summary_valid = true             # 当前图算子报告及汇总绑定通过
```

任何绑定、输出覆盖、来源、数值或配置检查失败都不算完成。`unverified-real-artifacts` / `unverified-tiny-artifacts` 是未验收状态；`tiny-interface-validation-only` 也不是官方权重产品。哈希和 manifest 不是签名，应保留可信生成与分发过程。

### 9.4 大内存是否意味着“一键成功”？

**支持有前提的一键编排，不承诺换机器后无需改代码即可成功。** 第 9.2 节的 runner 串接导出、逐组件验证/算子检查、可选端到端、汇总与打包；环境、完整权重、processor、资源与 provider 都是前提。本机没有执行官方 30B 验收，目标机可能暴露新的图、内核或精度问题。

- 默认未启用 `--run-end-to-end` 时，即使组件阶段通过，最终也应标为 `unverified-real-artifacts` 并非零退出；不能拿到已验收官方产品。
- 启用该选项时，必须完整跑过第 9.3 节的实际检查，不能只看到产物目录或手改 manifest 状态。
- 如果分组件操作，`--component` 只做导出，不替代 validate、inspect、real 端到端、aggregate 与 package。当前 real loader 仍加载全 checkpoint，不能靠逐组件命令承诺降低到 96 GiB 即可运行。
- 内存保护失败应更换资源或重新设计加载方案，不通过降低门槛、放宽容差、绕过证据检查“变绿”。
- `--force` 会覆盖对应产物，正式运行应选用独立 real 产品目录，避免覆盖仍需保留的 tiny 证据。

## 10. 版本管理（GitHub）

`.gitignore` 已排除：三个第三方源码克隆、`.venv/`、全部生成产物（`artifacts/*`、产品包）、任何权重文件。仓库本身只有工具代码 + 文档 + 参考元数据，体积极小。

```bash
git init
git add .
git commit -m "qwen3omni onnx export tool: 4-component pipeline + verification"
git branch -M main
git remote add origin git@github.com:JeremyKing10/qwen3omni-onnx-export.git
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
- **输出目录或 `--force` 被拒绝**：基础 `export_onnx.py` 输出限于 `artifacts/` 直接子目录；四组件产品须在工作区安全路径中。不要通过符号链接或把目标设为工作区根目录绕过检查。
- **tiny 与 real 报告冲突**：使用独立产品目录并保留所需证据；不要把 `--force` 当作无损操作或证据升级。重导出后必须重新验收。
- **换 Shape**：修改 profile 后重导出并重新验收，不复用旧 schema 或旧报告；换模型还需专门适配，见第 14 节。
- **BF16 喂数或 kernel 错误**：区分数据交换失败与算子内核缺失；当前有位保持 NPZ/IOBinding 路径，不将 BF16 偷换 FP32。CPU 小图交换成功不代表 Qwen 整图通过，换 CUDA 或 FP16 也须重新验收。
- **打包 `unverified-real-artifacts` 或非零退出**：检查是否缺 real 端到端、参考链、当前 schema v2 绑定或必要配置。`--run-end-to-end` 是必要流程选项，不是保证成功的开关。
- **`validation.json` 没有更新**：你可能用了 `--skip-shape-inference` 或更宽的容差，此时结果写入 `validation.diagnostic.json`；异常写入 `validation.failure.json`。只有严格通过才更新 `validation.json`。
- **出现 `.export.lock` 或 `.transaction-*`**：前者是并发互斥锁，正常结束不会遗留；后者是未正常结束的事务备份，工具不会自动删除，请先检查备份内容再手工处理。
- **另一个导出进程正在使用目标目录**：同一产品目录不允许并发导出，请等待或改用独立目录。

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

- `thinker_prefill` 输出每层 Key/Value，共 `2L` 个显式 Cache 张量；tiny `L=1`、2 个，官方配置 `L=48`、96 个。
- `thinker_decode` 输入/输出对应 `past_*` / `present_*`，自回归时逐层交接；官方 96 个尚未真实权重验收。
- past-sequence 为动态轴，其余 profile 维度固定；64 的导出约束上界不是全范围验证承诺。

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

## 14. 能否扩展到 DeepSeek、Kimi 等多模型？

**当前只适配 Qwen3-Omni Thinking，不能更换 `--model-path` 就自动导出其它模型。** DeepSeek、Kimi 的不同开放权重版本可能采用不同架构、缓存和算子；必须先选定具体仓库、版本、权重许可与目标设备。仅提供闭源 API 的模型没有可供本工具转换的本地权重，不能导出。

可以复用的通用部分：`onnx_artifact_utils.py` 的路径/external data/哈希与 dtype 工具、ORT 执行和比较、算子统计、schema v2 证据框架。它们不自动解决模型语义。

每个新模型仍需实现并验收：

1. config/架构识别、权重加载与来源约束；
2. forward Wrapper、Prefill/Decode 划分、Cache 布局和位置编码；
3. MoE 路由或其它数据相关控制流的可导出表达；
4. tokenizer/processor、多模态占位与宿主预处理；
5. 独立官方 golden reference → Wrapper → ORT 两层比较，不能共享有缺陷的预处理自验；
6. 多输入、动态长度、路由、dtype、异常和证据失效的边界测试，以及目标机器真实权重验收。

选定首个新模型后，再设计最小 `adapter registry` 接口；现在不新增多模型承诺、不实现 DeepSeek/Kimi adapter，也不重组现有根目录。
