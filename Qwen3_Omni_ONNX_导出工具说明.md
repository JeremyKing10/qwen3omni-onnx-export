# Qwen3-Omni ONNX 导出工具说明与运行结果

## 1. 结论

本工作区已经成功生成并验证 3 个 ONNX：

| ONNX | PyTorch 来源 | 用途 | 验证结果 |
|---|---|---|---|
| `artifacts/rmsnorm/model.onnx` | `Qwen3OmniMoeThinkerTextRMSNorm` | 验证基础归一化算子 | 通过 |
| `artifacts/moe_block/model.onnx` | `Qwen3OmniMoeThinkerTextSparseMoeBlock` | 验证真实 MoE 路由与专家计算 | 通过 |
| `artifacts/tiny_thinker/model.onnx` | `Qwen3OmniMoeThinkerTextModel + LM Head` | 验证一层完整文本 Thinker 前向图 | 通过 |

“通过”包含：

- PyTorch Dynamo ONNX 导出成功；
- ONNX Checker `full_check=True` 通过；
- 严格 Shape Inference 通过，未知 Tensor 和未知维度均为 0；
- ONNX Runtime `CPUExecutionProvider` 加载和执行成功；
- 两组输入均与 PyTorch 输出满足 `rtol=1e-4, atol=1e-5`；
- MoE 两组输入产生了两种不同的 Top-K 路由模式；
- 导出图只包含标准 `ai.onnx` domain，没有 TensorRT 或其他自定义 domain；
- ONNX external data 文件存在且 offset/length 范围合法。

这些 ONNX 使用固定 Transformers v5.2.0 中的真实 Qwen3-Omni 类和计算路径，但采用小配置、随机权重。它们用于验证导出接口、ONNX 算子和编译器兼容性，不等同于 30B 官方权重模型。

## 2. 为什么当前没有生成完整 30B 实权重 ONNX

目标检查点：

```text
Qwen/Qwen3-Omni-30B-A3B-Thinking
revision: 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b
```

已通过 Hugging Face 元数据核实：

```text
权重分片：16 个
权重总量：63,440,997,640 bytes，约 59.08 GiB
本机物理内存：48 GiB
```

权重本身已经超过本机物理内存。PyTorch 加载、前向计算、Dynamo 图捕获、ONNX 序列化还需要额外内存，因此在当前 Mac 上继续下载并尝试完整实权重导出有较高的系统内存耗尽风险，不能把它当作安全可执行步骤。

详细证据保存在：

```text
artifacts/real_thinking_metadata/config.json
artifacts/real_thinking_metadata/generation_config.json
artifacts/real_thinking_metadata/resource_assessment.json
```

完整 30B 导出应迁移到 Linux/NVIDIA 大内存机器。建议至少准备 128 GiB 主机内存；若在单 GPU 上加载 BF16 全量权重，还需要能容纳模型和导出过程开销的 GPU/统一内存，实际建议先根据目标导出器做资源预估。

## 3. 固定的软件和源码版本

```text
Python 3.11.9
PyTorch 2.8.0
Transformers 5.2.0
ONNX 1.22.0
ONNX Runtime 1.30.0
ONNXScript 0.7.2
```

Transformers 固定源码：

```text
目录：transformers-v5.2.0
commit：7d9754a05193eb79b1d86aa744b622b8068008cd
```

导出工具会检查 `transformers.__file__` 必须来自该目录，并核对 commit，防止误用另一个 wheel 或不同源码。

## 4. 文件及作用

### `qwen3_omni_onnx_cases.py`

公共模型定义和测试数据模块，负责：

- 构造小型 `Qwen3OmniMoeTextConfig`；
- 固定 `eager` Attention 和 `batched_mm` Experts；
- 包装 `Qwen3OmniMoeThinkerTextModel + LM Head`；
- 创建 RMSNorm、MoE Block、tiny Thinker 三种 case；
- 为每种 case 创建两组固定 Shape 输入；
- 捕获 MoE Top-K expert indices，生成路由证据；
- 核对本地 Transformers 源码路径和 commit；
- 计算文件 SHA-256，并原子写入 JSON。

小配置使用 1 层、hidden size 8、4 个专家、每个 token 选择 2 个专家。真实 Thinking 配置为 48 层、hidden size 2048、128 个专家、每个 token 选择 8 个专家。

### `export_onnx.py`

ONNX 生成工具，负责：

1. 构造指定 Qwen3-Omni case；
2. 运行 PyTorch 前向，保存两组参考输入和输出；
3. 确认 MoE 两组输入具有不同路由；
4. 调用 `torch.onnx.export(..., dynamo=True, opset_version=18)`；
5. 启用 external data；
6. 执行 ONNX Checker；
7. 保存配置、版本、源码路径、commit、输入输出哈希和路由信息。

出于安全考虑，输出目录必须是工作区 `artifacts/` 的直接子目录，`--force` 不能删除工作区或任意外部目录。

### `validate_onnx.py`

正确性验证工具，负责：

- 校验 ONNX、输入 NPZ 和参考输出的 SHA-256；
- 执行 ONNX Checker；
- 执行严格 Shape Inference；
- 检查 ONNX 输入输出名称与导出元数据完全一致；
- 使用 ONNX Runtime CPU 后端运行每组输入；
- 检查输出名称、Shape、dtype、NaN/Inf 和数值误差；
- 确认 MoE 至少覆盖两种不同路由；
- 将结果写入 `validation.json`。

### `inspect_onnx.py`

ONNX 结构和算子报告工具，负责：

- 输出模型输入、输出、Shape 和 dtype；
- 统计 `domain + op_type` 及节点数量；
- 递归检查控制流子图和 ONNX local function；
- 识别非标准 domain；
- 检查 external data 路径不能越出模型目录；
- 检查文件存在、非空、offset/length 合法并计算 SHA-256；
- 将结果写入 `operators.json`。

## 5. 使用方法

进入环境：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
source .venv/bin/activate
```

### 5.1 导出 RMSNorm

```bash
python export_onnx.py \
  --case rmsnorm \
  --output-dir artifacts/rmsnorm \
  --force

python validate_onnx.py \
  --case rmsnorm \
  --model artifacts/rmsnorm/model.onnx

python inspect_onnx.py \
  --model artifacts/rmsnorm/model.onnx \
  --fail-on-custom-domain
```

### 5.2 导出真实 Qwen3-Omni MoE Block

```bash
python export_onnx.py \
  --case moe_block \
  --output-dir artifacts/moe_block \
  --force

python validate_onnx.py \
  --case moe_block \
  --model artifacts/moe_block/model.onnx

python inspect_onnx.py \
  --model artifacts/moe_block/model.onnx \
  --fail-on-custom-domain
```

### 5.3 导出 tiny Thinker Text + LM Head

```bash
python export_onnx.py \
  --case tiny_thinker \
  --output-dir artifacts/tiny_thinker \
  --force

python validate_onnx.py \
  --case tiny_thinker \
  --model artifacts/tiny_thinker/model.onnx

python inspect_onnx.py \
  --model artifacts/tiny_thinker/model.onnx \
  --fail-on-custom-domain
```

## 6. 输出目录的数据流

每个 case 目录包含：

```text
model.onnx                  ONNX 图和小型 initializer
model.onnx.data             external data 权重
inputs_0.npz                导出输入
inputs_1.npz                第二组验证输入
reference_outputs_0.npz     第 1 组 PyTorch 输出
reference_outputs_1.npz     第 2 组 PyTorch 输出
export_metadata.json        配置、版本、哈希、路由和导出参数
validation.json             Checker、Shape、ORT 和数值结果
operators.json              接口、算子、domain 和 external data 报告
```

完整数据流：

```text
Transformers Qwen3-Omni PyTorch 类
    + 小配置/权重
    + 固定测试输入
        ↓
export_onnx.py
        ↓
model.onnx + model.onnx.data
        ├── validate_onnx.py
        │     └── Checker + Shape + ORT + PyTorch 数值对比
        └── inspect_onnx.py
              └── 输入输出 + domain + 算子 + external data
                    ↓
               自研编译器
```

## 7. 实际算子结果

### 7.1 RMSNorm

```text
Add, Mul, Pow, Reciprocal, ReduceMean, Sqrt
```

共 7 个节点，无自定义 domain。

### 7.2 Sparse MoE Block

```text
Cast, Clip, Div, GatherND, Gemm, Less, MatMul, Mul,
ReduceSum, Reshape, Sigmoid, Softmax, Split, Squeeze,
TopK, Unsqueeze
```

共 33 个节点，无自定义 domain。两组输入产生不同 Top-K 路由，均与 PyTorch 数值一致。

### 7.3 Tiny Thinker Text + LM Head

```text
Add, And, Cast, Clip, Concat, Cos, Div, Gather, GatherND,
Gemm, Less, LessOrEqual, MatMul, Mul, Neg, Pow, Reciprocal,
ReduceMean, ReduceSum, Reshape, ScatterElements, ScatterND,
Sigmoid, Sin, Slice, Softmax, Split, Sqrt, Squeeze, TopK,
Transpose, Unsqueeze, Where
```

共 142 个节点，无自定义 domain。该图覆盖：

- Token Embedding；
- 多模态 RoPE 的文本路径；
- Q/K/V Projection；
- 因果 Attention；
- RMSNorm；
- Top-K MoE Router；
- batched expert weight Gather 和 MatMul；
- LM Head logits。

## 8. 关键实现选择

### 为什么不用原始 eager Experts 循环

原始实现会根据输入的路由结果执行 Python 循环：

```text
one_hot → nonzero → for expert_idx → where → index_add_
```

这包含数据相关 Python 控制流，传统 tracing 容易只固化示例输入命中的专家。

Transformers v5.2.0 已提供等价的 `batched_mm` Experts 实现：

```text
TopK indices
→ Gather 对应专家权重
→ batched MatMul
→ routing weight
→ reshape + ReduceSum
```

本工具使用 Transformers 自己提供的 `batched_mm` 实现，不是自行修改模型数学定义。这样可以得到只含标准 ONNX 算子的动态路由图。

### 为什么暂时不用 KV Cache

Transformers 的 `DynamicCache` 是 Python 对象，不适合作为 ONNX 公共接口。首轮设置 `use_cache=False`。后续需要拆分：

```text
prefill.onnx
专门的 decode.onnx
```

并将每层 K/V 展平为显式张量输入输出。

## 9. 下一步迁移到真实权重

在大内存 Linux/NVIDIA 机器上：

1. 使用固定 revision 下载 Thinking 权重；
2. 安装与本工作区一致的 Transformers 源码；
3. 仅保留 `thinker.model + thinker.lm_head` 导出边界；
4. 设置 `attn_implementation="eager"`；
5. 设置 `experts_implementation="batched_mm"`；
6. 固定 `batch=1, sequence=16, use_cache=False`；
7. 使用 ONNX external data；
8. 运行同样的 Checker、Shape、数值和算子检查；
9. 再将 ONNX 提交给自研编译器。

真实权重导出完成前，不能把当前 tiny ONNX 描述为“Qwen3-Omni-30B 实权重 ONNX”。当前成果准确表述应为：

> 已跑通 Qwen3-Omni v5.2.0 文本 MoE 架构的标准 ONNX 导出与验证链路，并获得基础层、真实 MoE Block 和 tiny Thinker Text + LM Head 三个可执行 ONNX；完整 30B 实权重导出受本机内存限制，需迁移到大内存目标机。
