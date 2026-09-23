# Qwen3-Omni ONNX 导出工具：自检验证报告

> 本文区分**历史实验记录**与**本轮验收**，不将未执行的命令写成实测通过。
> 第 1～2 节保存 2026-09-22 的 macOS arm64、48 GiB 历史环境与输出摘录；461 节点 / 41 种算子、旧哈希和旧 e2e 结论只属于该批旧产物，不是本轮结果。
> 2026-09-23 修订：旧记录没有完整证明官方顶层语义，且本轮代码和证据格式已变化。不能只改数字、哈希或 JSON 将它们升级为 schema v2；必须重新导出、验证和打包。
> 当前脚本的推荐验收命令在第 4 节；本轮实测状态在第 0.1 节。本文保留的历史命令不是“所有命令均已实跑”的声明。

---

## 0. 结论先说清楚

### 已有证据的适用范围

历史日志记录了 tiny 随机权重四组件在指定样例下的导出、标准算子检查、Wrapper 与 ORT 数值对比及三步 Cache 回灌。tiny 文本只有 **1 层、2 个 KV 张量**，不是官方 48 层的 96 个；tiny vision/audio 均 1 层，audio 为 16 mel bins、20 帧。real 默认 audio profile 为 101 帧。

有限样例一致只支持相应环境、Shape、dtype、输入和容差内的结论，既不是数学等价证明，也不能排除 Wrapper 和验证器共同使用错误位置的情况。本轮要求将**官方顶层语义 → Wrapper**与**Wrapper → ORT**分成独立参考链。

### 尚未验收的部分

| 未验收项 | 边界 |
|---|---|
| 官方 30B 四组件及 real 端到端 | 本机 48 GiB 未安全加载约 59.08 GiB 权重；目标机仍需实测 |
| 官方规模算子、峰值内存与内核支持 | 不能由 tiny 计数或内存门槛推断；128/192 GiB 不是成功保证 |
| 全部动态长度 / 专家组合 | 64 是 Decode 导出约束上界，有限 profile 测试不覆盖全范围 |
| 真实媒体与生成质量 | 合成张量测试不是质量基准 |
| 完整 Qwen BF16 的 CPU/CUDA 执行 | 数据交换支持不等于所有算子内核支持，换 CUDA 不保证成功 |

### 0.1 本轮验收（2026-09-23）

- 持久回归入口：`python -B -m unittest discover -s tests -v`，测试位于 `tests/test_*.py`，公共产物逻辑在 `onnx_artifact_utils.py`。
- 已有局部测试事实：内存内 CPU IOBinding Cast/Identity 小图与 BF16 NPZ bit-preserving 往返成功；此结论仅覆盖 BF16 数据交换，不代表完整 Qwen BF16 内核可用。
### 本轮实测（2026-09-23，macOS arm64、48 GiB，仅 tiny）

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B run_local_thinking_pipeline.py --offline
```

- 持久 unittest：**59 项全部通过**（公共证据 24、语义 12、端到端证据 7、打包证据 16 分布在三个文件中）。
- 一键 tiny 全流程：退出码 **0**，并生成 `status=tiny-interface-validation-only`、`passed=true` 的 manifest。
- 产物统计（schema v2）：`vision_encoder 84`、`audio_encoder 65`、`thinker_prefill 160`、`thinker_decode 158`，合计 **467 节点、41 种标准算子、0 个自定义 domain**。
- 模型哈希前缀：`vision 97198aa0e3644a7b`、`audio 3678ad997d691a9d`、`prefill 8824b3c12dd642de`、`decode 5d3fd0358285e4fe`。
- 端到端 3 步 Decode，参考范围 `official_top_level_with_raw_synthetic_features`、`reference_positions=official_forward_independent_mrope_and_cache`；各阶段最大绝对误差：vision `5.59e-09`、audio `1.16e-09`、prefill `8.34e-07`、decode 三步均 `8.34e-07`。
- 三组早期回归（rmsnorm / moe_block / tiny_thinker）重新导出并严格验证通过。

**未覆盖**：官方 30B 权重、CUDA/其它 provider、完整 Qwen 的 FP16/BF16 图内核兼容性、真实磁盘峰值。下方历史数字与哈希属于旧产物。
- 旧产物必须重新导出验收；不允许仅编辑 metadata、JSON 报告或本报告数字来制造通过。

---

## 1. 环境前提

```text
机器      macOS arm64，内存 48 GiB
Python    3.11.9
PyTorch   2.8.0
ONNX      1.22.0
ORT       1.30.0
onnxscript 0.7.2
Transformers  v5.2.0（固定源码，commit 7d9754a05193eb79b1d86aa744b622b8068008cd）
```

---

## 2. 历史核实方案与输出摘录（2026-09-22，非本轮验收）

### 步骤 1：语法与源码来源

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
source .venv/bin/activate
python -m py_compile *.py
python -c "from qwen3_omni_onnx_cases import assert_transformers_provenance as f; print(f())"
```

输出：

```text
OK（编译无报错）

{'imported_file': '/Users/.../transformers-v5.2.0/src/transformers/__init__.py',
 'revision': '7d9754a05193eb79b1d86aa744b622b8068008cd'}
```

含义：所有脚本语法正确；当前 Python 导入的 `transformers` **确实来自固定源码目录且 commit 正确**，不是 PyPI wheel。

### 步骤 2：四组件全链路（一条命令）

```bash
python run_local_thinking_pipeline.py
```

输出（关键行）：

```text
[OK] vector=0 logits: max_abs=2.23517e-08, max_rel=1.65148e-06
[OK] vector=1 logits: max_abs=1.49012e-08, max_rel=1.57795e-05
[OK] distinct_routing_patterns=2
[OK] nodes=158
[OK] custom_domains=none
[OK] end_to_end=.../validation/end_to_end.json
[OK] total_nodes=461
[OK] unique_operators=41
[OK] status=tiny-interface-validation-only
[OK] local Thinking pipeline completed
```

### 步骤 3：读证据文件（最关键）

```bash
python - <<'PY'
import json
p="Qwen3-Omni-30B-A3B-Thinking-ONNX"
m=json.load(open(f"{p}/manifest.json"))
print("status:", m["status"], "| official_weights:", m["official_weights_included"])
for n,c in m["components"].items():
    print(f"  {n:16s} profile={c['profile']} validated={c['validated']} custom_domains={c['custom_domains']}")
e=json.load(open(f"{p}/validation/end_to_end.json"))
print("e2e passed:", e["passed"], "| decode_steps:", e["decode_steps"])
for stage,items in e["comparisons"].items():
    print(f"  {stage:14s} outputs={len(items)} all_passed={all(i['passed'] for i in items.values())}")
s=json.load(open(f"{p}/operators/summary.json"))
print("nodes:", s["total_node_count"], "| ops:", len(s["unique_operators"]))
PY
```

输出：

```text
status: tiny-interface-validation-only | official_weights: False
  vision_encoder   profile=tiny-fixed-shape validated=True custom_domains=[]
  audio_encoder    profile=tiny-fixed-shape validated=True custom_domains=[]
  thinker_prefill  profile=tiny-fixed-shape validated=True custom_domains=[]
  thinker_decode   profile=tiny-fixed-shape validated=True custom_domains=[]

e2e passed: True | decode_steps: 3
  vision           outputs= 2 all_passed=True
  audio            outputs= 1 all_passed=True
  prefill          outputs= 3 all_passed=True
  decode_step_1    outputs= 3 all_passed=True
  decode_step_2    outputs= 3 all_passed=True
  decode_step_3    outputs= 3 all_passed=True

nodes: 461 | ops: 41
```

端到端每一步的最大绝对误差：

```text
vision        worst_max_abs=5.59e-09
audio         worst_max_abs=1.16e-09
prefill       worst_max_abs=8.34e-07
decode_step_1 worst_max_abs=8.34e-07
decode_step_2 worst_max_abs=8.34e-07
decode_step_3 worst_max_abs=8.34e-07
```

单组件数值证据（`onnx/thinker_prefill/validation.json`）：

```text
checker: passed
shape_inference: passed, unknown_tensors=0, unknown_dimensions=0
routing_coverage: distinct_patterns=2, passed=True

  vector=0 routing=7ddf12dbd49f6a7b… unique_experts=[0,1,2,3]
      logits           allclose=True shape_match=True dtype_match=True max_abs=2.24e-08
      present_key_0    allclose=True shape_match=True dtype_match=True max_abs=2.38e-07
      present_value_0  allclose=True shape_match=True dtype_match=True max_abs=1.49e-08
  vector=1 routing=c8bc9b90c705d252… unique_experts=[0,1,2,3]
      logits           allclose=True shape_match=True dtype_match=True max_abs=4.47e-08
      present_key_0    allclose=True shape_match=True dtype_match=True max_abs=3.58e-07
      present_value_0  allclose=True shape_match=True dtype_match=True max_abs=1.49e-08
```

四个组件的模型哈希（`manifest.json`，用于确认文件未被替换）：

```text
vision_encoder   107ab651abe0706d…
audio_encoder    3662251850bf07be…
thinker_prefill  e9bce1965cfb7eb2…
thinker_decode   846aacca245bf8a3…
```

算子分布（`operators/summary.json`）：

```text
vision_encoder   78 节点
audio_encoder    65 节点
thinker_prefill 160 节点
thinker_decode 158 节点
合计 461 节点，41 种算子，0 个自定义 domain
Top10: Mul(61), Transpose(44), Unsqueeze(40), Add(39), Reshape(37),
       MatMul(25), Slice(19), Gemm(18), Gather(14), Concat(13)
```

### 步骤 4：反向测试（证明验证不是摆设）

```bash
rm -rf artifacts/_tamper
cp -r Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_prefill artifacts/_tamper
printf 'x' | dd of=artifacts/_tamper/model.onnx bs=1 seek=200 conv=notrunc status=none

python validate_onnx.py --model artifacts/_tamper/model.onnx            >/dev/null 2>&1; echo "退出码=$?"
python validate_onnx.py --model artifacts/rmsnorm/model.onnx --atol inf  >/dev/null 2>&1; echo "退出码=$?"
python export_onnx.py --case rmsnorm --output-dir . --force              >/dev/null 2>&1; echo "退出码=$?"
python validate_onnx.py --case rmsnorm --model Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_prefill/model.onnx >/dev/null 2>&1; echo "退出码=$?"
python validate_onnx.py --model artifacts/rmsnorm/model.onnx             >/dev/null 2>&1; echo "退出码=$?"
rm -rf artifacts/_tamper
```

输出（**真实 Python 进程退出码**）：

```text
--- 测试1 篡改 ONNX 一个字节 ---    python 退出码=1
--- 测试2 --atol inf 制造假通过 ---  python 退出码=1
--- 测试3 --force 指向工作区根 ---   python 退出码=1
--- 测试4 --case 传错 ---           python 退出码=1
--- 对照：正常验证 ---               python 退出码=0
```

对应报错信息：

```text
RuntimeError: ONNX 文件哈希与导出元数据不一致
ValueError: atol 必须是有限非负数，实际为 inf
ValueError: 输出目录必须位于 .../artifacts 下，实际为 /Users/.../qwen3-omni-onnx-work
ValueError: case 不一致：参数=rmsnorm 元数据=thinker_prefill
```

> ⚠️ **重要提醒**：不要用 `python xxx | tail -1; echo $?` 来判断退出码——`$?` 取到的是 `tail` 的退出码（永远是 0），必须用 `python xxx >/dev/null 2>&1; echo $?` 才能得到 Python 的真实退出码。

> 安全性说明：`validate_onnx.py` 只在**模型身份校验（case + ONNX 哈希）通过**之后才允许写/覆盖 `validation.json`。因此测试 4（`--case` 传错，属于调用方式错误）**不会**碰掉该模型原本有效的验证报告；只有测试 1（文件被篡改，哈希不一致）才会失效化旧报告——因为旧报告对新文件已不成立。

---

## 3. 本轮证据链要求（不等同于实测结果）

```text
固定 Transformers 来源、checkpoint 身份与导出时 source_snapshot 真实 bytes
        ↓
官方顶层语义 → Wrapper（独立位置/Cache 参考）
        ↓
导出 ONNX + external data + metadata + 输入/参考向量
        ↓
Checker + strict shape inference + 实际图域/算子清单
        ↓
Wrapper → ORT（记录 provider、逐张量 shape/dtype、有限容差、NaN/Inf）
        ↓
多组 MoE 路由 + 三步 Decode（tiny 2 KV；官方 96 KV 待 real 实测）
        ↓
schema v2 报告绑定当前 ONNX、external data、metadata 与向量
        ↓
算子汇总 / 打包重新核验 → manifest 状态
```

上图为验收要求：导出时真实源码 bytes 归档仍待实现与验收，不能把当前源码 hash/environment/git 声称为可恢复的 `source_snapshot/` 归档。`tools/` 是打包时快照，也不能代替导出源码。即使后续补齐源码归档，仍须恢复 bootstrap、固定 Transformers 与平台依赖；lock 当前 torch 项为 `torch==2.8.0`，不是 macOS wheel URL。

哈希用于发现意外混用或文件改变，不是签名、可信时间戳或真实性自证。若能同时改写产物与全部证据，哈希本身不能证明原始身份；正式可信分发需额外可信渠道。

---

## 4. 当前推荐验收命令（不是执行记录）

先恢复固定源码与依赖，再在工作区执行。以下不抑制 stderr，也不经 `grep/tail` 隐藏失败；执行结果需记录真实退出码。重导出会更新已有 tiny 产物，保留旧证据前请另行备份；不要覆盖 real 产品目录。

```bash
set -e
source .venv/bin/activate
python -B -m unittest discover -s tests -v
python run_local_thinking_pipeline.py
for c in rmsnorm moe_block tiny_thinker; do
  python export_onnx.py --case "$c" --output-dir "artifacts/$c" --force
  python validate_onnx.py --case "$c" --model "artifacts/$c/model.onnx"
  python inspect_onnx.py --model "artifacts/$c/model.onnx" --fail-on-custom-domain
done
```

反向场景使用 `tests/test_*.py` 的持久 unittest，在临时隔离目录验证，不需要照抄历史 `rm -rf/dd` 操作。`--case` 调用错误应保留有效旧报告；产物身份或绑定失效应阻止旧报告继续被采信。具体覆盖以本轮真实 unittest 输出为准。

BF16、整数和布尔张量须按真实 dtype 校验：例如 Vision `position_indices=int32`，Text 两种位置 mask 为 `bool`，Audio `cu_seqlens=int32`。不能把全部张量 cast 为 float32 来凑数值通过。逐组件接口见 `README.md` 第 6 节。

---

## 5. 附：怎么查看导出的 ONNX

- `.onnx` = 计算图（protobuf 二进制，文本编辑器打开是乱码，不要编辑）
- `.onnx.data` = 权重（protobuf 单文件 2 GB 上限，故外置）；两者必须同目录成对存放

三种查看方式：

1. **Netron 网页版（最直观）**：打开 <https://netron.app>，把 `model.onnx` 拖进去即可看到拓扑、算子属性、Shape/dtype。
2. **本仓库工具**：`python inspect_onnx.py --model <path>` 输出 JSON 报告。
3. **官方 API 打印可读图**：`python -c "import onnx; print(onnx.printer.to_text(onnx.load('<path>').graph))"`

详见 `README.md` 第 7.1 节。

## 6. 相关文件

| 文件 | 作用 |
|---|---|
| `Qwen3-Omni-30B-A3B-Thinking-ONNX/manifest.json` | 产品自述：状态、每组件哈希、接口、验证结论 |
| `.../validation/end_to_end.json` | 端到端证据：是否通过、绑定哪四个 ONNX、Decode 步数 |
| `.../onnx/<组件>/validation.json` | 单组件：每组输入误差、dtype/Shape、MoE 路由 |
| `.../operators/summary.json` | 各组件节点数 + 全局唯一算子 |
| `.../operators/all_operators.csv` | 完整算子清单（带 Component 列） |
| `README.md` | 使用说明与命令参考（第 7 节为自检体系） |
