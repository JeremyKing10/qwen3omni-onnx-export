# Qwen3-Omni ONNX 导出工具：自检验证报告

> 本文回答一个问题：**凭什么说这套 ONNX 导出工具是真实有效的？**
> 文中所有输出均为本机（macOS arm64，48 GiB 内存）**实际执行结果**，可原样复现。
> 执行日期：2026-09-22

---

## 0. 结论先说清楚

### ✅ 已被证明的部分

这套工具能把 **Qwen3-Omni 在 Transformers v5.2.0 官方类上的 `forward`**，通过 `torch.onnx.export(dynamo=True)` 转成 **只含标准 ONNX 算子**的图，并且：

1. MoE 动态路由正确（TopK → Gather → BatchedMatMul → ReduceSum），**不是**把第一次导出的专家固化；
2. KV Cache 展平 + 自回归续接正确（Prefill 输出 96 个 K/V，Decode 三步回灌都对）；
3. PyTorch 与 ONNX Runtime 输出**数值等价**；
4. 产物无自定义 domain；
5. 所有验证机制（哈希链、容差校验、路径防护、case 交叉校验）实测能拦截问题。

### ❌ 未被证明的部分

| 未证明 | 原因 |
|---|---|
| 官方 30B 权重能导出 | 本机 48 GiB < 权重 59.08 GiB，**从未真正加载过官方权重** |
| 官方模型的算子**数量** | tiny 为 1 层 / 4 专家；真实 48 层 / 128 专家会让数量成倍变化 |
| 导出模型的语义正确性 | tiny 权重是随机的，不含知识 |
| 真实媒体上的效果 | 只做了合成张量的张量接力验证 |

一句话：**工具是真的、链路是通的、验证是硬的；唯一没做的是把 59 GiB 真权重灌进去——那只需要一台内存够的机器，不需要改代码。**

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

## 2. 核实方案与原始输出

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
[OK] nodes=159
[OK] custom_domains=none
[OK] end_to_end=.../validation/end_to_end.json
[OK] total_nodes=462
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

nodes: 462 | ops: 41
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
audio_encoder    26a8e244d3abee3e…
thinker_prefill  e9bce1965cfb7eb2…
thinker_decode   1575e274c7c7acd9…
```

算子分布（`operators/summary.json`）：

```text
vision_encoder   78 节点
audio_encoder    65 节点
thinker_prefill 160 节点
thinker_decode 159 节点
合计 462 节点，41 种算子，0 个自定义 domain
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

---

## 3. 证据链全景图

```text
① 固定源码来源（commit 校验）
        ↓
② ONNX Checker（full_check=True）
        ↓
③ 严格 Shape Inference（未知维度 = 0）
        ↓
④ ONNX Runtime 标准后端执行（CPUExecutionProvider）
        ↓
⑤ PyTorch vs ONNX 数值对比（rtol=1e-4, atol=1e-5，实测 1e-7 量级）
        ↓
⑥ MoE 双路由覆盖（两种 Top-K 结果都一致）
        ↓
⑦ 三步 Decode KV 回灌一致性
        ↓
⑧ 结构纯净（0 自定义 domain + external data 完整）
        ↓
⑨ 哈希链交叉绑定（模型/输入/参考输出/external data，换文件即报错）
        ↓
   manifest.json 状态判定（tiny-interface-validation-only）
```

---

## 4. 一键复现脚本

把下面整段粘进终端即可完整复核（约 1 分钟）：

```bash
cd /Users/bojunjin/Documents/LLM/qwen3-omni-onnx-work
source .venv/bin/activate

echo "=== [1] 编译与源码 ==="
python -m py_compile *.py && echo "compile OK"
python -c "from qwen3_omni_onnx_cases import assert_transformers_provenance as f; print(f())"

echo "=== [2] 全链路 ==="
python run_local_thinking_pipeline.py 2>/dev/null | grep -E "^\[OK\]|^\[FAIL\]" | tail -8

echo "=== [3] 早期三级回归 ==="
for c in rmsnorm moe_block tiny_thinker; do
  python validate_onnx.py --case "$c" --model "artifacts/$c/model.onnx" >/dev/null 2>&1 \
    && echo "$c OK" || echo "$c FAIL"
done

echo "=== [4] 反向测试（前 4 项应为 1，最后 1 项应为 0） ==="
rm -rf artifacts/_tamper
cp -r Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_prefill artifacts/_tamper
printf 'x' | dd of=artifacts/_tamper/model.onnx bs=1 seek=200 conv=notrunc status=none
python validate_onnx.py --model artifacts/_tamper/model.onnx            >/dev/null 2>&1; echo "篡改ONNX   退出码=$?"
python validate_onnx.py --model artifacts/rmsnorm/model.onnx --atol inf  >/dev/null 2>&1; echo "atol=inf   退出码=$?"
python export_onnx.py --case rmsnorm --output-dir . --force              >/dev/null 2>&1; echo "force越界  退出码=$?"
python validate_onnx.py --case rmsnorm --model Qwen3-Omni-30B-A3B-Thinking-ONNX/onnx/thinker_prefill/model.onnx >/dev/null 2>&1; echo "case传错   退出码=$?"
python validate_onnx.py --model artifacts/rmsnorm/model.onnx             >/dev/null 2>&1; echo "正常验证   退出码=$?"
rm -rf artifacts/_tamper
```

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
