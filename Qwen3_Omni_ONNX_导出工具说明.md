# Qwen3-Omni ONNX 导出工具说明（历史文档，已并入 README）

> **本文件内容已并入 `README.md`。**
>
> 仍然有价值的三部分已经迁移：
>
> - 设计决策（为什么用 `batched_mm` Experts、为什么显式展平 KV Cache、为什么拆四组件）→ README 第 12 节
> - 早期三级回归模型的定位与算子统计 → README 第 12.4 节与第 7 节
> - 官方权重迁移步骤 → README 第 9 节（Linux 大内存机器）
>
> 本文件仅作为旧链接/历史记录保留，**新读者请直接看 `README.md`**。

## 仍然有效的三条结论

1. 当前成果是 **tiny 接口验证件**：结构与接口同官方一致，但权重是随机小配置；不得称为“Qwen3-Omni-30B 实权重 ONNX”。
2. 官方权重完整导出的唯一阻塞是内存：59.08 GiB 权重 > 本机 48 GiB RAM，工具已用 `--minimum-memory-gib` 门禁强制拦截。
3. 一切结论以可重跑的验证为准：`python run_local_thinking_pipeline.py`、`validate_onnx.py`、`inspect_onnx.py`，以及产品包里的 `manifest.json` / `validation/end_to_end.json` / `operators/summary.json`。
