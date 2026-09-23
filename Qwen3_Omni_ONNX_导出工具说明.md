# Qwen3-Omni ONNX 导出工具说明（历史文档，已并入 README）

> **本文件内容已并入 `README.md`。**
>
> 仍然有价值的三部分已经迁移：
>
> - 设计决策（为什么用 `batched_mm` Experts、为什么显式展平 KV Cache、为什么拆四组件）→ README 第 12 节
> - 早期三级回归模型的定位与算子统计 → README 第 12.4 节与第 7 节
> - 官方权重迁移步骤 → README 第 9 节（Linux 大内存机器）
>
> 本文件仅作为旧链接/历史记录保留，**新读者请直接看 `README.md`**；想手动复核"凭什么说这套工具有效"，看 `Qwen3_Omni_ONNX_自检验证报告.md`。

## 当前边界（2026-09-23 修订）

1. 当前目标仍是 **tiny 接口验证件**：使用官方类的小配置随机权重，不代表官方规模或所有执行路径。tiny 文本 1 层、2 个 KV 张量；官方文本 48 层、96 个 KV 张量尚未做真实权重验收。有限样例的数值一致不是数学等价证明。
2. 本机 48 GiB RAM 不具备安全加载约 59.08 GiB 官方权重的条件，但内存不是唯一未知项。大内存流程仍须检查峰值内存、磁盘、PyTorch 设备、ORT provider、dtype/算子内核和官方顶层语义；128/192 GiB 只是规划门槛，不保证成功。
3. 本轮证据要求采用 schema v2，绑定 ONNX、external data、metadata 和测试向量；导出时真实源码 bytes 的归档仍待实现与验收，当前仅源码 hash/environment/git 不能恢复源码。旧产物必须重新导出和验收，不能仅修改 JSON。哈希防止意外混用，不是签名或自证真实性；打包时 `tools/` 快照不能替代导出源码或可恢复环境。
4. BF16 的 NPZ 位保持与 CPU IOBinding Cast/Identity 小图数据交换已有成功测试；完整 Qwen BF16 图的内核支持仍取决于环境，不能断言 CPU 全部不支持或换 CUDA 即可。
5. 以 `README.md` 的条件式一键流程、`python -B -m unittest discover -s tests -v` 和重导出报告为准。本文件不把历史结果或尚未执行的命令写成本轮通过；多模型适配边界见 `README.md`，更换 `--model-path` 不会自动支持 DeepSeek/Kimi。
