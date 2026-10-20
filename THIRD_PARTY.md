# 模型来源与许可

根目录 [LICENSE](LICENSE) 为仓库初始化时的 MIT 许可；以下第三方组件保留自身的署名与许可。

- **Geoformer**：来自 [Microsoft AI2BMD](https://github.com/microsoft/AI2BMD) 的 Geoformer 分支，固定提交 `20ba4c1fe3045de862f03ddbb84c10811884f5f8`；[MIT 许可](baselines/geformer/LICENSE)。以残基 Cα 适配 IPE 编码器并接入本项目预测头。节点预测省略不参与预测的末层 IPE 更新，避免该无效分支溢出污染输出；保留参数格式及零梯度/AdamW 行为。
- **DeepInteract**：来自 [BioinfoMachineLearning/DeepInteract](https://github.com/BioinfoMachineLearning/DeepInteract)，固定提交 `c78d205465f02ee4ef751dbbafb7ec8f30c75c9a`；[GPL-3.0 许可](baselines/deepinteract/LICENSE)。适配完整物理链几何上下文与共享表位预测头，使用选中残基 kNN（含自环）、18 个真实欧氏距离 RBF 和 `O_query @ O_neighbor.T` 相对旋转；缺失原子及链断点由 mask 处理。不使用 DIPS 预训练、MSA 或原 ResNet 接触预测头。

其他几何基线和 LF3DRoPE 变体的实现定义见各模型 `model.py` 与对应配置；这些适配不表示完整复现上游论文的原任务与训练流程。
