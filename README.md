# CFHTransformer_0415

当前仓库整理为两部分：

- `v7回溯_0424/`：2026-04-24 回溯保存的旧版完整仓库内容，包含原来的 README 与代码。
- `v10/`：标准训练 baseline 版本，保留代码、配置、数据划分、日志与分析文件；未上传 checkpoint 模型权重。

## v10 说明

`v10/` 是一个标准训练的 baseline，规范补充了各种功能，比如 TensorBoard 和 test100首监控。

best 记录：

```text
128	best_after20	0.022305	0.799798	0.770271	0.440881	0.50	0.40
```

验证集阈值对应日志：

```text
2026-04-23 00:07:16 [INFO] Epoch 128/1300 | train_loss=0.0199 | val_loss=0.0176 | COn_f1=0.8128 | COnP_f1=0.7958 | COnPOff_f1=0.4868 | sig_onset=0.0032 sig_frame=0.0128 | thresh(on=0.50,fr=0.40) | lr=2.93e-04
2026-04-23 00:07:17 [INFO]   -> Best model saved! COnP_f1=0.7958
```
