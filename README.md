# CFHTransformer_0415

本仓库保存 CFHTransformer_0415 的代码、一次可复现的代表性训练产物，以及对应的评测结果说明。

## 保留的训练产物

| 类型 | 路径 | 说明 |
|---|---|---|
| 最优模型 | `run/20260416_194859_COnP/checkpoints/best_model.pt` | 当前保留的唯一 checkpoint（Best COnP_f1=0.7528） |
| 训练运行目录 | `run/20260416_194859_COnP/` | 当前保留的唯一运行目录 |
| 标准输出日志 | `run/20260416_194859_COnP/logs/train_stdout.log` | 训练期间的控制台输出 |
| 测试监控日志 | `run/20260416_194859_COnP/test_monitor.txt` | 每轮监控记录 |

仓库中其余运行目录与额外 checkpoint 已清理，不再保留，以减少仓库体积并突出最终结果。

## 推理（生成预测 JSON）

使用以下命令对测试集进行推理，生成 `pred_test_v2.json`：

```bash
CUDA_VISIBLE_DEVICES=2 python3 predict_to_json.py \
    --config config.yaml \
    --checkpoint run/20260416_194859_COnP/checkpoints/best_model.pt \
    --split test \
    --onset_thresh 0.2 \
    --frame_thresh 0.35 \
    --output pred_test_v2.json
```

## 评测命令

在服务器上使用如下命令进行评测：

```bash
python3 evaluate_github.py \
    /mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json \
    pred_test_v2.json \
    0.05
```

## 评测结果

```
         Precision  Recall   F1-score
COnPOff  0.413332   0.403068 0.407526
COnP     0.769747   0.753376 0.760358
COn      0.800514   0.783286 0.790635
gt note num: 31311.0  tr note num: 30530.0  song number: 100
```

## 指标汇总

| 指标 | Precision | Recall | F1-score |
|---|---:|---:|---:|
| COnPOff | 0.413332 | 0.403068 | 0.407526 |
| COnP | 0.769747 | 0.753376 | 0.760358 |
| COn | 0.800514 | 0.783286 | 0.790635 |

## 说明

`pred_test_v2.json` 为推理输出文件，由于仓库忽略规则默认不提交 `.json` 文件，因此以 README 的方式保留关键评测结果。
