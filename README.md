# CFHTransformer_0415

本仓库复现论文 *"Cycle Frequency-Harmonic-Time Transformer for Note-Level Singing Voice Transcription"* (ICME 2024)，并持续优化以逼近论文报告的指标。

## 版本历史

| 版本 | COn | COnP | COnPOff | 论文目标 COnP | 主要改动 |
|------|----:|-----:|--------:|-------------:|---------|
| v6 | 0.7906 | 0.7604 | 0.4075 | 0.8013 | 基线版本 |
| **v7** | **0.8022** | **0.7752** | **0.4464** | 0.8013 | CQT归一化 + onset类别平衡 + 400首训练 |

## v7 改动详情

1. **CQT 归一化** — 输入dB范围[-80,0]，原来直接送入Conv3d，现加入 `CQTNormalize` 层做零均值单位方差归一化
2. **Onset 类别平衡** — onset正样本占比<0.2%，原来`onset_weight=1.0`，现加入 `onset_pos_weight=5.0` 对正样本加权
3. **训练集扩充** — 从360首→400首（对齐论文：400 train + 100 test，无独立val集）
4. **阈值搜索优化** — 从10首歌扩大到30首，更细粒度阈值网格

## 环境与数据依赖

- Python 3.12, PyTorch, librosa, mir_eval
- CQT 预计算缓存：`/mnt/ssd/lian/论文复现/CFH-Transformer/cqt_cache_50ms/npy/`（288-bin, hop=800, 50ms/帧）
- 标注文件：`/mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json`

## 训练

```bash
# v7 训练（使用 config_v7.yaml）
CUDA_VISIBLE_DEVICES=3 python3 train_conp_v6_0415.py --config config_v7.yaml

# 训练在 screen 中后台运行
screen -S cft_v7
CUDA_VISIBLE_DEVICES=3 python3 train_conp_v6_0415.py --config config_v7.yaml
# Ctrl+A D 退出screen
```

训练产物保存在 `run/<timestamp>_COnP/` 下：
- `checkpoints/best_model.pt` — 最优模型（按 COnP F1 选择）
- `logs/train_stdout.log` — 训练日志
- `test_monitor.txt` — 每5 epoch 记录的指标

## 推理（生成预测 JSON）

```bash
# v7 推理（使用 v7 的配置、模型和阈值）
CUDA_VISIBLE_DEVICES=1 python3 predict_to_json.py \
    --config config_v7.yaml \
    --checkpoint run/<timestamp>_COnP/checkpoints/best_model.pt \
    --split test \
    --onset_thresh 0.50 \
    --frame_thresh 0.40 \
    --output pred_test_v7.json
```

> 注意：`--onset_thresh` 和 `--frame_thresh` 从训练日志中 `thresh(on=X,fr=Y)` 获取最优值。v7 最优阈值为 `on=0.50, fr=0.40`。

## 评测

```bash
python3 evaluate_github.py \
    /mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json \
    pred_test_v7.json \
    0.05
```

## 评测结果

### v7 (当前最佳)

```
         Precision  Recall   F1-score
COnPOff  0.451497   0.442353 0.446392
COnP     0.782801   0.769437 0.775213
COn      0.810266   0.796093 0.802231
gt note num: 31311.0  tr note num: 30632.0  song number: 100
```

| 指标 | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| COnPOff | 0.451497 | 0.442353 | 0.446392 |
| COnP | 0.782801 | 0.769437 | 0.775213 |
| COn | 0.810266 | 0.796093 | 0.802231 |

### v6 (基线)

```
         Precision  Recall   F1-score
COnPOff  0.413332   0.403068 0.407526
COnP     0.769747   0.753376 0.760358
COn      0.800514   0.783286 0.790635
gt note num: 31311.0  tr note num: 30530.0  song number: 100
```

## 与论文差距分析

当前 COnP=0.7752 vs 论文 0.8013，差距 2.6%。已知原因：

- 模型在 epoch ~80 后 val_loss 开始上升，存在过拟合
- 未来改进方向：更强正则化、GAP 改用 sum（论文原话 "sum up"）、输出头加 LayerNorm、学习率 warmup
