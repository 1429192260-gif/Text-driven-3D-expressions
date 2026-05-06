# When Words Smile Prior V2 改进实验报告

## 1. 实验目的

V1 使用固定情绪时间模板对官方生成序列做后处理，已经取得小幅提升。V2 在此基础上进一步改进为可学习的 prior adapter。

目标是验证：

```text
learned emotion prior adapter
```

是否能比手工模板更有效地调制 When Words Smile 生成的表情参数序列。

## 2. 方法

基础输入仍是 When Words Smile 官方 checkpoint 生成的序列：

```text
P = {p_1, p_2, ..., p_T}, p_t in R^53
```

V2 根据文本提取轻量情绪先验：

```text
emotion category
intensity
```

然后学习一个情绪相关的时间-维度调制函数：

```text
P'_t = P_t * F(e, s, t)
```

其中：

- `e` 是情绪类别。
- `s` 是情绪强度。
- `t` 是时间位置。
- `F` 是可学习 adapter。

为避免直接训练大模型，V2 只学习少量 adapter 参数，不修改 BERT、CVAE 或 Transformer 主体。

## 3. 训练设置

使用 dev 集进行 adapter 训练和验证：

```text
dev 前 1200 条: adapter train
dev 后 300 条: adapter validation
test: final evaluation only
```

训练配置：

```text
epochs: 200
lr: 0.05
best val MAE: 0.378464
```

训练曲线显示，validation MAE 在 50 到 60 epoch 左右达到最优，之后开始轻微回升，因此最终采用 best validation checkpoint。

## 4. Test 集结果

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| When Words Smile baseline | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |
| Prior V1 fixed template | 0.377198 | 0.558812 | 0.380854 | 0.653193 | 244.81 |
| Prior V2 learned adapter | 0.372688 | 0.549068 | 0.359706 | 0.616814 | 205.29 |

## 5. 结果分析

相比原始 When Words Smile baseline：

- MAE 从 `0.379475` 降至 `0.372688`。
- RMSE 从 `0.563167` 降至 `0.549068`。
- PPL 从 `262.39` 降至 `205.29`。
- Smoothness 从 `0.386786` 降至 `0.359706`。
- Acceleration 从 `0.663121` 降至 `0.616814`。

相比 V1，V2 也有稳定提升：

- MAE 进一步下降 `0.004510`。
- PPL 从 `244.81` 下降到 `205.29`。
- 序列变化更平滑。

这说明学习式 prior adapter 比固定规则模板更能适配数据分布。

## 6. 价值

V2 的意义在于，它已经不只是固定规则后处理，而是一个可训练的小模块：

```text
text-derived emotion prior -> learnable sequence modulation
```

这为后续真正接入中文情绪先验提供了一个清晰接口。后续只需要将当前英文关键词 prior 替换为：

```text
Chinese-Emotion-Small + rules_v4
```

即可形成中文情绪先验引导的动态表情序列生成方法。

## 7. 局限

当前 V2 仍然是 adapter 层级改进：

- 没有重训 When Words Smile 主体模型。
- prior 仍然来自英文关键词规则，主要用于适配 EmoAva 复现实验。
- 还没有把 prior 拼接进 Transformer / CVAE 的内部表示。

因此 V2 可以作为强于 V1 的中间版本，但不是最终方法。

## 8. 下一步

V3 建议做结构级融合：

```text
text embedding + emotion prior embedding -> sequence decoder
```

即把情绪先验放入模型输入或中间层，而不是只对输出序列做 adapter 调制。
