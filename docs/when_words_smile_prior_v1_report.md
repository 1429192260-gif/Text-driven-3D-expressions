# When Words Smile Prior V1 改进实验报告

## 1. 实验目的

本实验是基于 When Words Smile 复现结果的第一版改进尝试。

核心目标不是直接重训官方模型，而是在官方 checkpoint 输出的动态表情序列上，引入一个轻量的情绪先验调制模块，验证：

```text
emotion prior + temporal profile
```

是否能改善 `text -> expression sequence` 的生成结果。

需要说明的是，官方 EmoAva 数据集为英文文本，因此本实验先使用英文关键词版 emotion prior adapter 验证流程。后续迁移到中文任务时，可将该 adapter 替换为：

```text
Chinese-Emotion-Small + rules_v4
```

## 2. 方法

基础模型：

```text
When Words Smile official checkpoint
```

原始输出：

```text
P = {p_1, p_2, ..., p_T}, p_t in R^53
```

V1 改进流程：

```text
text -> emotion prior
P -> emotion-specific temporal modulation -> P'
```

情绪先验包括：

- emotion category
- intensity
- keyword scores

时间模板包括：

- happy
- sad
- angry
- surprise
- concern
- calm
- neutral

调制公式：

```text
P'_t[g_e] = P_t[g_e] * (1 + alpha * intensity * (w_e(t) - 1))
```

其中：

- `g_e` 是情绪相关参数组。
- `w_e(t)` 是情绪对应的时间曲线。
- `alpha` 是 dev 集选择出的调制强度。

## 3. Dev 集选择 alpha

在 dev 集上扫描：

```text
alpha = 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00
```

主要结果：

| Alpha | MAE | RMSE | Smoothness | Acceleration | PPL |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.383168 | 0.571361 | 0.388399 | 0.666431 | 278.01 |
| 0.10 | 0.382633 | 0.570545 | 0.387638 | 0.665156 | 272.56 |
| 0.20 | 0.382162 | 0.569792 | 0.386877 | 0.663880 | 267.95 |
| 0.30 | 0.381758 | 0.569104 | 0.386115 | 0.662605 | 264.15 |
| 0.40 | 0.381418 | 0.568480 | 0.385354 | 0.661329 | 261.11 |
| 0.50 | 0.381142 | 0.567921 | 0.384593 | 0.660054 | 258.87 |
| 0.60 | 0.380930 | 0.567427 | 0.383832 | 0.658780 | 257.42 |
| 0.80 | 0.380700 | 0.566634 | 0.382310 | 0.656229 | 256.84 |
| 1.00 | 0.380716 | 0.566103 | 0.380788 | 0.653678 | 259.63 |

综合 MAE 与 PPL，选择：

```text
alpha = 0.80
```

## 4. Test 集结果

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| When Words Smile baseline | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |
| Prior V1 | 0.377198 | 0.558812 | 0.380854 | 0.653193 | 244.81 |

## 5. 结果分析

V1 相比 baseline 有小幅但一致的提升：

- MAE 从 `0.379475` 降至 `0.377198`。
- RMSE 从 `0.563167` 降至 `0.558812`。
- PPL 从 `262.39` 降至 `244.81`。
- Smoothness 和 Acceleration 同时降低，说明序列变化略微更平稳。

这说明在不重训官方模型的条件下，情绪先验和动态时间模板能够对生成序列产生正向调制。

## 6. 当前局限

V1 是后处理式改进，仍然比较轻量：

- 没有修改官方模型结构。
- 没有把 prior 融入训练过程。
- 当前使用英文关键词 prior，只是为了适配 EmoAva 英文数据。
- 还不能代表最终中文情绪先验模型效果。

## 7. 后续方向

V2 建议升级为：

```text
text embedding + emotion prior -> sequence generator
```

也就是把 prior 从后处理模块前移到模型输入或中间层，而不是只调制输出。

迁移到中文任务时，将英文 prior adapter 替换为：

```text
Chinese-Emotion-Small + rules_v4
emotion / intensity / keyword scores
emotion-specific temporal profile
```

这样才能形成真正属于我们任务的中文情绪先验引导表情序列生成方法。
