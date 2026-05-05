# When Words Smile 复现与对比分析

## 复现范围

本次复现基于官方代码、官方 `EmoAva` 数据集和官方 baseline checkpoint。

已完成：

- 官方 test 集完整推理。
- 官方模型输出检查。
- mean / random / shuffle 三类简单 baseline 对比。
- MAE、RMSE、Smoothness、Acceleration、official-style PPL 评价。
- 前 10 条测试样本的案例级序列分析。
- 前 10 条测试样本的曲线可视化。
- 前 3 条测试样本的轻量表情 GIF 可视化。

尚未完成：

- 官方 `evaluate.py` 原始多进程版本在 Windows 中文路径下会报 `UnicodeEncodeError`。
- 原始单进程版本运行超过 30 分钟未结束。
- 因此本次 PPL 使用了等价快速实现，保留官方公式的核心形式。
- 训练复现尚未开始，当前阶段是 checkpoint inference reproduction。
- 官方 3D 渲染依赖 PyTorch3D / DECA / CUDA，当前先完成参数序列层面的复现与分析。
- 当前 Windows 环境无法直接安装 `pytorch3d==0.7.8`，因此官方 FLAME/DECA 渲染暂未完成。
- 完整训练尚未运行，但已完成官方结构的训练 smoke test。

## 数据与模型

数据：

```text
train: 12000
dev: 1500
test: 1500
expression shape: T x 53
```

模型：

```text
external/EmoAva/checkpoints/model.chkpt
```

完整测试集输出：

```text
outputs/when_words_smile_repro/test_parallel_full.pt
shape: 1500 x 255 x 53
```

结果文档：

```text
docs/when_words_smile_repro_full_eval.md
docs/when_words_smile_repro_ppl_fast.md
docs/when_words_smile_repro_case_analysis.md
docs/when_words_smile_ppl_formula_check.md
docs/when_words_smile_param_error_analysis.md
outputs/when_words_smile_repro/plots
outputs/when_words_smile_repro/light_face
outputs/when_words_smile_repro/train_smoke/model_smoke.chkpt
```

## 对比结果

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 0.696932 | 0.925793 | 0.000000 | 0.000000 | 1624594132.26 |
| Random baseline | 0.936255 | 1.240559 | 0.050528 | 0.075919 | 6607959965.88 |
| Shuffle baseline | 0.576919 | 0.833852 | 0.120418 | 0.209696 | 1720846.69 |
| When Words Smile checkpoint | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |

## 结果解读

### 0. 可视化结果

已生成 CPU 版本的参数序列可视化，不依赖 PyTorch3D / CUDA：

```text
outputs/when_words_smile_repro/plots/case_01.png
...
outputs/when_words_smile_repro/plots/case_10.png
outputs/when_words_smile_repro/plots/baseline_energy_summary.png
```

每个 case 图包含：

- 真实序列与预测序列的整体能量曲线。
- 预测序列中不同参数组的动态变化。
- 每一帧的参数 MAE。

这些图可以作为正式 3D 渲染前的轻量可视化复现结果。

此外，已生成轻量表情 GIF：

```text
outputs/when_words_smile_repro/light_face/case_01.gif
outputs/when_words_smile_repro/light_face/case_02.gif
outputs/when_words_smile_repro/light_face/case_03.gif
```

这不是官方 FLAME 渲染，而是一个基于 53 维参数分组的 2.5D 动态脸部代理，用于快速检查预测序列是否能产生可见的动态表情变化。

### 1. 官方模型明显优于简单 baseline

从 MAE 和 RMSE 看，官方 checkpoint 明显优于三类简单 baseline：

- 相比 `Mean baseline`，MAE 从 `0.696932` 降到 `0.379475`。
- 相比 `Random baseline`，MAE 从 `0.936255` 降到 `0.379475`。
- 相比较强的 `Shuffle baseline`，MAE 从 `0.576919` 降到 `0.379475`。

这说明模型不是简单生成平均表情，也不是随便复用其他样本的运动序列，而是确实从文本中学习到了对表情序列有用的信息。

### 2. PPL 指标差距非常明显

official-style PPL 中，官方模型为 `262.39`，远低于 baseline：

- `Mean baseline`: `1.62e9`
- `Random baseline`: `6.61e9`
- `Shuffle baseline`: `1.72e6`

这说明在官方概率评价视角下，官方模型生成序列更接近真实测试表情分布。

### 3. Smoothness / Acceleration 需要谨慎解释

Mean baseline 的 Smoothness 和 Acceleration 都是 `0`，因为它每一帧都相同，但这并不代表它好，只代表它没有动态变化。

官方模型的 Smoothness 和 Acceleration 更高，说明它生成了更明显的动态变化。对于动态表情任务，这类指标不能单独看“越低越好”，需要和参数误差、PPL、可视化一起分析。

较合理的解释是：

- Mean baseline 最平滑，但没有表情动态。
- Random / shuffle baseline 有动态，但与文本和真实序列匹配较差。
- 官方模型在保持较低参数误差的同时生成了明显动态变化。

### 4. 对我们后续改进的启发

When Words Smile 的优势主要体现在：

- 将任务从静态表情预测扩展到动态表情序列生成。
- 使用文本语义直接生成连续表情参数序列。
- 通过 CVAE / Transformer 建模同一句文本对应多种合理表情。
- 使用 PPL、动态流畅性、多样性等指标评价序列质量。

我们的改进空间：

- 官方模型主要面向英文文本。
- 它依赖文本编码隐式学习情绪，不显式建模中文情绪规则。
- 对中文弱情绪、细粒度情绪、程度副词和否定表达没有专门处理。
- 缺少针对不同情绪动态过程的显式时间模板，例如快起快落、慢起慢落、峰值保持等。

因此，我们后续可以在其 text-to-expression sequence 框架上加入：

```text
Chinese-Emotion-Small + rules_v4
emotion / intensity / keyword scores
emotion-specific temporal profile
```

形成：

```text
text embedding + Chinese emotion prior -> expression sequence
```

进一步可以扩展为：

```text
text feature + emotion prior + temporal profile -> expression sequence
```

这样既保留 When Words Smile 的动态序列生成能力，又能把我们已有的中文情绪先验变成可解释控制分支。

## 当前结论

本次复现证明，When Words Smile 的官方 checkpoint 可以在本地完整 test 集上生成 `1500 x 255 x 53` 的动态表情参数序列，并且显著优于 mean / random / shuffle 等简单 baseline。

此外，训练 smoke test 已通过，说明官方训练代码的数据读取、模型前向、反向传播、loss 计算和 checkpoint 保存都能在本地执行。smoke test 不代表完整训练复现，只说明训练链路可运行。

因此，它适合作为我们后续改进实验的基础模型或强相关对比方法。下一步应优先做两件事：

1. 基于官方输出做可视化案例分析，确认生成表情的直观效果。
2. 在官方 text-to-expression sequence 框架上加入中文情绪先验分支，设计先验增强消融实验。
