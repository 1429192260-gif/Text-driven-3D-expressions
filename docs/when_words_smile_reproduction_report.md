# When Words Smile 复现实验报告

## 1. 复现目标

本阶段目标是扎实复现 `When Words Smile: Generating Diverse Emotional Facial Expressions from Text` 的基础实验流程，为后续加入中文情绪先验改进做准备。

复现重点包括：

- 官方数据集读取。
- 官方 checkpoint 推理。
- 官方指标的可运行替代实现。
- 简单 baseline 对比。
- 序列曲线与轻量可视化。
- 训练流程 smoke test。

## 2. 实验环境

代码：

```text
external/EmoAva
```

数据：

```text
external/EmoAva/dataset
```

模型：

```text
external/EmoAva/checkpoints/model.chkpt
models/bert-base-cased
```

环境：

```text
conda env: cteg
python: 3.10
torch: 2.2.2
transformers: 4.39.3
```

## 3. 数据统计

数据规模：

```text
train: 12000
dev: 1500
test: 1500
```

每条样本由一段英文文本和一段 53 维表情参数序列组成：

```text
text -> expression sequence
expression shape: T x 53
```

详细统计见：

```text
docs/emoava_dataset_summary.md
```

## 4. 官方 checkpoint 推理复现

完整 test 集推理命令：

```shell
conda run -n cteg python translate.py ^
  -model "..\checkpoints\model.chkpt" ^
  -tokenizer_path "..\..\..\models\bert-base-cased" ^
  -save_path "..\..\..\outputs\when_words_smile_repro" ^
  -data_source "..\dataset" ^
  -save_name "test_parallel_full.pt" ^
  -max_seq_len 256 ^
  -src_len 128 ^
  -batch_size 64 ^
  -infer_mode "p" ^
  -seed 42 ^
  -cvae ^
  -no_cuda
```

输出：

```text
outputs/when_words_smile_repro/test_parallel_full.pt
shape: 1500 x 255 x 53
```

这说明官方 checkpoint 可以在本地完整生成测试集表情参数序列。

## 5. 指标复现

官方 `evaluate.py` 在 Windows 中文路径下使用 joblib 多进程会触发编码错误，单进程版本耗时过长。因此本次实现了等价 fast PPL。

等价性验证：

```text
docs/when_words_smile_ppl_formula_check.md
```

小样本验证中，fast PPL 与官方 SciPy 公式逐帧概率最大差异约为：

```text
1.17e-8
```

因此 fast PPL 可作为当前环境下的官方指标替代实现。

## 6. 对比实验

对比方法：

- Mean baseline
- Random baseline
- Shuffle baseline
- When Words Smile checkpoint

结果：

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 0.696932 | 0.925793 | 0.000000 | 0.000000 | 1624594132.26 |
| Random baseline | 0.936255 | 1.240559 | 0.050528 | 0.075919 | 6607959965.88 |
| Shuffle baseline | 0.576919 | 0.833852 | 0.120418 | 0.209696 | 1720846.69 |
| When Words Smile checkpoint | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |

结论：

- 官方模型显著优于 mean / random / shuffle baseline。
- PPL 差距尤其明显，说明官方模型生成结果更接近真实表情序列分布。
- Mean baseline 虽然平滑性最低，但没有动态变化，不能说明生成质量更好。

## 7. 可视化复现

已生成参数序列曲线图：

```text
outputs/when_words_smile_repro/plots
```

每个 case 图包含：

- 真实与预测整体能量曲线。
- 预测序列中不同参数组的动态变化。
- 每一帧 MAE。

由于当前 Windows 环境无法直接安装 PyTorch3D，官方 FLAME/DECA 渲染暂未完成。作为替代，已生成轻量 2.5D 表情 GIF：

```text
outputs/when_words_smile_repro/light_face/case_01.gif
outputs/when_words_smile_repro/light_face/case_02.gif
outputs/when_words_smile_repro/light_face/case_03.gif
```

该可视化不是官方 FLAME 渲染，仅用于快速观察生成序列的动态变化。

## 8. 训练流程验证

完整训练尚未运行，因为 CPU 环境下成本较高。但已完成训练 smoke test：

```text
outputs/when_words_smile_repro/train_smoke/model_smoke.chkpt
```

smoke test 验证了：

- 训练数据读取可用。
- BERT 编码器加载可用。
- CVAE/Transformer 前向传播可用。
- loss 计算可用。
- 反向传播和优化器更新可用。
- checkpoint 保存可用。

smoke test 输出：

```text
loss: 35.7182
kl_loss: 2.3721
one_step_loss: 17.4358
first_token_loss: 28.9235
```

## 9. 复现结论

本阶段已经完成 When Words Smile 的主要推理复现和实验对比。官方 checkpoint 能够在本地完整 test 集上生成动态 3D 表情参数序列，并且在 MAE、RMSE 和 PPL 上显著优于简单 baseline。

当前复现足以支撑后续工作：

```text
When Words Smile baseline
-> 中文情绪先验分支
-> prior-guided expression sequence generation
```

下一阶段建议在官方动态序列生成框架基础上，加入我们的：

```text
Chinese-Emotion-Small
rules_v4
emotion / intensity / keyword scores
emotion-specific temporal profile
```

这样可以从“复现论文”自然推进到“基于论文做中文场景改进”。
