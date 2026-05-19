# 官方 3D 表情生成后端接入方案

## 目标

我们当前前端已经可以生成 WhenWordsSmile / EmoAva 格式的连续 `53D` 表情参数。现在接入官方 3D 后端时，采用下面这条闭环：

```text
文本 -> 我们的 V6 前端 -> 53D 动态表情参数 -> 官方 DECA/FLAME -> 3D 表情视频
```

这里的 `53D` 与官方一致：

```text
前 50 维：FLAME expression code
后 3 维：jaw pose，下颌姿态
```

## 为什么优先用官方后端

官方 `external/EmoAva/src/visualize.py` 使用的是 DECA + FLAME 渲染链路，能直接把表达参数转成 3D 人脸表情视频。论文里这样表述最稳：后端不是我们临时画的 2D 图，而是复用 WhenWordsSmile/EmoAva 官方三维表情生成工具；我们的主要贡献集中在文本到动态表情参数的生成与改进。

## 本机检查结果

当前 Windows 本机可以读取数据和生成官方输入，但不能直接跑完整官方渲染，原因是：

```text
1. 当前环境没有 CUDA。
2. 当前环境没有 pytorch3d，官方渲染器默认依赖它。
3. 如果运行官方原始 visualize.py，还需要把 FLAME_albedo_from_BFM.npz 放到 external/EmoAva/src/data/。
```

不过我们已经确认 FLAME 几何模型文件存在，且 `cteg` 环境中官方 FLAME Decoder 可以在 CPU 上正常生成 `5023` 个顶点和 `68` 个 3D landmarks。这说明我们的参数格式和官方几何模型是能对接的，真正缺的是 GPU 渲染环境。

## 本地先导出官方渲染输入

在本机运行：

```powershell
& 'C:\ProgranData\anaconda\envs\cteg\python.exe' scripts\export_emoava_official_render_inputs.py `
  --methods gold,baseline,v6_sample `
  --cases 967,354,295,428,84 `
  --output-dir outputs/emoava_official_render_inputs/v6_showcase
```

输出位置：

```text
outputs/emoava_official_render_inputs/v6_showcase/
```

主要文件：

```text
gold_selected_exps.pkl
baseline_selected_exps.pkl
v6_sample_selected_exps.pkl
manifest.json
manifest.md
```

其中 `.pkl` 文件就是官方 `visualize.py --exp_path` 能读取的格式。`manifest.md` 记录了每个渲染视频对应的原始 case 编号和文本，方便汇报截图时对齐前端实验结果。

## GPU 服务器推荐环境

建议使用 Linux + NVIDIA GPU，优先选下面这种配置：

```text
系统：Ubuntu 20.04 / 22.04
GPU：RTX 3090 / RTX 4090 / A10 / A100 均可
显存：16GB 以上较稳
Python：3.10
CUDA：与 PyTorch 2.2.2 兼容
```

渲染不是训练大模型，成本主要来自安装 PyTorch3D 和逐帧生成视频。先租一张 3090/4090 级别 GPU 跑通即可，不需要一开始就上很贵的多卡机器。

## 服务器安装步骤

进入仓库后：

```bash
conda create -n cteg python=3.10 -y
conda activate cteg
cd external/EmoAva/src
pip install -r requirements.txt
```

安装 PyTorch3D。官方 README 给的是源码安装：

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

如果运行官方原始 `visualize.py`，还需要下载 `FLAME_albedo_from_BFM.npz`，并放到：

```text
external/EmoAva/src/data/FLAME_albedo_from_BFM.npz
```

如果只跑我们包装脚本的 `--geometry-only` 模式，可以先不放这个 400MB 纹理文件，先生成官方 FLAME 几何表情视频。

## 服务器渲染命令

推荐先跑 V6 最佳版本：

```bash
python scripts/render_emoava_official_deca.py \
  --exp-path outputs/emoava_official_render_inputs/v6_showcase/v6_sample_selected_exps.pkl \
  --output-dir outputs/emoava_official_render_videos/v6_sample \
  --device cuda \
  --geometry-only \
  --no-detail
```

再跑 baseline 对比：

```bash
python scripts/render_emoava_official_deca.py \
  --exp-path outputs/emoava_official_render_inputs/v6_showcase/baseline_selected_exps.pkl \
  --output-dir outputs/emoava_official_render_videos/baseline \
  --device cuda \
  --geometry-only \
  --no-detail
```

如果想渲染真实测试集 ground truth：

```bash
python scripts/render_emoava_official_deca.py \
  --exp-path outputs/emoava_official_render_inputs/v6_showcase/gold_selected_exps.pkl \
  --output-dir outputs/emoava_official_render_videos/gold \
  --device cuda \
  --geometry-only \
  --no-detail
```

输出视频会类似：

```text
outputs/emoava_official_render_videos/v6_sample/row_00_case_0967.mp4
outputs/emoava_official_render_videos/v6_sample/row_01_case_0354.mp4
```

## 汇报展示建议

建议展示三组视频：

```text
1. Gold：真实表情参数渲染结果，作为上限参考。
2. Baseline：复现 WhenWordsSmile baseline 后生成的 3D 表情。
3. V6 sample gate：我们当前最佳改进方法生成的 3D 表情。
```

讲法可以是：

```text
前面的实验只评估了 53D 参数误差和平滑性；现在我们进一步把参数接入官方 DECA/FLAME 后端，验证这些数值改进是否能转化为可观察的三维表情动画改进。
```

## 下一步

优先顺序：

```text
1. 在本机导出 V6/baseline/gold 的官方渲染输入。
2. 租 GPU 服务器并安装 PyTorch3D。
3. 用 geometry-only 模式先跑通官方 3D 几何视频。
4. 如果时间允许，再补 FLAME_albedo_from_BFM.npz，尝试官方完整纹理可视化。
5. 把生成视频截图/动图放进 PPT，说明从文本到 3D 动态表情的完整流程已经闭环。
```

## 服务器环境检查

安装完依赖后，可以先运行：

```bash
python scripts/check_emoava_official_3d_env.py --device cuda --instantiate-deca
```

如果输出里 `pytorch3d: ok`、`cuda available: True`，并且 `DECA geometry-only init: ok`，就可以开始跑上面的渲染命令。

如果 `FLAME_albedo_from_BFM.npz` 显示 `optional missing`，不影响 `--geometry-only --no-detail` 的几何表情渲染；只有想渲染官方纹理/细节版本时才需要补这个文件。
