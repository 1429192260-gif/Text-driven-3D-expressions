# Mac Codex Official Render Phase 2 Prompt - 2026-07-09

把下面这段发给 Mac 端 Codex，用于接入官方 EmoAva / DECA / FLAME 渲染迁移。

```text
你现在是 Mac 端 Codex，正在继续 Phase 2：迁移官方 EmoAva / DECA / FLAME 3D 渲染资产和输入。

请先阅读：
- docs/official_render_phase2_handoff_2026-07-09.md
- docs/official_3d_backend_integration.md
- docs/mac_phase1_verified_2026-07-09.md

Windows 端已准备 Phase 2 迁移包：
- project_mac_transfer_official_render_phase2.zip
- 大小约 286.62MB

你的目标不是强行在 Mac 上完成 CUDA/PyTorch3D 渲染。第一目标是：
1. 解压 Phase 2 包到仓库根目录。
2. 确认官方 DECA/FLAME assets 已归位。
3. 确认官方 render inputs 已归位。
4. 重新运行 export_emoava_official_render_inputs.py，验证 53D 输入可再生成。
5. 运行 check_emoava_official_3d_env.py --device cpu，报告缺失依赖。
6. 如果 PyTorch3D/DECA 在 Mac 可用，再尝试 --instantiate-deca。
7. 不要把 external/EmoAva 官方 3D 渲染和 Phase 1 主线研究混在同一个结论里。

请执行：

unzip ~/Downloads/project_mac_transfer_official_render_phase2.zip -d .

test -f external/EmoAva/src/data/generic_model.pkl
test -f external/EmoAva/src/data/default_code_trevor_emoca2.pkl
test -f outputs/emoava_official_render_inputs/v6_showcase/v6_sample_selected_exps.pkl
test -f outputs/when_words_smile_prior_v6/test_mixture_gate_sample.pt

.venv/bin/python scripts/export_emoava_official_render_inputs.py \
  --methods gold,baseline,v6_sample \
  --cases 967,354,295,428,84 \
  --output-dir outputs/emoava_official_render_inputs/v6_showcase_mac_check

.venv/bin/python scripts/check_emoava_official_3d_env.py --device cpu

如果环境检查显示 pytorch3d 缺失，这是预期风险，不要在 Mac 上硬修太久。把完整输出发回 Windows 端。

如果 Mac 环境里刚好有 PyTorch3D 或可以顺利安装，再尝试：

.venv/bin/python scripts/check_emoava_official_3d_env.py --device cpu --instantiate-deca

请最终返回：
- Phase 2 包是否解压成功
- assets 是否存在
- export 脚本是否成功
- check env 输出
- 是否存在 rendered mp4
- 如果失败，完整错误栈和缺失依赖名

注意：
- external/EmoAva/checkpoints/model.chkpt 没有包含在本包里，因为从导出的 .pkl 做官方 DECA 渲染不需要它。
- cloud-rendered official mp4 视频没有在 Windows 项目本地找到。如果用户还有云端输出，需要单独下载。
```

