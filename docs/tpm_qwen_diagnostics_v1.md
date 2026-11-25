# TPM-MI 图文 Qwen：结果分析与图片错配诊断

本入口接在已经完成的 A/B/C 实验后面。默认读取 `outputs_vlm_gold/tpm_qwen_visual_min28_v1`，新结果写入 `outputs_vlm_gold/tpm_qwen_diagnostics_v1`。

不修改原 TPM-MI 或图文 Qwen 模块，不改变它们绑定的代码哈希，不重新训练任何模型，也不重新选择融合权重。

## 运行什么

1. **CPU 分析**：复现原逐样本预测的指标；输出完整官方测试集、原有去重子集、有效图片数 0/1/2+ 分组；计算 C−B 的配对 95% bootstrap 区间。
2. **GPU 冒烟**：用原输入重提取少量正确图特征，与缓存核对；换成错配像素后验证词特征变化，文本 token、视觉 token 预算及位置保持原样。
3. **冻结错配推理**：仅对官方 test 的 Qwen 图片输入作干预，复用现有三个种子的辅助头与原 CRF。TPM-MI 继续用正确图片，其 emissions 直接复用。没有 train/dev 特征提取或训练阶段。

## 安装与命令

在 AutoDL 执行定向安装，无需切换分支，不覆盖以前手动安装的未跟踪实验文件：

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM &&
git fetch origin &&
git restore --source=origin/codex/tpm-mi-llm-20261117 --worktree -- \
    scripts/evaluate_tpm_qwen_diagnostics_v1.py \
    scripts/run_tpm_qwen_diagnostics_v1.sh \
    docs/tpm_qwen_diagnostics_v1.md
```

指定已完成的图文实验目录和新的诊断目录，在同一终端运行：

```bash
export TPM_VISUAL_RUN_DIR=outputs_vlm_gold/tpm_qwen_visual_min28_v1
export TPM_DIAG_RUN_DIR=outputs_vlm_gold/tpm_qwen_diagnostics_v1

# 只分析现有结果，不加载 Qwen、不使用 GPU 推理。
bash scripts/run_tpm_qwen_diagnostics_v1.sh analyze

# 检查真实模型下正确图特征复现与固定尺寸错配。
bash scripts/run_tpm_qwen_diagnostics_v1.sh smoke
```

smoke 通过后启动完整诊断：

```bash
nohup bash scripts/run_tpm_qwen_diagnostics_v1.sh all > tpm_qwen_diagnostics_v1.log 2>&1 &
tail -f tpm_qwen_diagnostics_v1.log
```

`all` 会复用已完成的 analyze 和 smoke；错配特征按 64 条分块缓存，可续跑。只有 Qwen 测试特征和辅助头推理会新增 GPU 计算。查看简明结果：

```bash
bash scripts/run_tpm_qwen_diagnostics_v1.sh summary
```

继续使用现有 Conda `base` 和 `mner-vlm` 环境，不安装依赖。可设置 `TPM_CONDA_BIN`、`TPM_VLM_ENV`。默认错配随机种子和 bootstrap 随机种子均为 1234，bootstrap 重采样 10,000 次。可在首次 prepare 前设置 `TPM_DIAG_MISMATCH_SEED`、`TPM_DIAG_BOOTSTRAP_SEED`、`TPM_DIAG_BOOTSTRAP_REPLICATES`、`TPM_DIAG_SHARD_SIZE`；改变配方或脚本后换一个新诊断目录。不能根据 test 结果反复挑选错配种子。

## 错配控制了什么

- 给每个帖子从**相同有效图片数量**的其他帖子中，按固定随机种子抽取一个 donor 帖子。抽样允许重复，同一个错配表共享到全部模型种子。
- donor 必须来自不同的“精确文本或图像哈希”连通组，排除相同文本、重复图片以及传递关联。构造错配表不读取实体标注。
- 先照原方案处理正确图片，保留实际 `input_ids`、`attention_mask`、`image_grid_thw` 和词对齐位置。将 donor 图片缩放到这些原始视觉网格对应的尺寸，关闭 donor 的再次自动 resize，只替换 `pixel_values`。因此文本长度、视觉 token 数、图像数量和位置完全不变。donor 可能需要改变长宽比，这是固定网格控制的明确处理。
- 不存在有效图片，或同数量组中没有不关联 donor 的记录，保持原 C 的特征和预测。报告同时给出全体、实际可错配子集、去重子集和多图子集，不能把未干预记录当作错配成功记录。
- C 的现有辅助头、CRF 转移、主模型 emissions 和融合系数完全固定。评估前必须用原 C 的缓存 logits 精确复现其逐样本预测。

## 置信区间的口径

使用实体 TP/FP/FN 重算每次重采样的**语料级 micro-F1**，不平均逐帖 F1。以测试集内精确文本/共享图片 SHA256 的连通组为重采样单位；每次对 B/C 以及全部已拟合模型种子使用同一组抽样，随后计算种子平均的 C−B。报告百分位 95% 区间。

这个区间表示**给定当前已训练模型时，测试样本的不确定性**，不完整覆盖重新训练的不确定性。独立组不足两组时不报告区间；空图片分组也不能作性能解释。区间跨 0 时，不能据此声称统计显著。

## 输出文件与解释

- `analysis/analysis_report.json`：完整/去重结果、图片数分组、C−B 的 `delta_pp` 和 `ci95_pp`，均以百分点表示。
- `mismatch_plan.json`：固定的 donor 对应关系、可干预记录和未干预原因。
- `smoke/completed.json`：原正确图特征复现误差、错配后的像素/词特征变化。
- `evaluation/mismatch_report.json`：A/B/C/D 的 F1，C−D 的逐种子差值和平均百分点差；`eligible_mismatch` 是主要干预子集。
- `evaluation/seed*_mismatched_predictions.jsonl`：D 的逐样本预测。

若正确图片 C 优于错配图片 D，支持当前模型对图文对应内容敏感；仍不能仅凭这一项声称具备更强的跨图关系推理。诊断是在看到主结果后提出的，报告标记为 post-hoc；错配只运行一个固定抽样，不提供错配抽样之间的方差或显著性结论。原官方测试集此前已使用过，不称为新盲测。

本地 CPU 验证覆盖原生 Transformers 4.51.3 下的网格/像素控制、小图、置信区间、原预测重放与源目录不改写；完整 3B 模型冒烟与错配指标需要在 AutoDL GPU 上实测。
