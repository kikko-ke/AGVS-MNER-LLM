# 缓存融合与 Qwen 7B 规模对照

本入口复用已经完成的 3B 图文实验 `outputs_vlm_gold/tpm_qwen_visual_min28_v1`，不改写旧模块、协议、预测或 TPM-MI 检查点。两个研究问题分别运行，结果目录独立。

## 定向更新

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM &&
git fetch origin &&
git restore --source=origin/codex/tpm-mi-llm-20261117 --worktree -- \
    scripts/run_tpm_qwen_studies_v1.py \
    scripts/run_tpm_qwen_studies_v1.sh \
    scripts/download_qwen7b_v1.py \
    scripts/evaluate_tpm_qwen_diagnostics_v1.py \
    docs/tpm_qwen_studies_v1.md
```

`evaluate_tpm_qwen_diagnostics_v1.py` 是未改动的既有统计工具，这里列出是为确保依赖存在。不要切换分支或覆盖旧 TPM-MI/图文模块。

## 1. 缓存融合：不提取新特征，不训练

保留原两套头、它们原来验证集选出的融合权重以及 TPM-MI 的输出，将两套完整 emission 作凸组合：

`e_mix = (1-alpha) * e_text3b + alpha * e_multimodal3b`

alpha=0、1 分别精确还原旧 B、C。只在 seed 1234 的官方 dev 上搜索 `{0, 0.25, 0.5, 0.75, 1}`，F1 平局取较小 alpha；随后固定到全部配对种子。测试集不搜索 alpha。这是全局标量插值先导实验，不是已训练的词级门控。

```bash
export TPM_VISUAL_RUN_DIR=outputs_vlm_gold/tpm_qwen_visual_min28_v1
export TPM_FUSION_RUN_DIR=outputs_vlm_gold/tpm_qwen_fusion_v1
bash scripts/run_tpm_qwen_studies_v1.sh fusion
```

该模式只用现有 Conda base 的 CPU/CRF，不加载 Qwen。输出 `calibration/dev_report.json` 和 `evaluation/study_report.json`；测试评估会先精确重放原 B/C 逐样本预测，再报告插值方案相对两者的 F1 差值及配对区间。若选择了端点，这是有效的无额外收益结果，不再根据 test 挑其他比例。

## 2. 7B：文本/图文成对对照

只换成同系列 Qwen2.5-VL-7B-Instruct，TPM-MI 底座、原始数据划分、图片预算、固定投影 seed、768 维特征、BiGRU 结构和配对模型种子均沿用原实验。7B 的两个小头重新训练；不能直接套用 3B 的头。Qwen 本身保持 NF4/BF16、完全冻结，特征以 batch 1 提取；辅助头的 batch/学习率/epoch 与原 3B 一致。

先下载到独立权重目录，完成后冒烟：

```bash
export TPM_QWEN7B_MODEL=/root/autodl-tmp/pretrained_models/Qwen2.5-VL-7B-Instruct
bash scripts/run_tpm_qwen_studies_v1.sh download7b
bash scripts/run_tpm_qwen_studies_v1.sh smoke7b
```

下载器使用官方 `Qwen/Qwen2.5-VL-7B-Instruct`，在首次网络请求时解析并固定 revision，重试保留缓存；已有完整权重则跳过网络。它检查必需文件、配置和索引分片存在性/大小，不宣称代替完整权重校验或真实模型加载。指向 3B 或其他模型目录时拒绝覆盖。

smoke 检查两个分支的词级对齐、图片对文本状态的影响和小头前向/反向，并记录模型加载及特征计算阶段的峰值已分配显存。能否在实例上运行以这次真实模型 smoke 为准，不保证显存占用或耗时。

先只跑 train/dev：

```bash
nohup bash scripts/run_tpm_qwen_studies_v1.sh pilot7b > tpm_qwen_7b_v1.log 2>&1 &
tail -f tpm_qwen_7b_v1.log
```

或直接运行全部配对种子和最终测试（不要与 pilot 同时启动）：

```bash
nohup bash scripts/run_tpm_qwen_studies_v1.sh all7b > tpm_qwen_7b_v1.log 2>&1 &
tail -f tpm_qwen_7b_v1.log
```

若已完成 pilot，用 `all7b` 继续会复用已完成阶段。两种 7B 分支共享同一固定随机投影，各自拟合相同结构的头；它们在 seed 1234 dev 上分别从原网格 `{0, 0.25, 0.5, 1}` 选择分数权重，并固定到全部种子。另报告双方都固定为 1 的对照。特征按原 shard_size（通常 64）同时缓存两条分支，可续跑；头训练中断后从头开始。

7B 结果默认为 `outputs_vlm_gold/tpm_qwen_7b_v1/evaluation/study_report.json`，同时比较 3B 文本、3B 图文、7B 文本、7B 图文。重点字段：`visual_gain_7b`、`visual_gain_7b_shared1`、`scale_text_7b_minus_3b`、`scale_multimodal_7b_minus_3b`。不能把 7B 图文总分上升自动解释为视觉增益扩大；应比较 7B 图文−7B 文本与旧 3B 图文−3B 文本。

## 汇总与变量

```bash
bash scripts/run_tpm_qwen_studies_v1.sh summary-fusion
bash scripts/run_tpm_qwen_studies_v1.sh summary7b
```

继续使用现有 Conda base 与 mner-vlm，不安装/升级依赖。变量包括 `TPM_VISUAL_RUN_DIR`、`TPM_FUSION_RUN_DIR`、`TPM_7B_RUN_DIR`、`TPM_QWEN7B_MODEL`、`TPM_CONDA_BIN`、`TPM_VLM_ENV`、`CUDA_VISIBLE_DEVICES`。每个目录的代码、数据、模型和配方均绑定协议，改变后换新目录。统计区间复用按精确文本/图像组配对重采样的方法，仅表示给定已拟合模型时的样本不确定性。

这两项研究是在查看已有 test 结果后提出的后续研究，协议记录为 post-hoc；所有新选择仍只在 train/dev 完成，官方 test 不重新称作未见盲测。现有错配诊断与三种子结果保留，不以新结果覆盖。真实 7B GPU smoke 和识别指标需在 AutoDL 上运行，本地 CPU 验证不代表已有性能提升。
