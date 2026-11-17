# TPM-MI + frozen BERT/Qwen word features

本入口不依赖 AGVS 的网络、checkpoint、内部划分或旧实验完成报告。新增文件不会更改旧训练入口，结果写到 `outputs_vlm_gold/tpm_mi_llm_v1`。

## 基线是什么

Huang et al., *MNER-MI: A Multi-image Dataset for Multimodal Named Entity Recognition in Social Media*, LREC-COLING 2024.

- 论文：<https://aclanthology.org/2024.lrec-main.1001/>
- 官方实现：<https://github.com/JinFish/MNER-MI>
- 对照的模型文件 blob：`b7e3de6187808bf95b3d54778efbb36440d3a98b` (`models/BERTModels.py`)

实现保持 BERT + ViT + 图像位置嵌入 + 图像 Transformer + 逐层视觉 K/V 前缀 + 线性分类 + CRF 的拓扑。不能称为逐位复刻或声称已经复现论文分数。本实现明确处理以下实现差异，并在协议中记录：

1. 官方代码把 `(batch, image, hidden)` 输入默认序列优先的 Transformer；这里显式使用 `batch_first=True`，避免不同帖子互相注意。
2. 原词只做一次 WordPiece 分词，直接转换 token ID，避免再次 `encode_plus` 导致标签错位。
3. 缺失和填充图像使用掩码；无有效图片的样本也能得到有限输出。
4. checkpoint 选择和主评估都采用全部原词上的实体边界、类型严格匹配 micro-F1。超出 64 位置窗口的原词映射为 O。

请将论文中的基线描述为 **TPM-MI 的经核对实现 / audited reproduction**，公开这些差异并报告自己的复现分数。

## 数据与运行环境

- 使用原始官方 train/dev/test（6856 / 860 / 860），不是旧的 5484/1372 内部划分或 4388 条拟合集。
- 默认来源：`dataset/text/MNER-MI_train.txt`、`MNER-MI_val.txt`、`MNER-MI_test.txt`。
- 图片默认为 `dataset/images`，保留每条记录中的图像排列。无法解码的图像不移动其他图片的槽位，记录后用掩码排除。
- 默认本地权重位于项目的上一级 `pretrained_models/`：`bert-base-uncased`、`ViTB-16`、`Qwen2.5-VL-3B-Instruct`。
- 底座训练、CRF 和融合评分使用现有 Conda `base`；Qwen/BERT 特征提取和辅助头训练使用现有 `/root/autodl-tmp/envs/mner-vlm`。脚本不安装或升级依赖。
- 需要一张 CUDA GPU，默认 `CUDA_VISIBLE_DEVICES=0`。所有模型从本地加载。

## AutoDL 命令

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM
bash scripts/run_tpm_llm_v1.sh smoke
nohup bash scripts/run_tpm_llm_v1.sh pilot > tpm_mi_llm_v1.log 2>&1 &
tail -f tpm_mi_llm_v1.log
```

`smoke` 检查真实模型的前向/反向、batch 隔离、Qwen 特征和辅助头。它不做正式训练。

`pilot` 提取官方 train/dev 的冻结特征，训练 seed 1234 的底座与两种辅助分支，在官方 dev 选择融合系数。最后输出 `calibration/dev_report.json`，**不计算 test 指标**。

开发集试验完成后，可继续同一输出目录中的三种子完整比较：

```bash
nohup bash scripts/run_tpm_llm_v1.sh all >> tpm_mi_llm_v1.log 2>&1 &
```

`all` 复用已经完成并通过哈希核验的步骤，补 seed 42/2026；然后提取 test 特征，按固定系数评估。测试报告为 `evaluation/evaluation_report.json`，逐帖子预测另存 JSONL。官方测试集在项目早期已经使用过，这一点在报告中保留，不称为未接触过的盲测。

## 训练和对照

- TPM-MI：batch 8，AdamW 统一 lr=5e-5、weight decay=0.01，15 epoch 上限，1% 预热，梯度裁剪 1；dev 连续 3 epoch 不提升则停止，保存 dev 最优 checkpoint。
- Qwen 只运行文本解码器，4-bit NF4，bfloat16 计算，完全冻结；不输入图片、不生成实体答案。
- 辅助 BERT 是额外的冻结 BERT，底座自己的 BERT 仍属于 TPM-MI。
- 取最后一个重叠辅助子词，Qwen 用固定 seed 1234 的随机投影统一到 768 维。BERT 已是 768 维。
- BiGRU 辅助头为 LayerNorm + 双向 GRU（每方向 128）+ 9 标签线性层，仅用官方 train 的真实标注训练 6 epoch；batch 16，lr=1e-3，weight decay=0.01。
- 同时训练 LayerNorm + Linear 的线性头，作为去掉 BiGRU 的对照。
- 只在 seed 1234 的官方 dev 搜索系数 `{0, 0.25, 0.5, 1}`；每分支独立选择，平局选较小系数。之后固定并共享到三个种子。0 表示辅助分支没有带来 dev 改善。
- 主系统是 `tpm`、`tpm_bert_head`、`tpm_qwen_head`。额外报告 linear、去 clip、双方共用系数 1 的比较。去 clip 保留中心化，不重新选系数。
- test 的重叠过滤子集按与官方 train **和 dev** 的精确文本/图像连通分量生成。样本数由实际数据决定，不能沿用旧实验的 834。
- 不预设 TPM-MI 上必然有增益；旧 AGVS 的分数不能改名字后用于本实验。

## 进度与路径调整

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM
ps -eo pid,etime,args | grep '[r]un_tpm_llm_v1'
/root/miniconda3/bin/python scripts/run_tpm_llm_v1.py --stage status
tail -n 40 tpm_mi_llm_v1.log
```

在第一次 `prepare` 前，可用环境变量指定路径和 batch：

```bash
TPM_DATA_DIR=/absolute/path/to/original/text \
TPM_IMAGE_DIR=/absolute/path/to/images \
TPM_BERT_MODEL=/absolute/path/to/bert-base-uncased \
TPM_VIT_MODEL=/absolute/path/to/ViTB-16 \
TPM_QWEN_MODEL=/absolute/path/to/Qwen2.5-VL-3B-Instruct \
TPM_BATCH_SIZE=4 TPM_RUN_DIR=outputs_vlm_gold/tpm_mi_llm_b4_v1 \
bash scripts/run_tpm_llm_v1.sh smoke
```

后续命令保持相同变量。配置、数据、代码与权重绑定到协议；更改配置或实现后用新的 `TPM_RUN_DIR`。脚本会拒绝把不同协议的文件混用。

已完成阶段可复用。底座若在训练中途终止，重新运行会从预训练权重和 epoch 1 重启该种子；不声称恢复 optimizer 状态。旧实验目录不被删除或覆盖。

完整的模型/数据训练尚需在 AutoDL GPU 上运行；仓库 CPU 测试只核对关键计算与数据行为，不等价于实测识别增益。
