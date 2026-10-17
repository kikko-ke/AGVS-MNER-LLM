# 研究计划：G-MEG（Gold-conditioned MLLM Evidence Gating）

## 动机

现有 Qwen3-VL 教师在训练缓存上的实体 F1 只有 0.5060。直接对全部 token 做 KL 蒸馏会把错误边界、错误类型和过多的非实体预测传给学生；已有实验中 Full KD（0.8318）没有超过同配置的无教师基线（0.8324）。

LLM 仍然包含数据集标注没有提供的信息：一个正确实体由哪张图支撑。因此主方法不再让 LLM 重复教授已有的 BIO 标签。正式缓存使用 `--teacher-task grounding`，把训练集 gold 实体交给 VLM，只让它补充跨图像证据；旧缓存也可通过 gold-consistency gate 用于先导实验。

## 方法

对训练集中的教师实体 `e=(start,end,type,confidence,image_ids)` 定义可靠性门：

`z_e = 1[教师 span/type 与 gold 实体完全一致] * 1[confidence >= tau] * 1[所选证据槽位有效]`

只有 `z_e=1` 的实体进入辅助任务。教师目标是五路分布 `q_e=[img0,img1,img2,img3,NONE]`；空 `image_ids` 对应 NONE，明确教授学生何时不应使用视觉信息。学生把实体中每个词的首个 WordPiece 表征 `h_i` 与四张图的原始 ViT CLS 表征 `v_m` 投影到同一空间，避免从图像槽位嵌入中走捷径：

`p(m|i) = softmax([cos(W_t h_i, W_v v_0..3), s_NONE] / T)`

预测的图像概率直接进入 NER 决策路径：

`g_i = sum_m p(m|i) * v_m`

`h'_i = LayerNorm(h_i + visual_strength * gate_i * (AGVS_i + tanh(alpha) * W_g g_i))`

其中 `p(NONE)` 自动吸收不需要视觉证据的概率质量。为解决第一版实验中共享辅助梯度造成的负迁移，正式模型增加两项保护：非实体词接受总质量受限的弱 NONE 监督，使背景 token 在推理时关闭视觉残差；教师 grounding 损失默认使用 detach 后的 BERT/ViT 特征，只更新证据选择头。视觉残差系数从 0 初始化，NER 主损失可在证据有用时自行增大它。辅助头不只是训练时旁路，推理时学生仍会自行预测实体相关图片并控制视觉融合。

Grounding 损失为：

`L_ground = - sum_i w_i * sum_m q_i,m log p(m|i) / sum_i w_i`

同一实体的 token 权重之和等于“教师置信度 × 类别权重”，避免多词实体仅因长度获得更大损失。默认类别权重按训练集实体频次的逆平方根计算并截断到 2.0，减少 PER 对辅助任务的支配；必须报告关闭该权重的消融。

最终目标：

`L = L_CRF + lambda_ground * L_ground`

默认 `tau=0.9`、`T=0.1`、`lambda_ground=0.1`，背景 NONE 总权重为实体监督质量的 0.25。原始 token KL 与全局 image KL 默认权重均为 0，仅保留作消融。门控使用的 gold 只来自训练集，验证与测试不产生教师损失；部署时不调用 LLM。

## 核心贡献表述

1. 提出可靠性门控的多模态 LLM 教师，只提取与人工标注一致实体的视觉证据，隔离生成式教师的标签噪声。
2. 提出含 NONE 门的实体级多图 grounding 目标，并让预测分布直接控制 NER 视觉融合，把 LLM 的图片证据转化为学生的实际决策路径。
3. 教师完全离线，学生推理速度、API 成本和数据隐私边界与原 AGVS 一致。

论文中应在实验得到支持后再写“提升性能”；在此之前可主张方法解决了 noisy teacher supervision，并通过消融验证机制。

## 必做实验

| 实验 | token KL | global image KL | gated grounding | 目的 |
| --- | ---: | ---: | ---: | --- |
| AGVS | 0 | 0 | 0 | 严格基线 |
| Raw token KD | 0.5 | 0 | 0 | 复现噪声标签影响 |
| Global image KD | 0 | 0.2 | 0 | 样本级图像监督 |
| Proposed | 0 | 0 | 0.1 | 主方法 |
| Proposed + raw KD | 0.1 | 0 | 0.1 | 使用独立 mner cache，判断 token KD 是否仍有互补性 |

主方法先固定同一 seed、batch size 和学习率，搜索 `lambda_ground={0.02,0.05,0.1,0.2}` 与 `tau={0.5,0.7,0.8}`。选择只能依据验证集 F1，测试集只在配置确定后运行一次。至少报告 3 个随机种子的均值和标准差。

附加消融：自由 MNER 缓存 vs gold-conditioned grounding、去掉 confidence gate、使用 `--no-llm-grounding-fusion` 让辅助头不进入融合、使用 `--no-llm-grounding-detach-backbone` 恢复共享梯度、将 background ratio 设为 0、去掉 NONE、把实体级 target 换成全局 image score、随机打乱教师 image_ids、固定 slot-0、CLIP 图文相似度教师，以及按实体类型和图片数量分组报告 F1。正式实验还应加入图片顺序置换一致性，并至少人工核查 100–200 个实体的图片证据。

## 已观察到的负结果

在 UNI、seed 1234、batch size 4 上，第一版直接融合取得测试 F1 0.8286；关闭融合但保留共享 grounding 梯度取得 0.8260，均未超过 0.8318 基线。该结果表明问题不只来自融合残差，grounding 辅助梯度本身也会使共享表示偏离 NER 目标。因此后续主实验必须使用梯度隔离、背景 NONE 门控和零初始化残差，并将旧实现作为负迁移消融完整报告。

## 训练前审计

先运行：

```bash
python scripts/audit_llm_grounding.py \
  --teacher-task grounding \
  --input dataset/text/MNER-UNI_train.txt \
  --cache teacher_cache/qwen_grounding/MNER-UNI_train.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images
```

若某阈值下可用实体过少，优先降低阈值；不要重新打开 raw token KD。审计应记录可用实体数、覆盖样本数、类别分布和过滤原因，作为论文教师质量分析的一部分。
