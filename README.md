# AGVS-MNER-LLM

面向多图像多模态命名实体识别（MNER）的 LLM 增强实验项目。学生保留 AGVS-MNER 的多图顺序建模、全局视觉锚点、patch 选择与 CRF 解码；LLM 只在训练阶段提供实体到图片的证据标注。

## 当前研究主线

当前主方法是 LGEI（LLM-Guided Evidence Intervention，LLM 引导证据干预）：

1. 视觉语言模型读取训练样本的 gold 实体和最多四张图片，判断每个实体应依赖哪些图片，或输出 `NONE`。教师不能修改实体边界和类型。
2. 缓存读取器只接受与 gold 完全一致、超过置信度阈值且图片可解码的教师实体。
3. 学生在同一批次处理两个视图：原始完整图片视图，以及仅保留教师证据图片的 routed view。只有 `NONE` 证据的样本在 routed view 中关闭视觉分支；没有可靠教师信号的样本保持原视图。
4. routed view 接受 gold CRF 监督，并作为停止梯度的教师约束完整视图的实体 token 分布。两个前向复用同一组 dropout 随机数，路由非空时保持原始视觉强度，使差异只来自图片证据选择。
5. 推理仍使用原始完整视图 AGVS；不需要 LLM、教师缓存、额外路由器或第二次前向。

默认训练目标为：

`L = L_CRF(full) + 0.5 L_CRF(routed) + 0.2 * T^2 * KL(stopgrad(p_routed/T) || p_full/T)`，其中 `T=2`。

旧的 token KD、global image KD、grounding selector/fusion 和 utility gate 保留为消融。已有结果显示 998 条连续样本组成的 utility pilot 存在域偏差，不能作为主方法监督。

## 项目结构

- `models/agvs_mner.py`：AGVS 学生模型与可选教师头
- `modules/dataset.py`：MNER-MI/UNI 数据读取与 WordPiece 标签对齐
- `modules/llm_dataset.py`：教师缓存校验和实体图片目标构建
- `modules/llm_trainer.py`：证据路由、双视图 CRF 与一致性损失
- `modules/trainer.py`：训练、验证、epoch-0 回退和 checkpoint 管理
- `scripts/build_teacher_cache.py`：离线生成教师缓存
- `scripts/audit_llm_grounding.py`：审计缓存覆盖、类别和证据模式
- `run.py`：原始 AGVS 基线入口
- `run_llm.py`：LLM 监督训练入口
- `docs/research_plan.md`：方法、实验协议与消融

## 训练前检查

完整 UNI grounding 缓存应覆盖 10,227/10,229 个训练样本。先运行：

```bash
python scripts/audit_llm_grounding.py \
  --teacher-task grounding \
  --input dataset/text/MNER-UNI_train.txt \
  --cache teacher_cache/siliconflow_qwen3vl32b_grounding/MNER-UNI_train.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --thresholds 0.9
```

`BASELINE` 必须指向同一代码、数据和标签口径下严格基线的 `best_model.pt`。训练入口会先评估并保存 epoch 0；后续训练若未提高验证 F1，基线 checkpoint 会保留。

```bash
export BASELINE=/path/to/strict_baseline/best_model.pt

python -u run_llm.py \
  --dataset UNI \
  --data-path dataset/text \
  --image-path dataset/images \
  --twitter2017-image-path dataset/twitter2017_images \
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased \
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16 \
  --teacher-cache teacher_cache/siliconflow_qwen3vl32b_grounding \
  --teacher-cache-task grounding \
  --init-checkpoint "$BASELINE" \
  --output-dir outputs_llm/lgei_r05_k02_t2_seed1234 \
  --device cuda \
  --num-epochs 5 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 1e-5 \
  --bert-lr 2e-6 \
  --vit-lr 1e-6 \
  --warmup-ratio 0.1 \
  --patience 3 \
  --seed 1234 \
  --image-dropout-prob 0 \
  --llm-token-loss-weight 0 \
  --llm-image-loss-weight 0 \
  --llm-grounding-loss-weight 0 \
  --llm-grounding-confidence 0.9 \
  --llm-grounding-background-ratio 0 \
  --no-llm-grounding-fusion \
  --no-llm-utility-gate \
  --llm-routed-crf-loss-weight 0.5 \
  --llm-consistency-kl-weight 0.2 \
  --llm-consistency-temperature 2 \
  --skip-test
```

`--skip-test` 用于所有开发实验。先依据验证集锁定方法、权重和随机种子协议，最终正式运行时删除该参数，测试集只用于最终报告。

## 必做对照

- Strict AGVS：不使用教师信号。
- Continued training：从同一 baseline 微调，routed CRF 和 KL 均为 0。
- Route only：routed CRF 为 0.5，KL 为 0。
- Consistency only：routed CRF 为 0，KL 为 0.2。
- LGEI：routed CRF 为 0.5，KL 为 0.2。
- Shuffled evidence：在样本内随机打乱 `image_ids`，检验证据内容而非额外训练量。
- Legacy grounding/utility：作为负结果与机制比较。

单种子只有在 LGEI 相对 epoch 0 和 continued-training 对照的验证 F1 均提高至少 0.003 时，才进入三种子复验。论文报告至少三个固定种子的均值、标准差，以及 LOC/MIS/ORG/PER 分项结果。

