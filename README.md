# AGVS-MNER-LLM

面向多图像多模态命名实体识别（MNER）的 LLM 增强实验项目。学生保留 AGVS-MNER 的多图顺序建模、全局视觉锚点、patch 选择与 CRF 解码，并以严格验证集审计决定 LLM 的使用方式。

## 当前主线：V6.1 锁定实体的多模态训练增强

V5.1 受保护适配器稳定地保持了强基线，但没有产生有意义的 validation 增益：
epoch 0 为 `0.8179`，后续最佳仍约为 `0.8179`，未达到预注册的 `0.8209`。
解释拟合损失持续下降而 NER 不提高，说明问题不是主干漂移，而是现有自然语言
解释与 validation 错误之间的监督相关性不足。V5 因此判定为 **no-go**，不运行
test，也不继续调 adapter 权重。

V6 不再蒸馏 LLM 的标签或隐藏表征。初版 V6 让模型返回完整句子和实体 span，
20 次 smoke 只有 7 个合法响应（5 accepted、2 abstained），补跑失败样本仍为
0/13，主要错误是实体 surface 漂移和 span 计算错误，因此判定协议 no-go。
V6.1 改为让 Qwen3-VL 只返回实体之间的 `context_segments`；程序按固定顺序插回
每个 gold 实体的原始 token，自动计算 span 并重建 BIO 标签。从结构上模型不再
能够修改、遗漏、重排或重叠实体。提示词禁止引入新专名；程序还会拒绝上下文中
新增的账号/URL/hashtag、未引用实际图片、改写幅度异常和超过 BERT 长度预算。
安全 abstention 会进入缓存但不会进入
训练集。validation/test 原样复制，部署也不需要 LLM。

### V6.1-A：20 次请求的 smoke（CPU/无卡模式即可）

先选择 1,000 个包含 MIS/ORG 的训练样本，但用 `--limit 20` 只请求前 20 个。
这样 smoke 通过后删除 `--limit 20` 即可在同一缓存上断点续传，不会重复付费。

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM

export LLM_BASE_URL=https://api.siliconflow.cn/v1
export LLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
test -n "$LLM_API_KEY" || { echo "缺少 LLM_API_KEY"; exit 1; }

mkdir -p augmentation_cache/qwen3vl32b_v61 logs
set -o pipefail

python -u scripts/build_llm_augmentation_cache.py \
  --input dataset/text/MNER-UNI_train.txt \
  --output augmentation_cache/qwen3vl32b_v61/MNER-UNI_train.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased \
  --target-types MIS ORG \
  --route-samples 1000 \
  --route-seed 20261027 \
  --max-seq-length 64 \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --timeout 300 \
  --max-tokens 1024 \
  --max-retries 5 \
  --request-delay 1 \
  --max-consecutive-failures 20 \
  --limit 20 \
  --resume \
  2>&1 | tee -a logs/v61_augmentation_smoke.log

python -u scripts/audit_llm_augmentation.py \
  --input dataset/text/MNER-UNI_train.txt \
  --cache augmentation_cache/qwen3vl32b_v61/MNER-UNI_train.jsonl \
  2>&1 | tee logs/v61_augmentation_smoke_audit.log
```

Smoke 只检查协议：应有 20 条合法缓存响应、0 条未恢复的格式失败、至少 10 条
accepted，并且 accepted 同时覆盖 MIS 与 ORG。未通过时先审计拒绝原因，不能直接
扩大请求。它不使用 validation/test，也不用于声称性能提升。

### V6.1-B：扩到 1,000 个训练请求并构建配对数据

Smoke 通过后，原命令保持所有参数不变，仅删除 `--limit 20` 再运行。缓存达到
1,000 条合法响应后重新执行 audit，然后构建 LLM 增强组和等量重复原样本对照组：

```bash
python -u scripts/build_augmented_dataset.py \
  --input-dir dataset/text \
  --cache augmentation_cache/qwen3vl32b_v61/MNER-UNI_train.jsonl \
  --output-dir dataset/v61_llm_aug_seed20261027 \
  --control-output-dir dataset/v61_duplicate_control_seed20261027
```

两个训练集的新增样本数完全一致；对照组新增的是同一批 source sample 的原始
副本，因此可以区分“LLM 上下文多样性”与“更多训练步/类别重采样”。先用 seed
1234、相同超参数分别从头训练，始终加 `--skip-test`。V6.1 只有在 validation 上
同时比原始严格基线和 duplicate control 高至少 `0.003` 时才进入三种子确认；
否则停止。达到 `0.86` 仍是实验目标，不是预先保证的结果。

## 历史 V5.1：受保护的 LLM 证据残差适配器

V4 严格共识已经判定为 **no-go**：其开发集基线 F1 为 `0.828393`，所选
原子编辑 oracle 为 `0.854460`，但三角色严格共识只接受 1 个中性编辑，最终
F1 与基线完全相同；放宽规则均降低 F1。因此 V5 不再让 LLM 直接修改 CRF
输出，也不再追加 API 请求。

V5 复用已经生成的完整 grounding 缓存。只接受同时满足以下条件的训练实体：

1. LLM 的 span/type 与训练集 gold 实体完全一致；
2. `confidence >= 0.90`；
3. `evidence` 非空，且至少引用一张当前可解码的有效图片。

V5.0 直接继续微调整个基线时，证据模型与无证据对照都从 epoch-0 的
`0.8179` 降到约 `0.814`，说明主要问题是强基线的继续训练过拟合；V5.0 相对
同轮对照也没有正增益。V5.1 因此冻结 BERT、ViT、原 AGVS 融合、NER 分类器
和 CRF，并加入零初始化的残差适配器。适配器读取部署时同样可得的文本表示与
AGVS 视觉上下文；LLM 解释只对该适配器的实体表示提供训练监督。零初始化保证
epoch 0 与基线逐值一致，冻结主干阻止基线漂移。推理、validation 和 test 都
不读取解释、LLM 或教师缓存，不增加推理 API。

当前缓存审计得到 `9,838` 个可用实体、覆盖 `7,094` 个训练样本；解释文本
重复率约 `0.5%`。预计算产物同时记录训练文件与 grounding 缓存 SHA-256，
训练时不匹配会直接中止。

### V5-A：一次性预计算解释嵌入

这一步不调用 LLM API。GPU 推荐；CPU 也可运行但更慢。

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM

python -u scripts/build_llm_evidence_embeddings.py \
  --input dataset/text/MNER-UNI_train.txt \
  --teacher-cache teacher_cache/siliconflow_qwen3vl32b_grounding/MNER-UNI_train.jsonl \
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased \
  --output teacher_cache/siliconflow_qwen3vl32b_grounding/MNER-UNI_train.evidence_embeddings.pt \
  --device cuda \
  --batch-size 64 \
  --max-length 96 \
  --confidence-threshold 0.90
```

输出还会生成同名 `.meta.json`。已经生成且 hash 正确时不应重复覆盖；确需
重建时显式追加 `--overwrite`。

### V5-B：只看 validation 的安全微调 pilot

先从严格基线 checkpoint 初始化。下面只启用 V5 损失，显式关闭所有历史
LLM 损失和推理融合，并用 `--skip-test` 封存 test。epoch 0 会先保存原基线，
后续只有 validation F1 严格提高才替换它。

```bash
BASELINE=outputs_llm/strict_baseline_seed1234/uni_20261019_174846/best_model.pt
EVIDENCE=teacher_cache/siliconflow_qwen3vl32b_grounding/MNER-UNI_train.evidence_embeddings.pt

test -f "$BASELINE" || { echo "找不到严格基线"; exit 1; }
test -f "$EVIDENCE" || { echo "找不到 V5 解释嵌入"; exit 1; }

python -u run_llm.py \
  --dataset UNI \
  --data-path dataset/text \
  --image-path dataset/images \
  --twitter2017-image-path dataset/twitter2017_images \
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased \
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16 \
  --teacher-cache teacher_cache/siliconflow_qwen3vl32b_grounding \
  --teacher-cache-task grounding \
  --llm-evidence-embedding-cache "$EVIDENCE" \
  --output-dir outputs_llm/v51_protected_evidence_w002_seed1234 \
  --init-checkpoint "$BASELINE" \
  --device cuda \
  --num-epochs 5 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 1e-4 \
  --bert-lr 0 \
  --vit-lr 0 \
  --warmup-ratio 0.01 \
  --weight-decay 0.01 \
  --patience 3 \
  --seed 1234 \
  --image-dropout-prob 0 \
  --llm-token-loss-weight 0 \
  --llm-image-loss-weight 0 \
  --llm-grounding-loss-weight 0 \
  --llm-routed-crf-loss-weight 0 \
  --llm-consistency-kl-weight 0 \
  --no-llm-grounding-fusion \
  --no-llm-utility-gate \
  --llm-evidence-loss-weight 0.02 \
  --llm-evidence-contrastive-weight 0 \
  --llm-evidence-max-entities 8 \
  --llm-evidence-protected-adapter \
  --llm-evidence-adapter-size 128 \
  --llm-evidence-adapter-l2-weight 0.01 \
  --skip-test \
  2>&1 | tee logs/train_v51_protected_evidence_w002_seed1234.log
```

历史预注册要求同结构的 adapter control，并要求 V5.1 最佳 validation F1 至少
比 epoch 0 和 control 各高 `0.003`。实际实验中 epoch 0 为 `0.8179`，epoch
1--5 分别约为 `0.8175/0.8179/0.8176/0.8175/0.8170`；虽然证据损失从
`0.1448` 降至 `0.0972`，性能门仍未通过。因此该阶段已经停止，不应再运行
control 或 test 来追逐偶然波动。

## 历史 V4：单编辑、多角色保守共识

V3 已完成并判定为 **no-go**：在旧 `meta_evaluation` 上，完整校准器仅取得
`+0.00081` F1，只实现约 `1.55%` 的原子 oracle 增益；LLM 自报置信度也没有
区分有益和有害编辑。该划分的结果已经被查看，因此永久退出 V4 的提示词、
路由、阈值和模型选择。

V4 保留不变的 CRF top-5 和严格原子编辑候选，但每次 API 请求只展示一个
编辑，并让三个互相独立的角色审查：

1. `text_verifier`：只接收文本，绝不上传图片；
2. `visual_verifier`：只有直接可见证据才能支持 baseline 或 edit；
3. `adversarial_critic`：专门寻找“采用该编辑会出错”的反证。

响应不再包含 LLM 自报 `confidence` 或自由 span。只有三角色齐全、critic 返回
`NO_OBJECTION`、两个 verifier 都不支持 baseline，且至少一个 verifier 明确支持
edit 时才接受编辑；任何缺失、冲突、非法输出或 provenance 不匹配都回退 CRF
top-1。每个样本最多审查并应用一个最低 margin 编辑。

### V4-A：先用 20 条 meta-train 调通协议

这一步共 `20 × 3 = 60` 次请求，只检查格式、图片引用和断点续传，不用于
选择性能阈值。请求缓存不读取 gold，也不需要 GPU。

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM

test -n "$LLM_BASE_URL" || { echo "缺少 LLM_BASE_URL"; exit 1; }
test -n "$LLM_API_KEY"  || { echo "缺少 LLM_API_KEY"; exit 1; }
test -n "$LLM_MODEL"    || { echo "缺少 LLM_MODEL"; exit 1; }

mkdir -p atomic_cache/qwen3vl32b_v4 logs
set -o pipefail

python -u scripts/build_llm_atomic_multirole_cache.py \
  --candidates atomic_cache/qwen3vl32b_v3/meta_candidates.jsonl \
  --manifest dataset/atomic_preflight_seed20261022/atomic_preflight_manifest.json \
  --roles meta_train \
  --route-samples 20 \
  --max-edits-per-sample 1 \
  --scope all \
  --output atomic_cache/qwen3vl32b_v4/meta_train_smoke_reviews.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --timeout 300 \
  --max-tokens 512 \
  --max-retries 5 \
  --request-delay 1 \
  --resume \
  2>&1 | tee -a logs/atomic_multirole_meta_train_smoke.log

wc -l atomic_cache/qwen3vl32b_v4/meta_train_smoke_reviews.jsonl
```

正常完成应为 60 行。中断后原命令再次执行即可从复合主键继续。

### V4-B：冻结协议后运行 100 条 meta-calibration pilot

这一步固定选择 `meta_calibration` 中原子 margin 最低的 100 个不同样本，
每个样本一个编辑、三个角色，共 300 次请求。评估脚本是唯一读取这些开发集
gold 的 V4 程序。

```bash
python -u scripts/build_llm_atomic_multirole_cache.py \
  --candidates atomic_cache/qwen3vl32b_v3/meta_candidates.jsonl \
  --manifest dataset/atomic_preflight_seed20261022/atomic_preflight_manifest.json \
  --roles meta_calibration \
  --route-samples 100 \
  --max-edits-per-sample 1 \
  --scope all \
  --output atomic_cache/qwen3vl32b_v4/meta_calibration_reviews.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --timeout 300 \
  --max-tokens 512 \
  --max-retries 5 \
  --request-delay 1 \
  --resume \
  2>&1 | tee -a logs/atomic_multirole_meta_calibration.log

python -u scripts/evaluate_atomic_multirole_pilot.py \
  --input dataset/atomic_preflight_seed20261022/MNER-UNI_meta.txt \
  --candidates atomic_cache/qwen3vl32b_v3/meta_candidates.jsonl \
  --reviews atomic_cache/qwen3vl32b_v4/meta_calibration_reviews.jsonl \
  --manifest dataset/atomic_preflight_seed20261022/atomic_preflight_manifest.json \
  --meta-role meta_calibration \
  --output atomic_cache/qwen3vl32b_v4/meta_calibration_evaluation.json \
  2>&1 | tee logs/evaluate_atomic_multirole_meta_calibration.log
```

Pilot 预先锁定的通过条件为：完整三角色响应覆盖率至少 `95%`；接受至少 20
个编辑；接受编辑中 beneficial 至少 `80%`、harmful 不超过 `10%`；正确
top-1 保持率至少 `98%`；实现所选编辑 oracle 增益至少 `50%`；总 F1 至少
提高 `0.003`。任一条件失败就停止 V4，不扩大 API 请求，也不访问旧
`meta_evaluation` 或官方 validation/test。通过后才创建全新的、从未查看的
holdout，并从头训练不接触该 holdout 的基础模型进行正式确认。

## 历史 V3：原子编辑审查与监督校准

CRF top-5 的正确序列覆盖充分，但让 Qwen3-VL 直接选择整条序列会破坏大量本来正确的 top-1。V3 将每个候选分解为严格的单实体原子编辑：删除误检实体、补回漏检实体、同边界类型替换、同类型重叠边界替换。Qwen3-VL 只独立审查每个固定编辑，并输出结构化证据；它不能生成实体、改写候选或直接决定最终序列。一个小型监督校准器结合 CRF margin、编辑结构和 LLM 结构化审查特征，最多接受一个编辑，否则回退 CRF top-1。

严格验证集审计得到：基线 F1 `0.8179`，全部 top-5 oracle `0.8977`，只允许上述四类单一原子编辑时 oracle 仍为 `0.8867`；最低 margin 的 800 条路由样本 oracle 为 `0.8686`。因此候选空间足以超过 `0.86`，实际模型仍须在无泄漏 preflight 中实现至少 `60%` 原子 oracle 增益，才进入正式验证。

### 1. 创建无泄漏 preflight 划分

只拆分原训练集：72% 训练基础模型，8% 选择基础模型，20% 生成基础模型未见过的 meta 预测。meta 再固定拆为 60% 校准器训练、20% policy 校准、20% 独立 go/no-go 评估。官方 validation/test 不参与这个阶段。

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM

python -u scripts/create_atomic_preflight_split.py \
  --input dataset/text/MNER-UNI_train.txt \
  --output-dir dataset/atomic_preflight_seed20261022 \
  --dataset UNI \
  --seed 20261022
```

### 2. 训练只在 72/8 划分上选择的基础模型

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u run.py \
  --dataset UNI \
  --data-path dataset/atomic_preflight_seed20261022 \
  --image-path dataset/images \
  --twitter2017-image-path dataset/twitter2017_images \
  --bert-model /root/autodl-tmp/pretrained_models/bert-base-uncased \
  --vit-model /root/autodl-tmp/pretrained_models/ViTB-16 \
  --output-dir outputs_atomic/preflight_base_seed20261022 \
  --device cuda \
  --num-epochs 15 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 5e-5 \
  --bert-lr 1e-5 \
  --vit-lr 5e-6 \
  --warmup-ratio 0.01 \
  --weight-decay 0.01 \
  --patience 3 \
  --seed 20261022 \
  --skip-test

BASE_PRE=$(ls -dt outputs_atomic/preflight_base_seed20261022/uni_* | head -1)/best_model.pt
test -f "$BASE_PRE" || { echo "找不到 preflight checkpoint"; exit 1; }
echo "$BASE_PRE"
```

### 3. 导出 meta 原子候选并请求 1,000 条 LLM 审查

候选导出和 LLM 请求均不读取 gold label。`--route-queries 1000` 只审查原子 margin 最低的 1,000 条样本；缓存支持 `--resume`。

```bash
mkdir -p atomic_cache/qwen3vl32b_v3 logs

python -u scripts/export_atomic_edit_candidates.py \
  --checkpoint "$BASE_PRE" \
  --input-file dataset/atomic_preflight_seed20261022/MNER-UNI_meta.txt \
  --source-ids dataset/atomic_preflight_seed20261022/MNER-UNI_meta.txt.source_ids.json \
  --output atomic_cache/qwen3vl32b_v3/meta_candidates.jsonl \
  --split-label meta \
  --k 5 \
  --oversample-factor 20 \
  --route-queries 1000 \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  2>&1 | tee logs/export_atomic_meta.log

test -n "$LLM_BASE_URL" || { echo "缺少 LLM_BASE_URL"; exit 1; }
test -n "$LLM_API_KEY"  || { echo "缺少 LLM_API_KEY"; exit 1; }
test -n "$LLM_MODEL"    || { echo "缺少 LLM_MODEL"; exit 1; }

python -u scripts/build_llm_atomic_review_cache.py \
  --candidates atomic_cache/qwen3vl32b_v3/meta_candidates.jsonl \
  --output atomic_cache/qwen3vl32b_v3/meta_reviews.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --timeout 300 \
  --max-tokens 1024 \
  --max-retries 5 \
  --request-delay 1 \
  --scope routed \
  --resume \
  2>&1 | tee -a logs/atomic_review_meta.log
```

### 4. 构建监督样本，训练 full 与 structure-only 对照

下面的 gold-aware builder 是唯一读取 meta gold 的阶段。Dinkelbach `q` 只用 `meta_train` 的 top-1 计数估计；`meta_calibration` 只选择阈值，`meta_evaluation` 只做最终 preflight 评估。

```bash
python -u scripts/build_atomic_calibrator_data.py \
  --input dataset/atomic_preflight_seed20261022/MNER-UNI_meta.txt \
  --candidates atomic_cache/qwen3vl32b_v3/meta_candidates.jsonl \
  --reviews atomic_cache/qwen3vl32b_v3/meta_reviews.jsonl \
  --manifest dataset/atomic_preflight_seed20261022/atomic_preflight_manifest.json \
  --output atomic_cache/qwen3vl32b_v3/calibrator_examples.jsonl

python -u scripts/train_atomic_calibrator.py \
  --data atomic_cache/qwen3vl32b_v3/calibrator_examples.jsonl \
  --output atomic_cache/qwen3vl32b_v3/full_calibrator.pt \
  --policy-output atomic_cache/qwen3vl32b_v3/full_policy.json \
  --feature-mode full \
  --epochs 300 \
  --minimum-top1-preservation 0.98 \
  --seed 20261022 \
  --device cpu

python -u scripts/train_atomic_calibrator.py \
  --data atomic_cache/qwen3vl32b_v3/calibrator_examples.jsonl \
  --output atomic_cache/qwen3vl32b_v3/structure_calibrator.pt \
  --policy-output atomic_cache/qwen3vl32b_v3/structure_policy.json \
  --feature-mode structure-only \
  --epochs 300 \
  --minimum-top1-preservation 0.98 \
  --seed 20261022 \
  --device cpu
```

### 5. 只在独立 meta-evaluation 上作 go/no-go 决策

```bash
python -u scripts/evaluate_atomic_calibrator.py \
  --data atomic_cache/qwen3vl32b_v3/calibrator_examples.jsonl \
  --checkpoint atomic_cache/qwen3vl32b_v3/full_calibrator.pt \
  --policy atomic_cache/qwen3vl32b_v3/full_policy.json \
  --structure-checkpoint atomic_cache/qwen3vl32b_v3/structure_calibrator.pt \
  --structure-policy atomic_cache/qwen3vl32b_v3/structure_policy.json \
  --role meta_evaluation \
  --device cpu \
  --output atomic_cache/qwen3vl32b_v3/meta_evaluation.json \
  2>&1 | tee logs/evaluate_atomic_meta.log
```

`go_no_go.passed=true` 同时要求：实现至少 `60%` 的本路由原子 oracle 增益、保留至少 `98%` 原本完全正确的 top-1、接受的有益编辑多于有害编辑，并且 full 模型比独立训练的 structure-only 对照高至少 `0.003` F1。未通过时停止，不访问官方 validation/test，也不宣称达到 `0.86`。

## 历史 V2：直接 top-k 重排

已完成的验证集图片消融为：完整图片 `0.8179`、关闭图片 `0.8152`、按相同图片掩码打乱图片 `0.8170`。正确图文对应相对随机图片只增加约 `0.00087` F1，因此当前 AGVS checkpoint 的主要视觉增益来自通用视觉偏置，而不是图片语义。继续调 grounding fusion、utility gate 或 LGEI 不再作为性能主线，也不需要继续生成教师缓存。

CRF top-k 上限审计已经通过。严格基线验证 F1 为 `0.8179`，oracle@3 为 `0.8803`、oracle@5 为 `0.8977`、oracle@10 为 `0.9128`。这说明正确序列经常已经存在于前五个候选中，当前主要问题是排序。第一版“隐藏全部 CRF 先验、让 Qwen3-VL 直接五选一”的 200 条 pilot 未通过：整体只实现 `6.3%` 的可用 oracle 增益；随机对照组 F1 从 `0.8516` 降到 `0.8148`。低 margin 组有正增益（`0.5783 -> 0.6190`），但破坏正确 top-1 的比例过高。

第二版改为 **MLLM 基线锚定残差验证**：只对最低 CRF margin 样本请求 Qwen3-VL，明确标出 CRF top-1 为默认基线；只有模型指出具体的边界、类型、漏检或误检错误时才能选择一个固定替代候选。候选仍不可修改，CRF 分数和其余候选 rank 仍不进入提示词。

整个流程分为三个互相隔离的阶段：候选导出不读取 `label`，LLM 请求只读取候选、文本与图片，最后的评估脚本才读取 gold。候选使用不透明 ID；提示只公开哪个 ID 是基线，不公开数值分数或替代候选的 rank。缺失/非法响应、低置信度、`uncertain=true`、没有具体基线错误类型或决策 provenance 不匹配时回退到 CRF top-1；候选 artifact 自身的 hash/provenance 损坏时直接中止。

先在验证集导出全部 top-5 候选，只对 100 个最低 CRF margin 样本执行第二版 pilot：

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -o pipefail

test -n "$LLM_BASE_URL" || { echo "缺少 LLM_BASE_URL"; exit 1; }
test -n "$LLM_API_KEY"  || { echo "缺少 LLM_API_KEY"; exit 1; }
test -n "$LLM_MODEL"    || { echo "缺少 LLM_MODEL"; exit 1; }

BASELINE=/path/to/strict_baseline/best_model.pt
test -f "$BASELINE" || { echo "找不到基线 checkpoint"; exit 1; }

mkdir -p rerank_cache/qwen3vl32b_anchor_v2 logs

python -u scripts/export_crf_rerank_candidates.py \
  --checkpoint "$BASELINE" \
  --split val \
  --k 5 \
  --oversample-factor 20 \
  --uncertainty-queries 100 \
  --random-control-queries 0 \
  --route-seed 20261020 \
  --output rerank_cache/qwen3vl32b_anchor_v2/val_candidates.jsonl \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  2>&1 | tee logs/export_rerank_val.log

python -u scripts/build_llm_rerank_cache.py \
  --candidates rerank_cache/qwen3vl32b_anchor_v2/val_candidates.jsonl \
  --output rerank_cache/qwen3vl32b_anchor_v2/val_decisions.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --scope pilot \
  --resume \
  2>&1 | tee -a logs/rerank_pilot.log

python -u scripts/evaluate_llm_reranker.py \
  --input dataset/text/MNER-UNI_val.txt \
  --candidates rerank_cache/qwen3vl32b_anchor_v2/val_candidates.jsonl \
  --decisions rerank_cache/qwen3vl32b_anchor_v2/val_decisions.jsonl \
  --confidence-thresholds 0.80 0.85 0.90 0.95 \
  --output rerank_cache/qwen3vl32b_anchor_v2/pilot_metrics.json \
  2>&1 | tee logs/evaluate_rerank_anchor_v2_pilot.log
```

第二版 pilot 的 go/no-go 门槛仍为：实现这 100 条低 margin 样本 oracle 增益的至少 `60%`，并保留至少 `95%` 原本 top-1 完全正确的样本。在达到门槛之前不要补齐验证集、冻结 policy 或访问测试集。后续正式路由规则必须只由验证集确定，并在测试前冻结。

选择一个已经通过门槛的阈值（下面以 `0.90` 为例），重新执行评估并写出不可变 policy：

```bash
FROZEN_THRESHOLD=0.90

python -u scripts/evaluate_llm_reranker.py \
  --input dataset/text/MNER-UNI_val.txt \
  --candidates rerank_cache/qwen3vl32b_top5/val_candidates.jsonl \
  --decisions rerank_cache/qwen3vl32b_top5/val_decisions.jsonl \
  --confidence-thresholds "$FROZEN_THRESHOLD" \
  --freeze-policy rerank_cache/qwen3vl32b_top5/frozen_policy.json \
  --frozen-threshold "$FROZEN_THRESHOLD" \
  2>&1 | tee logs/freeze_rerank_policy.log

python -u scripts/build_llm_rerank_cache.py \
  --candidates rerank_cache/qwen3vl32b_top5/val_candidates.jsonl \
  --output rerank_cache/qwen3vl32b_top5/val_decisions.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --scope all \
  --policy rerank_cache/qwen3vl32b_top5/frozen_policy.json \
  --resume \
  2>&1 | tee -a logs/rerank_val_all.log

python -u scripts/evaluate_llm_reranker.py \
  --input dataset/text/MNER-UNI_val.txt \
  --candidates rerank_cache/qwen3vl32b_top5/val_candidates.jsonl \
  --decisions rerank_cache/qwen3vl32b_top5/val_decisions.jsonl \
  --scope all \
  --policy rerank_cache/qwen3vl32b_top5/frozen_policy.json \
  --output rerank_cache/qwen3vl32b_top5/val_full_metrics.json \
  2>&1 | tee logs/evaluate_rerank_val_all.log
```

只有完整验证集仍支持方案时才执行一次正式测试。三个测试阶段都要求显式解锁；builder/evaluator 强制 `scope=all`，evaluator 只使用 policy 中唯一的冻结阈值：

```bash
python -u scripts/export_crf_rerank_candidates.py \
  --checkpoint "$BASELINE" \
  --split test \
  --allow-test \
  --k 5 \
  --oversample-factor 20 \
  --output rerank_cache/qwen3vl32b_top5/test_candidates.jsonl \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  2>&1 | tee logs/export_rerank_test.log

python -u scripts/build_llm_rerank_cache.py \
  --candidates rerank_cache/qwen3vl32b_top5/test_candidates.jsonl \
  --output rerank_cache/qwen3vl32b_top5/test_decisions.jsonl \
  --image-root dataset/images \
  --twitter-image-root dataset/twitter2017_images \
  --base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --model "$LLM_MODEL" \
  --scope all \
  --policy rerank_cache/qwen3vl32b_top5/frozen_policy.json \
  --allow-test \
  --resume \
  2>&1 | tee logs/rerank_test.log

python -u scripts/evaluate_llm_reranker.py \
  --input dataset/text/MNER-UNI_test.txt \
  --candidates rerank_cache/qwen3vl32b_top5/test_candidates.jsonl \
  --decisions rerank_cache/qwen3vl32b_top5/test_decisions.jsonl \
  --scope all \
  --policy rerank_cache/qwen3vl32b_top5/frozen_policy.json \
  --allow-test \
  --output rerank_cache/qwen3vl32b_top5/test_metrics.json \
  2>&1 | tee logs/evaluate_rerank_test.log
```

## 历史 LGEI 实验

LGEI（LLM-Guided Evidence Intervention，LLM 引导证据干预）的设计为：

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
- `modules/crf_topk.py`：CRF top-k 解码与精确语料级 micro-F1 oracle
- `modules/crf_candidates.py`：无 gold 的结构约束、WordPiece 到词级候选转换
- `modules/crf_edit_decomposition.py`：将 top-k 差异限制为四类严格单实体编辑
- `modules/reranker.py`：不透明候选、pilot 路由、provenance 与安全回退策略
- `modules/atomic_calibrator.py`：原子编辑特征、监督校准器、保守 policy 与 oracle
- `modules/trainer.py`：训练、验证、epoch-0 回退和 checkpoint 管理
- `scripts/build_teacher_cache.py`：离线生成教师缓存
- `scripts/audit_llm_grounding.py`：审计缓存覆盖、类别和证据模式
- `scripts/evaluate_crf_topk_oracle.py`：评估受约束 LLM 重排的候选空间上限
- `scripts/export_crf_rerank_candidates.py`：导出不含 gold 的 top-5 候选
- `scripts/build_llm_rerank_cache.py`：生成可断点续传的 Qwen 候选选择缓存
- `scripts/evaluate_llm_reranker.py`：唯一读取 gold 的候选重排评估阶段
- `scripts/create_atomic_preflight_split.py`：建立 72/8/20 和 60/20/20 无泄漏划分
- `scripts/export_atomic_edit_candidates.py`：无 gold 导出 top-5 严格原子编辑
- `scripts/build_llm_atomic_review_cache.py`：生成可续传的独立 LLM 编辑审查
- `scripts/build_atomic_calibrator_data.py`：唯一读取 meta gold 并构建监督目标的阶段
- `scripts/train_atomic_calibrator.py`：训练校准器并只在 meta-calibration 冻结 policy
- `scripts/evaluate_atomic_calibrator.py`：在 meta-evaluation 检验 full、结构对照和扰动对照
- `run.py`：原始 AGVS 基线入口
- `run_llm.py`：LLM 监督训练入口
- `docs/research_plan.md`：方法、实验协议与消融

## 历史 LGEI 训练前检查

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

