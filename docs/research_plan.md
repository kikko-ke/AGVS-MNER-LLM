# 研究计划：MLLM 原子编辑审查与保守校准

## 当前主方法：V4 单编辑、多角色共识

V3 在已经锁定的 meta evaluation 上未通过：基线 F1 `0.83891`，full 为
`0.83972`，只实现 `1.55%` 的原子 oracle 增益；full 相对 structure-only
仅高 `0.00076`。因此旧 `meta_evaluation` 已经暴露并永久封存，不能再参与
V4 的提示词、路由、投票、阈值或模型选择。

V4 对每个请求只展示一个固定原子编辑，分别调用互相不可见输出的
`text_verifier`、`visual_verifier` 和 `adversarial_critic`。输出只包含严格
枚举的 verdict、证据目标、reason code 和有效图片槽位；删除 V3 中失真的
自报 confidence，也不允许模型生成自由 span 或 BIO 序列。文本角色不会上传
图片；视觉角色的支持结论必须引用实际可解码图片；critic 的具体反对意见拥有
否决权。

固定共识为：三个角色完整且 provenance 一致；critic=`NO_OBJECTION`；文本和
视觉角色均不支持 baseline；二者至少一个明确支持 edit。否则一律回退 CRF
top-1。每个样本最多选择一个最低 margin 编辑，任何缓存缺失、格式错误、指纹
失配或角色冲突也回退 top-1。

开发分两步：先在 `meta_train` 固定 20 个样本调通 schema（60 次请求），然后
冻结协议，在 `meta_calibration` 按无 gold 的最低 margin 规则固定选择 100 个
不同样本（300 次请求）并只评估一次。通过条件预先锁定为：角色完整覆盖率
`>=0.95`、accepted `>=20`、beneficial precision `>=0.80`、harmful rate
`<=0.10`、top-1 exact preservation `>=0.98`、normalized selected-edit oracle
gain `>=0.50`、语料级 F1 gain `>=0.003`。失败即停止；不得在这 100 条结果上
反复修改 prompt 后仍称其为独立验证。

若 pilot 通过，正式证据必须重新创建未见 holdout：排除旧 meta 的全部 source
ID，用新 seed 在原 base pool 中预先抽样并保存 ID/hash；旧 checkpoint 因见过
这些样本不能用于正式结论，必须从头训练一个排除新 holdout 的基础模型。所有
V4 协议和策略冻结后才打开新 holdout 的 gold。正式门槛恢复 normalized oracle
gain `>=0.60`，并增加独立训练的 structure-only 对照，要求 full 至少高
`0.003` F1。官方 validation/test 已在历史开发中多次查看，只能作为补充报告。

## 历史 V3：原子编辑审查

V2 整序列五选一未达到 go/no-go。V3 保留 top-5 候选覆盖，只提取相对 CRF top-1 的四类严格单实体修改：删除误检、添加漏检、同边界类型修改、同类型重叠边界修改。Qwen3-VL 独立输出每个固定编辑的 `KEEP_BASELINE/APPLY_EDIT/ABSTAIN`、文本支持、视觉支持、引用图片和决定性 span。LLM 不生成 BIO 序列，也不直接作最终选择。

监督校准器使用 CRF 结构特征和 LLM 结构化字段，回归语料级 F1 的 Dinkelbach 编辑增益，同时预测 beneficial/neutral/harmful。推理时每条样本最多接受一个编辑；schema、fingerprint、数据 hash 或预测异常均回退 top-1。训练目标和 policy 阈值分别来自 `meta_train` 与 `meta_calibration`，最终 go/no-go 只在 `meta_evaluation` 执行。

原训练集固定拆为 72% `base_train`、8% `base_dev`、20% `meta`，基础模型不接触 meta；meta 再按 60/20/20 拆分。官方 validation/test 在 preflight 通过前保持封存。严格单编辑 oracle 为 `0.8867`，最低 margin 800 条路由 oracle 为 `0.8686`，所以主方案的 go/no-go 为：至少实现 `60%` 原子 oracle 增益、保持 `98%` 原本完全正确的 top-1、有益编辑多于有害编辑、full 比独立训练的 structure-only 对照至少高 `0.003` F1。完整操作命令见 README。

## 历史 V2：整序列候选重排

严格基线验证 F1 为 `0.8179`。CRF 候选审计得到 oracle@3 `0.8803`、oracle@5 `0.8977`、oracle@10 `0.9128`；top-1 的 483 个实体集合错误样本中，top-5 可完整恢复 172 个。K=5 在候选复杂度和理论余量之间最合适：达到 `0.86` 需要实现约 `53%` 的完整 top-5 oracle 增益。

第一版盲重排 pilot 没有通过：200 条样本的最佳整体增益只有 `+0.00125` F1，归一化 oracle 增益 `6.3%`；随机对照组明显负增益。低 margin 组从 `0.5783` 提升到 `0.6190`，说明无标签 CRF margin 可以定位可干预样本，但直接五选一会破坏过多正确 top-1。

第二版采用 MLLM 基线锚定残差验证。AGVS 生成五个合法 BIO 序列，提示明确公开哪个不透明 ID 是 CRF top-1 基线，但不公开 CRF 分数和替代候选 rank。Qwen3-VL 默认保留基线；选择替代项时必须声明一个具体的 boundary/entity_type/missing_entity/spurious_entity 错误。缺失/非法响应、越界候选、决策 hash 不匹配、`uncertain=true` 或置信度不足时均回退 top-1。

为阻止标签泄漏，流水线物理拆分为三个程序：

1. `export_crf_rerank_candidates.py` 使用 inference-only dataset，只访问 `text` 和 `images`，导出候选、checkpoint hash 与只覆盖有序 `text+images` 投影的输入 hash；改变 gold 标签不会改变该 hash 或 pilot 抽样。
2. `build_llm_rerank_cache.py` 只读取候选 artifact 和图片，调用 Qwen3-VL；它不导入 gold evaluator。
3. `evaluate_llm_reranker.py` 才读取 `label`，计算整体 micro F1、路由子集 oracle、实际获得的 oracle 增益比例和 top-1 保持率。

第二版先固定做 100 次验证集 pilot，只选择无标签 top1-top2 margin 最低的样本，不再请求已经证实有害的随机对照组。go/no-go 条件是实现这 100 条样本 oracle 增益至少 `60%`，且保持至少 `95%` 原本完全正确的 top-1。在通过前不补齐验证集，也不访问测试集；通过后再用验证集冻结 margin 路由规则、prompt、LLM 型号和置信度阈值。

## 历史训练期方案与问题诊断

### 问题诊断

严格修复 WordPiece 对齐和评估口径后，同配置无教师 AGVS 在 UNI 上取得约 0.8184 验证 F1 和 0.8324 测试 F1。此前记录的 0.8257 验证 F1 / 0.8318 测试 F1 属于 `token KD=0.5, image KD=0.2` 的 Full KD 实验，并非严格基线。已有三类 LLM 方案没有形成稳定增益：

- 原始教师实体预测 F1 约 0.506，直接 token KD 会传播边界与类型错误。
- grounding selector 主要学习“图片是否描绘实体”，不等于图片是否帮助 NER；其融合和辅助梯度实验低于基线。
- 998 条 utility pilot 是连续 sample ID 0--999，完全没有覆盖 Twitter2017 域；当前 gate 又只消费 `p(NONE)`，没有直接使用教师 `image_ids`。其 0.8321 测试 F1 与 0.8318 基线相当，但验证 F1 更低，不能视为增益。

现有完整 grounding 缓存覆盖 10,227/10,229 个训练样本和几乎全部 gold 实体，且包含具体图片编号。LGEI 直接利用这份覆盖完整的训练期特权信息，不再训练弱测试时路由器。

### 历史方法：LGEI

教师实体为 `e=(start,end,type,confidence,image_ids)`。可靠性门只接受满足以下条件的实体：

`z_e = 1[span/type 与 gold 完全一致] * 1[confidence >= tau] * 1[图片槽位有效]`

对样本 `x`，将所有可靠实体的 `image_ids` 取并集得到证据集合 `S(x)`：

- `S(x)` 非空：routed view 仅保留证据图片；
- 存在可靠实体但所有实体均为 `NONE`：routed view 关闭视觉分支；
- 没有可靠实体：不实施干预，继续使用完整视图。

完整视图和 routed view 使用同一个学生参数。两个前向保持 batch 形状相同并复用 CPU/CUDA RNG 状态，使 dropout 与 image dropout 的随机变量一致。对于非空 route，视觉融合强度沿用完整视图的有效图片比例；因此干预改变的是证据内容，不混入图片数量造成的额外幅度衰减。

设 `p_f`、`p_r` 为完整视图和 routed view 的 token emissions，`M_e` 为被教师覆盖且 route 确实发生变化的实体首 WordPiece：

`L = L_CRF(full) + lambda_r L_CRF(routed) + lambda_k T^2 sum_(i in M_e) w_i KL(stopgrad(p_r_i/T) || p_f_i/T)`

默认 `tau=0.9`、`lambda_r=0.5`、`lambda_k=0.2`、`T=2`。routed CRF 让模型在去除干扰图后学习 gold 序列；KL 把这一更干净视图的实体判断蒸馏回完整视图。推理只执行完整视图一次，不需要 LLM、缓存或路由器。

该实现是样本级 evidence intervention：多实体样本共享教师证据图片的并集。论文不得将它表述为严格逐实体前向路由。

### 安全微调

主实验从严格基线 `best_model.pt` 开始，以较低学习率微调 3--5 个 epoch。训练器在任何更新前先评估 epoch 0 并保存 checkpoint；后续验证 F1 没有严格提高时不会用同分或更差模型覆盖基线。

推荐参数：

- other LR `1e-5`
- BERT LR `2e-6`
- ViT LR `1e-6`
- image dropout `0`
- patience `3`
- routed CRF `0.5`
- consistency KL `0.2`
- temperature `2`

旧 `grounding selector loss` 必须设为 0，grounding fusion 和 utility gate 必须关闭；否则实验同时混入已知负结果。代码会拒绝同时启用 routed intervention 与旧 grounding loss。

### 开发与测试协议

所有参数选择必须带 `--skip-test`，只看验证集。单种子完整方案必须同时满足：

1. epoch 0 验证 F1 与严格基线相差不超过 0.0005；
2. 相对 epoch 0 提高至少 0.003；
3. 相对同学习率、同 epoch 的 continued-training 对照提高至少 0.003。

通过后固定设置，使用种子 `42、1234、2026`：平均提升至少 0.003，至少两个种子为正，任一种子下降不超过 0.003。全部配置冻结后才去掉 `--skip-test`，运行正式测试。

当前测试集已经在早期开发中被多次查看，因此论文应如实说明开发历史。若需要更严格的无泄漏证据，应增加封存的新划分或外部数据集。

### 实验矩阵

| 实验 | routed CRF | KL | 教师 route | 目的 |
| --- | ---: | ---: | --- | --- |
| Strict AGVS | 0 | 0 | 无 | 基线 |
| Continued training | 0 | 0 | 无 | 排除额外微调收益 |
| Route only | 0.5 | 0 | 正确 | 检验去噪视图监督 |
| Consistency only | 0 | 0.2 | 正确 | 检验特权视图蒸馏 |
| LGEI | 0.5 | 0.2 | 正确 | 主方法 |
| Shuffled evidence | 0.5 | 0.2 | 样本内打乱 | 检验证据语义 |
| Random matched route | 0.5 | 0.2 | 匹配 route 大小 | 排除仅减少图片数量的效果 |

补充分析按实体类型、数据域、图片数量、`NONE/all/subset/single` 证据模式分组。至少人工核查 100--200 个实体的图片证据，并报告缓存覆盖率和过滤原因。

## 论文贡献边界

实验得到支持后可以主张：

1. 提出结构受约束的 MLLM 重排，把开放式生成限制为 CRF 合法候选选择，避免 LLM 改写边界和类型；
2. 提出全样本候选重排与冻结置信度回退策略，在保持强 top-1 的同时只接受可审计的 LLM 决策；
3. 提供候选上限、视觉消融和失败回退的完整审计，区分“候选缺失”与“排序错误”。

该方法在推理时需要 MLLM API；缓存记录每次成功请求的延迟、尝试次数与服务端返回的 token usage（若接口提供），论文必须报告调用覆盖、延迟、费用、非法响应率及无 LLM top-1 回退结果。pilot 和完整验证集通过前，只能把性能提升写作研究假设，不能预先声称达到 0.86。
