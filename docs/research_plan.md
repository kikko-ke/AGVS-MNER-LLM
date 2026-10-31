# 研究计划：选样作用的事后匹配重采样对照

## 当前补充实验：选择信号与训练特征混杂

数据构建脚本显示，历史 duplicate control 的 344 个新增源样本就是 LLM
生成并通过规则验证的 accepted 源 ID。因此该臂间接依赖 LLM；已有实验证明
不了 LLM 选样的独立价值，也不能把 accepted 解释为真实语义质量。当前是
查看既有结果后提出的新假设，须保留原协议和负结果，不回溯改成预注册成功。

唯一新增训练臂为原训练集加 344 条特征匹配抽样的原句。抽样池与原实验相同，
来自 augmentation metadata 中 1,000 个训练候选。精确匹配 domain、gold 实体
类别存在签名及前四张可解码图片数，固定预算的组内随机交换优化原始词数、
WordPiece 数、gold 实体计数及粗分箱分布。候选没有 LLM 语义或生成评分；
只有匹配目标分布间接来自参考臂。允许与参考臂重叠，记录重叠比例与分层强制
重叠下界，绝不为制造差距而排除 accepted 样本。

在训练前固定 sampling seed=20261031、交换次数=50,000、最大候选池 SD
标准化差异=0.10、最大参考重叠率=0.80。数据门槛通过后用 seed 1234、42、2026
及原有全部训练超参数，从头训练新臂，validation-only，和既有 accepted-source
原句重采样结果逐种子比较。数据抽样不读 val/test 标签或成绩，生成阶段结束，
不再发 API 请求。不根据模型结果重复选抽样种子。新实验属于事后探索，不能
把当前已反复查看的 test 当成独立留出集；若要独立确认，需新划分/外部数据。
实现：`scripts/build_matched_resampling_control.py`，命令和审计说明见 README。

### 已观察到的验证集差值及下一项比较

2026-10-31 匹配数据审计通过：最大标准化差异 0.007254，参考重叠率 41.57%。
metrics.csv 的四位小数结果为：seed 1234 的参考/匹配为 0.8187/0.8149，
42 为 0.8200/0.8145，2026 为 0.8182/0.8171。平均差值约 +0.0035，3/3 为正。
这涉及同一个固定匹配抽样集合上的三次训练，不是三个独立的选样实验。

在观察上述 validation 结果后，测试补充分析固定全部三个种子的现有 dev-best
checkpoint，不进一步调参。主比较固定为 accepted-source 原句重采样减去
feature-matched 原句重采样的单模型 F1；次比较为两臂各自与相同严格基线
按历史 0.7/0.3 融合的 F1 差值。两项都完整报告，不能按结果替换主比较。
实现为 `scripts/evaluate_matched_resampling.py`，读取原 final-test 报告绑定的
policy/confirmation/checkpoint hash，复现旧结果并报告新臂。此时测试结果未知。
代码测试只验证评估实现，不替代实际模型评估。

LLM 与自动校验器共同决定 acceptance，不能仅凭此实验分离两者作用，也不能
将 acceptance 解释为真实多模态语义质量。早期 test 已用于开发，当前也是
事后假设；补充测试不是独立确认，三种子差值不等于显著性结论。

## 已完成实验：固定七组 CRF 分数融合

V6.1 单模型 pilot 已完成：严格基线 dev F1 `0.8142`、duplicate control
`0.8187`、LLM augmentation `0.8175`（seed 1234，均 epoch 3 最佳）。LLM 臂
比 duplicate control 低 `0.0012`，原协议判定 **no-go**，不追认成成功。
最终 1,000 条已处理请求中有 344 accepted、100 abstained、556 rejected；
缓存生成阶段结束。

用户将新探索目标改为小幅性能提高。仅使用现有三个 checkpoint，固定比较：
control 单模型；control 与 baseline 按 `0.9/0.1、0.8/0.2、0.7/0.3` 融合；
control 与 LLM augmentation 按完全相同三组权重融合。对 emission、CRF start、
end、pair transition 一起混合，得到两个完整序列分数的加权和；不投票 BIO 标签，
不改变结构约束或重新训练。模型逐个前向 validation 后在 CPU 上解码。

首先检查模型配置一致（训练数据/输出路径可不同）、validation SHA256 相同，
并检查单模型重放等价与已记录 dev F1 的复现。gold 只用于统一的 seqeval 评估和
互补错误诊断，不进入融合分数、逐样本路由或权重生成。权重集合在评估前固定，
只在 validation 上选择一次，同时保存所有七项结果与策略的 artifact hash。

新探索只有 LLM 融合超过单独 control 和最好的无 LLM 同规模融合，才观察到
LLM 的附加收益；打平保留无 LLM 策略。验证集选择不构成独立确认或显著性证据，
历史 test 多次查看的局限仍保留。不得因普通集成增益而宣称 LLM 有效，也不得
在 test 上搜索权重。实现见 `scripts/evaluate_crf_ensemble.py`，命令见 README。

七组探索实际结果（seed 1234）：control `0.818665`，最优无 LLM 融合
`0.7 control + 0.3 baseline = 0.824223`，最优 LLM 融合 `0.821140`。
故冻结普通融合 `0.7/0.3` 后，只在 seed 42、2026 上训练严格基线与 duplicate
control，并使用 `scripts/confirm_crf_ensemble.py` 从原 policy JSON 读取固定
权重评估。此阶段没有权重参数搜索、不需要新 LLM 模型或 API 请求。
报告每种子的 baseline/control/ensemble F1，以及相对同种子更强单模型的增益。
确认均值不混入用于选权重的 seed 1234。两个新种子的复现仍使用同一 validation
划分；不得当作独立数据验证或显著性检验，也不构成 LLM 有效的证据。

固定权重确认实际结果：seed 42 的 baseline/control/ensemble 分别为
`0.815851/0.819984/0.821907`；seed 2026 分别为
`0.819431/0.818218/0.821794`。两种子都超过各自更强的单模型，平均增益
`0.002143`。因此将普通融合权重冻结，使用 `evaluate_frozen_crf_test.py`
评估 seed 42、1234、2026 的三个方法。原策略和确认报告通过 SHA256 绑定，
checkpoint 必须与两阶段既有 hash 一致。完整报告三种子结果、均值和样本标准差，
不按 test 选择权重、模型或种子。历史 test 已用于早期开发的局限必须披露。

## 历史 V6.1：Locked-Entity Multimodal Augmentation

V5.1 的冻结主干和零初始化残差成功排除了继续微调过拟合，但 epoch 0 与后续
最佳 validation F1 都约为 `0.8179`，未达到 `+0.003` 的预注册门槛。与此同时，
解释拟合损失从 `0.1448` 降到 `0.0972`，说明适配器确实学习了教师解释，只是
该信号没有纠正验证错误。因此 V5.1 判定为 no-go，不查看 test，不继续搜索权重。

V6 将 MLLM 从“标签教师/表征教师”改为“受约束的训练数据生成器”。初版要求
MLLM 返回完整 token 序列和实体 span，20 次 smoke 仅 7 次形成合法缓存且失败
样本补跑为 0/13；主要错误为实体 surface 漂移、实体重叠和 span 越界，故协议
no-go。V6.1 对每个只从
训练集选出的样本，程序从 gold BIO 提取不可变实体契约
`C={(surface_i,type_i,order_i)}`。Qwen3-VL 接收文本与最多四张图片，只能改写
实体以外的上下文，并必须引用实际发送的图片。模型只返回实体之间的 N+1 个
上下文片段；程序确定性插回 N 个实体、计算 span 并由契约重建 BIO 标签，因此
surface、数量、顺序和类型不再属于生成空间。提示词禁止新增专名；程序拒绝新增
账号/URL/hashtag、无图片依据、过短/过长、改写不足/过度或超出 62 个
WordPiece 的结果。无法安全改写时允许 abstain，abstain 只缓存不训练。

首轮固定 route seed `20261027`，目标类别为弱项 MIS/ORG，预先选择 1,000 个
训练样本。只请求前 20 条做 schema smoke；accepted、abstained 和未通过安全门的
rejected 都形成可审计且不会进入训练的 resolved cache record，网络/API 失败则
保持 unresolved 以便恢复。要求 resolved=20、accepted 至少 10 条且同时覆盖
MIS/ORG。通过后在同一 metadata/cache 上断点
续传至 1,000 个请求。正式 pilot 生成两个新增数量相同的数据臂：

1. `LLM augmentation`：原训练集加 accepted 图像一致改写；
2. `duplicate control`：原训练集加同一 source ID 的原句副本。

两臂使用相同 seed、epoch、batch size、优化器和 validation-only 选择，从头训练，
不从已经看过 validation 的 checkpoint 继续微调。只有 LLM 臂同时超过原严格
基线和 duplicate control 至少 `0.003` 才进入 `42/1234/2026` 三种子确认；平均
增益仍须至少 `0.003`、至少两个种子为正、任一种子下降不超过 `0.003`。协议
冻结前始终 `--skip-test`。这一路线的论文贡献是“受实体契约保护、图像一致的
MLLM 数据增强”，而不是宣称 MLLM 本身能准确预测 MNER 标签。

## 历史 V5.1：Protected Evidence Adapter

V4 严格共识在冻结 pilot 上没有获得任何增益：基线 F1 `0.828393`，所选编辑
oracle `0.854460`，严格共识只接受一个中性编辑并回到 `0.828393`；放宽共识
规则全部恶化。因此 V5 放弃“让 LLM 选择或修改最终标签”，改用已经缓存的
实体级视觉解释作为训练期特权信息。

对训练实体 `e=(start,end,type,confidence,image_ids,evidence)`，可靠性门为：

`z_e = 1[span/type 与 gold 完全一致] * 1[confidence >= .90] * 1[evidence 非空] * 1[至少一个实际有效图片槽位]`

V5.0 从强基线继续微调整个模型后，证据实验与无证据对照都从 `0.8179` 降到
约 `0.814`，且 V5.0 未优于同轮对照。因此 V5.1 冻结全部既有基线参数，
关闭其训练态 dropout，并加入零初始化的 token emission 残差适配器。适配器
输入由冻结的文本状态与 AGVS 视觉上下文拼接得到，因而训练和推理输入一致。
冻结 BERT 将 LLM 解释编码为 `r_e`，适配器实体表示为 `a_e`：

`emission = emission_base + AdapterClassifier(a)`

`L = L_CRF + lambda_e * sum_e w_e (1 - cos(Proj(a_e), stopgrad(r_e))) + lambda_d ||delta||^2`

可选的 batch 内对比项只作为消融，不在首轮 pilot 启用。首轮设
`lambda_e=0.02`、`lambda_d=0.01`，适配器宽度 128。LLM 解释向量只在训练中
使用；部署保留小型适配器，但不调用 LLM、不读取缓存。配对的 adapter control
使用完全相同结构和优化设置，仅令 `lambda_e=0`。

当前数据审计：完整 grounding 缓存含 10,227 条记录；阈值 `.90` 后有 9,838
个可用视觉实体，覆盖 7,094 个训练样本，类别为 MIS 1,358、LOC 1,357、ORG
1,441、PER 5,682。解释平均 34.5 词，中位数 33，重复率约 0.5%。嵌入 artifact
保存训练文件和 grounding 缓存 SHA-256，训练时强制核验 provenance。

开发协议仍是 validation-only：从严格基线初始化，先记录 epoch 0，再与完全
相同低学习率/epoch 的 continued-training 对照比较。单种子至少同时满足
`V5.1 - epoch0 >= .003` 和 `V5.1 - adapter_control >= .003` 才进入三种子确认；所有
设置冻结前不得查看 test。精确命令见 README。

## 历史 V4：单编辑、多角色共识

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

V6 实验得到支持后可以主张：

1. 提出训练期 MLLM 图像一致上下文增强，以不可变实体契约把开放式生成限制在非实体上下文，避免把弱 LLM 标签作为监督；
2. 提出实体 surface/order/type、实际图片引用、专名启发式、改写幅度、WordPiece 预算和 artifact provenance 的自动可靠性门；
3. 通过 source-matched duplicate control 区分语义增强与额外训练步数/重采样收益，同时保持原 AGVS 推理结构、速度和 CRF 解码，部署时不需要 MLLM API。

只有 validation-only pilot 与三种子对照达到预注册门槛后，才能把性能增益写成
论文结论；在此之前只能称为研究假设，不能预先宣称达到 `0.86`。最终论文还
必须报告请求数、accepted/abstain/失败比例、目标类别覆盖、改写幅度、引用图片数、
缓存生成模型与成本，并将早期多次查看官方 test 的开发历史列为限制。V5 的解释
覆盖率、类别分布和负结果作为方法选择审计报告，不得包装成正向主结果。
