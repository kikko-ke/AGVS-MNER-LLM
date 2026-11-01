# 训练内部的 LLM 上下文一致性试验

2026-11-01 服务器实验已完成：原始训练=0.8050，Dropout=0.8111，LLM
改写=0.8103。LLM 相对两控制的差值为 +0.005323/-0.000812，未通过继续
门槛，停止该候选。以下保留原固定协议，不能用这个结果宣称改写独立增益。
另见 [后续选样对照](dropout_selection_pilot.md)。过去固定测试负结果不变。

## 诊断与待验证假设

现有审计发现 MI 的 accepted 源有 219 条，其中 MIS/ORG 实体较多，常见
mention 更集中；但加入整个训练集后，MIS 占比只增加约 1.02 个百分点，
PER 减少约 1.27 个百分点。UNI 同样没有很大的整体分布漂移。MI 增加
MIS 原句重复后，该类 validation F1 仍下降约 1.19 个百分点。因此不能把
原句重采样当成有效少数类增强，也不能从这些统计直接推断性能下降原因。
同规模 accepted/matched 的测量分布很接近；MI 的候选与 accepted 又是
UNI 中相同源记录的子集，两个数据集的证据并不完全独立。

下一假设是：不重复原句、不把生成上下文直接当成新增 gold 序列，而让
模型对同一受保护实体在原句和 LLM 改写中的预测保持一致。沿用 legacy
AGVS 架构和原始 CRF 监督，每个原始训练样本每轮出现一次。只有原句的
主损失使用 gold 标签；改写只增加停止梯度教师的一致性损失。

## 固定协议

只先做 MI，训练 seed=1234，不请求 LLM、不增加缓存。从原始 **train**
创建固定 seed=20261101 的内部约 80/20 划分；规范化、忽略大小写的文本
与有序图片名相同者放在同组。按训练实体类型出现情况和列出的图片域
分层，划分不依赖缓存是否 accepted、模型错误或任何 dev/test 成绩。
单例分层留在训练，因此比例是近似值。原始 train、划分文件及源行号均
绑定 SHA256。官方 val/test 文件不参与本试验。

已有缓存是在原始 train 上预先生成的；仅内部 train 的 accepted 改写用于
训练。内部 dev 的缓存响应内容与状态跳过，不用作监督或选样。历史上
已经反复查看官方测试结果，这个新划分不会消除历史局限，也不构成新的
独立最终测试集。

固定三臂，主数据、模型初始化、shuffle、主视图 dropout 流和优化器一致：

| 臂 | 主训练 | 额外视图 |
| --- | --- | --- |
| none | 原句 gold CRF | 无 |
| dropout | 原句 gold CRF | 对相同 eligible 源再次前向，独立 dropout |
| llm_context | 原句 gold CRF | 已有 LLM 改写，再次前向 |

dropout 与 llm_context 用同一 eligible 集合、同一原词位置、每轮同一额外
样本前向预算，前两轮都只做主训练；最终轮数由相同 early-stopping 规则
决定。控制组仍共享 LLM+校验器决定的
eligible 群体，故此比较是 **在同一来源群体内的改写内容增量**，不是
完全独立于 LLM 的选样因果试验。两视图使用同一组图片。

保护实体必须逐一保持相同 surface、标签和顺序；只对齐各原词的第一个
WordPiece，任一视图中被截断或空分词的整个实体跳过。每个 eligible
样本的实体权重和为 1，实体间等权、实体内原词等权。损失为
`CRF(original) + 0.5 * T² * KL(stopgrad(original/T) || view/T)`，T=2；
KL 按原始 batch 大小平均，稀疏群体不被自动放大。教师是当前学生在
原句上的停止梯度分布，**不是新的 LLM 实体预测**。只要求实体位置一致，
不要求改写的背景 O 词或长度与原句相同。

辅助前向使用独立 Torch RNG 并恢复 RNG 状态，避免改变后续主训练 dropout
和数据顺序；warmup=2 epochs，epoch 上限=15、patience=3，其余沿用
batch=4、lr=5e-5、BERT=1e-5、ViT=5e-6、dropout=0.2 等现有设置。
主 CRF 的 loss reduction 保持不变。损失数值通过测试不等于性能有效。

## 运行与停止标准

需要有 GPU 的 AutoDL 实例。已存在 MI 投影缓存及其 `.meta.json`，不需要
设置 API Key。在项目根目录执行：

```bash
bash scripts/train_context_consistency_pilot.sh
```

脚本前台串行三次训练，实时 tee 到 `logs/context_internal_*.log`；模型在
`outputs_context_internal/{none,dropout,llm_context}_seed1234/mi_*/`。
最后重新前向内部 dev，核验原生 CRF 与已保存最佳 F1，生成
`outputs_context_internal/pilot_*.json`，包括 checkpoint SHA256 与配置。
这里的 F1 不可直接与旧的官方 validation/test F1 比高低。

固定工程门槛：LLM 臂在该内部 dev 上相对 **两个** 控制都至少 +0.003
（+0.30 F1 百分点）才进入后续种子复验；否则停止这个候选，不扫权重，
不扩展 UNI，不以去掉控制或只挑类别/种子来构造增益。这不是统计显著性
标准；单种子通过也不能写成稳定提升。通过后还须另行固定复验协议。

脚本不会覆盖非空输出目录，也没有中断续训功能。若中途失败，保留日志
和模型后排查，不直接删除结果重抽划分或重跑挑选成绩。
