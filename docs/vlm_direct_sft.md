# 金标准直接监督的多模态大模型基线

2026-11-03 新路线实现，尚无服务器 GPU 冒烟结果或实际 F1。目标机器由用户
报告为 RTX 4090 D，24,564 MiB 总显存、驱动 580.76.05。旧环境 Python3.12 /
torch2.3.0+cu121 / transformers4.44.2 保留，用独立 Python3.11 环境运行。

## 动机与结论边界

此前改写一致性没有超过普通 Dropout；同样原句 Dropout 下，联合选样
0.808046、特征匹配0.807335，差值仅+0.000711，未通过+0.003继续门槛。
这些路线停止，不追加 API 教师缓存。新方案让 Qwen2.5-VL-3B 直接学习原始
训练集的金标准实体，是一个新的可验证假设。基础模型使用监督微调本身
不构成新的 AGVS 组件贡献；尚不能宣称更高性能或达到0.86。

## 固定数据与评估

复用已有 MI `context_internal_mi_seed20261101/context_internal_manifest.json`，
不重新抽划分，不读官方 val/test。原内部 train 全量用于训练；内部 dev
按 source_id 的固定 hash（seed20261103）选择最多128条开发面板，不按
标签、模型错误或图像效果挑选。面板用于这次有限开发，不是独立最终测试。
此前已经查看官方 test 的历史局限仍保留。

所有原词及其金标准实体保留，不按 BERT WordPiece 截断。目标为
`{"entities":[{"start":0,"end":1,"label":"PER"}]}`，半开原词跨度，类型
LOC/MIS/ORG/PER，不允许重叠。推理提示只含编号原词和图片，不含金标签。
训练仅对 assistant 答案（含结束符）计算交叉熵，精确核验多模态 prompt
token 前缀后屏蔽前缀；不对图像 token 和用户输入计算目标损失。

每条最多使用原先前四槽中可解码的图片。缺失/损坏图片按原 JPEG 修复规则
检查；短边小于28像素的图片先按比例放大，宽高比超过200的图片不送入
Qwen处理器。全部无效时保留该样本的文本输入，不删除困难样本。图片与数据文件
有 hash 绑定。JSON/跨度错误统一计作空预测并保留失败记录，不按金标准
修复响应，不丢弃无效输出，不重试直到正确。

指标为完整原词跨度的实体级 micro P/R/F1，并报告逐类计数和无效输出率。
它与旧的 BERT 截断评估、官方验证/测试分数不是相同口径，**不能直接相减**。
后续性能比较必须在同一面板/完整 gold 集合上重评估严格基线，以及同一个
LLM 的文本/普通多图输入控制；不能将换大模型的收益归给选图模块。

## 显存配置与环境

固定 Qwen2.5-VL-3B-Instruct：语言注意力 q/k/v/o 层 rank8/alpha16 LoRA，
dropout0.05，视觉与其余骨干冻结。NF4 double-quant、BF16计算，SDPA，
非重入 gradient checkpointing；不依赖 FlashAttention。每图最多约128个
视觉 token，最多4图，batch1、梯度累积8，序列预算2048，生成最多256。
全量训练之前审计 token 预算，超限时报 source_id 并停止，不静默裁剪标签。
该配置是24GB上的候选方案，只有实际 GPU 检查能证明相应样本可运行。

独立环境固定 torch2.6.0+cu124 / torchvision0.21.0 / transformers4.51.3 /
PEFT0.15.2 / accelerate1.6.0 / bitsandbytes0.45.5，见 requirements-vlm.txt。
语言 LoRA 是唯一可训练参数；视觉模块保持 eval。只保存 adapter 与 processor，
不保存一套完整量化大模型。安装与模型需预留约20GiB磁盘空间，不自动清理旧结果。

依据：[PyTorch官方旧版本安装](https://pytorch.org/get-started/previous-versions/)、
[Transformers4.51.3 Qwen2.5-VL实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)、
[PEFT量化微调](https://huggingface.co/docs/peft/v0.15.0/en/developer_guides/quantization)。
官方实现支持不等于已验证本项目24GB显存或性能。

## 执行阶段

在 AutoDL 项目根目录依次执行：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/setup_vlm_qlora_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /root/autodl-tmp/envs/mner-vlm
bash scripts/vlm_ner_pilot.sh smoke
```

smoke 阶段转换数据、下载一次开源权重，按输入长度/图片数选择8个训练
样本做8次真实梯度更新，并测试2个训练输入的生成。报告在
`outputs_vlm_gold/mi_multi_image_smoke/smoke_report.json`。它记录峰值 allocated/
reserved 显存、有限非零梯度、token长度、耗时；不宣称开发/测试准确率。
smoke 完成不会自动启动长训练。失败时保留failure.json与日志排查；非空
输出不会被覆盖。不能将本地 mock 模型测试当成真正 GPU 或模型可用证明。

只有通过相同数据、模型文件/关键代码hash和配置的真实smoke报告后，才可另行执行：

```bash
bash scripts/vlm_ner_pilot.sh train
```

固定2 epochs、lr1e-4、warmup0.03、weight_decay0.01，每轮在固定128条
面板上生成评估，同时保留epoch0基线。允许epoch0胜出，不强制制造提升。
输出 `outputs_vlm_gold/mi_multi_image_seed1234/pilot_report.json` 和
`best_adapter/`；日志 `logs/vlm_*.log`。此阶段只建立可用性基线，不扩展
UNI、不搜索选图模块或提示、不读取官方测试。其后是否继续要根据实际
学习曲线、格式失败率、同口径控制和可用预算重新固定协议。
