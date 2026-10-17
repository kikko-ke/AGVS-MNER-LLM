# AGVS-MNER-LLM

面向多图像多模态命名实体识别（MNER）的 LLM 增强实验项目。它以现有 AGVS-MNER 为学生模型骨架，保留多图像顺序建模、全局视觉锚点、patch 选择和 CRF 解码，同时增加可替换的大语言模型教师链路。

## 研究主线

当前主线是 C-VUG（Counterfactual Visual Utility Gating，反事实视觉效用门控）：

1. 离线教师比较“只看文本”和“文本加图像”两种实体分类条件，输出 `text_sufficient`、`visual_utility`、真正提供额外消歧信息的图片编号及判断置信度。
2. 训练集标注只负责可靠性门控：教师实体必须与 gold 的边界和类型完全一致，并超过置信度阈值；教师不能改变 NER 标签。
3. 连续效用被编码为四个图片槽位加 NONE 的软目标。文本已经充分、视觉效用低或没有有用图片时都监督 NONE；背景词接受弱 NONE 监督。
4. 教师损失只更新效用选择头。学生用预测的 NONE 概率抑制原 AGVS 视觉分支，门控强度从零开始，初始模型严格等于基线。
5. 部署时不需要调用 LLM，只有训练好的学生模型参与预测。

这避免了低质量教师 BIO 预测直接污染 NER 监督，并把 LLM 的作用集中在它相对有价值的跨图像证据判断上。教师模型可以使用 Qwen2.5-VL、Qwen3-VL、InternVL 或任意兼容 OpenAI Chat Completions 的视觉语言模型。旧的 token KL 和全局 image KL 仍保留用于消融，但默认关闭。

## 项目结构

AGVS-MNER-LLM/
- models/agvs_mner.py: AGVS 学生模型与实体—图像 grounding 辅助损失
- modules/dataset.py: 沿用原始 MNER-MI 数据格式
- modules/llm_dataset.py: 叠加教师缓存的 Dataset
- modules/llm_trainer.py: 将教师信号送入模型的训练器
- llm/schema.py: 教师 JSON 校验与软标签转换
- llm/prompts.py: 结构化多图像 MNER 提示词
- llm/client.py: OpenAI-compatible LLM 客户端
- scripts/build_teacher_cache.py: 离线生成教师缓存
- scripts/audit_llm_grounding.py: 统计不同置信度阈值下可用的 grounding 监督
- run.py: 原始 AGVS 基线入口
- run_llm.py: LLM 蒸馏训练入口
- configs/llm_distill.yaml: 实验配置样例
- docs/research_plan.md: 研究问题、消融和后续路线

数据目录不复制到新项目；训练时通过 --data-path、--image-path 和 --twitter2017-image-path 指向现有数据目录即可。

## 安装和训练

    cd D:\code\pycharmProjects\AGVS-MNER-LLM
    pip install -r requirements.txt

主方法只需要为训练集生成 gold-conditioned utility 缓存。它与旧 `mner`、`grounding` 缓存语义不同，必须使用新目录：

    python scripts/build_teacher_cache.py --teacher-task utility --input D:\code\pycharmProjects\AGVS-MNER\dataset\text\MNER-UNI_train.txt --output .\teacher_cache\qwen_utility\MNER-UNI_train.jsonl --image-root D:\code\pycharmProjects\AGVS-MNER\dataset\images --twitter-image-root D:\code\pycharmProjects\AGVS-MNER\dataset\twitter2017_images --bert-model ..\pretrained_models\bert-base-uncased --base-url https://your-endpoint/v1 --api-key $env:LLM_API_KEY --model your-vision-language-model

生成缓存后训练：

    python scripts/audit_llm_grounding.py --teacher-task utility --utility-threshold 0.5 --input D:\code\pycharmProjects\AGVS-MNER\dataset\text\MNER-UNI_train.txt --cache .\teacher_cache\qwen_utility\MNER-UNI_train.jsonl --image-root D:\code\pycharmProjects\AGVS-MNER\dataset\images --twitter-image-root D:\code\pycharmProjects\AGVS-MNER\dataset\twitter2017_images

    python run_llm.py --dataset UNI --data-path D:\code\pycharmProjects\AGVS-MNER\dataset\text --image-path D:\code\pycharmProjects\AGVS-MNER\dataset\images --twitter2017-image-path D:\code\pycharmProjects\AGVS-MNER\dataset\twitter2017_images --bert-model ..\pretrained_models\bert-base-uncased --vit-model ..\pretrained_models\ViTB-16 --teacher-cache .\teacher_cache\qwen_utility --teacher-cache-task utility --device cuda --llm-token-loss-weight 0 --llm-image-loss-weight 0 --llm-grounding-loss-weight 0.1 --llm-grounding-confidence 0.9 --llm-utility-threshold 0.5 --llm-grounding-background-ratio 0.25 --llm-grounding-detach-backbone --no-llm-grounding-fusion --llm-utility-gate --llm-utility-gate-init 0

缓存生成是离线步骤，训练循环不会请求外部 LLM。若不提供缓存，直接运行原始 run.py 即可得到 AGVS 基线。
启用 grounding loss 时，训练入口会检查缓存任务、文件存在性和可用实体数，避免因路径或阈值错误静默退化成普通基线。旧缓存可用 `--teacher-cache-task mner` 做先导实验，但其中的 confidence 含义不同，不作为正式主结果。

## 建议对照

- AGVS baseline：不使用教师信号。
- Raw token KD：旧的 token 软标签蒸馏，作为负结果/消融。
- Global image KD：旧的样本级图像分布蒸馏，作为消融。
- LLM visual utility：反事实视觉效用监督与 NONE 抑制门（主方法）。
- LLM grounding：只判断实体是否被图片描绘（负结果/消融）。
- Shared-gradient grounding：使用 `--no-llm-grounding-detach-backbone`，验证负迁移隔离的必要性。
- No background NONE：设置 `--llm-grounding-background-ratio 0`，验证背景门控的必要性。
- No gold-consistency gate：取消边界/类型一致性门控，验证可靠性门控的必要性。
- Confidence：比较 0.5/0.7/0.8/0.9 阈值。
- Image-count：按 1/2/3/4 张图分别报告 F1。
- Teacher ablation：比较纯文本 LLM、带图像的 VLM，以及不同教师模型。

教师输出只作为训练时的视觉 grounding 信号；实体边界最终仍由学生的 CRF 决定。方法定义和实验矩阵见 `docs/research_plan.md`。

