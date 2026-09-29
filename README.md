# AGVS-MNER-LLM

面向多图像多模态命名实体识别（MNER）的 LLM 增强实验项目。它以现有 AGVS-MNER 为学生模型骨架，保留多图像顺序建模、全局视觉锚点、patch 选择和 CRF 解码，同时增加可替换的大语言模型教师链路。

## 研究主线

推荐先做“多模态 LLM 教师 -> AGVS 学生”的蒸馏路线：

1. 离线教师读取一条文本和最多四张有序图像，输出结构化实体候选：start/end、实体类型、支撑图像编号、置信度，以及每张图的证据分数。
2. scripts/build_teacher_cache.py 将教师输出转换为固定长度的 token 软标签和图像分布，写入 JSONL 缓存。
3. 学生训练时仍由 BERT + ViT + AGVS + CRF 完成推理，同时加入 token KL 蒸馏和图像注意力蒸馏。
4. 部署时不需要调用 LLM，只有训练好的学生模型参与预测。

这样可以把 LLM 的跨图像语义判断引入训练，又保留 BIO/CRF 的稳定边界和较低推理成本。教师模型可以使用 Qwen2.5-VL、Qwen3-VL、InternVL 或任意兼容 OpenAI Chat Completions 的视觉语言模型。

## 项目结构

AGVS-MNER-LLM/
- models/agvs_mner.py: AGVS 学生模型，增加 LLM 蒸馏损失接口
- modules/dataset.py: 沿用原始 MNER-MI 数据格式
- modules/llm_dataset.py: 叠加教师缓存的 Dataset
- modules/llm_trainer.py: 将教师信号送入模型的训练器
- llm/schema.py: 教师 JSON 校验与软标签转换
- llm/prompts.py: 结构化多图像 MNER 提示词
- llm/client.py: OpenAI-compatible LLM 客户端
- scripts/build_teacher_cache.py: 离线生成教师缓存
- run.py: 原始 AGVS 基线入口
- run_llm.py: LLM 蒸馏训练入口
- configs/llm_distill.yaml: 实验配置样例
- docs/research_plan.md: 研究问题、消融和后续路线

数据目录不复制到新项目；训练时通过 --data-path、--image-path 和 --twitter2017-image-path 指向现有数据目录即可。

## 安装和训练

    cd D:\code\pycharmProjects\AGVS-MNER-LLM
    pip install -r requirements.txt

先为 train、val、test 分别生成同名教师缓存：

    python scripts/build_teacher_cache.py --input D:\code\pycharmProjects\AGVS-MNER\dataset\text\MNER-UNI_train.txt --output .\teacher_cache\MNER-UNI_train.jsonl --image-root D:\code\pycharmProjects\AGVS-MNER\dataset\images --twitter-image-root D:\code\pycharmProjects\AGVS-MNER\dataset\twitter2017_images --bert-model ..\pretrained_models\bert-base-uncased --base-url https://your-endpoint/v1 --api-key $env:LLM_API_KEY --model your-vision-language-model

生成缓存后训练：

    python run_llm.py --dataset UNI --data-path D:\code\pycharmProjects\AGVS-MNER\dataset\text --image-path D:\code\pycharmProjects\AGVS-MNER\dataset\images --twitter2017-image-path D:\code\pycharmProjects\AGVS-MNER\dataset\twitter2017_images --bert-model ..\pretrained_models\bert-base-uncased --vit-model ..\pretrained_models\ViTB-16 --teacher-cache .\teacher_cache --device cuda

缓存生成是离线步骤，训练循环不会请求外部 LLM。若不提供缓存，直接运行原始 run.py 即可得到 AGVS 基线。

## 建议对照

- AGVS baseline：不使用教师信号。
- LLM-token：只启用 token 软标签蒸馏。
- LLM-image：只启用图像证据分布蒸馏。
- Full：同时启用两种蒸馏。
- Image-count：按 1/2/3/4 张图分别报告 F1。
- Teacher ablation：比较纯文本 LLM、带图像的 VLM，以及不同教师模型。

教师输出只作为训练信号；实体边界最终仍由学生的 CRF 决定。后续可以加入 LoRA 视觉语言模型、实体级对比学习、图像关系图和在线候选重排。

