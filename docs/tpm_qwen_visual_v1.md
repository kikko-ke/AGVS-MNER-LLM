# TPM-MI：纯文本 Qwen 与图文 Qwen 的配对对照

本实验直接回答：**已有的 Qwen 文本增强有效后，让同一个冻结 Qwen 读取原始多图，能否进一步提高实体识别 F1？**

| 系统 | TPM-MI 输入 | Qwen 输入 | 需要新训练的部分 |
| --- | --- | --- | --- |
| A：`tpm` | 文本＋原始多图 | 无 | 复用原检查点 |
| B：`tpm_qwen_text` | 文本＋原始多图 | 原始文本 | 复用原文本辅助头 |
| C：`tpm_qwen_multimodal` | 文本＋原始多图 | 原始多图＋原始文本 | 新的相同结构 BiGRU 辅助头 |

主比较是 **C−B**，另外报告 B−A、C−A，以及 B/C 都使用融合权重 1 的对照。原始 B 的验证集权重不变；C 在同一 seed 1234 的官方 dev 上用原网格 `{0, 0.25, 0.5, 1}` 选权重，平局取小值，随后固定到全部配对种子。没有提高也照实保留结果。

## 与已有实验的关系

- 以 [原 TPM-MI 实验](tpm_mi_llm_v1.md) 的输出目录为来源，默认 `outputs_vlm_gold/tpm_mi_llm_v1`。不是旧 AGVS 的内部划分实验。
- 原入口、原模块的源文件保持不变。原协议会检查代码哈希，因此不要修改 `modules/tpm_llm_v1/*.py` 或原 Python 入口。
- 新入口只读取原检查点的 emissions、Qwen 文本头和文本 logits，不重新训练 TPM-MI，不改写原目录。
- 新结果默认保存为 `outputs_vlm_gold/tpm_qwen_visual_v1`，两个目录不能相同或互相嵌套。
- 完整沿用来源协议中的官方 train/dev/test、配对种子、辅助头学习率、epoch、batch 和固定投影 seed。默认种子为 1234、42、2026。
- 先前使用过的官方 test 不重新称为盲测。新 pilot 不计算 test 指标；`all` 才执行最终测试比较。

## 图文特征如何提取

Qwen2.5-VL-3B-Instruct 保持 4-bit NF4、BF16 计算，全部冻结，不做 LoRA。不生成实体字符串，也不使用 caption 或额外提示词。

输入格式是按数据中原顺序排列的原始图片前缀，然后紧接原始文本：

```text
<|vision_start|><|image_pad|><|vision_end|> … 原始帖子文本
```

采用 Qwen 原生 processor 和完整 VL 前向：视觉编码、视觉 token 注入和多模态位置编码均由模型负责。全部图片位于文本之前，因此在因果注意力下每个文本位置都能读取全部图片的视觉 token。没有对图像做单独的第三条分类分支。

原生 processor 会按分辨率扩展 image token。不能拿未扩展字符串的字符位置直接索引模型输出。这里先对原文本独立分词，核对它与多模态 token 序列的末尾**完全相等**，再将原词对应的最后一个子词位置平移到真实序列中。核对失败会停止，不猜测偏移。

只取最后层隐藏状态。为避免计算每个图像/文本位置上的巨大词表 logits，将输出词表层替换成 `Identity`；原始视觉模块和语言解码器不变。CPU 测试将这个读出与未替换模型的最后层隐藏状态逐值比较。实现对应固定版本 [Transformers 4.51.3 Qwen VL 模型](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py) 和 [processor](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py)。

继续使用与 B 相同的固定随机投影得到 768 维词特征。C 的 BiGRU 头从相同配对 seed 重新训练，结构与 B 一致，693,513 个可训练参数。融合只更新原词首个 WordPiece 的 emission，沿用原中心化、截断和 CRF。

最多使用记录中的 4 张图片，不主动缩减张数。按照原图像审计跳过缺失/损坏图片，保留其余有效图片的相对顺序；每条记录的使用槽位、跳过槽位和数量都保存。全无有效图片的记录走文本输入，并明确计入 fallback 数量。

## AutoDL 执行

继续使用现有 Conda `base` 与 `/root/autodl-tmp/envs/mner-vlm`，脚本不安装或升级环境。

如果使用本地更新包，先把 `tpm_qwen_visual_v1_update.zip` 上传至 `/root/autodl-tmp`，解压到已有代码目录。更新包只含本实验的新增文件和 README，不含原 TPM-MI 模块、数据、模型或结果：

```bash
unzip -o /root/autodl-tmp/tpm_qwen_visual_v1_update.zip -d /root/autodl-tmp/AGVS-MNER-LLM
```

如果本更新已经推送至 GitHub，也可以从包含它的 `codex/tpm-mi-llm-20261117` 分支更新代码：

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM
git fetch origin
git switch codex/tpm-mi-llm-20261117
git pull --ff-only origin codex/tpm-mi-llm-20261117
```

两种更新方式任选其一，随后执行：

```bash
cd /root/autodl-tmp/AGVS-MNER-LLM

# 填已有 TPM-MI 实验的真实输出路径。默认路径相同时这行可以省略。
export TPM_SOURCE_RUN=outputs_vlm_gold/tpm_mi_llm_v1
export TPM_VISUAL_RUN_DIR=outputs_vlm_gold/tpm_qwen_visual_v1

# 先检查原协议、真实图文特征、图像敏感性，以及辅助头前向/反向。
bash scripts/run_tpm_qwen_visual_v1.sh smoke

# 只跑 train/dev：提取特征，训练 seed 1234 的 C 头并选融合系数。
nohup bash scripts/run_tpm_qwen_visual_v1.sh pilot > tpm_qwen_visual_v1.log 2>&1 &
tail -f tpm_qwen_visual_v1.log
```

pilot 结束后，在相同终端变量设置下运行完整配对比较：

```bash
nohup bash scripts/run_tpm_qwen_visual_v1.sh all >> tpm_qwen_visual_v1.log 2>&1 &
tail -f tpm_qwen_visual_v1.log
```

`all` 复用完成的特征、seed 1234 头及验证集系数，补齐来源协议的其他种子，再生成 test 特征和 logits，最后在原 `base` 环境用原 CRF 计算 A/B/C。同一输出目录有锁，不能同时启动 pilot 与 all。

**来源文件要求：**smoke/pilot 至少需要原协议、已完成的原 dev calibration、seed 1234 的 Qwen 文本头和 dev emissions。`all` 需要全部对应种子的原 Qwen 文本头、test logits 和 test emissions；缺少时给出具体路径并停止。请先用原入口完成这些阶段，不能拿 AGVS 结果替代 TPM-MI。

## 显存与续跑

- 图文特征以 batch 1 提取；辅助头继续使用原来的 batch（通常为 16），不改变头训练配方。
- 每张图默认最少 3,136、最多 200,704 像素（约 4–256 个合并视觉 token，实际由 processor 的尺寸取整决定）。完整图文序列上限 2,048 token，原文本上限沿用 512。超限直接报错，不截断原词或丢掉图片。
- 每 64 条记录原子保存一个带哈希核验的特征分块。重跑同一命令会复用已完成分块。未完成的辅助头训练从头开始，不声称恢复 optimizer 状态。
- 不承诺 GPU 用时或实测 F1；先以真实模型 smoke 确认当前 GPU、权重及图片能运行。

如 smoke 显存不足，在新目录使用更低图像分辨率，所有后续命令保留相同变量：

```bash
export TPM_VISUAL_RUN_DIR=outputs_vlm_gold/tpm_qwen_visual_128_v1
export TPM_VISUAL_MAX_PIXELS=100352
bash scripts/run_tpm_qwen_visual_v1.sh smoke
```

另外可设置 `TPM_VISUAL_MIN_PIXELS`、`TPM_VISUAL_MAX_INPUT_TOKENS`、`TPM_VISUAL_SHARD_SIZE`、`TPM_CONDA_BIN`、`TPM_VLM_ENV`、`CUDA_VISIBLE_DEVICES`。改变图像预算、实现或分块配置后必须使用新目录。模型和数据位置沿用来源协议，不在新实验里另换模型。

## 查看结果

```bash
/root/miniconda3/bin/python scripts/run_tpm_qwen_visual_v1.py --stage status
```

自定义了目录时加 `--run "$TPM_VISUAL_RUN_DIR"`。主要产物：

- `smoke/completed.json`：真实图像像素变化对文本隐藏状态的影响、输入压力样本与头训练检查。
- `features/qwen_multimodal_train.summary.json` 等：图像数量、无有效图像记录数和最大序列长度。
- `features/qwen_multimodal_*.pt`：词级特征、对齐掩码、逐记录图片使用明细。
- `calibration/dev_report.json`：A/B/C 的 seed 1234 dev 指标、C 的全部候选权重和最终权重。
- `evaluation/comparison_report.json`：全 860 条官方 test 及与 train/dev 去重子集的三组配对结果，还有按有效图片数 0、1、2+ 分组的诊断指标。
- `evaluation/seed*_predictions.jsonl`：各系统逐记录预测。

主结果字段为 `summary.official_test_full.paired_deltas.multimodal_minus_text`，包含每种子的 F1 差值、均值、百分点均值和正向种子数。`multimodal_minus_text_shared1` 是双方固定权重 1 的比较。F1 原值为 0–1；`mean_percentage_points` 已乘 100。

若 C 优于 B，可支持当前配置中图像条件化有额外收益；图片数分组不是人工标注的“图像消歧/冲突”子集。此实验本身不能证明已解决某类多图冲突，也不预设一定有提升。
