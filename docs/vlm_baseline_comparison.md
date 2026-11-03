# QLoRA 与原严格基线的完整原词跨度比较

用户已回传服务器完整dev评估日志：

| 范围 | 样本数 | QLoRA F1 | TP/FP/FN | 格式错误 |
| --- | ---: | ---: | --- | ---: |
| full_dev | 1372 | 0.79010417 | 1517/377/429 | 1 |
| dev_panel | 128 | 0.82779456 | 137/23/34 | 0 |
| dev_nonpanel | 1244 | 0.78654887 | 1380/354/395 | 1 |

最佳epoch2的面板逐条预测复现成功。完整dev上的LOC/MIS/ORG/PER F1分别为
0.80601093/0.59899749/0.69043152/0.89926843。该结果来自用户的服务器日志，
不是本地模拟测试产生的真实性能。

用户随后回传同口径基线比较日志，原none臂权重位于
`outputs_context_internal/none_seed1234/mi_20261101_211430/best_model.pt`，
历史截断口径F1=0.8049645390070922；原生预测复现通过。终端六位小数结果为：

| 范围 | 原none臂基线F1 | QLoRA F1 | QLoRA减基线 |
| --- | ---: | ---: | ---: |
| full_dev | 0.804965 | 0.790104 | -0.014860 |
| dev_panel | 0.825581 | 0.827795 | +0.002213 |
| dev_nonpanel | 0.802997 | 0.786549 | -0.016448 |

当前固定两轮、多图QLoRA配置未超过原基线：完整dev落后约1.49个F1百分点，
非面板部分落后约1.64个百分点。面板微小优势不足以支持整体提升或替换基线。
该结论限于这两个冻结候选及现有内部dev，不说明所有QLoRA配置均无效。
现有输出应保留。后续先用已保存逐条预测做逐类差值与错误交集分析，区分
共同错误、QLoRA新增正确实体和破坏的基线正确实体；再决定是否投入新的
训练预算，不按小面板分数直接追加轮次或宣称选图收益。

## 固定比较对象

采用 `outputs_context_internal/none_seed1234/mi_*/best_model.pt`。此臂仅用
同一5484条内部train的原句金标准训练，没有额外一致性损失、选样重采样或
教师损失。按原来的完整内部dev选取最佳模型，历史截断口径F1约0.8050。
不能使用在原始完整train上训练的旧checkpoint，因为那会接触当前内部dev。

两个模型均在已固定的1372条内部dev上按完整原词实体跨度评分，另列128条
面板和其余1244条。保留基线原来的WordPiece窗口、分词器、ViT图片处理和
原生CRF路径；不改为长窗口或滑窗，不使用新的约束解码。将每个原词第一片
的预测映射回原位置，预测X按原评估规则转O。未进入窗口或空分词的原词也
保留，预测填O；跨截断边界的金标准实体仍是原来的完整实体。

基线使用完整dev选轮，而QLoRA使用128条面板选轮；两模型的架构、训练预算、
分词和图像处理均不同。该比较只回答现有冻结模型在同一gold上的表现，不能
作为受控组件增益或独立测试证明。非面板部分也属于历史开发使用过的内部dev。

## 执行

同步新增的 `scripts/evaluate_baseline_fullwords.py` 后，可以先在当前环境做
无模型的只读核验，不需要GPU或旧环境的CRF依赖：

```bash
python -u scripts/evaluate_baseline_fullwords.py --check-only
```

正式前向使用原BERT/ViT环境：torch2.3.0 / transformers4.44.2，并具备原来的
pytorch-crf、seqeval、Pillow等依赖。保留独立的 `mner-vlm` 环境，不向其中
补装或升级这些依赖。如果旧环境为Conda base，执行：

```bash
conda run --no-capture-output -n base python -u scripts/evaluate_baseline_fullwords.py
```

若旧环境另有名称，将 `-n base` 换成其名称或 `-p /原环境路径`。脚本核对
版本，错环境时在创建输出目录之前停止。模型初始化与分词都只使用配置里
已有的本地BERT/ViT目录，不下载模型。原配置的batch size保持不变。

默认入口自动定位唯一的none臂checkpoint，并从 `outputs_context_internal/pilot_*.json`
中查找绑定相同checkpoint SHA256和完整配置的原报告。若有多个运行/报告，
应指定原来那个，不按新分数挑选：

```bash
python -u scripts/evaluate_baseline_fullwords.py --checkpoint /原none臂/best_model.pt --reference-report /原pilot报告.json
```

脚本还核验固定内部划分、prepared数据和图像hash，重算已有QLoRA逐条预测的
指标并核对其报告，保证样本和金标准一致。基线预测的原生截断口径F1须与
历史pilot及metrics.csv最大值都相差不超过0.0005，否则保留诊断并停止比较。
预测时使用不加载标签的 `MultimodalInferenceDataset`，不向模型传入gold。

成功后输出到新的 `outputs_vlm_gold/baseline_fullwords_seed1234_dev_evaluation/`：

- `comparison_report.json`：三个范围下的基线/QLoRA指标、差值、逐类计数，
  基线旧口径复现、原词覆盖率及文件hash。
- `baseline_predictions.jsonl`：映射回原词位置的逐条预测与可见原词索引。
- 失败时保存 `failure.json`；原口径不复现时另存 `native_reproduction.json`。

不训练、不更新权重、不改变任何checkpoint选择，不读取官方val/test，
不覆盖已有输出。成功后先判断现有QLoRA相对于基线的实际结果，再固定独立
文本QLoRA对照的训练协议；推理时给多图adapter去图只是输入消融，不等同
独立训练的文本控制。

## 离线错误对照

已实现 `scripts/audit_baseline_qlora_errors.py`，使用上述两个评估目录中的
逐条预测和同一prepared manifest。在当前 `mner-vlm` 或原base环境均可执行，
只依赖标准库及项目的纯Python评分函数，不导入Torch、Transformers、PEFT、
Pillow，不读取模型权重或图片，不产生GPU前向、训练或API请求：

```bash
python -u scripts/audit_baseline_qlora_errors.py
```

先核对source_id集合、prepared数据和已有QLoRA预测hash、所用评分实现，
再从原始响应严格解析并复算两模型在三个范围内的全部指标。baseline预测
没有被旧comparison报告直接绑定hash，因此本次核对其全部指标和原词可见
位置，并新增SHA256记录；不能把它表述成已经恢复了缺失的历史hash证明。

每个范围和LOC/MIS/ORG/PER分别记录七组互斥的实体集合：

| 字段 | 含义 |
| --- | --- |
| common_tp | 两模型都识别正确的gold实体 |
| recovered_gold | QLoRA识别正确、基线漏掉的gold实体 |
| lost_gold | 基线识别正确、QLoRA漏掉的gold实体 |
| both_missed_gold | 两模型都漏掉的gold实体 |
| shared_fp | 两模型共有的误报 |
| removed_baseline_fp | 换成QLoRA后消失的基线误报 |
| introduced_vlm_fp | QLoRA新增的误报 |

按实体集合核对TP、FP、FN守恒；不以样本F1平均替代实体micro F1。
同一条样本可能既补对实体又破坏其他实体，所以还单列两模型均全对、
仅基线全对、仅QLoRA全对、两者同错及异错的样本数。格式失败仍按空预测
参与实体计分，另保留失败标记；空gold样本的实体集合可能恰好为空而格式
仍错误，报告同时保留这两种信息。

输出到新的 `outputs_vlm_gold/baseline_qlora_error_audit/`：

- `error_audit_report.json`：三个范围的逐类差值、七组实体计数和样本结局。
- `by_type.csv`：三个范围共12行类别统计，便于比较哪一类贡献下降。
- `error_cases.jsonl`：全部样本的原词、gold、两个模型预测、实体变化及QLoRA原响应。
- `review_cases.jsonl`：固定seed20261103，按六种错误变化类别各取最多8个
  非面板案例，去重并标注所属类别，不按错误严重程度挑选。

该分析使用gold，属于诊断，不能将这些集合直接当作推理编辑策略或融合收益。
当前只判断互补错误的结构和数量，不搜索新阈值，不从dev拟合选择器，
不追加训练。原始预测、报告和训练文件保持不变；已有审计目录不会覆盖。
