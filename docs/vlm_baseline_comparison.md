# QLoRA 与原严格基线的完整原词跨度比较

用户已回传服务器完整dev评估日志：

| 范围 | 样本数 | QLoRA F1 | TP/FP/FN | 格式错误 |
| --- | ---: | ---: | --- | ---: |
| full_dev | 1372 | 0.79010417 | 1517/377/429 | 1 |
| dev_panel | 128 | 0.82779456 | 137/23/34 | 0 |
| dev_nonpanel | 1244 | 0.78654887 | 1380/354/395 | 1 |

最佳epoch2的面板逐条预测复现成功。完整dev上的LOC/MIS/ORG/PER F1分别为
0.80601093/0.59899749/0.69043152/0.89926843。该结果来自用户的服务器日志，
不是本地模拟测试产生的真实性能；下一步尚待服务器运行基线比较。

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
