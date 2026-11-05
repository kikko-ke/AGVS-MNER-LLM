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

## 已回传的错误对照结果

用户服务器日志确认，完整dev中QLoRA补对116个gold实体、丢失188个基线
正确实体，净少72个TP；消除295个基线FP但新增259个FP，净少36个FP。
非面板部分对应补对109、丢失176、消除FP270、新增FP242。

| 类型 | 基线F1 | QLoRA F1 | 补对 / 丢失gold | 消除 / 新增FP |
| --- | ---: | ---: | ---: | ---: |
| LOC | 0.819277 | 0.806011 | 21 / 32 | 47 / 43 |
| MIS | 0.663304 | 0.598997 | 42 / 66 | 91 / 120 |
| ORG | 0.675522 | 0.690432 | 31 / 25 | 62 / 62 |
| PER | 0.895268 | 0.899268 | 22 / 65 | 95 / 34 |

MIS的漏检和误报均退步。ORG相对基线改善；PER的F1略升主要来自减少误报，
其正确实体仍减少43个。不能把所有类别的F1变化直接相加来解释micro F1。
这些变化支持存在互补预测，但不支持直接用QLoRA替换基线或按类别拼接。

## MIS错例关系核查

新增入口复用前一步已冻结的预测、评分实现和审计报告，不加载模型或图片。
在当前 `mner-vlm` 终端和项目根目录运行：

```bash
python -u scripts/audit_mis_errors.py
```

先复核已有 `error_audit_report.json` 的输入SHA256、source_id及全部原计数。
按gold实体的FN视图和预测实体的FP视图分别归类，类型混淆方向始终是
`gold类型->预测类型`：

| 类别 | 定义 |
| --- | --- |
| no_overlap | gold漏检时没有重叠预测；FP视图下则没有重叠gold |
| type_only | 边界完全相同，类型不同 |
| boundary_only | 只与一个同类型实体重叠，边界不同 |
| boundary_and_type | 只与一个不同类型实体重叠，边界也不同 |
| multiple_overlaps | 重叠多个实体，保留歧义，不贪心选择其中一个 |
| malformed_response | QLoRA输出格式错误导致整条空预测，单列于真正的无重叠输出 |

分类描述的是输出跨度之间的关系，不足以证明漏检的语义原因。例如错位
索引也可能表现为no_overlap。一次类型/边界错误可能同时产生FN和FP，
两视图不能相加当作独立错误总数，也不修改gold或模型响应。

每个范围分别统计两模型全部MIS FN/FP、QLoRA丢失/补对的MIS、引入/消除
的MIS FP，以及共同漏检和共同FP。每个计数必须还原前一步相应MIS总数。
完整dev的MIS gold共411个，QLoRA丢失66、补对42、共同漏检106。
服务器已于2026-11-04回传关系分类结果，见下文。

输出目录 `outputs_vlm_gold/mis_error_diagnosis/` 包含：

- `mis_diagnosis_report.json`：完整dev、面板、非面板的计数、类型混淆方向和边界关系。
- `mis_errors.jsonl`：每个MIS实体错误的原词、gold、两模型预测及重叠候选。
- `review_cases.jsonl` 与 `review_examples.txt`：从非面板中按固定seed20261103，
  对丢失、引入FP和补对三组的每种关系分类各取最多2个实体错误案例。

终端同时打印主要计数和最多6个实际文本案例，优先展示不同关系分类。
接下来用这些固定案例核对MIS标注范围、类型和边界；本步骤不追加训练、
不修改提示，不从dev拟合编辑策略。旧训练、预测、评分和审计文件保持原样。

## 已回传的MIS关系分类与案例

用户提供的服务器日志显示原审计复核通过，统计如下。这里仅列相对基线
丢失的正确MIS和新增的MIS误报，不代表QLoRA的全部FN/FP：

| 范围 / 错误视图 | 总数 | 无重叠 | 仅类型 | 仅边界 | 类型及边界 | 多重叠 | 格式失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_dev / lost_gold | 66 | 43 | 9 | 7 | 3 | 4 | 0 |
| full_dev / introduced_vlm_fp | 120 | 45 | 35 | 28 | 6 | 6 | 0 |
| dev_nonpanel / lost_gold | 59 | 37 | 8 | 7 | 3 | 4 | 0 |
| dev_nonpanel / introduced_vlm_fp | 112 | 41 | 33 | 26 | 6 | 6 | 0 |

完整dev丢失的MIS中43/66（65.2%）没有重叠预测；新增MIS误报中
35/120（29.2%）与其他类型gold边界完全相同，另有28/120（23.3%）
属于同类型的边界偏差。35个精确跨度的类型错误由PER->MIS 21个、
ORG->MIS 10个、LOC->MIS 4个组成。完整dev唯一一条格式失败没有贡献
这两组MIS错误，因此不能将该类退步归因于JSON格式问题。

固定非面板案例提供以下可直接观察的证据：

| source_id | 原词片段 | gold与两模型预测的差别 |
| --- | --- | --- |
| 1737 | Carrera Targa | gold与基线为MIS，QLoRA整条为空预测 |
| 1616 | ISAC | gold与基线为MIS，QLoRA在同一跨度预测ORG |
| 2122 | In May | QLoRA新增MIS，该跨度没有重叠gold；Yujoko仍正确预测PER |
| 1877 | Garba pandal | gold与基线为LOC，QLoRA在同一跨度预测MIS |
| 2607 | running man | gold为MIS，基线漏检，QLoRA补对 |
| 5040 | Russians | gold与QLoRA为MIS，基线在同一跨度预测LOC |

这些案例说明既有漏检，也有类型混淆和局部补对，尚不能证明图片导致
退步。`In May`的错误还不能区分过度识别与原词索引偏移。
source_id1877的`Indore`被原gold标为PER，而两模型均预测LOC，是需要
人工核查的标注疑点；保持原gold计分，不据此修正标签、将gold当规则
或推断全数据集标注质量。以上解释只使用已回传文本，没有检查对应图片。

## 独立文本训练对照协议（已完成）

现有 `scripts/run_vlm_ner.py` 已支持 `--input-mode text`；
`scripts/evaluate_vlm_ner.py` 从训练profile读取输入模式，也支持该控制。
不需要修改旧训练或评估实现。此次只补齐原协议中尚缺的独立文本控制，
用于判断当前多图输入相对文本是否有增益，不据MIS案例调整提示或采样。

- 从同一原始Qwen2.5-VL-3B本地权重、seed1234新建adapter，不续训多图adapter。
- 使用同一prepared manifest的5484条train、128条选轮面板和1372条dev，
  保留全部原词、gold与空实体样本。唯一输入处理差异是不送入图片。
- 保持原系统提示、语言LoRA、量化、优化器、样本顺序、batch1/累积8、
  2轮与生成设置；正式训练共1372次优化器更新。相同样本和更新预算不等于相同FLOPs。
- 与多图组一样在epoch0/1/2中按同一面板micro F1选最佳，平分保留较早轮次。
  文本组需要自己的GPU smoke，不能使用多图组的smoke报告。
- 训练结束先复现最佳面板，再一次性生成完整内部dev；比较整体及逐类
  P/R/F1、格式失败率，同时报告full_dev、dev_panel和dev_nonpanel。
  不按其中最高的分组结论挑选模型，不新增官方val/test访问。

进入项目目录并激活原 `mner-vlm` 环境后，先单独运行冒烟：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/run_vlm_ner.py --stage smoke --input-mode text \
  --output outputs_vlm_gold/mi_text_smoke
python -m json.tool outputs_vlm_gold/mi_text_smoke/smoke_report.json
```

确认文本冒烟报告的passed为true，且profile.input_mode为text后，再运行
独立的正式训练命令；冒烟脚本不会自动执行本阶段：

```bash
python -u scripts/run_vlm_ner.py --stage train --input-mode text \
  --smoke-report outputs_vlm_gold/mi_text_smoke/smoke_report.json \
  --output outputs_vlm_gold/mi_text_seed1234
```

该训练入口会核对smoke与正式训练的完整profile相同。正式训练完成后：

```bash
python -u scripts/evaluate_vlm_ner.py \
  --run outputs_vlm_gold/mi_text_seed1234 \
  --output outputs_vlm_gold/mi_text_seed1234_dev_evaluation
```

所有输出位于新目录。文本组两轮训练与完整内部dev评估现已完成，用户回传
的日志确认最佳面板逐条预测复现成功；完整dev格式失败3条，全部位于非面板。
以下是真实服务器日志结果，本地单元测试只核验实现，不产生这些性能数据。

## 三组冻结候选的完整内部dev结果

| 范围 | 样本数 | 原none臂基线F1 | 文本QLoRA F1 | 多图QLoRA F1 | 多图减文本（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_dev | 1372 | 0.804965 | 0.781185 | 0.790104 | +0.8919 |
| dev_panel | 128 | 0.825581 | 0.795252 | 0.827795 | +3.2542 |
| dev_nonpanel | 1244 | 0.802997 | 0.779835 | 0.786549 | +0.6714 |

文本组完整dev为TP1503 / FP399 / FN443，P=0.79022082、R=0.77235355；
多图组为TP1517 / FP377 / FN429。多图相对文本净增加14个TP、减少22个FP，
但相对原基线仍落后1.4860个F1百分点，文本组落后2.3780个百分点。
不能把面板上的3.25个百分点差值推广到完整dev，其非面板差值只有0.67个百分点。

| 类型（full_dev） | 文本QLoRA F1 | 多图QLoRA F1 | 多图减文本（百分点） |
| --- | ---: | ---: | ---: |
| LOC | 0.780161 | 0.806011 | +2.5850 |
| MIS | 0.607277 | 0.598997 | -0.8280 |
| ORG | 0.682396 | 0.690432 | +0.8036 |
| PER | 0.891676 | 0.899268 | +0.7592 |

MIS在文本组也低于原基线0.663304，因此不能把此前的MIS短板全部归因于
推理图片干扰。当前比较说明这两个独立训练、分别按同一面板规则选出的
adapter中，多图组的整体表现略高；尚不能分离多图训练与推理图像输入
各自的作用。保留三组结果，不按类型从dev gold拼接模型，也不据此直接扩训。

单种子开发集上的差值仅支持现有配置的比较，不能解释成AGVS组件贡献、
统计显著性或独立测试泛化；内部dev已有的开发使用历史继续保留。

## 冻结多图adapter的推理去图消融（已完成）

新增 `scripts/evaluate_vlm_no_images.py`，保持多图组的同一个best_adapter、
基础模型、processor、系统提示、编号原词与生成配置，仅在推理时去掉全部
图片。该结果与上面的独立文本训练组不同，不能代替该训练控制。

只需同步这个新入口，不修改任何旧训练、评估或审计代码。当前 `mner-vlm`
环境、项目根目录中执行：

```bash
python -u scripts/evaluate_vlm_no_images.py
```

需要GPU。可选地先用 `--check-only` 核验文件和既有指标，不导入模型库、
不创建输出目录；默认正式执行已经包含这一步，不必重复运行。

入口核验完整多图训练profile、基础模型/adapter/实现hash、相同数据与样本、
已保存预测的严格解析和全部指标，并与既有离线错误审计绑定的原报告/预测
SHA256核对。后者补充了旧完整dev报告没有直接绑定自身预测文件的问题。
加载并冻结多图adapter后，先用原图片重跑128条面板，逐条复现成功才开始
去图生成。全部1372条原词与gold保留、每条去图生成一次，格式失败仍计空预测。
原本无有效图片的样本，去图前后输入相同，其实体预测/格式状态也必须复现；
否则保存诊断并停止。结束时再次校验冻结文件与adapter没有变化。

新输出 `outputs_vlm_gold/mi_multi_image_seed1234_no_images/` 包括：

- `ablation_report.json`：三个固定范围的原图/去图P/R/F1、逐类差值及所有输入hash。
- `no_image_predictions.jsonl`：全部原词跨度预测和原始响应。
- `panel_reproduction.json` 与 `matched_panel_predictions.jsonl`：原图面板复现检查。
- `zero_image_reproduction.json`：原本无图样本的预测复现检查。
- `config.json`，失败时另存 `failure.json`；不覆盖已有结果。

原图完整dev分数来自已绑定并复核的既有预测，本步骤只重跑原图面板；
不是重新生成整套原图预测。终端差值方向固定为原图F1减去图F1。
该消融只回答固定多图adapter对有无图片输入的敏感性；图片数量、视觉token
和输入分布都会随去图变化，不能由此证明正确图片语义的贡献，也不能证明
AGVS选图组件有效。本步骤不训练、不改提示、不重新选轮、不读官方val/test。

## 已回传的去图结果

用户服务器已完成1372条去图预测并生成 `ablation_report.json`，终端结果为：

| 范围 | 多图adapter原图F1 | 同一adapter去图F1 | 原图减去图（百分点） |
| --- | ---: | ---: | ---: |
| full_dev | 0.790104 | 0.775092 | +1.5012 |
| dev_panel | 0.827795 | 0.798780 | +2.9014 |
| dev_nonpanel | 0.786549 | 0.772858 | +1.3691 |

逐类原图减去图的F1差值分别为LOC +1.3047、MIS +1.9176、ORG -0.3367、
PER +2.1440个百分点。以上数值采用终端六位小数日志，不据此反推未提供
的实体计数、格式失败数或更高精度分数。

这说明该冻结多图adapter在当前内部dev上有图优于去图，MIS也从原图片
输入中受益。独立文本adapter的MIS略高于多图adapter，与同一个多图adapter
去图后MIS下降并不矛盾：前者比较两套训练权重，后者固定权重只改变输入。
去图同时改变视觉token数量和输入分布，还不能将全部1.50个百分点解释为
图片语义对应关系的收益。四个现有条件的完整dev结果按F1依次是原none臂
0.804965、多图QLoRA原图0.790104、独立文本QLoRA 0.781185、
多图adapter去图0.775092；当前QLoRA配置仍未超过原基线。

## 同图片数的冻结错配图对照（已完成）

新增 `scripts/evaluate_vlm_mismatched_images.py`。固定同一多图adapter与
既有原图/去图控制，用seed20261104分别在面板和非面板中按有效图片数量
分组，将同组图片bundle作确定性双射置换。分配只读取source_id、固定面板
成员关系、图片数和图片SHA256，不读取原词、标签、模型错误或分数。
bundle内部图片顺序保留，
每条接收同样数量的图片、每个入选bundle恰好使用一次，且接收方原图片
与替换图片的SHA256集合必须完全不重叠。

用确定性二分图匹配寻找完整置换，避免贪心分配导致最后一条被迫使用原图。
原本无图的样本和无法完成完整无重叠置换的整个分区/图片数桶均明确列出，
不从不可匹配的桶里只挑容易替换的个体。主比较的full_dev/panel/nonpanel
都是这一固定输入侧入选集合的子集；终端前缀为 `eligible_` 并同时打印
入选数/原范围样本数。不能将这个子集的F1直接冒充完整1372条dev分数。

同一入选集合分别重算原图、去图和错配图P/R/F1，前两组使用已冻结的
逐条预测，不重跑整套控制。先复核原图训练/完整dev/审计及去图报告的
profile、hash、source_id和全部指标，再用真实原图重新生成128条选轮面板。
逐条复现成功后，先审计所有错配组合的prompt预算，再对每条入选样本
生成一次错配预测；超限或格式问题不裁剪或重试，格式错误仍计空预测。
结束时再次核验输入、置换计划与adapter文件没有变化。

在原 `mner-vlm` 环境与项目根目录执行，需要GPU：

```bash
python -u scripts/evaluate_vlm_mismatched_images.py
```

可选 `--check-only` 只核对冻结控制与置换覆盖范围，不加载模型、不创建输出。
默认正式执行已包含这些核验。先前的 `evaluate_vlm_no_images.py` 和
`evaluate_vlm_ner.py` 必须保留原字节，新入口直接复用它们，不修改旧文件。

新输出 `outputs_vlm_gold/mi_multi_image_seed1234_mismatched_images/` 包括：

- `image_permutation.json`：固定seed、接收方/供图方ID、原图/替换图路径及hash、排除的整桶。
- `mismatch_report.json`：入选/原范围ID、三条件同集合指标、逐类差值、输入hash及预算审计。
- `mismatched_predictions.jsonl`：全部入选样本的错配响应、预测和格式错误记录。
- `matched_panel_predictions.jsonl` 与 `panel_reproduction.json`：原图面板复现。
- `config.json`，失败时另存 `failure.json`；不覆盖已有输出。

图片不跨面板/非面板边界交换，因此三个范围各自保持单条图片数与入选集合
的bundle分布；单条分辨率及视觉token数可能改变。
不同来源且无相同hash的图片也不保证语义无关，不能把这当作
无混杂的语义因果估计。只做一次固定置换，不按分数更换seed、选择模型或
训练新模块。用户现已回传服务器错配图结果，见下文。

## 已回传的错配图结果与阶段结论

2026-11-05收到用户服务器日志：固定seed20261104下，完整dev入选
1372/1372、面板128/128、非面板1244/1244，排除的图片数桶为空。
原图面板复现、错配prompt预算审计及全部错配生成均通过，输出
`mi_multi_image_seed1234_mismatched_images/mismatch_report.json`。
因此本次的eligible范围恰好等于各自原范围，可直接做全量三条件比较。
附件开头重复粘贴的去图日志按同一个既有结果记录，不计为多个独立复验。

| 范围 | 原配图F1 | 去图F1 | 错配图F1 | 原配图减错配图（百分点） |
| --- | ---: | ---: | ---: | ---: |
| full_dev | 0.790104 | 0.775092 | 0.760778 | +2.9326 |
| dev_panel | 0.827795 | 0.798780 | 0.755287 | +7.2508 |
| dev_nonpanel | 0.786549 | 0.772858 | 0.761301 | +2.5247 |

F1和差值分别按终端精度记录；独立四舍五入后不要求显示值严格相减一致。
完整dev错配图比去图还低约1.43个百分点，非面板低约1.16个百分点。
完整dev逐类原配图减错配图的F1差值为：

| 类型 | 差值（百分点） |
| --- | ---: |
| LOC | +3.6780 |
| MIS | +4.1746 |
| ORG | +3.0163 |
| PER | +2.1136 |

该冻结adapter在完整dev和非面板上都呈现原配图优于去图、去图优于错配图。
保持单条图片数和分区内图片bundle总体分布后，原始图文配对仍优于这一次
固定错配，支持其对图文对应关系敏感；并非只要送入同样数量的图片即可。
这一结果不证明原数据实际存在大量错配，也不证明存在可靠的推理选图器。
单条分辨率/视觉token变化、单次置换和开发集历史的解释边界继续保留，
不能据此宣称纯语义因果效应、统计显著性或AGVS组件增益。

当前完整dev的冻结候选/输入条件汇总如下，只有前三行是独立训练候选：

| 训练候选 | 推理输入 | F1 |
| --- | --- | ---: |
| 原none臂BERT/ViT基线 | 原配图 | 0.804965 |
| 两轮多图QLoRA | 原配图 | 0.790104 |
| 两轮文本QLoRA | 文本 | 0.781185 |
| 同一个多图adapter | 去图 | 0.775092 |
| 同一个多图adapter | 固定错配图 | 0.760778 |

基线建立与输入对照阶段至此完成。当前多图QLoRA虽然使用了图片信息，
整体仍比原none臂落后约1.49个F1百分点；视觉输入对照的正差值不能替代
相对严格基线的性能提升。MIS在两个独立QLoRA训练候选中均弱于原基线，
且同一多图adapter去图/错配后MIS进一步下降，不能将该短板简单归为
图片输入造成的问题。

保留现有权重、逐条预测和原基线控制，本阶段不追加训练或新的输入消融。
若继续方法开发，应另外固定单一训练侧假设和预算，以原配图完整dev上的
实体识别表现为主要比较，保留原基线0.804965及多图QLoRA 0.790104两个
控制；去图/错配只作辅助诊断。当前结果不足以替某个尚未实现的训练改动
背书，也不按dev gold拼接不同类别的预测或修改原标签。

随后完成25%训练侧整组图像移除试验，仍用原配图选轮及评估。完整dev
F1为0.790034，较旧多图低约0.007个百分点，较原none臂低约1.49个百分点。
该候选按预定门槛收束。设置和逐类结果见[固定训练图像移除试验](vlm_image_dropout.md)。

## 逐词 BIO 监督试验

固定一个逐词 BIO 输出候选，用同一原配图、内部划分、两轮预算和 seed 1234
检验召回差距。assistant 目标为 `{"tags":"..."}`，每个原词对应一个数字，
0 为 O，1/2、3/4、5/6、7/8 分别表示 LOC、MIS、ORG、PER 的 B/I。
评估将合法序列还原为原词跨度，非法长度和续接标签计为空预测并单独计数。

入口 `scripts/run_vlm_bio.py` 提供 `--stage smoke`、`--stage train` 和
`--stage evaluate`。冒烟使用八条训练样本并覆盖最长原词序列。正式训练前
逐条核验原标注往返转换、2048 token 总预算和 256 token 答案预算。通过
专属冒烟后训练，按原配图面板选轮，复现面板后评估全部内部 dev。

完整 dev 同时报召回、F1、逐类结果和格式失败数。原多图 F1 为 0.790104，
原 none 臂为 0.804965。继续门槛沿用原基线加 0.003，约 0.807965。

服务器完成八步 GPU 冒烟，6856 条训练与内部 dev 样本通过 token 预算检查。
两轮训练选中 epoch 2，评估复现面板后生成其余 1244 条预测。回传终端 F1 如下。

| 范围 | 原 none 臂 | 原多图 QLoRA | 逐词 BIO | 相对原多图（百分点） |
| --- | ---: | ---: | ---: | ---: |
| full_dev | 0.804965 | 0.790104 | 0.802725 | +1.2620 |
| dev_panel | 0.825581 | 0.827795 | 0.817910 | -0.9884 |
| dev_nonpanel | 0.802997 | 0.786549 | 0.801264 | +1.4715 |

完整 dev 的 TP/FP/FN 从 1517/377/429 变为 1532/339/414，召回提高
0.77 个百分点。MIS F1 从 0.598997 升至 0.638701，误报从 148 降至
92，正确实体从 239 降至 236。MIS 召回下降约 0.73 个百分点。完整 dev
有 14 条格式失败，按原协议计为空预测。

BIO 是当前完整 dev F1 最高的 QLoRA 候选。它比原 none 臂少命中 57 个
实体，召回低约 2.93 个百分点，F1 低约 0.224 个百分点。按预设的
0.807965 门槛收束这次两轮试验，归档 BIO 结果及原 none 臂主对照。
