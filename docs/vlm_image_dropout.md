# 固定25%训练图像移除试验

2026-11-05固定协议。此前多图QLoRA完整内部dev F1=0.790104，独立文本
QLoRA=0.781185，原none臂基线=0.804965。同一个多图adapter去图为
0.775092、固定错配图为0.760778。输入对照说明当前adapter对图文对应
关系敏感，但不证明原训练数据存在大量错配，也不保证图像移除训练有效。

本次只检验一个假设：在训练中混入少量无图视图，是否能增强文本识别与
缺失视觉输入时的回退能力，同时保持原配图上的识别表现。去掉有用图片
也可能降低性能，判断以真实原配图完整dev为主。它是普通监督训练控制，
不宣称新AGVS结构、视觉选图器或已实现的性能增益。

## 固定改动与预算

- 仅对训练输入做整组图像移除，概率固定0.25。不替换错配图、不增删原词
  或gold、不新增样本或损失、不增加额外模型前向。
- 每条训练样本每轮恰好出现一次。是否移除由
  `SHA256("20261105:epoch:source_id")`的前8字节无符号整数是否小于
  `2**62`决定，不消耗Python/Torch随机状态，不读取文本、标签或错误。
  每轮实际比例不强制恰好25%，两个epoch的具体ID保存在dropout_plan.json。
- 从相同本地Qwen2.5-VL-3B基础权重和seed1234新建LoRA，不续训旧adapter，
  也不接着训练冒烟后的参数。
- 同一5484条内部train、128条选轮面板、1372条完整dev；旧prepared manifest
  与原词完整跨度评分保持。验证和推理始终使用原配图，不做随机移除。
- 其余设置与已完成多图控制相同：冻结视觉、语言q/k/v/o的rank8/alpha16
  LoRA、dropout0.05、NF4/BF16、lr1e-4、warmup0.03、weight_decay0.01、
  batch1/累积8、2轮、max_length2048、生成256。正式训练共1372次更新，
  样本顺序与原控制相同；相同更新预算不等于相同计算量。
- 在epoch0/1/2的原配图面板micro F1中选最佳，平分保留较早轮次，允许
  epoch0胜出。完整dev和错配图分数不用于挑选epoch。

只做该p=0.25、seed1234的一个候选，不扫描概率或轮数。继续门槛预先定为
原配图完整dev F1比原none臂至少高0.003，即约0.807965，同时完整报告
非面板与逐类结果。达到后才考虑另行固定多种子复验；未达到则停止该候选。
这个开发门槛不代表统计显著性，历史内部dev的使用局限继续保留。

## 独立入口与保护

新入口为 `scripts/run_vlm_image_dropout.py`，包含smoke/train/evaluate
三个显式阶段，不修改此前任何训练、评估或输入消融文件。

入口复核原多图控制的训练报告、关键代码、基础模型、数据和最佳面板预测，
要求固定超参数一致。新profile额外绑定移除规则、新入口代码和原控制报告
及adapter hash；新格式避免被旧评估入口误当作普通多图训练。
正式训练需要自己的真实GPU冒烟，旧多图/文本冒烟报告不能放行。

冒烟取8条训练输入做8次真实更新，强制交替4个原图视图和4个无图视图，
并分别检查原图/无图生成。这里的强制视图比例用于检查硬件与梯度，
不模拟正式训练25%的频率，不宣称held-out准确率。正式训练前核验原图
与无图的完整assistant目标mask和token预算，超限停止，不裁剪gold。

训练只保存adapter，最终报告绑定最佳adapter文件hash。独立评估先核对
移除计划、更新次数、选轮规则、最佳面板预测与adapter hash，再加载冻结
权重并复现原配图面板；成功后才评估其余1244条。全部原词和原gold保留，
格式错误计空预测，新评估报告同时记录逐条预测文件hash。不读取官方
val/test、不调用LLM API，不覆盖已有输出。

## 服务器执行

使用已有mner-vlm环境和本地模型，只需同步这一个新脚本，不重新安装依赖。
在项目根目录逐阶段运行：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/run_vlm_image_dropout.py --stage smoke
python -m json.tool outputs_vlm_gold/mi_image_drop25_smoke/smoke_report.json
```

只有报告passed为true后，另行执行正式训练：

```bash
python -u scripts/run_vlm_image_dropout.py --stage train
```

正式训练结束后评估已选中的adapter：

```bash
python -u scripts/run_vlm_image_dropout.py --stage evaluate
```

默认输出分别为 `outputs_vlm_gold/mi_image_drop25_smoke/`、
`outputs_vlm_gold/mi_image_drop25_seed1234/`、
`outputs_vlm_gold/mi_image_drop25_seed1234_dev_evaluation/`。
每个阶段单独执行，冒烟不会自动开始正式训练。失败保留failure.json；
已有目录拒绝覆盖。不要使用旧 `evaluate_vlm_ner.py` 评估新schema的候选。

## 服务器结果

2026-11-05回传日志确认GPU冒烟完成8次更新及两种视图生成，峰值已分配
显存约7.88 GiB。正式训练选中epoch 2，原配图面板逐条复现通过，完整
1372条内部dev评估已写入报告。以下为原配图、完整原词跨度F1。

| 范围 | 原none臂 | 旧多图QLoRA | 25%图像移除训练 | 相对旧多图（百分点） |
| --- | ---: | ---: | ---: | ---: |
| full_dev | 0.804965 | 0.790104 | 0.790034 | -0.0070 |
| dev_panel | 0.825581 | 0.827795 | 0.823881 | -0.3914 |
| dev_nonpanel | 0.802997 | 0.786549 | 0.786811 | +0.0262 |

完整dev的TP/FP/FN由旧多图的1517/377/429变为1522/385/424，格式失败
由1条增至3条。MIS F1从0.598997升至0.611386，增加1.24个百分点，
仍低于原基线0.663304。ORG从0.690432降至0.674157，下降1.63个百分点。
整体F1相对旧多图变化约−0.007个百分点，较原none臂低约1.49个百分点。
按预先固定的0.807965继续门槛收束该候选，保留原none臂为主对照。

评估报告路径在退出清理异常之前已打印。异常发生于Python和Conda移除
`/tmp`临时目录时，日志显示`Stale file handle`。用户随后在服务器读回报告，
确认`panel_reproduction.passed`为真且预测文件SHA256与报告一致，校验
输出完整dev F1为0.7900337399429016。
