# v1.1 / v1.2 离线标注页

双击同目录的 `annotate.html`，在常用桌面浏览器打开即可。页面无需安装、无需服务器，所有文件都在本机读取。先读负责人提供的同版指南（仓库现为 [试标指南 v1.2](../../docs/data/annotation_pilot.md)），使用负责人发放的同版任务包；两位标注者各自独立完成，不互看答案。

1. 输入自己的匿名 ID，选 `training`（练习）或 `pilot`（试标），加载初始 JSON 任务包。明确点选本批“此前看过目标回复”“此前读过核验评论”两个是/否。个别事件不同，可添加暴露例外并写说明；例外覆盖该事件的批次声明。
2. **S-only**：逐事件阅读上下文、对方话轮和完整回复。切分诊断事件先在字间点击“·”放置/取消边界，再“提交切分”；允许零内部边界，提交后不能修改。之后按冻结片段标注，不能改片段。
3. 每片段先选判断状态。可判断时选至少一个动作；不可分多功能才多选，并手选原因代码。拿不准时用 `uncertain`、选候选和原因；上下文不足用 `insufficient_context` 和原因。`OTHER` 必须描述功能。COMMIT / SOCIAL 的 subtype 可留空。
4. v1.1 四特征分别选“有 / 无 / 无法确定”，均无默认。选“有”后，在相应黄色原文框内拖选文字，点“添加这段证据”；可重复添加多段，不填偏移。选“无法确定”写短原因。v1.2 字段见下节。边界问题点“对此片段提出边界异议”，选择问题并写一句建议；需合并时指出另一片段 ID。
5. **F/G**：先选 F1 可接受单元、首选及 mode，锁定后不能回改。再逐候选判断 G、写理由、勾证据消息并锁定。没有候选也须注明原因并锁定。全批 F1/G 锁定后才能加载同批同版、同事件集合的 stage2；未完成前不要打开该文件。揭示回复后标 F2 的实际承接：按首次出现顺序添加组，同组可多选、单元不重复；无匹配/不足选对应状态。再完成 S。F 单元异议可单独补记。
6. 点“检查并导出”。未完成时页面按事件列缺项，不能下载；全部完成后下载 `<batch_id>-<annotator_id>.json`。保留各自原始结果，按负责人约定单独交付，勿提交真实任务包或结果到公共仓库。

标完并导出后点“清除本机草稿”。

## 自己切分模式

主人在本机单人标注时，可加载 `self_segmentation: true` 的 S-only 或 F/G stage2 包；每事件 `segments: []`，切分诊断省略或为 false。练习包见 `fixtures/synthetic/annotation_packages/self_seg_s_only.json`。
先在回复原文的字间点选边界，再点“确定切分”；消息两端自带边界，片段不跨消息，首尾空白会裁掉，纯空白片段会丢弃。
随后直接给自己切出的 `s1, s2, …` 标动作和该版本特征；点“重新切分”并确认，会清空本事件全部 S 标注。
导出以这些片段作为 `s.segments`，附 `s.segmentation_source: "self"`，不含 `free_segmentation`；未启用此模式的包仍按原预切流程操作。

**保存与继续**：页面自动保存本浏览器草稿。刷新后，用相同匿名 ID、phase 加载原初始包即可恢复；F/G 已加载的 stage2 也会恢复。浏览器清理数据、隐私模式、移动页面路径可能使草稿不可用。完成后及时导出备份；可先加载原初始包，再“导入已导出的 JSON 继续”。F/G 导入后仍须加载原 stage2 来核对冻结片段；F1/G 保持锁定。浏览器禁止存储时，页面会提示，请保持页面打开直到导出。

仓库的 `fixtures/synthetic/annotation_packages/` 仅为独立编造的操作练习；`s_only.json` 是两事件 S 包，`fg_stage1.json` / `fg_stage2.json` 是两阶段练习。`example_fg_*` 是权威教学记录的匹配包，含已公开答案，不作盲标。

整理者在仓库根校验（标注者无需运行命令）：

```sh
.venv/bin/python scripts/annotation_package_check.py 包.json
.venv/bin/python scripts/annotation_export_check.py 结果.json --package 初始包.json
# F/G 结果在上一条命令末尾再加：--package stage2包.json
```

格式通过只说明结构与定位有效，不代表人工标注正确或一致性通过。F1/G 的锁定是操作约束，无法证明标注者此前没有看过另一个文件；请如实记录暴露情况。


## 只做 G 模式

加载 `stage: "g_only"` 的初始包，只显示当前语境 x 和候选历史 c → r，不标 F/S、不揭示目标回复，也不加载 stage2。
每个候选选择 applicable / not_applicable / uncertain，填写理由并勾选至少一条 c/r/x 消息证据，再逐事件点“锁定 G”；锁定后不可修改。
可选 permission 仅在候选上方显示称呼、调侃许可和备注作参考，不改变校验规则，不进入导出。
批次暴露声明、事件例外、草稿恢复和导出方式沿用原流程；练习包见 `fixtures/synthetic/annotation_packages/g_only.json`。

## v1.2 字段

任务包的 `protocol_version` 决定字段：v1.1 沿用原四特征，v1.2 为八项表达特征（含候选 intensification）及 `content_attributes.self_disclosure`，均无默认且须逐项选择。
evaluation 选 positive / negative / mixed / absent / unclear；present 类值拖选证据（mixed 分别定位褒贬），unclear 填短原因；irony present 还须定位字面与反向语境并说明相反立场。
REQUEST 索取自身关照时 subtype 选 seek_care，其他请求留 null；自己切分、G-only、F/G 流程通用，导出保留任务包协议版本。练习包为 `v12_s_only.json`、`v12_self_seg_s_only.json`。
