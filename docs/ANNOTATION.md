# 标注说明

本仓库只附独立编造的格式/教学示例，不包含负责人的个人标注任务或答案。公共来源任务包须完成具体数据来源、使用范围与许可核验，按约定方式单独分发。

## CPED 试标入口

**[首轮指南 v1.1](data/annotation_pilot.md)** 是 [Issue #3](https://github.com/Api-19/desktop-ai-companion-team/issues/3) 的现行规范（`rcpem-public-pilot-v1.1`）：12 个动作、四个独立互动特征、完整动作顺序，并附 F/G、上下文与信息单元规则。

现有 50 条材料草案继续使用，按指南校对后分为 10 条 S 校准、30 条 S 验证和 10 条封存目标回复的 F/G 批；其中 4 条先做独立切分诊断。具体规则已经确定用于试标，尚未获得人工一致性验证。

[独立编造的记录例](data/annotation_record.example.json) 含教学答案，不可当作盲标包。标注使用[离线标注页](../tools/annotation_page/README.md)，页面导出 JSON；在页面发布前不开始正式标注。旧策略页一次只收一个标签，`m1_labeler.py` 还固定 A–E，不能直接承载新版动作序列。下面的两标签练习包仅演示旧格式，不定义本轮策略。

## 已有实现

`src/companion/admin/labeling.py` 实现任务格式校验和标注存储，`focus_labeling.py` 实现关注点标注。对应页面由本地 Admin 提供；Admin 默认需要本机私人配置，本仓库不附带任何人的 Admin 配置、一次性登录码或已完成标注。

可从 `tests/admin/test_label_pages.py` 和 `test_focus_label_pages.py` 学习如何使用临时目录、合成任务和本地模拟请求验证页面，不需要模型密钥。

## 练习任务包

`fixtures/synthetic/team_labeling_tasks.json` 是独立编造的格式示例：

- `labels` 定义标签代码、名称及解释。
- `items` 使用随机任务 ID，包含 `context`、`partner`、`reply` 数组。
- `rounds` 定义轮次、展示顺序和可选开放日期。

标注只依据可见上下文，不推测两人的真实身份、关系或隐藏经历。不确定时记录疑问，交任务负责人裁决；不要为完成数量而猜测。

建议先让两位标注者独立完成同一小批合成任务，再讨论分歧、修改规范，最后开始正式任务。最终答案与评分不要混入待标任务包。标注输出放在本地忽略目录，按任务负责人约定单独交付，不自动提交到本代码仓库。

## 开发验证

```sh
env PYTHONPATH=src .venv/bin/python -m pytest -q tests/admin/test_label_pages.py tests/admin/test_focus_label_pages.py tests/scripts/test_m1_labeler.py
```

`m1_labeler.py` 另有真实 provider 模式；运行上述测试不需要该模式，不要为练习标注开启付费调用。
