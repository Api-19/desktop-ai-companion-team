# 标注说明

现阶段只使用 `fixtures/synthetic/` 中独立编造的示例。给队员分发真实来源的任务包，需要另行取得具体数据授权并审核；本仓库不包含负责人的个人标注任务或答案。

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
