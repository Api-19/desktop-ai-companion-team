# 协作快照验证记录

日期：2026-10-17。环境：本机 macOS / Apple Silicon、Python 3.12.14。测试在团队仓库新建的独立 `.venv` 中执行，使用合成数据、临时目录、fake provider 和本机回环通信。

| 命令 | 退出码 | 原始计数或结果 |
| --- | --- | --- |
| `.venv/bin/python -m pip install -e '.[dev,body,local-speech]'` | 0 | 现有依赖安装成功 |
| `.venv/bin/python -m pip check` | 0 | No broken requirements found |
| `env PYTHONPATH=src .venv/bin/python -m companion.cli healthcheck` | 0 | healthy / development |
| `env PYTHONPATH=src .venv/bin/python -m pytest -q tests` | 0 | 6839 passed, 1 skipped in 207.04s |
| `env PYTHONPATH=src .venv/bin/python -m pytest -q tests/scripts/test_team_share_privacy.py tests/scripts/test_verify_no_private_data.py` | 0 | 17 passed in 0.40s |
| `.venv/bin/ruff check scripts/verify_no_private_data.py tests/scripts/test_team_share_privacy.py` | 0 | All checks passed |
| `.venv/bin/python scripts/verify_no_private_data.py` | 0 | 已跟踪及未忽略的新文件未命中预设隐私路径和密钥规则 |
| `git diff --check` | 0 | 无空白错误 |

完整测试运行后，隐私路径检查另增加了嵌套目录、习惯卡、数据库和私钥文件名的四个参数化场景，表中的 17 项聚焦测试包含这四项。未将聚焦计数与完整测试计数相加。

合成标注练习包也通过现有任务格式校验。完整套件中的一个跳过项为只适用于 Windows 的身份测试；本次不提供 Windows 实机验证。

## 导出边界

- 162 个产品源码文件与导出时原项目一致；通用语言习惯学习算法全部保留。
- 原项目文件和 Git 状态经对比保持不变。
- 仅导出 `src/`、产品相关 `scripts/`、`tests/`、`fixtures/synthetic/`、通用 `config/` 和构建配置，团队说明重新编写。
- 未导入原项目 `.git`、私人数据、运行数据、测试结果、私人配置、录音、模型、历史日志和原项目的本地代理会话。
- 未导出原项目本机代理调度脚本及其协作规则测试；这些依赖原工作区的契约与历史文档，属于负责人本机流程。产品功能测试保留。
- 原隐私检查脚本在团队版中增加了首次提交前未跟踪文件、嵌套私人路径、常见个人产物、软链接及额外密钥格式检查。

这些检查不证明未知自然语言内容绝无私人信息。当前导出使用通用实现和合成数据，后续新增数据仍须人工核验来源。GitHub Actions 已配置，远端执行结果须发布后另行查看；此记录不代表真实 provider、真实硬件、主观风格效果或生产验收。
