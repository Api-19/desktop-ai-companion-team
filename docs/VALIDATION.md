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


## 适配性修复验证（2026-10-17）

初始 GitHub Linux 运行暴露了两类问题：测试子进程写死仓库 `.venv/bin/python`，而 CI 没有创建该环境；以及 Mac 专用稳定性进程/原生资源采样测试没有标明平台范围。

修复提交 `4916c71`：CI 在 Mac 和 Linux 各自创建独立虚拟环境；测试子进程使用正在运行测试的解释器。原生 Mac 进程测试明确标注平台限制，参数测试使用 mock 验证受支持分支，原有 Linux/Windows 拒绝测试继续执行。未移除测试文件、未通过命令排除失败用例、未放宽产品的平台检查，也未改产品源码。

| 本机验证命令 | 退出码 | 原始结果 |
| --- | --- | --- |
| `env PYTHONPATH=src .venv/bin/python -m pytest -q tests/scripts/test_build_paired_snippets.py tests/scripts/test_m1_labeler.py tests/scripts/test_m1_pools.py tests/scripts/test_bodylink_soak.py tests/scripts/test_bodylink_soak_privacy.py tests/scripts/test_bodylink_soak_resources.py` | 0 | 261 passed in 24.83s |
| `env PYTHONPATH=src .venv/bin/python -m pytest -q -rs tests` | 0 | 6843 passed, 1 skipped in 202.94s |
| `.venv/bin/ruff check tests/scripts/test_build_paired_snippets.py tests/scripts/test_m1_labeler.py tests/scripts/test_m1_pools.py tests/scripts/test_bodylink_soak.py tests/scripts/test_bodylink_soak_privacy.py tests/scripts/test_bodylink_soak_resources.py tests/integration/test_bodylink_soak_processes.py` | 0 | All checks passed |

远端 Mac/Linux 两项任务及原始日志见 [修复提交的 Actions 运行](https://github.com/Api-19/desktop-ai-companion-team/actions/runs/39044113212)。本机的一个跳过项仍为 Windows 身份测试。Linux 的平台跳过只涉及 Mac 专用真实进程和 libproc 采样，通用隐私/协议/传输/数据处理测试继续运行；完整 Windows 支持尚未验收。

## 路线说明的共享边界

本仓库新增的 [团队路线 v2](TECHNICAL_ROUTE_V2.md) 是重新编写的通用方法与进度说明，附 [可派发任务](TEAM_TASKS.md)。没有复制原项目路线文档中的私人案例日期/数量、切分规模、个人关系分析结论、私人标注进度或个人机器信息。该审阅只针对获准查看的路线说明和共享代码，没有读取私人数据目录或评分答案。
