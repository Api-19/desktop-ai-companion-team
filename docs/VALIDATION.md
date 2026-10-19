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


首轮远端修复运行中，Linux 原始结果为 `6759 passed, 85 skipped in 124.45s`。Mac 为 `1 failed, 6842 passed, 1 skipped in 285.33s`，唯一失败是多轮排期测试给第一轮仅留了 0.6 秒，受共享 runner 调度影响发生漏槽。

该测试随后调整为在 4 秒观测期内按 3 秒间隔完成两轮；仍断言两轮均成功、没有失败/中止/漏槽和没有重复事件，不改变产品的心跳、超时、停止上界或正式两小时判定。新增的诊断输出仅含轮次计数与清理计数。旧的故意重叠/漏槽测试和短时运行不得算作两小时通过的检查继续保留。最终结果以 [仓库最新 Actions](https://github.com/Api-19/desktop-ai-companion-team/actions) 为准。


计时测试修正后的本机聚焦验证：

`env PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration/test_bodylink_soak_processes.py::test_steady_run_completes_multiple_separate_slots tests/integration/test_bodylink_soak_processes.py::test_steady_run_does_not_overlap_or_replay_slots tests/scripts/test_bodylink_soak.py::test_all_real_run_tests_use_explicit_short_duration` → exit 0，`3 passed in 5.98s`。

## PR #4 配置提示与握手测试修复（2026-10-19）

配置失败提示只显示固定类别，不回显异常原文、配置值或本机路径；README 已补对应排查表。Node 子进程测试显式使用 UTF-8 解码。这些调整不构成 Windows 原生产品验收。

旧 Ubuntu 检查唯一失败发生在集成测试建连阶段：身体收到 Welcome 后，大脑的 session-start 回调尚未写入 session ID，测试立即比较两端状态。传输约定允许先发 Welcome 再执行大脑回调，因此修复测试夹具，在原有 10 秒上限内等待两端注册同一 session；未改变产品回调顺序、超时或重连策略。

新增测试通过事件控制故意延后大脑回调，在旧夹具上复现 `session_id != None` 的断言失败（1 failed），修复后连同相关用例通过。保留重连身份、旧轮次丢弃、音频清理等原断言，没有跳过失败用例或放宽会话身份要求。

| 命令 | 退出码 | 原始结果 |
| --- | --- | --- |
| `env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_cli.py tests/runtime/test_settings.py tests/scripts/test_ticl_review_pack.py tests/integration/test_bodylink_session_lifecycle.py` | 0 | 97 passed in 8.71s |
| `.venv/bin/ruff check src/companion/cli.py src/companion/runtime/settings.py tests/test_cli.py tests/scripts/test_ticl_review_pack.py tests/integration/test_bodylink_session_lifecycle.py` | 0 | All checks passed! |
| `.venv/bin/python scripts/verify_no_private_data.py` | 0 | No private paths or secret patterns found in tracked and unignored new files. |

已扫描 SettingsError / load_settings 的生产构造与脚本入口。完整测试和 Mac/Linux CI 的最终证据附在 [PR #4](https://github.com/Api-19/desktop-ai-companion-team/pull/4)，以对应提交的实际检查结果为准。测试只使用合成数据与本地模拟，不代表真实模型或硬件验收。
