# 桌面 AI 陪伴机器人 · 团队协作版

用于共同开发桌面 AI 陪伴机器人的通用实现。包含对话编排、风格学习算法、隐私处理、记忆管理、语音适配、表情、身体通信协议和本地管理界面。

此仓库只包含通用代码、配置和合成测试数据。个人聊天、个人学习成果、习惯卡、个人画像、检索索引、密钥、模型文件和原项目 Git 历史不随代码提供。没有个人数据时仍可运行健康检查和合成测试；真实语音和模型对话需要各开发者自己的模型及配置。

## 本地开始

需要 Python 3.11 或 3.12、Git 和系统 `openssl`（部分身体通信测试生成临时测试证书）。以下命令在仓库根目录执行，适用于 macOS/Linux：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,body,local-speech]'
env PYTHONPATH=src .venv/bin/python -m companion.cli healthcheck
.venv/bin/python scripts/verify_no_private_data.py
env PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

`local-speech` 安装本地语音依赖，不下载语音模型。完整测试还会使用其传递依赖 NumPy；只学习对话和隐私模块时可先安装 `.[dev]` 并仅运行对应目录的测试。Windows 的解释器路径为 `.venv\Scripts\python.exe`，PowerShell 可用 `$env:PYTHONPATH = "src"` 设置路径。部分存储、管理和稳定性工具有 POSIX / macOS 限制，以上整套测试尚未在 Windows 验收。CI 分别验证 Mac 和 Linux，并明确列出平台跳过项；详见 [验证记录](docs/VALIDATION.md)。

## 代码导航

| 路径 | 内容 |
| --- | --- |
| `src/companion/contracts/` | 对话、隐私、记忆、语音与身体消息的共享协议 |
| `src/companion/dialogue/` | 提示构建、模型调用接口、预算、输出检查和编排 |
| `src/companion/importer/`、`privacy/`、`style/` | 导入、筛选、授权与通用风格学习实现 |
| `src/companion/memory/` | 授权、记忆存储、检索和删除 |
| `src/companion/speech/`、`runtime/` | 语音适配和运行时组装 |
| `src/companion/bodylink/`、`expression/` | 身体通信、设备接口及表情控制 |
| `src/companion/admin/` | 本地管理、审核与标注页面 |
| `scripts/` | 数据处理、离线评估、硬件部署和检查工具 |
| `fixtures/synthetic/`、`tests/` | 合成数据、fake provider 与自动化测试 |

源码文件名中的 `private` 指隐私数据处理功能，可以共享；实际个人数据和处理产物不能提交。`config/privacy/veiled_terms.json` 是通用隐私风险词表，不是个人语言画像。

## 团队协作

先读 [技术路线 v2 与实现进度](docs/TECHNICAL_ROUTE_V2.md)、[同步状态](docs/SYNC_STATUS.md) 和 [可派发任务](docs/TEAM_TASKS.md)。

按 [协作与数据边界](docs/COLLABORATION.md) 分配任务，通过分支和 Pull Request 提交代码。

- 标注工作先读 [标注说明](docs/ANNOTATION.md)，从合成任务包练习。
- 硬件工作先读 [硬件入口](docs/HARDWARE.md)，通过 fake 设备与协议测试开发。
- 每次提交前运行隐私检查；检查只识别预设规则，仍需人工审核新增数据和文档。

此版本是开发协作快照。测试通过不表示真实模型对话质量、真实硬件或跨平台功能已经验收。
