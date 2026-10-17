# 协作与数据边界

## 仓库关系

团队仓库维护可共享的通用实现。完整私人工作目录由项目负责人保留，个人学习产物不进入共享仓库，也不迁移原项目提交历史。

团队修改通过 Pull Request 合并；负责人审核后，将需要的代码变更带回完整项目。不要把完整私人目录反向同步到团队仓库，也不要用 `git add -f` 绕过数据忽略规则。

## 任务分工

| 任务 | 主要目录 | 验证入口 |
| --- | --- | --- |
| 标注工具、任务格式、标注体验 | `admin/labeling.py`、`admin/focus_labeling.py` | `tests/admin/test_label_pages.py`、`test_focus_label_pages.py` |
| 身体通信、按键、静音、音频设备 | `bodylink/`、`contracts/body.py` | `tests/bodylink/`、`tests/contracts/test_body_protocol.py` |
| 表情和运动 | `expression/`、`runtime/speech_expression.py` | `tests/expression/`、`tests/integration/test_speech_expression_faults.py` |
| 对话与通用学习算法 | `dialogue/`、`privacy/`、`style/` | 对应单测及集成测试 |

这些是分工建议，共享仓库成员可以看到仓库内全部内容。需要进一步限制某人的数据访问时，另行分发最小任务包。

## 提交流程

1. 为一个明确任务建分支，写明涉及模块和验收条件。
2. 使用合成输入和 fake provider 开发，运行对应测试。
3. 在仓库根目录运行 `.venv/bin/python scripts/verify_no_private_data.py`；会扫描已跟踪文件和未被忽略的新文件。
4. 用 `git status --short` 和 GitHub Desktop 的 diff 复核准备提交的文件，尤其是新增 JSON、文本、HTML 和文档。
5. 提交 PR，说明修改内容、实际运行的命令与结果，以及是否涉及接口变化。
6. 负责人审核后合并。建议在 GitHub 设置可用时为默认分支配置 PR 审核规则。

GitHub Actions 包含预设隐私规则检查和合成测试。不要向 Actions 配置个人模型密钥、真实数据路径或个人学习产物。自动检查不能识别所有私人自然语言，不代替数据来源核验和人工审核。

## 共享与保留

可以共享算法、协议、工具、普通默认配置和人工编造的测试示例。禁止共享原始或仅改名的私人聊天、个人风格样本、确认后的习惯卡、画像、向量与索引、记忆库、私人评估答案与评分、密钥及录音。

私人产物可能是 JSON、JSONL、文本或数据库，不以文件扩展名判断是否可共享。通用算法里的规则和系统提示属于实现；从某个人聊天中总结出来的内容属于个人成果，应留在私人配置中。

## 初次发布

新仓库的代码检查和测试完成后，在 GitHub Desktop 点击 Publish repository，并保持 Keep this code private 勾选。确认发布的是本团队目录，再邀请队员。没有许可证文件时，不应对外声明开源授权。
