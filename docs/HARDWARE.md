# 硬件入口

## 结构

桌面端负责对话与语音编排，身体端通过 `bodylink` 交换控制消息、状态和音频帧。协议定义在 `src/companion/contracts/body.py`，传输层在 `bodylink/transport.py`，桌面服务与身体客户端分别在 `brain.py` 和 `body.py`。

`body_main.py` 组合音频、按键、静音状态和表情设备；`body_devices.py` 提供音频设备接口。当前开发静音实现和终端按键不等于实体开关、GPIO 按键已经实现；新增真实设备适配时保留协议层与设备层边界。

## 无设备开发

```sh
env PYTHONPATH=src .venv/bin/python -m pytest -q tests/contracts/test_body_protocol.py tests/bodylink tests/expression tests/integration/test_bodylink_session_lifecycle.py
```

这些测试包含 fake 设备、临时证书和本机回环通信，不证明实体硬件验收。设备接线、供电和实体静音需要单独的任务及真机验证。

## 已有工具

- `scripts/bodylink_pair.py`：生成配对证书。每套开发设备独立生成，私钥和证书放在仓库外，不复用负责人的证书。
- `scripts/pi_body/deploy.sh`：树莓派部署工具。先用 `--dry-run <自己的 SSH 目标>` 阅读将执行的命令；不要未经授权操作他人的设备。
- `scripts/pi_body/kiki-body.env.example`：公开配置模板，真实主机地址和设备参数只在本机填写。
- `scripts/bodylink_soak*.py`：通信稳定性实验工具；先阅读 `--help` 和对应 `tests/scripts/test_bodylink_soak*.py`，报告放在忽略的 `test-results/` 下。

初次加入团队不需要执行部署、生成证书或连接真实设备；先通过无设备测试，再按分配的任务推进。

## 平台验证

GitHub Actions 分别在 Mac 和 Linux 运行相同的完整测试命令。`bodylink_soak` 的真实进程与 libproc 资源采集明确仅支持 macOS，相关原生测试在 Linux 标注跳过；通用协议、传输、隐私、参数边界与不支持平台拒绝测试仍执行。Mac 的短时合成测试也不能代替两小时真机稳定性证明。Windows 完整支持另行验证。
