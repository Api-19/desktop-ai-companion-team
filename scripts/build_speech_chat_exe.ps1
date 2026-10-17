param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Output = "dist\speech-chat"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = if ([IO.Path]::IsPathRooted($Python)) { $Python } else { Join-Path $root $Python }
$outputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $root $Output }

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python interpreter not found: $pythonPath"
}

& $pythonPath -m pip install --disable-pip-version-check "pyinstaller>=6.10,<7"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller installation failed" }

& $pythonPath -m PyInstaller --noconfirm --clean --onefile --console `
    --name speech-chat `
    --paths (Join-Path $root "src") `
    --collect-submodules companion `
    --runtime-hook (Join-Path $root "scripts\pyi_speech_chat_cwd.py") `
    --distpath $outputPath `
    --workpath (Join-Path $root "build\speech-chat") `
    --specpath (Join-Path $root "build") `
    (Join-Path $root "src\companion\runtime\speech_chat.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$configSource = Join-Path $root "config"
$configTarget = Join-Path $outputPath "config"
if (Test-Path -LiteralPath $configTarget) {
    Remove-Item -LiteralPath $configTarget -Recurse -Force
}
New-Item -ItemType Directory -Path $configTarget -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $configSource "default.toml") -Destination $configTarget

$readme = Join-Path $outputPath "README.txt"
@"
语音对话 Windows 发布包

1. 复制 config\default.toml 为本机配置，并按项目文档填写 private_data_dir。
2. 将已批准的本地 speech 模型目录和运行时配置放入 private_data_dir\speech。
3. 在 Windows 环境变量中设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY（不要写入文件）。
4. 确认 runtime.toml 中的输入/输出设备选择器匹配本机设备。
5. 双击 speech-chat.exe，按空格开始/结束一轮，按 Q 退出。

本程序只支持 Windows AMD64；模型、音频设备、API 配置和物理静音能力不会被打进 exe。
"@ | Set-Content -LiteralPath $readme -Encoding UTF8

Write-Host "Built: $(Join-Path $outputPath 'speech-chat.exe')"
