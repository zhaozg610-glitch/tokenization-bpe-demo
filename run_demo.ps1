param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$demoRoot = $PSScriptRoot
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Please use Python 3.12"; import tiktoken'
    if ($LASTEXITCODE -ne 0) { throw '请使用 Python 3.12，并先按 README 安装 requirements.txt。' }
    & $Python -X utf8 (Join-Path $demoRoot 'prepare_cache.py')
    if ($LASTEXITCODE -ne 0) { throw '官方词表缓存准备失败。' }
    & $Python -X utf8 (Join-Path $demoRoot 'run_all.py')
    if ($LASTEXITCODE -ne 0) { throw '完整验证未通过。' }
    & $Python -X utf8 (Join-Path $demoRoot 'classroom_snippets.py')
    if ($LASTEXITCODE -ne 0) { throw '课堂短代码验证未通过。' }
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
}
