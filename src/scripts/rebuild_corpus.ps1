# 从原始数据重建训练语料 dataset/corpus/prompts.jsonl。
# 用法: powershell -ExecutionPolicy Bypass -File src\scripts\rebuild_corpus.ps1
#      -Out dataset\corpus      输出目录（默认 dataset\corpus）
#      -SkipFetch               跳过联网抓取，只用磁盘上已有的原始数据
# 输入: dataset/danbooru_api、dataset/danbooru_cn、dataset/wikidata 或联网抓取
# 输出: <Out>/prompts.jsonl 与 <Out>/manifest.json
# 预期: 每一步打印条数；任一步失败立即退出，不覆盖已有输出

param(
    [string]$Out = "dataset\corpus",
    [switch]$SkipFetch
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
$env:PYTHONPATH = "src"

function Step($name) { Write-Host "== $name ==" }

Step "1/6 danbooru 标签全库"
if ((-not $SkipFetch) -and (-not (Test-Path "dataset\danbooru_api\tags.jsonl"))) {
    python -m moss_small_encoder.sources.danbooru --download
} else {
    Write-Host "  跳过抓取或文件已存在"
}

Step "2/6 中英对照表"
if ((-not $SkipFetch) -and (-not (Test-Path "dataset\danbooru_cn\tags.csv"))) {
    python -m moss_small_encoder.sources.danbooru_cn --download
} else {
    Write-Host "  跳过抓取或文件已存在"
}

Step "3/6 维基实体"
if ((-not $SkipFetch) -and (-not (Test-Path "dataset\wikidata\entities.jsonl"))) {
    python -m moss_small_encoder.sources.wikidata --out dataset\wikidata\entities.jsonl
} else {
    Write-Host "  跳过抓取或文件已存在"
}

Step "4/6 知识桶"
python -m moss_small_encoder.build --config src\configs\dataset_kb.yaml --out dataset\kb

Step "5/6 老八桶回放"
if (-not (Test-Path "dataset\prompts.jsonl")) {
    python -m moss_small_encoder.build --config src\configs\dataset_v1.yaml --out dataset
} else {
    Write-Host "  dataset\prompts.jsonl 已存在，跳过"
}

Step "6/6 合并"
python -m moss_small_encoder.merge --out $Out --sort length

Write-Host "完成: $Out\prompts.jsonl"
