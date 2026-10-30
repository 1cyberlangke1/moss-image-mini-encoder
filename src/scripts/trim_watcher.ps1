param(
    [int]$PidToWatch = 0,
    [int]$IntervalSec = 30
)

# 如果未显式传 PID，自动寻找正在跑训练的 python 进程
if ($PidToWatch -le 0) {
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*moss_small_encoder.train.run*" } |
        Select-Object -First 1
    if ($proc) {
        $PidToWatch = $proc.ProcessId
        Write-Host "已自动绑定训练进程 PID: $PidToWatch"
    } else {
        Write-Host "未找到正在运行的训练进程"
        exit 0
    }
}

# 确保在 64 位系统下正确引入 EmptyWorkingSet
$sig = '[DllImport("psapi.dll")] public static extern int EmptyWorkingSet(IntPtr hwProc);'
Add-Type -MemberDefinition $sig -Name "PsapiWatcher" -Namespace "Win32" -ErrorAction SilentlyContinue

while ($true) {
    $p = Get-Process -Id $PidToWatch -ErrorAction SilentlyContinue
    if (-not $p) {
        break
    }
    try {
        [Win32.PsapiWatcher]::EmptyWorkingSet($p.Handle) | Out-Null
    } catch {
        # 忽略进程关闭瞬间的偶发异常
    }
    Start-Sleep -Seconds $IntervalSec
}
