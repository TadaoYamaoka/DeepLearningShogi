param(
  [string]$Engine = "$PSScriptRoot\..\usi\x64\Release\dlshogi_tensorrt.exe",
  [string]$Model = "$PSScriptRoot\..\artifacts\models\ny.onnx",
  [int]$Nodes = 32,
  [int]$Threads = 0,
  [int]$Slots = 2,
  [int]$Batch = 128,
  [string]$Log = "$PSScriptRoot\..\artifacts\usi_smoke.log"
)
$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\TensorRT-8.6.1.6\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;' + $env:PATH
$enginePath = [IO.Path]::GetFullPath($Engine)
$modelPath = [IO.Path]::GetFullPath($Model)
$logPath = [IO.Path]::GetFullPath($Log)
$start = New-Object Diagnostics.ProcessStartInfo
$start.FileName = $enginePath
$start.WorkingDirectory = [IO.Path]::GetFullPath("$PSScriptRoot\..")
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$process = New-Object Diagnostics.Process
$process.StartInfo = $start
[void]$process.Start()
$stderr = $process.StandardError.ReadToEndAsync()
$writer = New-Object IO.StreamWriter($logPath, $false)
function Send-Usi([string]$line) {
  $writer.WriteLine('> ' + $line); $writer.Flush()
  $process.StandardInput.WriteLine($line); $process.StandardInput.Flush()
}
function Read-Until([string]$pattern, [int]$timeoutSeconds = 1800) {
  $timer = [Diagnostics.Stopwatch]::StartNew()
  while ($true) {
    $task = $process.StandardOutput.ReadLineAsync()
    while (-not $task.Wait(1000)) {
      if ($timer.Elapsed.TotalSeconds -gt $timeoutSeconds) { throw "USI timeout waiting for $pattern" }
    }
    $line = $task.Result
    if ($null -eq $line) { $process.WaitForExit(); throw ('Engine exited (' + $process.ExitCode + '): ' + $stderr.Result) }
    $writer.WriteLine($line); $writer.Flush()
    if ($line.StartsWith('info ') -or $line -eq 'readyok' -or $line.StartsWith('bestmove ')) { Write-Output $line }
    if ($line -match $pattern) { return }
  }
}
try {
  Send-Usi 'usi'; Read-Until '^usiok$' 10
  Send-Usi "setoption name DNN_Model value $modelPath"
  Send-Usi "setoption name Threads value $Threads"
  Send-Usi "setoption name DNN_Slots value $Slots"
  Send-Usi "setoption name DNN_Batch_Size value $Batch"
  Send-Usi 'setoption name Mate_Root_Search value 0'
  Send-Usi 'setoption name DfPn_Hash value 64'
  Send-Usi 'setoption name PV_Mate_Search_Threads value 0'
  Send-Usi 'setoption name Resign_Threshold value 0'
  Send-Usi 'isready'; Read-Until '^readyok$'
  Send-Usi 'usinewgame'
  Send-Usi 'position startpos'
  Send-Usi "go nodes $Nodes"
  Read-Until '^bestmove ' 30
  Send-Usi 'quit'
  if (-not $process.WaitForExit(10000)) { throw 'Engine failed to exit' }
  if ($process.ExitCode -ne 0) { throw ('Engine failed: ' + $stderr.Result) }
  Write-Output "USI smoke passed: $logPath"
} finally {
  if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
  $writer.WriteLine($stderr.Result); $writer.Dispose(); $process.Dispose()
}
