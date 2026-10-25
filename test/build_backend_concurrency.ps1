param(
  [ValidateSet('TensorRT','ONNX')][string]$Backend='TensorRT',
  [ValidateSet('Release','Debug')][string]$Configuration='Release',
  [string]$NyugyokuFeatures='true',
  [string]$Output='artifacts/multigpu_review/backend_concurrency'
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location $root
try {
  $folder=[IO.Path]::GetFullPath($Output)
  New-Item -ItemType Directory -Force $folder | Out-Null
  $project=if($Backend -eq 'TensorRT'){'usi/usi.vcxproj'}else{'usi_onnxruntime/usi_onnxruntime.vcxproj'}
  & MSBuild.exe $project "/p:Configuration=$Configuration" /p:Platform=x64 "/p:NyugyokuFeatures=$NyugyokuFeatures" "/p:OutDir=$folder/" "/p:IntDir=$folder/obj/" /p:TargetName=backend_concurrency "/p:ForceImportAfterCppTargets=$PSScriptRoot/backend_concurrency.targets" /m /nologo /v:minimal *> "$folder/build.log"
  if($LASTEXITCODE -ne 0){Get-Content "$folder/build.log" -Tail 30;throw 'Backend concurrency test build failed'}
  Write-Output "built $folder/backend_concurrency.exe"
} finally {
  Pop-Location
}
