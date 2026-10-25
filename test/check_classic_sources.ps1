param([string]$Project)
$ErrorActionPreference='Stop'
[xml]$document=[IO.File]::ReadAllText([IO.Path]::GetFullPath($Project))
$projectDirectory=[IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Project))
$vendoredRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../lc0/src/'))
foreach($oldDirectory in @('../usi/lc0','../usi/lc0_shogi')) {
  if(Test-Path -LiteralPath (Join-Path $PSScriptRoot $oldDirectory)) {
    throw "Obsolete nested lc0 directory remains: $oldDirectory"
  }
}
$ns=New-Object Xml.XmlNamespaceManager($document.NameTable)
$ns.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
$classic=@('neural/shared_params.cc','search/classic/search.cc','search/classic/node.cc','search/classic/params.cc','search/classic/stoppers/timemgr.cc','search/classic/stoppers/stoppers.cc')
$utility=@('optionsdict.cc','optionsparser.cc','logging.cc','random.cc','string.cc','commandline.cc','configfile.cc','filesystem.win32.cc','filesystem.posix.cc')
foreach($source in $document.SelectNodes('//m:ClCompile[@Include]',$ns)) {
  $name=$source.Include.Replace('\','/')
  if($name -match '(^|/)(UctSearch|Node|Message|PvMateSearch)\.cpp$') {throw "Legacy USI search linked: $name"}
  if($name -match '(^|/)lc0/src/(.*)$') {
    $relative=$Matches[2]
    $resolved=[IO.Path]::GetFullPath((Join-Path $projectDirectory $name))
    if(-not $resolved.StartsWith($vendoredRoot,[StringComparison]::OrdinalIgnoreCase)) {
      throw "Source outside the single vendored lc0 tree: $resolved"
    }
    if($relative -notin $classic -and -not ($relative.StartsWith('utils/') -and $relative.Substring(6) -in $utility)) {
      throw "Non-allowlisted upstream implementation: $name"
    }
  }
}
Write-Output "classic source boundary verified: $Project"
