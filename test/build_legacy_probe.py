"""Rebuild the pinned legacy engine with root-NN probes or inference counters.

All generated sources and binaries stay under artifacts. No legacy search is
linked into the new engine. Extraction from Git remains reproducible after the
old USI search files have been removed.
"""
import argparse
import pathlib
import subprocess
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "0cc16854474ab3c7c4f7be66c624f9f9001f7543"
NS = "http://schemas.microsoft.com/developer/msbuild/2003"
ET.register_namespace("", NS)


def git_file(name):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{REVISION}:{name}"], cwd=ROOT
    ).decode("utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyugyoku", action="store_true")
    parser.add_argument("--onnx", action="store_true", help="Pinned old search and NN with restored ORT 1.20.1 dependencies")
    parser.add_argument("--nnstats", action="store_true", help="Count completed EvalNode inference instead of printing root NN probes")
    args = parser.parse_args()
    prefix = "stats_" if args.nnstats else "probe_"
    folder = ROOT / "artifacts" / "baseline" / (prefix + ("onnx_" if args.onnx else "") + ("ny" if args.nyugyoku else "no_ny"))
    folder.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-tree", "--name-only", REVISION, "usi/"], cwd=ROOT
    ).decode().splitlines()
    for name in names:
        path = pathlib.PurePosixPath(name)
        if len(path.parts) == 2 and path.suffix in {".cpp", ".h"}:
            text = git_file(name)
            if args.nnstats and path.name == "UctSearch.cpp":
                marker = '#include "UctSearch.h"'
                assert text.count(marker) == 1
                text = text.replace(marker, marker + '''
namespace {
std::atomic<uint64_t> inference_evaluations{0}, inference_batches{0};
}
void DumpLegacyNeuralStats() {
    std::cout << "info string nnstats evaluations "
              << inference_evaluations.load(std::memory_order_relaxed)
              << " batches " << inference_batches.load(std::memory_order_relaxed)
              << std::endl;
}
''')
                marker = 'grp->nn_forward(thread_id, policy_value_batch_size, features1, features2, y1, y2);'
                assert text.count(marker) == 1
                text = text.replace(marker, marker + '''
    inference_evaluations.fetch_add(policy_value_batch_size, std::memory_order_relaxed);
    inference_batches.fetch_add(1, std::memory_order_relaxed);
''')
            elif args.nnstats and path.name == "UctSearch.h":
                text += '\nvoid DumpLegacyNeuralStats();\n'
            elif args.nnstats and path.name == "main.cpp":
                marker = 'else if (token == "setoption") {'
                assert text.count(marker) == 1
                text = text.replace(marker, '''else if (token == "nnstats") {
            StopUctSearch();
            if (th.joinable()) th.join();
            DumpLegacyNeuralStats();
        }
        ''' + marker)
            elif path.name == "UctSearch.cpp":
                marker = "*policy_value_batch[i].value_win = (float)*value;"
                assert text.count(marker) == 1
                text = text.replace(marker, marker + '''
        if (node == tree->GetCurrentHead()) {
            std::ostringstream probe;
            probe << std::setprecision(9) << "info string nnprobe value " << (float)*value;
            for (int j = 0; j < child_num; ++j)
                probe << ' ' << uct_child[j].move.toUSI() << ':' << uct_child[j].nnrate;
            std::cout << probe.str() << std::endl;
        }
''')
            (folder / path.name).write_text(text, encoding="utf-8")
    project_directory = ROOT / ("usi_onnxruntime" if args.onnx else "usi")
    if args.onnx:
        for name in ["nn_onnxruntime.cpp", "nn_onnxruntime.h"]:
            (folder / name).write_text(git_file("usi_onnxruntime/" + name), encoding="utf-8")
    project = ET.fromstring(git_file(project_directory.name + "/" + project_directory.name + ".vcxproj"))
    for item in project.findall(f".//{{{NS}}}ClCompile[@Include]"):
        old = item.attrib["Include"]
        resolved = (project_directory / pathlib.Path(old)).resolve()
        if resolved.parent in {ROOT / "usi", project_directory}:
            resolved = folder / resolved.name
        item.set("Include", str(resolved))
    for item in project.findall(f".//{{{NS}}}CustomBuild[@Include]"):
        item.set("Include", str((project_directory / item.attrib["Include"]).resolve()))
    for settings in project.findall(f".//{{{NS}}}ItemDefinitionGroup/{{{NS}}}ClCompile"):
        ET.SubElement(settings, f"{{{NS}}}AdditionalIncludeDirectories").text = ";".join(
            map(str, [folder, ROOT / "usi", ROOT / "cppshogi"])
        ) + ";%(AdditionalIncludeDirectories)"
        ET.SubElement(settings, f"{{{NS}}}AdditionalOptions").text = "/utf-8 %(AdditionalOptions)"
    for item in project.findall(f".//{{{NS}}}IncludePath"):
        item.text = str(folder) + ";" + str(ROOT / "usi") + ";" + str(ROOT / "cppshogi") + ";" + (item.text or "")
    text = ET.tostring(project, encoding="unicode")
    if args.onnx:
        # Both probes use identical runtime versions; the old NN source itself
        # remains unchanged, including its input-feature and output handling.
        text = text.replace("..\\packages\\", str(ROOT / "packages") + "\\")
        text = text.replace("OnnxRuntime.DirectML.1.7.0", "OnnxRuntime.DirectML.1.20.1")
        text = text.replace("AI.DirectML.1.4.2", "AI.DirectML.1.15.2")
        if args.nyugyoku:
            text = text.replace("ONNXRUNTIME;", "NYUGYOKU_FEATURES;ONNXRUNTIME;")
    if not args.nyugyoku:
        text = text.replace("NYUGYOKU_FEATURES;", "").replace(" -DNYUGYOKU_FEATURES", "")
    text = text.replace("$(Platform)\\$(TargetedSDKConfiguration)", "$(IntDir)")
    project_path = folder / "legacy_probe.vcxproj"
    project_path.write_text(text, encoding="utf-8")
    subprocess.run(["MSBuild.exe", str(project_path), "/p:Configuration=Release", "/p:Platform=x64", "/m", "/nologo", "/v:minimal"], check=True, cwd=ROOT)
    print(folder / "x64" / "Release" / ("dlshogi_onnxruntime.exe" if args.onnx else "dlshogi_tensorrt.exe"), flush=True)


if __name__ == "__main__":
    main()
