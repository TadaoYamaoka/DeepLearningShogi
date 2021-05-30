import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('onnxfile')
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--gpu_id', '-g', type=int, nargs='+', default=0)
args = parser.parse_args()

procs = []
for gpu_id in args.gpu_id:
    procs.append(subprocess.Popen([args.command, args.onnxfile, '-b', str(args.batchsize), '-g', str(gpu_id)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

for proc in procs:
    proc.wait()
    print(f"{' '.join(proc.args)}, returncode={proc.returncode}")
    for line in proc.stdout.readlines():
        sys.stdout.buffer.write(line)
    for line in proc.stderr.readlines():
        sys.stderr.buffer.write(line)
