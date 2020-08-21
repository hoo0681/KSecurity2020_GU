#require sudo apt install upx
import os
import argparse
import subprocess
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datadir", required=True, help="directory for packing")
parser.add_argument("-o", "--output", required=True, help="output directory")
args = parser.parse_args()

if not os.path.exists(args.datadir):
    parser.error("ember model {} does not exist".format(args.modelpath))
if not os.path.exists(args.output):
    os.makedirs(args.output)

for _file in tqdm.tqdm(os.listdir(args.datadir)):
    path = os.path.join(args.datadir, _file)
    output = os.path.join(args.output, _file)

    command = ['upx -l ' + path]
    r = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.STDOUT)
    NotPacker = r.communicate()[0].decode('utf-8')
    
    if "NotPackedException" in  NotPacker:
        command = ['upx -o ' + output + ' ' + path]
        subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.STDOUT)
    else:
        command = ['cp ' + path + ' ' + output]
        subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.STDOUT)
    
print("Done")
