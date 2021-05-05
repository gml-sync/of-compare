import subprocess
from glob import glob
import shutil

def subp_run_str(cmd, output=True):
    print('RUN:', cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    if output:
        for line in process.stdout:
            print(line.decode(), end='')
    rc = process.poll()
    return rc

def subp_bash(cmd):
    subp_run_str("bash -c '" + cmd + "'")

subp_bash('cd scripts/validation; bash IRR-PWC_sintel.sh')

# for filepath in sorted(glob('/content/irr/saved_check_point/pwcnet/eval_temp/IRR_PWC/flo/frames/in/*')):
#     filename = filepath.split('/')[-1]
#     #shutil.copyfile(filepath, '/content/sintelall/MPI-Sintel-complete/training/frames/out/frame_' + filename.split('_')[1] + '.png')
#     shutil.copyfile(filepath, '/content/sintelall/MPI-Sintel-complete/training/frames/out/' + filename)
