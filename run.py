import os
import subprocess

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




# Split frames
def frame_ffmpeg_split(video_path, output_folder):
    subp_run_str(['ffmpeg -i ' + video_path +
            ' -qscale:v 2 ' + output_folder + '/frame_%04d.jpg'])


# Effective path must be '/'
os.makedirs('frames')
frame_ffmpeg_split('vids/09_l4.mkv', 'frames')