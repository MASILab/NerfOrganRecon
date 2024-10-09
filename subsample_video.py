"""
Michael Kim
Vanderbilt University
MASI Lab
2024


Subsmaples organ videos (so that we end up with roughly 1750 frames per video, but can specify however you want)

These are then broken into a specified number of groups  (used 5 groups of 350 frames each)
"""

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math

def pa():
    parser = argparse.ArgumentParser(description='Subsample organ videos')
    parser.add_argument('root_dir', type=str, help='root directory of video dataset')
    parser.add_argument('--num_frames', type=int, default=1750, help='Number of frames that we want to extract (approximately)')
    parser.add_argument('--groups', type=int, default=1, help='Number of groups to break the frames into')
    parser.add_argument('--path_to_colmap_script', type=str, default='/NERF2MESH/nerf2mesh/scripts/colmap2nerf.py', help='Path to the colmap2nerf script')
    return parser.parse_args()


args = pa()
root = Path(args.root_dir)
num_groups = args.groups 

#find the video (should be in the root directory)
videos = [x for x in root.iterdir() if 'mp4' in x.suffix or 'MOV' in x.suffix]
if len(videos) > 1:
    raise ValueError("More than one video found in the root directory")

video = videos[0]

#determine the total time of the video
#cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 {}".format(video)
#num_frames = int(subprocess.check_output(cmd, shell=True))
time_cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}".format(video)
time_s = float(subprocess.check_output(time_cmd, shell=True))

#determine the frame rate to sample at
sample_rate = math.ceil(args.num_frames / time_s)
print("Total video time: {}".format(time_s))
print("Sampling at rate: {}".format(sample_rate))

#now, subsample the video with ffmpeg
frames_dir = root / 'frames'
frames_dir.mkdir(exist_ok=True)
#before running, check to make sure that the frames directory is empty
if len(list(frames_dir.iterdir())) == 0:
    #raise ValueError("Frames directory is not empty. Please remove all files before running this script")
    cmd = "ffmpeg -i {} -qscale:v 1 -qmin 1 -vf \"fps={}\" {}/%04d.jpg".format(video, sample_rate, frames_dir)
    print("Running command: {}".format(cmd))
    subprocess.run(cmd, shell=True)

#now, we need to break the frames into 5 groups of 350 frames each, evenly spaced
if not (root / 'Group1').exists() or len([x for x in (root / 'Group1').iterdir()]) == 0:
    
    imgs = [x for x in frames_dir.iterdir() if 'jpg' in x.suffix]
    imgs.sort()
    imgs = np.array(imgs)

    idxs = np.linspace(0, len(imgs), len(imgs)).astype(int)
    groups = idxs % num_groups

    for i in tqdm(range(num_groups)):
        group = groups == i
        #group = group.astype(int)
        group_img_dir = root / 'Group{}'.format(i+1) / 'images'
        group_img_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs[group]:
            cmd = "ln -s {} {}".format(img, str(group_img_dir / Path(img).name))
            subprocess.run(cmd, shell=True)

#now, we need to run COLMAP on each of the groups
group_dirs = [x for x in root.iterdir() if 'Group' in x.name]
colmap_script = Path(args.path_to_colmap_script)

for group in group_dirs:
    cmd = "python {} --images {} --run_colmap --hold 0".format(colmap_script, group / 'images')
    print("Running command: {}".format(cmd))
    subprocess.run(cmd, shell=True)

#python scripts/colmap2nerf.py --images ./data/custom/images/ --run_colmap
