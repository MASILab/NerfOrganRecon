# NerfOrganRecon
How to run code to reconstuct an organ surface from an .mp4 video

Author: Michael Kim

michael.kim@vanderbilt.edu

## Link to singularity image

https://vanderbilt.box.com/s/ppwfzob96vzggl544yhk8pv87rkm9qya

## Tested Environment

-Ubuntu 20.04

-Quadro RTX 5000 GPU

-CUDA 12.4

-NVIDIA Driver Version: 550.100

## NOTES

In order to have compatibility with a newer version of torch and CUDA, some of the source code was changed slightly from the original repository. It did not affect the results in any way when tested. The changes are as follows:

- changed all lines in `/NERF2MESH/nerf2mesh/meshutils.py` to be `pml.PureValue` instead of `pml.AbsoluteValue` as the `pymeshlab` Classes changed names
  
- changed lines in `/NERF2MESH/nerf2mesh/nerf/renderer.py` in the function `mark_unseen_triangles()` from `RasterizeGLContext(db=False)` to `RasterizeCudaContext()` to prevent having to access OpenGL in the singularity. Also made the same change on line 128 for the same file. If this ends up causing an issue, you may also be able to run the nerf2mesh training code by calling `xvfb-run` similar to the preprocessing script (but using the original code before changes to `renderer.py`).

## How to Run

### Setup

First step is to create an output directory and put your `.mp4` file into it. Also create a `temp` directory that will be used later:

```
mkdir OUTPUTS TEMP
mv </path/to/mp4> OUTPUTS
```
***
### Preprocessing

Next, you will need do some preprocessing of the video. First step is to subsample the video into separate images:

```
# the ---home flag requires an absolut path. You can get this by running 'readlink -f TEMP' or just knowing what it is
singularity exec -e --contain --env QT_QPA_PLATFORM=offscreen --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B outputs:/OUTPUTS nerf2mesh.simg xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python3 /SCRIPTS/subsample_video.py /OUTPUTS --num_frames <NUMBER_OF_FRAMES> --groups <NUMBER_OF_GROUPS>
```

This script will subsample the video into separate groups of images, where each group is subsampled equidistantly and sequentially. For instance:
- Group1: Frame 1, Frame 4, ..., Frame N-2
- Group2: Frame 2, Frame 5, ..., Frame N-1
- Group3: Frame 3, Frame 6, ..., Frame N

So if you would like to get 1750 frames total per video, with 350 frames per separate group, you would run:
```
singularity exec -e --contain --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B outputs:/OUTPUTS nerf2mesh.simg python3 /SCRIPTS/subsample_video.py /OUTPUTS --num_frames 1750 --groups 5
```

Keep in mind that GPUs have only so much memory, so trying to extract too many frames per video may result in an OOM error when training the model for surface reconstruction.

Also, note that if you wanted to inspect the images, they would be located in the `outputs/frames` directory. If you look into `outputs/GroupX/images`, you will see a bunch of symlinks that appear invalid but will work in the singularity if you continue using the same directory binding calls (do NOT delete/modify these links/files, as the subsequent steps expect them to be named in a particular way).

EDIT: Please note that sometimes the above command to separate the video into separate images may throw some errors. If you are running into issues, please try modifying the command as follows:

```
singularity exec -e --contain --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B outputs:/OUTPUTS nerf2mesh.simg bash -c "export DISPLAY=:1; export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; export QT_QPA_PLATFORM=offscreen; python3 /SCRIPTS/subsample_video.py /OUTPUTS --num_frames 1750 --groups 5"
```

This SHOULD fix the issues (COLMAP needs to see the DISPLAY sometimes even though it should be running in headless mode).

***
Next is to run carvekit to mask out the object in each image frame. The background should be relatively homogenous to make the extraction much easier. If the video has a lot of background heterogeneity, it may result in a worse surface in the end:
```
singularity exec -e --contain --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B outputs:/OUTPUTS nerf2mesh.simg bash -c "source /CARVEKIT/carvekit/bin/activate; bash /SCRIPTS/run_carvekit.sh /OUTPUTS"
```
***

### Training nerf2mesh model

Now that carvekit is run, we can train the nerf2mesh model. For each specific `Group` you have created, run the command:
```
singularity exec -e --contain --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B OUTPUTS:/OUTPUTS -B  nerf2mesh.simg python3 /NERF2MESH/nerf2mesh/main.py /OUTPUTS/GroupGROUPNUM --workspace /OUTPUTS/GroupGROUPNUM/part1 -O --data_format colmap --bound 1 --dt_gamma 0 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --iters 10000 --decimate_target 1e5 --sdf
```

When running, will need to bind the corresponding driver file to `/usr/lib/x86_64-linux-gnu/libcuda.so`. Otherwise, the code may not run properly because it cannot find the corresponding files that allow the code to interact with the GPU on your machine. The machine used to build/test this singularity had the driver located at `/usr/lib/x86_64-linux-gnu/libcuda.so.550.100`. Check to see where the NVIDIA driver is on your machine, as it may be in a different location or have a different name due to version differences. For Ubuntu 20.04, if you check the directory `/usr/lib/x86_64-linux-gnu/`, there may be a file called `libcuda.so`, which is a symlink to the driver (there may be a few symlinks chained, so you can get the true location of the file using `readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so`). So if you wanted to run `Group1` through the nerf2mesh pipeline and your NVIDIA driver file is `/usr/lib/x86_64-linux-gnu/libcuda.so.XXX.XXX`, you would run:

```
singularity exec -e --contain --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B /usr/lib/x86_64-linux-gnu/libcuda.so.XXX.XXX:/usr/lib/x86_64-linux-gnu/libcuda.so OUTPUTS:/OUTPUTS nerf2mesh.simg python3 /NERF2MESH/nerf2mesh/main.py /OUTPUTS/Group1 --workspace /OUTPUTS/Group1/part1 -O --data_format colmap --bound 1 --dt_gamma 0 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --iters 10000 --decimate_target 1e5 --sdf
```

If you encounter an error regarding `pymeshlab` and an `Undefined symbol`, then you will need to alter the call slightly: (EDIT: should be fixed now where this doesn't happen)

```
singularity exec -e --contain --nv --home /fs5/p_masi/choc8/NeRFsingularityTest/TEMP -B /tmp:/tmp -B /usr/lib/x86_64-linux-gnu/libcuda.so.550.107.02:/usr/lib/x86_64-linux-gnu/libcuda.so -B outputs:/OUTPUTS nerf2mesh.simg bash -c "export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/pymeshlab/lib:$LD_LIBRARY_PATH; python3 /NERF2MESH/nerf2mesh/main.py /OUTPUTS/Group1 --workspace /OUTPUTS/Group1/part1 -O --data_format colmap --bound 1 --dt_gamma 0 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --iters 10000 --decimate_target 1e5 --sdf"
```

***
If the above runs successfully, you should be able to run part2 as well with a very similar command (part2 refines the mesh further):
```
singularity exec -e --contain --nv --home </absolute/path/to/TEMP/dir> -B /tmp:/tmp -B /usr/lib/x86_64-linux-gnu/libcuda.so.XXX.XXX:/usr/lib/x86_64-linux-gnu/libcuda.so OUTPUTS:/OUTPUTS nerf2mesh.simg python3 /NERF2MESH/nerf2mesh/main.py /OUTPUTS/Group1 --workspace /OUTPUTS/Group1/part1 -O --data_format colmap --bound 1 --dt_gamma 0 --stage 1 --iters 5000 --lambda_normal 1e-2 --refine_remesh_size 0.01 --sdf
```

If you would like to specify other options (like a higher lambda on the laplacian regularization), please take a look at the nerf2mesh github: https://github.com/ashawkey/nerf2mesh



