#!/bin/bash
if [[ -z $1 ]]; then
    echo "Usage: $0 <path_to_organdir>" #/fs5/p_masi/kimm58/DeformSurfReg/nerf2mesh_processing/pancreas
    exit 1
fi

bgscript="/NERF2MESH/nerf2mesh/scripts/remove_bg.py"

organ_root=$(readlink -f $1)

find $organ_root -mindepth 1 -maxdepth 1 -type d -name 'Group*' | while read -r dir; do
    #imgdir
    imgdir=${dir}/images
    #run removal
    python3 ${bgscript} ${imgdir}
done
