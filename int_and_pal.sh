#!/bin/bash

source .venv/bin/activate

# need to enter image path
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_image>"
    deactivate
    exit 1
fi

# get paths to be used
img_path=$1

out_path="${img_path%.png}_with_intrinsic/"
if [ ! -d "$out_path" ]; then
    echo "Directory does not exist. Creating directory: $out_path"
    mkdir -p "$out_path"
else
    echo "Directory already exists: $out_path"
fi

# intrinsic decomposition
python3 intrinsic_shading/decompose.py --img "$img_path" --out "$out_path"

# palette extraction
alb_path="${out_path}albedo.png"
read -p "Enter number of colours in palette: " n_col
python3 palette_extraction/extractor.py --img "$alb_path" --out "$out_path" --n_col "$n_col"

# palette refinement
obj_ori_path="${out_path}original_palette.obj"
read -p "Enter lambda factor: " lambda
python3 palette_extraction/refiner.py --img "$alb_path" --obj "$obj_ori_path" --out "$out_path" --lambda_f "$lambda"

# mvc decomposition
obj_ref_path="${out_path}refined_palette.obj"
python3 MVC/mvc.py --img "$alb_path" --pal "$obj_ref_path" --out "$out_path"

# intrinsic reconstruction
new_alb_path="${out_path}new_albedo.png"
shd_path="${out_path}shading.png"
python3 intrinsic_shading/reconstruct.py --alb "$new_alb_path" --shd "$shd_path" --out "$out_path"

deactivate