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

out_path="${img_path%.png}_without_intrinsic/"
if [ ! -d "$out_path" ]; then
    echo "Directory does not exist. Creating directory: $out_path"
    mkdir -p "$out_path"
else
    echo "Directory already exists: $out_path"
fi

# palette extraction
read -p "Enter number of colours in palette: " n_col
python3 palette_extraction/extractor.py --img "$img_path" --out "$out_path" --n_col "$n_col"

# palette refinement
obj_ori_path="${out_path}original_palette.obj"
read -p "Enter lambda factor: " lambda
python3 palette_extraction/refiner.py --img "$img_path" --obj "$obj_ori_path" --out "$out_path" --lambda_f "$lambda"

# mvc decomposition
obj_ref_path="${out_path}refined_palette.obj"
python3 MVC/mvc.py --img "$img_path" --pal "$obj_ref_path" --out "$out_path"

deactivate