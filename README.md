# Multi-resolution 3D Face Reconstruction from Single-view RGB Images

## Dataset
We use the 300W-LP and AFLW2000 dataset pre-processed by [Nonlinear-3DMM](https://github.com/tranluan/Nonlinear_Face_3DMM). The dataset format:

data_path/
    ├── image
    ├── mask
    ├── obj
    ├── filelist
    ├── meanface
    └── process



## Usage
### Training
    CUDA_VISIBLE_DEVICES=0 python main.py -c save_ckpt_name

### Finetune
    CUDA_VISIBLE_DEVICES=0 python fintune.py -c load_ckpt_name -o output_save_dir
