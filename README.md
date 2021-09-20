# Multi-resolution 3D Face Reconstruction from Single-view RGB Images

## Environment
* python 3.6
* Pytorch 1.4.0
* Kaolin 0.1.0
* torch_geometric 1.6.0
* Pytorch3D 0.5.0

## Dataset
We use the 300W-LP and AFLW2000 dataset pre-processed by [Nonlinear-3DMM](https://github.com/tranluan/Nonlinear_Face_3DMM). The dataset format:

    data_path/
        ├── image
        ├── mask
        ├── obj
        ├── filelist
        ├── meandata
        └── process_obj



## Usage
### Training
    CUDA_VISIBLE_DEVICES=0 python main.py -c save_ckpt_name

### Finetune
    CUDA_VISIBLE_DEVICES=0 python fintune.py -c load_ckpt_name -o output_save_dir
