# Multi-resolution 3D Face Reconstruction from Single-view RGB Images

## Usage
### Training
    CUDA_VISIBLE_DEVICES=0 python main.py -c save_ckpt_name

### Finetune
    CUDA_VISIBLE_DEVICES=0 python fintune.py -c load_ckpt_name -o output_save_dir
