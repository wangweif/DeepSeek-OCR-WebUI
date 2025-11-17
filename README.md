
```bash
git clone https://github.com/neosun100/DeepSeek-OCR-WebUI.git
cd DeepSeek-OCR-WebUI


# Create and activate conda environment (REQUIRED)
conda create -n deepseek-ocr-mlx python=3.11
conda activate deepseek-ocr-mlx
```


# mac

```bash
## Install PyTorch with MPS support
pip install torch torchvision

## Install required packages
pip install transformers==4.46.3 tokenizers==0.20.3
pip install fastapi uvicorn PyMuPDF Pillow
pip install einops addict easydict matplotlib

## Or install all dependencies at once
pip install -r requirements-mac.txt

## Verify installation (optional)
./verify_mac_env.sh
python web_service_unified.py
```

# linux平台
```bash
## Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

## Install dependencies
pip install transformers==4.46.3 tokenizers==0.20.3
pip install fastapi uvicorn PyMuPDF Pillow
pip install einops addict easydict matplotlib

## Start service (auto-detects CUDA backend)
CUDA_VISIBLE_DEVICES=7 python web_service_unified.py
http://192.168.8.88:8011
```