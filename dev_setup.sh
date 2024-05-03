pip install -U torch torchvision torchaudio

pip install torchdiffeq matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections wandb
pip install hydra-core opencv-python torch-fidelity webdataset einops pytorch_lightning
pip install torchmetrics --upgrade
pip install opencv-python causal-conv1d

cd dis_causal_conv1d && pip install -e . && cd ..
cd dis_mamba && pip install -e . && cd ..

#wandb.Video() need it
pip install moviepy imageio

pip install scikit-learn --upgrade
pip install transformers==4.36.2

# (optional) for generating the hilbert path
pip install numpy-hilbert-curve

# (optional)  to use the ucf101 frame extracting
pip install av

#for FDD metrics
# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
