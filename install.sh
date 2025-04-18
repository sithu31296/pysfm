## install third-parties
# RoMa
cd third_party
git clone https://github.com/Parskatt/RoMa.git
cd RoMa
pip install -e .
cd ..

# SuperGLUE
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git

# DepthPro
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
source get_pretrained_models.sh
cd ..

# MoGe
pip install git+https://github.com/microsoft/MoGe.git


pip install -e .
