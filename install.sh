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

# SupeRANSAC
git clone https://github.com/danini/superansac.git
sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev cmake build-essential libboost-all-dev libeigen3-dev
cd superansac/
pip install .

cd ../..

# UFM
git clone --recursive https://github.com/UniFlowMatch/UFM.git
cd UFM
cd UniCeption
pip install -e .
cd ..
pip install -e .


pip install -e .
