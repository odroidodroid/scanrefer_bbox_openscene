# 3D visual grounding team project

### Download sourcefile

    git clone https://github.com/odroidodroid/scanrefer_bbox_openscene

### Dataset Preparation

    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip
    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip
    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_openseg.zip

### Build ops_dcnv3

    cd ops_dcnv3
    sh make.sh

### Requirements

    apt install libglfw-dev
    pip install pyglfw
    pip install scipy imageio tensorboardx open3d plyfile
	pip install openai
    git clone https://github.com/openai/CLIP.git
    
 ### Run Visualization
 

    python run/visualize_bbox.py
    val number : {type number from 0 to 140}
    
   ### If you got glfw error/warning & if you use docker
   

    export DISPLAY=:1.0
