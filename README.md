# <p align=center>`RoSeg: Robust Segmentation with Depth Estimation for Polyp Images`</p>

Authors : [Yeonkyu Kwak](mailto:yeonkyu0820@g.skku.edu), [Jongpil Jeong](mailto:jpjeong@skku.edu)

## Abstract
Precise polyp segmentation in colonoscopy images is critical for early diagnosis of colorectal cancer but remains challenging due to illumination variations, boundary ambiguities, and complex backgrounds. In this study, we propose Robust Segmentation with Depth Estimation (RoSeg), a Transformer-CNN model that leverages a geometry prior to enhance segmentation accuracy. To model spatial dependencies without physical depth sensors, we introduce a Geometry Self-Attention (GSA) mechanism guided by pseudo-depth maps derived from RGB inputs. Furthermore, we employ a complementary encoder structure that integrates DFormerv2 for global context and residual blocks for local detail preservation. Through comprehensive experiments on five benchmark datasets, including ablation studies and qualitative analyses, we rigorously validated the effectiveness of the proposed components. The ablation results empirically confirm that the geometry prior is essential for spatial awareness, while qualitative evaluations demonstrate RoSeg's superior capability in delineating fine boundaries and detecting small polyps. RoSeg achieved state-of-the-art results on the challenging ETIS-LaribPolypDB and CVC-ColonDB datasets and demonstrated strong robustness in cross-dataset and unseen dataset experiments. 

## 1. Get Start

**0. Install**

```bash
conda create -n RoSeg python=3.10 -y
conda activate RoSeg

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
pip install gradio_imageslider gradio==4.29.0 matplotlib
conda install scikit-learn -y
conda install -c conda-forge albumentations -y
pip install xformers
pip install ptflops
```

**1. Download Datasets and Checkpoints.**

**Datasets:**
  
The datasets used in this study are publicly available at: 
- Kvasir-SEG: [here](https://datasets.simula.no/kvasir-seg/). 
- CVC-ClinicDB: [here](https://polyp.grand-challenge.org/CVCClinicDB/). 
- ETIS-LaribpolypDB: [here](https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=share_link). 
- CVC-ColonDB: [here](https://drive.google.com/drive/folders/1-gZUo1dgsdcWxSdXV9OAPmtGEbwZMfDY?usp=share_link).

You can also download Train/Test datasets seperated by Pranet
- [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
- [Google Drive Link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

**Checkpoints**
- DFormerv2_Small_pretrained.pth: [here](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/pretrained).
- depth_anything_v2_vits.pth: [download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true).
<details>
<summary>Orgnize the checkpoints and dataset folder in the following structure:</summary>
<pre><code>

```shell
<RoSeg>
|-- <backbone>
    |-- <pretrained>
      |-- <DFormerv2_Small_pretrained.pth>
|-- <DepthAnythingV2>
    |-- <checkpoints>
      |-- <depth_anything_v2_vits.pth>
<datasets>
|-- <DatasetName1>
|-- <DatasetName2>
|-- ...
```

</code></pre>
</details>

**Train/Test**
- To train and test the model, please run `train.ipynb`.
