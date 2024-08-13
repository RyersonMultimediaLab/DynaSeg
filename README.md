# DynaSeg

## Overview
DynaSeg is a deep dynamic fusion method for unsupervised image segmentation that effectively balances feature similarity and spatial continuity. It automates parameter tuning through dynamic weighting, allowing for efficient segmentation without extensive hyperparameter adjustments. DynaSeg integrates seamlessly with existing segmentation networks and has been tested to achieve state-of-the-art performance on multiple benchmark datasets.
![Alt text](assets/DynamicWeightingSchemeforUnsupervisedImageSegmentation.png)
## Installation
To set up DynaSeg, you need to install the following OpenMMLab packages:

- **MIM** >= 0.1.5
- **MMCV-full** >= v1.3.14
- **MMDetection**
- **MMSegmentation**
- **MMSelfSup**

Install the required packages using the following commands:

```bash
pip install openmim mmdet mmsegmentation mmselfsup
mim install mmcv-full
``` 


## Usage

### Data Preparation
To prepare the necessary data for training, follow these steps:

1. **Download Datasets**: Download the training set and validation set of the COCO dataset, along with the stuffthing map.

2. **Unzip and Organize Data**: Unzip the data and organize it into the following directory structure:
data/
```kotlin
data/
├── curated
│   ├── train2017
│   │   ├── Coco164kFull_Stuff_Coarse_7.txt
│   ├── val2017
│   │   ├── Coco164kFull_Stuff_Coarse_7.txt
├── coco
│   ├── annotations
│   │   ├── train2017
│   │   │   ├── xxxxxxxxx.png
│   │   ├── val2017
│   │   │   ├── xxxxxxxxx.png
│   ├── train2017
│   │   ├── xxxxxxxxx.jpeg
│   ├── val2017
│   │   ├── xxxxxxxxx.jpeg
``` 

The curated directory contains the data splits for unsupervised segmentation, structured according to the splits used by PiCIE.
## License
This project is licensed under the [MIT License](LICENSE).

## Citation
If you use DynaSeg in your research, please cite our work:
```sql
@article{guermazi2024dynaseg,
  title={DynaSeg: A deep dynamic fusion method for unsupervised image segmentation incorporating feature similarity and spatial continuity},
  author={Guermazi, Boujemaa and Ksantini, Riadh and Khan, Naimul},
  journal={Image and Vision Computing},
  pages={105206},
  year={2024},
  publisher={Elsevier}
}
```
