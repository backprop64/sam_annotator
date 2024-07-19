# GUI for interactive segmentation with Segment Anything Model
A simple opencv gui with minimal dependencies that lets a user annotate instance segmenation data with Segment Anything Model turning user clicks into masks. This annotate assumes that you're creating class-agnostic detection dataset meaning everything gets stored with the same class ID. 

This was developed to quickly annotate instance segementation datasets used for training a mouse detection model (DAMM)

## (1) Setup our codebase locally 

```bash
$ conda create -n sam_annotator python=3.9 
$ conda activate sam_annotator
$ git clone https://github.com/backprop64/sam_annotator
$ pip install -r sam_annotator/requirements-gpu.txt
```
---

## (2) Get Segment Anything Model Weights (from [SAM repo](https://github.dev/facebookresearch/segment-anything))

- Download one of the models and add it to the argument when runninng the sam_data_annotator.py script (see below)
- smaller models will be faster: the size of the models from smallest to largest are the 'vit_b', 'vit_l', and 'vit_h'

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## (3) Start annotating

to start the GUI, make sure you are in your conda enviornment and run:

