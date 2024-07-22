# GUI for interactive segmentation with Segment Anything Model
A simple opencv gui with minimal dependencies that lets a user annotate instance segmenation data with Segment Anything Model turning user clicks into masks. This annotate assumes that you're creating class-agnostic detection dataset meaning everything gets stored with the same class ID. 

This was developed to quickly annotate instance segementation datasets used for developing a mouse detection model (DAMM)

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
- smaller models will be faster (...larger models will be more accurate): the size of the models from smallest to largest are the 'vit_b', 'vit_l', and 'vit_h'

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## (3) Start annotating

### Arguments:
run sam_data_annotator.py with the following arguments: 
- `--images_path`: Path to folder containing images for annotation
- `--metadata_path`: Path to metadata file (optional for first-time use)
- `--sam_weights_path`: Path to SAM model weights

**Note**: If you're starting annotations for the first time, the metadata file will be created automatically in the image folder.

### Controls
Once an image is displayed, use the following controls:

#### clicking objects to create masks (giving sam point prompts)
- **Left click**: Add a foreground point (pixel belonging to the object)
- **Right click**: Add a background point (pixel not belonging to the object)

#### Navigation
- **space**: Start annotating the next instance within the same image
  - Clears current point prompt
  - Adds mask/box to annotation
- **esc**: Save current annotation and go to the next image
  - Triggers SAM to encode the next image (may take a few seconds depending on hardware)
- **q**: Quit GUI
  - Current image annotation won't be saved
  - All previous annotations will be saved

## Citing our annotation tool 

If this tool was useful for your project, please cite us!

```

@article{kaul2024damm,
      author    = {Gaurav Kaul and Jonathan McDevitt and Justin Johnson and Ada Eban-Rothschild},
      title     = {DAMM for the detection and tracking of multiple animals within complex social and environmental settings},
      journal   = {bioRxiv},
      year      = {2024}
}
```

