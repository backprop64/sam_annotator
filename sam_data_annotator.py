import argparse
import copy
import json
import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm
from imantics import Mask
from segment_anything import sam_model_registry, predictor

parser = argparse.ArgumentParser(
    description = "This script is qqused for annotating data and saving them as coco style annotations"
)

parser.add_argument(
    "--images_path",
    help="input path to a folder of images you would like to create annotations for",
)

parser.add_argument(
    "--metadata_path",
    default='none/',
    help="output path to a folder where you would like to store annotations",
)

parser.add_argument(
    "--sam_weights_path",
    default='none/',
    help="outputh path to a folder where you would like to store annotations",
)

args = parser.parse_args()
try:
    if 'vit_h' in args.sam_weights_path:
        sam = sam_model_registry['vit_h'](checkpoint=args.sam_weights_path)
    if 'vit_l' in args.sam_weights_path:
        sam = sam_model_registry['vit_l'](checkpoint=args.sam_weights_path)
    if 'vit_b' in args.sam_weights_path:
        sam = sam_model_registry['vit_b'](checkpoint=args.sam_weights_path)
except:
    print('invalid segment anything model weights, checkout the readme for help')
    exit()

# Check if gpu (cuda) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")
sam.to(device)
SAM = predictor.SamPredictor(sam)

images_to_annotate = [os.path.join(args.images_path,img_path) for img_path in os.listdir(args.images_path) if img_path[-4:]=='jpeg' or img_path[-3:] == 'jpg']

# create metadata file if it doesnt exits
if args.metadata_path == 'none/':
    metadata = {'annotations':list()}
    metadata_path = os.path.join(args.images_path,'metadata.json')
    if not os.path.isfile(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    else:
        print("metadata file found at" + metadata_path)
        print("please specify using --metadata_path argumant")
        exit()
else:
    try:
        metadata = json.load(open(args.metadata_path))
        metadata_path = args.metadata_path
        annotations = metadata['annotations']
        imgs_already_annotated = [ann['file_name'] for ann in annotations]
        images_to_annotate = [img for img in images_to_annotate if os.path.join(os.path.normpath(img).split(os.sep)[-2],os.path.normpath(img).split(os.sep)[-1]) not in imgs_already_annotated]
        print(len(images_to_annotate), 'images left to annotate')

    except:
        print('failed to load', args.metadata_path)
        exit()
        
datapoint_template = {
    "file_name": "",
    "height": 0,
    "width": 0,
    "image_id": 0,
    "annotations": [],
}

annotaion_template = {
    "bbox": [],
    "bbox_mode": "BoxMode.XYXY_ABS",
    "category_id": 0, # all 
    "segmentation": [],
}

def make_masks(event, x, y, flags, param):
    global ix, iy, points,bbox,point_lables,polygon_verticies,segmentations, instances,original_img, img
    ## this block will a polygon for a single instance
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        point_lables.append(1)
        img = cv2.circle(img, (x, y), 1, (7,99,4), 2)


    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        point_lables.append(0)
        img = cv2.circle(img, (x, y), 1, (30,17,155), 2)


    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:

        pts = np.array(points)
        pts_lables = np.array(point_lables)

        masks, qualities, __ = SAM.predict(point_coords=pts, point_labels=pts_lables)
        polygon_verticies = []
        
        img = copy.deepcopy(original_img)
        binary_mask = masks[np.argmax(qualities)]
        polygons = Mask(binary_mask).polygons()

        for v in polygons.points:
            v = v.reshape((-1, 1, 2))
            polygon_verticies.append(v)

        segmentations = polygons.segmentation
        bbox = polygons.bbox()
        img = cv2.polylines(img, polygon_verticies, isClosed=True, color=(38,220,244), thickness=1) 
        img = cv2.rectangle(img, pt1=bbox.min_point,pt2=bbox.max_point, color=(38,220,244), thickness=1) 

        for ann in instances:
            img = cv2.polylines(img, ann['polygon_verticies'], isClosed=True, color=(186, 82, 15), thickness=2) 
            img = cv2.rectangle(img, pt1=ann['bbox'].min_point,pt2=ann['bbox'].max_point, color=(38,220,244), thickness=1) 


        for foreground,pt in zip(point_lables,points):
            if foreground == 1:
                img = cv2.circle(img, pt, 1, (7,99,4), 2)
            else:
                img = cv2.circle(img, pt, 1, (30,17,155), 2)    


random.shuffle(images_to_annotate)

for img_path in tqdm(images_to_annotate):
    
    instances = []
    points = []
    bbox = []
    point_lables = []
    polygon_verticies = []
    segmentations = []

    original_img = cv2.imread(img_path)
    img = copy.deepcopy(original_img)
    
    SAM.set_image(original_img)

    window_name = img_path.split(os.sep)[-1]
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, make_masks)

    datapoint = copy.deepcopy(datapoint_template)

    datapoint["file_name"] = os.path.join(os.path.normpath(img_path).split(os.sep)[-2],os.path.normpath(img_path).split(os.sep)[-1])
    datapoint["height"] = img.shape[0]
    datapoint["width"] = img.shape[1]
    datapoint["image_id"] = len(metadata['annotations'])

    while True:
        cv2.imshow(window_name, img)

        if cv2.waitKey(10) == 32: ## go to next instance 
            img = cv2.polylines(img, polygon_verticies, isClosed=True, color=(186, 82, 15), thickness=2) 
            instances.append({"segmentation": segmentations,"polygon_verticies":polygon_verticies,"bbox":bbox})

            points = []
            bbox = []
            point_lables = []
            polygon_verticies = []
            segmentations = []

            
        if cv2.waitKey(10) == ord('q'):
            print('saved metadata:', len(metadata['annotations']), 'images annotated')
            print('exiting')
            exit()
                

        if cv2.waitKey(10) == 27:

            for ann in instances:
                instances_annotation = copy.deepcopy(annotaion_template)
                seg = ann["segmentation"]
                seg = [s for s in seg if len(s) > 6]

                seg_lengths = [len(s) for s in seg]
                if len(seg_lengths) == 0:
                    print('seg is empty')
                    continue

                if min(seg_lengths) > 6:
                    instances_annotation["bbox"] = [ann['bbox'].min_point,ann['bbox'].max_point]

                    instances_annotation["segmentation"] = seg
                    instances_annotation["category_id"] = 0
                    datapoint["annotations"].append(instances_annotation)
                else:
                    print('seg is too small',seg)

                    continue

            metadata['annotations'].append(datapoint)
            print(len(datapoint["annotations"]), 'instances addded to image')
            print('saved metadata checkpoint:', len(metadata['annotations']), 'images annotated')

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            break # exit Inf loop and go to next image
    cv2.destroyAllWindows()
