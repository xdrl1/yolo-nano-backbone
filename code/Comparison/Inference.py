import cv2
import os
import torch
from yolox.utils import postprocess
from yolox.exp import get_exp
from yolox.data.data_augment import preproc as preprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import numpy as np
import random
#/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python

#paths
CONFIG_FILE_PATH = "../exps/example/custom/yolox_nano_deploy_relu_bird.py"
WEIGHTS_FILE_PATH = "../../float/baseline.pth"
ANNOTATION_FILE_PATH = "../../drone2021_copy/annotations/split_val_coco.json"
image_dir = "../../drone2021_copy/images"

# Load the YOLOX model
def load_model(config_file_path, weights_file_path):
    exp = get_exp(config_file_path, None)
    model = exp.get_model()
    model.eval()
    ckpt = torch.load(weights_file_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model

# Perform inference using the model
def inference(model, img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image {img_path} does not exist.")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read the image {img_path}. The image may be corrupted or the path is incorrect.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    exp = get_exp(CONFIG_FILE_PATH)
    # Preprocess the image
    img_preprocessed, _ = preprocess(img_rgb, exp.test_size, swap=(2,0,1))
    img_tensor = torch.from_numpy(img_preprocessed).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        #print(img_tensor.size()) ([1, 3, 2176, 3840])
        predictions = model(img_tensor)
        print("prediction size is" ,predictions.size())
        print("shape of tensor", img_tensor.size())
        predictions = postprocess(predictions, exp.num_classes, exp.test_conf, exp.nmsthre)

    return predictions, img

# Plot bounding boxes
def plot_bboxes(image,predictions,img_id):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Draw ground truth boxes in green
    # Get the annotations for the picked image
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    for ann in annotations:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)


    # Draw predicted boxes in red
    for pred in predictions:
        print(pred)
        # Iterate over each prediction tensor
        for bbox in pred:
            print(bbox)
            # Extract and convert the bounding box coordinates
            x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
            print(x1,y1,x2,y2)
            # Check for valid coordinates
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                continue  # Skip invalid boxes

            # Create the rectangle patch and add it to the plot
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.savefig(f"/home/vitis-ai-user/yolonano/code/Comparison/images2.2/{img_id}_test.png")

# Load the YOLOX model
model = load_model(CONFIG_FILE_PATH, WEIGHTS_FILE_PATH)

# Load the COCO annotations
coco = COCO(ANNOTATION_FILE_PATH)
img_ids = coco.getImgIds()
#chosoe random pic
for i in range(5    ):
    img_id = random.choice(img_ids)
    image_info = coco.loadImgs(img_id)[0]
    #get path to pic
    IMAGE_PATH = image_dir + '/' + image_info['file_name']

    # Get predictions from the model
    predictions, original_img = inference(model, IMAGE_PATH)
    
    if predictions is not None:
    # save as image
        plot_bboxes(original_img, predictions,img_id)
    else:
        print(f"No predictions were made for image with ID {img_id}")