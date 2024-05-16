import os
import cv2

mask_dir = "~/data/bbd/bbd2k5-images-osm"
mask_dir = os.path.expanduser(mask_dir)

output_annotations_file = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017.json"
output_annotations_file = os.path.expanduser(output_annotations_file)

annotations = []

for filename in mask_dir():
    if filename.endswith(".png"):
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            polygon = contour.squeeze().tolist()
            annotations.append({
                "image_id": filename,
                "polygon": polygon
            })

import json
with open(output_annotations_file, "w") as f:
    json.dump(annotations, f)
