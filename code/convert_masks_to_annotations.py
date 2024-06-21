import os
import cv2
import json
import numpy as np

original_dir = "~/data/bbd/bbd2k5-images-image"
umring_dir = "~/data/bbd/bbd2k5-images-umring"
original_dir = os.path.expanduser(original_dir)
umring_dir = os.path.expanduser(umring_dir)

output_train_file = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017.json"
output_train_file = os.path.expanduser(output_train_file)
output_val_file = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_val2017.json"
output_val_file = os.path.expanduser(output_val_file)

annotations_train = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "buildings"}]
}

annotations_val = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "buildings"}]
}

image_id_train = 0
image_id_val = 0
annotation_id_train = 0
annotation_id_val = 0

filenames = [f for f in os.listdir(original_dir) if f.endswith("jpg") or f.endswith("png")]
filenames.sort()

split_index = 10
train_filenames = filenames[:split_index]
val_filenames = filenames[split_index:split_index * 2]

def process_files(filenames, original_dir, umring_dir, annotations, image_id_start, annotation_id_start):
    image_id = image_id_start
    annotation_id = annotation_id_start

    for i, filename in enumerate(filenames):
        original_path = os.path.join(original_dir, filename)
        umring_path = os.path.join(umring_dir, filename.replace("-image", "-umring"))

        original_img = cv2.imread(original_path)
        if original_img is None:
            print(f"Failed to read original image: {original_path}")
            continue
        height, width, _ = original_img.shape

        umring_img = cv2.imread(umring_path, cv2.IMREAD_GRAYSCALE)
        if umring_img is None:
            print(f"Failed to read OSM image: {umring_path}")
            continue

        _, binary_umring_img = cv2.threshold(umring_img, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_umring_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print(f"No contours found in image: {filename}")
            continue

        image_info = {
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width
        }
        annotations["images"].append(image_info)

        for contour in contours:
            if cv2.contourArea(contour):
                segmentation = contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                }
                annotations["annotations"].append(annotation_info)
                annotation_id += 1

        image_id += 1

    return image_id, annotation_id

image_id_train, annotation_id_train = process_files(train_filenames, original_dir, umring_dir, annotations_train, image_id_train, annotation_id_train)
image_id_val, annotation_id_val = process_files(val_filenames, original_dir, umring_dir, annotations_val, image_id_val, annotation_id_val)

with open(output_train_file, "w") as f:
    json.dump(annotations_train, f, indent=4)

with open(output_val_file, "w") as f:
    json.dump(annotations_val, f, indent=4)

print(f"Training annotations saved to {output_train_file}")
print(f"Validation annotations saved to {output_val_file}")
