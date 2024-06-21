import os
import json

existing_annotations = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017.json"
output_annotations = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017_coco.json"
existing_annotations = os.path.expanduser(existing_annotations)
output_annotations = os.path.expanduser(output_annotations)

with open(existing_annotations, "r") as f:
    existing_annotations = json.load(f)

coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "building"}]
}

image_id_mapping = {}
for idx, image in enumerate(existing_annotations["images"]):
    image_id_mapping[image["id"]] = idx
    image["id"] = idx
    coco_annotations["images"].append(image)

# Update annotation image IDs and append to COCO annotations
for annotation in existing_annotations["annotations"]:
    annotation["id"] = len(coco_annotations["annotations"])
    annotation["image_id"] = image_id_mapping[annotation["image_id"]]
    coco_annotations["annotations"].append(annotation)

# Save annotations to the new file
with open(output_annotations, "w") as f:
    json.dump(coco_annotations, f, indent=4)

print(f"Annotations saved to {output_annotations}")
