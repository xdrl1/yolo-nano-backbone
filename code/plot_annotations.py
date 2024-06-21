import os
import cv2
import json

output_annotations = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017_coco.json"
output_annotations = os.path.expanduser(output_annotations)

original_dir = "~/data/bbd/bbd2k5-images-image"
original_dir = os.path.expanduser(original_dir)

output_dir = "./plots/overlaid_images"
os.makedirs(output_dir, exist_ok=True)

with open(output_annotations, "r") as f:
    coco_annotations = json.load(f)

image_bbox_dict = {}

for annotation in coco_annotations["annotations"]:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]

    if image_id not in image_bbox_dict:
        image_bbox_dict[image_id] = []

    image_bbox_dict[image_id].append(bbox)

for image_id, bboxes in image_bbox_dict.items():
    image_filename = coco_annotations["images"][image_id]["file_name"]
    #print(image_filename)

    image_path = os.path.join(original_dir, image_filename)
    original_image = cv2.imread(image_path)

    for bbox in bboxes:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)





   # overlaid_image = original_image.copy()


    # mask = np.zeros_like(original_image[:, :, 0])
    # cv2.fillPoly(mask, [np.array(segmentation)], 255)

    # overlaid_image = cv2.addWeighted(original_image, 0.5, cv2.cvtColor(cv2.merge([mask, mask, mask]), cv2.COLOR_GRAY2BGR), 0.5, 0)

    overlaid_image_path  = os.path.join(output_dir, f"overlaid_{image_filename}")
    cv2.imwrite(overlaid_image_path , original_image)

    print(f"Overlay saved to {output_dir}")
