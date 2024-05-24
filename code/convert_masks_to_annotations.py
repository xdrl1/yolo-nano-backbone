import os
import cv2
import json
import matplotlib.pyplot as plt

original_dir = "~/data/bbd/bbd2k5-images-image"
osm_dir = "~/data/bbd/bbd2k5-images-osm"
original_dir = os.path.expanduser(original_dir)
osm_dir = os.path.expanduser(original_dir)

output_annotations_file = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017.json"
output_annotations_file = os.path.expanduser(output_annotations_file)

annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "buildings"}]
}

image_id = 0
annotation_id = 0

for filename in os.listdir(original_dir):
    if filename.endswith("jpg") or filename.endswith("png"):
        original_path = os.path.join(original_dir, filename)
        osm_path = os.path.join(osm_dir, filename)
        original_img = cv2.imread(original_path)
        if original_img is None:
            print(f"Failed to read original image: {original_path}")
            continue
        height, width, _ = original_img.shape

        # Read OSM image in grayscale
        osm_img = cv2.imread(osm_path, cv2.IMREAD_GRAYSCALE)
        if osm_img is None:
            print(f"Failed to read OSM image: {osm_path}")
            continue

        # Find contours in the OSM image
        contours, _ = cv2.findContours(osm_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print(f"No contours found in image: {filename}")
            continue

        # Append image info to annotations
        image_info = {
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width
        }
        annotations["images"].append(image_info)

        for contour in contours:
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

        # Overlay contours on the original image for verification
        cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2)

        # Save plot for verification
        plot_path = os.path.join("plots", f"{filename}.png")
        os.makedirs("plots", exist_ok=True)
        plt.figure()
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Contours for {filename}")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

# Save annotations to the JSON file in COCO format
with open(output_annotations_file, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations saved to {output_annotations_file}")
