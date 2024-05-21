import os
import cv2
import json
import matplotlib.pyplot as plt

mask_dir = "~/data/bbd/bbd2k5-images-"
mask_dir = os.path.expanduser(mask_dir)

output_annotations_file = "~/alan/buildings/yolo-nano-backbone/code/datasets/COCO/annotations/instances_train2017.json"
output_annotations_file = os.path.expanduser(output_annotations_file)

annotations = []

for i, filename in enumerate(os.listdir(mask_dir)):
    if i == 2:
        break

    if filename.endswith(".png"):
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Failed to print an image{mask_path}")
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print(f"No contours found in image: {filename}")
            continue

        for contour in contours:
            polygon = contour.squeeze().tolist()
            annotations.append({
                "image_id": filename,
                "polygon": polygon
            })

        print(f"Polygon for {filename}: {polygon}")

        plt.figure()
        plt.imshow(mask, cmap='gray')
        # contour = contour.squeeze()
        # plt.plot(contour[:, 0], contour[:, 1], 'r')
        # plt.title(f"Contours for {filename}")
        # plt.show()
        if contour.size > 0:
            contour = contour.squeeze()
            if contour.ndim == 2:
                plt.plot(contour[:, 0], contour[:, 1], 'r')
            plt.title(f"Contours for {filename}")
            # Save plot instead of show for debugging
            plot_path = os.path.join("plots", f"{filename}.png")
            os.makedirs("plots", exist_ok=True)
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            plt.close()


with open(output_annotations_file, "w") as f:
    json.dump(annotations, f)
