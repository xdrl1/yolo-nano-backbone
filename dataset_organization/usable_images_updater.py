import json
import os
from pycocotools.coco import COCO


#Use this script with caution it will delete not usable images and modify the annotation file. Make a copy of your data before using!!!
# I reduced the dataset from 60gb to around 5gb.


# Checks if a bird moved by at least 5 pixels
def bird_moving(image_id1, image_id2, coco, bird_category_id=0, displacement_threshold=5 ):
    # Get the annotations for the current image IDs
    annotations1 = coco.loadAnns(coco.getAnnIds(imgIds=image_id1, catIds=[bird_category_id]))
    annotations2 = coco.loadAnns(coco.getAnnIds(imgIds=image_id2, catIds=[bird_category_id]))

    # Check if annotations are not empty
    if annotations1 and annotations2:
        bbox1 = annotations1[0]['bbox']
        bbox2 = annotations2[0]['bbox']

        # Calculate the displacement of the bounding box
        displacement_x = abs(bbox2[0] - bbox1[0])
        displacement_y = abs(bbox2[1] - bbox1[1])

        if displacement_x >= displacement_threshold or displacement_y >= displacement_threshold:
            return True

    return False
# returns several information about the bird dataset. The usable_image_list is teh most important one, it is used in the delete function. Check line 87
def usable_images(annotations_file):
    try:
        # Load the COCO annotations file
        coco = COCO(annotations_file)
    except Exception as e:
        print(f"Error loading annotations file: {e}")
        return -1

    # Get all image IDs in the dataset
    all_image_ids = coco.getImgIds()


    # Initialize counters
    images_without_birds_count = 0
    images_with_large_boxes_count = 0
    usable_images = 0
    image_list_no_birds = []
    usable_image_list = []

    # Iterate through each image
    for i in range(len(all_image_ids) - 1):
        image_id1 = all_image_ids[i]
        image_id2 = all_image_ids[i + 1]



        # Get the annotations for the current image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id1))

        # Check if there are any annotations with category 0 (bird)
        has_bird = any(ann['category_id'] == 0 for ann in annotations)

        if not has_bird:
            # If no bird annotations are found, increment the images_without_birds_count
            images_without_birds_count += 1
            image_list_no_birds.append(image_id1)

        # Check if there are any large bounding boxes (greater than 40x40)
        has_large_boxes = any(
            (ann['category_id'] == 0) and (ann['bbox'][2] > 40 and ann['bbox'][3] > 40)
            for ann in annotations
        )

        if has_large_boxes:
            # If large bounding boxes are found, increment the images_with_large_boxes_count
            images_with_large_boxes_count += 1
        # Check for usable images
        if has_bird and has_large_boxes and bird_moving(image_id1, image_id2, coco, bird_category_id=0, displacement_threshold=5):
            usable_images += 1
            usable_image_list.append(image_id1)
    return images_without_birds_count, images_with_large_boxes_count, image_list_no_birds,usable_images,usable_image_list #I know they are redundant, but troughout the process we were interested in different parameters


# this function deletes all images which are not in the usable_image_list and also updates the annotation file.
def delete_not_usable_images(annotations_file, images_dir):
    # Load the COCO annotations file
    coco = COCO(annotations_file)
    #get usable image ids
    usable_image_list = usable_images(annotations_file)[4]

    # Get all image IDs in the dataset
    all_image_ids = coco.getImgIds()

    # delete all non usable images
    updated_annotations = []
    updated_images = []

    for image_id in all_image_ids:
        if image_id not in usable_image_list:
            image_file_path = os.path.join(images_dir, coco.loadImgs(ids=[image_id])[0]['file_name'])
            os.remove(image_file_path)

    for ann in coco.dataset['annotations']:
        if ann['image_id'] in usable_image_list:
            updated_annotations.append(ann)

    for img in coco.dataset['images']:
        if img['id'] in usable_image_list:
            updated_images.append(img)

    #update json file
    coco.dataset['annotations'] = updated_annotations
    coco.dataset['images'] = updated_images
    with open(annotations_file, "w") as f:
        json.dump(coco.dataset, f)


if __name__ == "__main__":
    # Path to annotation files and image dir
    annotations_file = '/Volumes/Externe_SSD/Engineering_Project/drone2021_copy/annotations/split_train_coco.json'
    annotations_file_val = '/Volumes/Externe_SSD/Engineering_Project/drone2021_copy/annotations/split_val_coco.json'
    images_dir = '/Volumes/Externe_SSD/Engineering_Project/drone2021_copy/images/'

    delete_not_usable_images(annotations_file, images_dir)
    delete_not_usable_images(annotations_file_val, images_dir)
    print("done")


