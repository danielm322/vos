from collections import defaultdict
import json
import numpy as np


# Instantiate BDD parser
class BDDParser:
    def __init__(self, anns_file):
        with open(anns_file, 'r') as f:
            bdd = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.annId_dict = {}
        self.im_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {'categories': bdd['categories']}
        for ann in bdd['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann
        for img in bdd['images']:
            # Remove leading zeros from filenames
            # img['file_name'] = str(int(img['file_name'].split('.')[0])) + '.jpg'
            self.im_dict[img['id']] = img

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def get_img_info(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [self.im_dict[im_id] for im_id in im_ids]

# Make sure we are editing the correct database
print("Subsetting BDD...")
# Load json and instantiate parser
# In this case the dataset is located outside the project folder, inside the parent folder of the project
bdd_annotations_file = "../../bdd100k/val_bdd_converted.json"
bdd = BDDParser(bdd_annotations_file)

# Select the first x number of images to subset
number_of_selected_images = 200

# total number of images
total_images = len(bdd.get_imgIds())
print("Total images: ", total_images)
# Choose randomly the images
# np.random.seed(40)
# selected_indexes = np.random.permutation(total_images)[:number_of_selected_images]
# Or choose the first x number of images
selected_indexes = [i for i in range(number_of_selected_images)]

# Get images ids
img_ids = bdd.get_imgIds()
selected_img_ids = [img_ids[i] for i in selected_indexes]

# Subset the dictionary
ann_ids = bdd.get_annIds(selected_img_ids)
annotations = bdd.load_anns(ann_ids)
images_info = bdd.get_img_info(selected_img_ids)

# Build new dictionary
subset_bdd_dict = {
    'categories': bdd.categories_original['categories'],
    "type": "instances",
    'images': images_info,
    'annotations': annotations,
}
print(f"Saved {len(images_info)} images")
# Save dictionary as json
with open("../../bdd100k/val_bdd_converted_subset.json", "w") as outfile:
    json.dump(subset_bdd_dict, outfile)
