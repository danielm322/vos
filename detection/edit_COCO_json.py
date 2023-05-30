from collections import defaultdict
import json
import numpy as np


# Instantiate coco parser
class COCOParser:
    def __init__(self, anns_file):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        # Dict of id: category pairs
        self.cat_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {'categories': coco['categories']}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {'licenses': coco['licenses']}
        self.info_dict = {'info': coco['info']}
        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann
        for img in coco['images']:
            # Remove leading zeros from filenames
            img['file_name'] = str(int(img['file_name'].split('.')[0])) + '.jpg'
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        # Licenses not actually needed per image
        # for license in coco['licenses']:
        #     self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

    def get_img_info(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [self.im_dict[im_id] for im_id in im_ids]

print("Subsetting COCO...")
# Load json and instantiate parser
coco_annotations_file = "../../id_bdd_ood_coco/annotations/instances_val2017_ood_wrt_bdd_rm_overlap.json"
coco= COCOParser(coco_annotations_file)

# Select the first x number of images to subset
number_of_selected_images = 100

# total number of images
total_images = len(coco.get_imgIds())
print("Total images: ", total_images)
# Choose randomly the images
# np.random.seed(40)
# selected_indexes = np.random.permutation(total_images)[:number_of_selected_images]
# Or choose the first x number of images
selected_indexes = [i for i in range(number_of_selected_images)]

# Get images ids
img_ids = coco.get_imgIds()
selected_img_ids = [img_ids[i] for i in selected_indexes]

# Subset the dictionary
ann_ids = coco.get_annIds(selected_img_ids)
anns = coco.load_anns(ann_ids)
imgs_info = coco.get_img_info(selected_img_ids)

# Build new dictionary
subset_coco_dict = {
    'info': coco.info_dict['info'],
    'licenses': coco.licenses_dict['licenses'],
    'images': imgs_info,
    'annotations': anns,
    'categories': coco.categories_original['categories']
}
print(f"Saved {len(imgs_info)} images")
# Save dictionary as json
with open("../../id_bdd_ood_coco/annotations/instances_val2017_ood_wrt_bdd_rm_overlap_subset.json", "w") as outfile:
    json.dump(subset_coco_dict, outfile)
