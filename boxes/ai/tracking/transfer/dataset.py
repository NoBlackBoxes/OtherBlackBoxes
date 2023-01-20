import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pycocotools.coco import COCO

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None, target_transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = self.targets[idx,:]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, target

# Load dataset
def prepare(dataset_name, split):

    # Filter train and test datasets
    image_paths, targets = filter(dataset_name)

    # Split train/test
    num_samples = len(targets)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = (image_paths[train_indices], targets[train_indices])
    test_data = (image_paths[test_indices], targets[test_indices])

    return train_data, test_data

# Filter dataset
def filter(dataset_name):

    # Specify paths
    coco_folder = '/home/kampff/Dropbox/Voight-Kampff/Technology/Datasets/coco'
    annotations_path = coco_folder + '/annotations/person_keypoints_' + dataset_name + '.json'
    images_folder = coco_folder + '/' + dataset_name

    # Initialize the COCO API
    coco=COCO(annotations_path)

    # Select all people: category and image IDs
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids )

    # Select annotations of images with only one person with a visible nose
    image_paths = []
    targets = []
    for i in img_ids:
        ann_ids = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)

        # Individuals
        if len(annotations) > 1:
            continue

        # No crowds
        if annotations[0]['iscrowd']:
            continue

        # Extract relevant keypoints
        keypoints =     annotations[0]['keypoints']
        nose_x =        keypoints[0]
        nose_y =        keypoints[1]
        nose_visible =  keypoints[2]
        l_eye_x =       keypoints[3]
        l_eye_y =       keypoints[4]
        l_eye_visible = keypoints[5]
        r_eye_x =       keypoints[6]
        r_eye_y =       keypoints[7]
        r_eye_visible = keypoints[8]

        # Visible nose and eyes
        if (nose_visible == 0) or (l_eye_visible == 0) or (r_eye_visible == 0):
            continue
        
        # Big face
        eye_distance = abs(l_eye_x - r_eye_x) + abs(l_eye_y - r_eye_y)
        if eye_distance < 40:
            continue

        # Isolate image path
        img = coco.loadImgs(annotations[0]['image_id'])[0]

        # Normalize nose centroid
        width = img['width']
        height = img['height']
        x = np.float32(nose_x) / width
        y = np.float32(nose_y) / height
        target = np.array([x, y], dtype=np.float32)

        # Store dataset
        image_paths.append(images_folder + '/' + img['file_name'])
        targets.append(target)

    return np.array(image_paths), np.array(targets)

#FIN