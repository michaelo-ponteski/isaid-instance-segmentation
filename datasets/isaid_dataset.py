# minimalny przykład w notebooku
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torch
import numpy as np
import os

class ISaidDataset(Dataset):
    def __init__(self, dataset_root, split='train'):
        self.images_path = os.path.join(dataset_root, split, 'images')
        self.json_path = os.path.join(dataset_root, split, f'instance_only_filtered_{split}.json')
        with open(self.json_path) as f:
            self.coco = json.load(f)
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        # image_id -> annotations
        self.imgid_to_ann = {}
        for ann in self.annotations:
            self.imgid_to_ann.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_path, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_id = img_info['id']
        annos = self.imgid_to_ann.get(img_id, [])

        boxes, labels, masks = [], [], []
        for ann in annos:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            mask = Image.new('L', img.size, 0)
            for poly in ann['segmentation']:
                xy = [tuple(poly[i:i+2]) for i in range(0, len(poly), 2)]
                ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
            masks.append(torch.tensor(np.array(mask), dtype=torch.uint8))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0,img.size[1],img.size[0]), dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": torch.tensor([img_id])}
        return img, target
