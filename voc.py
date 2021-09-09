import torch
import torchvision
import transforms as T
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from PIL import Image

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

#类似于处理Coco
class ConvertVOC(object):
    def __call__(self, image, target):
        w, h = image.size
        
        # guard against no boxes via resizing
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(target["labels"], dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        target['image_id'] = image_id

        area = torch.tensor(target["area"])
        iscrowd = torch.tensor(target["iscrowd"])
        target["area"] = area
        target["iscrowd"] = iscrowd

        # target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        return image, target

class VOCDataset(torchvision.datasets.VOCDetection):

    # def __init__(self, root, year='2012',
    #              image_set='train',
    #              transforms=None):
    #     self.ds = super().__init__(root, year, image_set, transforms)
    #     return self.ds


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        image = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        # if(index==0): print('target: ', target)

        # image, target = super().__getitem__(index)

        target = target['annotation']
        labels_name = [t['name'] for t in target['object']]
        labels = [VOC_CLASSES.index(name) for name in labels_name]
        boxes = [[int(t['bndbox']['xmin']), int(t['bndbox']['ymin']), int(t['bndbox']['xmax']), 
            int(t['bndbox']['ymax'])] for t in target['object']]

        # target = {"boxes": boxes, "labels": labels, "labels_name": labels_name}

        # 从Image_id开始是为了适应COCO数据加入的信息
        iscrowd = [int(t['occluded']) for t in target['object']]
        # print(boxes)
        areas = [(b[2] - b[0])*(b[3] - b[1]) for b in boxes]
        target = {"boxes": boxes, "labels": labels, "image_id": index, "iscrowd": iscrowd, "area": areas}
        # print(target)


        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

def get_voc(data_path, year="2012", image_set="train", transforms=None):
    t = [ConvertVOC()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    ds = VOCDataset(data_path, year, image_set, transforms=transforms)


    num_classes = len(VOC_CLASSES)
    return ds, num_classes