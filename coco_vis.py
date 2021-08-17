from __future__ import print_function, division
from coco_utils import get_coco, get_coco_kp, get_coco_api_from_dataset

trainset = get_coco('H:/Datasets/COCO/022719/', 'train', transforms=None)

coco = get_coco_api_from_dataset(trainset)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
# print(cats)
print('number of categories: ', len(cats))
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = [cat['id'] for cat in cats]
print('COCO ids: \n{}'.format(nms))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

print(len(trainset))

classnames = {cat['id']: cat['name'] for cat in cats}


import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings("ignore") #忽略什么的警告？

plt.ion()

def draw_bbox(bbox, ax, polygons, color, name):
    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
#     [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    [bbox_x, bbox_y, bbox_x2, bbox_y2] = bbox
    bbox_w = bbox_x2 - bbox_x
    bbox_h = bbox_y2 - bbox_y
    
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
    np_poly = np.array(poly).reshape((4,2))
    polygons.append(Polygon(np_poly))
    color.append(c)
    
    # show catogory name
    ax.text(bbox_x, bbox_y, name, size = 6, color='black', bbox=dict(boxstyle="square",facecolor=c,alpha=0.8,
                                                                      ))

#随机展示十组样本
plt.figure()
for n in range(20):
#     i = 7830
    i = random.randint(0, len(trainset)-1)

    print('Index: ', i)
    image, target = trainset[i]
#     print(image)
#     print(target)
    plt.imshow(image)
    
    ax = plt.gca()
    
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    print('number of boxes: ', len(target['boxes']))
    nms=[classnames[t.item()] for t in target['labels']]
    print('class name: \n{}\n'.format(' '.join(nms)))

    for j in range(len(target['boxes'])):
        bbox = target['boxes'][j]

        name = classnames[target['labels'][j].item()]
        draw_bbox(bbox, ax, polygons, color, name)
        
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)
    plt.savefig('./samples/' + str(target['image_id'].item())+'.png', dpi=300)


    plt.pause(2)
    plt.show()
    ax.clear()
