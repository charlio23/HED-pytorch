from pycocotools.coco import COCO
from tqdm import tqdm
import os
import numpy as np
import skimage.io as io
from PIL import Image

rootDir = "../train2017/"
annotationPath = "./annotations_trainval2017/annotations/instances_train2017.json"

coco = COCO(annotationPath)

category = "person"
catIds = coco.getCatIds(catNms=[category])
imgIds = coco.getImgIds(catIds=catIds)
rootDirImg = rootDir + "images/"

for imgId in tqdm(imgIds):
    image = coco.loadImgs(imgId)[0]
    imageName = image["file_name"]
    print(imageName)
    if not os.path.isfile(rootDirImg + imageName):
        print(imageName, "not found, downloading...")
        io.imsave(rootDirImg + imageName, io.imread(image['coco_url']))