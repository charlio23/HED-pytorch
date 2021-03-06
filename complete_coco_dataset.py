from pycocotools.coco import COCO
from tqdm import tqdm
import os
import numpy as np
import skimage.io as io
from PIL import Image

rootDir = "../val2017/"
annotationPath = "./annotations_trainval2017/annotations/instances_val2017.json"

coco = COCO(annotationPath)

category = "person"
catIds = coco.getCatIds(catNms=[category])
imgIds = coco.getImgIds(catIds=catIds)
rootDirImg = rootDir + "images/"

size = (400,400)

for imgId in tqdm(imgIds):
    image = coco.loadImgs(imgId)[0]
    imageName = image["file_name"]
    if not os.path.isfile(rootDirImg + imageName):
        print(imageName, "not found, downloading...")
        io.imsave(rootDirImg + imageName, io.imread(image['coco_url']))
    Image.open(rootDirImg + imageName).resize(size, Image.ANTIALIAS).save(rootDirImg + imageName)