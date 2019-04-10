from pycocotools.coco import COCO
from tqdm import tqdm
import os
import numpy as np
from skimage.morphology import skeletonize
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

def grayTrans(img):
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img


def drawEdges(segment, width=1):
    contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.drawContours((segment*0.0).astype(np.uint8), contours, -1, 255, width).get()
    return edges

def drawSkeleton(segment):
    return skeletonize(segment)

annotationPath = "./annotations_trainval2017/annotations/instances_train2017.json"

outputDir = "train2017/groundTruth/"

coco = COCO(annotationPath)

category = "person"
catIds = coco.getCatIds(catNms=[category])
imgIds = coco.getImgIds(catIds=catIds)

os.makedirs(outputDir + category + "/edges/", exist_ok=True)
os.makedirs(outputDir + category + "/skeletons/", exist_ok=True)

for imgId in tqdm(imgIds):
    start = time.time()
    image = coco.loadImgs(imgId)[0]
    imageName = image["file_name"]
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds)
    annotations = coco.loadAnns(annIds)
    end = time.time()
    print("Load time: ", end-start)
    start = time.time()
    annList = [coco.annToMask(annotation) for annotation in annotations]
    end = time.time()
    print("Mask time: ", end-start)
    start = time.time()
    merge = np.vectorize(lambda x, y: np.uint8(255) if (x > 0.5 or y > 0.5) else np.uint8(0))
    edges = drawEdges(annList[0])
    skeleton = drawSkeleton(annList[0])
    for segment in annList[1:]:
        new_edges = drawEdges(segment)
        new_ske = drawSkeleton(segment)
        edges = merge(edges,new_edges)
        skeleton = merge(skeleton,new_ske)
    end = time.time()
    print("Edge/Ske time: ", end-start)
    start = time.time()
    grayTrans(edges).save(outputDir + category + "/edges/" + imageName)
    grayTrans(skeleton).save(outputDir + category + "/skeletons/" + imageName)
    end = time.time()
    print("Saving time: ", end-start)
