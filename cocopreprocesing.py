from pycocotools.coco import COCO
from tqdm import tqdm
import os
import numpy as np
from skimage.morphology import skeletonize
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
from scipy.ndimage.morphology import distance_transform_edt as bwdist

def grayTrans(img):
    img = img*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img


def drawEdges(segment, width=1):
    contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.drawContours((segment*0.0).astype(float), contours, -1, 1.0, width).get()
    return edges

def drawSkeleton(segment, edge):
    skeleton = skeletonize(segment)
    dist = 2.0*bwdist(1.0 - edge)
    make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)

    scale = make_scale(dist, skeleton).astype(np.uint8)
    return scale

annotationPath = "./annotations_trainval2017/annotations/instances_train2017.json"

outputDir = "../train2017/groundTruth/"

coco = COCO(annotationPath)

category = ""
catIds = coco.getCatIds(catNms=[category])
imgIds = coco.getImgIds(catIds=catIds)

os.makedirs(outputDir + category + "/edges/", exist_ok=True)
os.makedirs(outputDir + category + "/skeletons/", exist_ok=True)

for imgId in tqdm(imgIds):
    exit()
    start = time.time()
    image = coco.loadImgs(imgId)[0]
    imageName = image["file_name"].replace('.jpg','.png')
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds)
    annotations = coco.loadAnns(annIds)
    end = time.time()
    print("Load time: ", end-start)
    start = time.time()
    annList = [coco.annToMask(annotation) for annotation in annotations]
    if len(annList) == 0:
        continue
    end = time.time()
    print("Mask time: ", end-start)
    start = time.time()
    edges = drawEdges(np.copy(annList[0]))
    skeleton = drawSkeleton(np.copy(annList[0]), edges)
    print(len(annList))
    for segment in annList[1:]:
        new_edges = drawEdges(np.copy(segment))
        new_ske = drawSkeleton(np.copy(segment), new_edges)
        edges = np.maximum(edges,new_edges)
        skeleton = np.maximum(skeleton,new_ske)
    end = time.time()
    print("Edge/Ske time: ", end-start)
    start = time.time()
    ed_img = grayTrans(edges)
    ske_img = Image.fromarray(skeleton, 'L')
    ed_img.save(outputDir + category + "/edges/" + imageName)
    ske_img.save(outputDir + category + "/skeletons/" + imageName)
    end = time.time()
    print("Saving time: ", end-start)
