import numpy as np

import sys

import glob
import os

#Optional. To visualize data
import cv2

# mocapRootDir = '/run/media/hjoo/disk/data/Penn_Action/labels'
inputDir_root = '/run/media/hjoo/disk/data/3dpw/imageFiles'
img_outputDir_root = '/run/media/hjoo/disk/data/3dpw/densepose_img'
json_outputDir_root = '/run/media/hjoo/disk/data/3dpw/densepose'


if not os.path.exists(img_outputDir_root):
    os.mkdir(img_outputDir_root)

if not os.path.exists(json_outputDir_root):
    os.mkdir(json_outputDir_root)

# inputFolder=$1
# outputFolder=$2
# #./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images $outputFolder --write_images_format jpg
# echo ./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images ${outputFolder}_img --write_images_format jpg --write_json $outputFolder

seqList = sorted(glob.glob('{0}/*'.format(inputDir_root)) )

for i, inputPath in enumerate(seqList):
    
    seqName = os.path.basename(inputPath)
    print(seqName)

    # if not ("outdoors_fencing_01" in seqName or "downtown_walking_00"  in seqName or "outdoors_fencing_01" in seqName):
    #   continue

    outputFolder_img = os.path.join(img_outputDir_root,seqName)
    outputFolder_pkl = os.path.join(json_outputDir_root,seqName)

    if not os.path.exists(outputFolder_img):
        os.mkdir(outputFolder_img)

    # if not os.path.exists(outputFolder_json):
    #     os.mkdir(outputFolder_json)

    # cmd_str = "cd /home/hjoo/codes/openpose; ./build/examples/openpose/openpose.bin --image_dir {0} --write_images {1} --write_images_format jpg --write_json {2}".format(inputPath,
    #                                                                                                                                                 outputFolder_img, outputFolder_json)

    # cmd_str = "python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml model_final_5f3d7f.pkl \"{}/*.jpg\" dp_contour,bbox -v --output {}".format(inputPath, outputFolder_img)
    # print(cmd_str)
    # os.system(cmd_str)
    #./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images ${outputFolder}_img --write_images_format jpg --write_json $outputFolder

    from apply_net import caller
    
    params = ['dump','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'--output','{}.pkl'.format(outputFolder_pkl),'-v']
    caller(params)

    params = ['show','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'dp_contour,bbox','--output','{}/output.jpg'.format(outputFolder_img),'-v']
    caller(params)


