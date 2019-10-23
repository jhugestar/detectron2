from apply_net import denseposeRunner

import sys

import glob
import os

def runPennAction(startIdx, endIdx):
    # mocapRootDir = '/run/media/hjoo/disk/data/Penn_Action/labels'
    g_bIsDevfair = False
    if os.path.exists('/private/home/hjoo'):
        g_bIsDevfair = True

    if g_bIsDevfair:
        inputDir_root = '/private/home/hjoo/data/pennaction/frames'
        img_outputDir_root = '/private/home/hjoo/data/pennaction/densepose_img'
        json_outputDir_root = '/private/home/hjoo/data/pennaction/densepose'
    else:
        assert False

    if not os.path.exists(img_outputDir_root):
        os.mkdir(img_outputDir_root)

    if not os.path.exists(json_outputDir_root):
        os.mkdir(json_outputDir_root)

    # inputFolder=$1
    # outputFolder=$2
    # #./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images $outputFolder --write_images_format jpg
    # echo ./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images ${outputFolder}_img --write_images_format jpg --write_json $outputFolder

    # seqList = sorted(glob.glob('{0}/*'.format(inputDir_root)) )

    for seqIdx in range(startIdx, endIdx):
        
        seqName = '{:04d}'.format(seqIdx)
        print(seqName)
        inputPath = os.path.join(inputDir_root,seqName)

        # if not ("outdoors_fencing_01" in seqName or "downtown_walking_00"  in seqName or "outdoors_fencing_01" in seqName):
        #   continue

        outputFolder_img = os.path.join(img_outputDir_root,seqName)
        outputFolder_pkl = os.path.join(json_outputDir_root,seqName+'.pkl')  

        if not os.path.exists(outputFolder_pkl):
            print(">>> Running:{}".format(outputFolder_img))
            params = ['dump','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'--output',outputFolder_pkl,'-v']
            denseposeRunner(params)
        else:
            print(">>> Already exists:{}".format(outputFolder_img))


        # if not os.path.exists(outputFolder_img):
        #     os.mkdir(outputFolder_img)
        #     print(">>> Running:{}".format(outputFolder_img))
        #     params = ['show','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'dp_contour,bbox','--output','{}/output.jpg'.format(outputFolder_img),'-v']
        #     denseposeRunner(params)
        
        # else:
        #     print(">>> Already exists:{}".format(outputFolder_img))



def runPennAction_img(startIdx, endIdx):
    # mocapRootDir = '/run/media/hjoo/disk/data/Penn_Action/labels'
    g_bIsDevfair = False
    if os.path.exists('/private/home/hjoo'):
        g_bIsDevfair = True

    if g_bIsDevfair:
        inputDir_root = '/private/home/hjoo/data/pennaction/frames'
        img_outputDir_root = '/private/home/hjoo/data/pennaction/densepose_img'
        json_outputDir_root = '/private/home/hjoo/data/pennaction/densepose'
    else:
        assert False

    if not os.path.exists(img_outputDir_root):
        os.mkdir(img_outputDir_root)

    if not os.path.exists(json_outputDir_root):
        os.mkdir(json_outputDir_root)

    # inputFolder=$1
    # outputFolder=$2
    # #./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images $outputFolder --write_images_format jpg
    # echo ./build/examples/openpose/openpose.bin --image_dir $inputFolder --write_images ${outputFolder}_img --write_images_format jpg --write_json $outputFolder

    # seqList = sorted(glob.glob('{0}/*'.format(inputDir_root)) )

    for seqIdx in range(startIdx, endIdx):
        
        seqName = '{:04d}'.format(seqIdx)
        print(seqName)
        inputPath = os.path.join(inputDir_root,seqName)

        # if not ("outdoors_fencing_01" in seqName or "downtown_walking_00"  in seqName or "outdoors_fencing_01" in seqName):
        #   continue

        outputFolder_img = os.path.join(img_outputDir_root,seqName)
        outputFolder_pkl = os.path.join(json_outputDir_root,seqName+'.pkl')  

        # if not os.path.exists(outputFolder_pkl):
        #     print(">>> Running:{}".format(outputFolder_img))
        #     params = ['dump','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'--output',outputFolder_pkl,'-v']
        #     denseposeRunner(params)
        # else:
        #     print(">>> Already exists:{}".format(outputFolder_img))


        if not os.path.exists(outputFolder_img):
            os.mkdir(outputFolder_img)
            print(">>> Running:{}".format(outputFolder_img))
            params = ['show','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'dp_contour,bbox','--output','{}/output.jpg'.format(outputFolder_img),'-v']
            denseposeRunner(params)
        
        else:
            print(">>> Already exists:{}".format(outputFolder_img))


if __name__ == "__main__":

    interval = 20
    for i in range(0,2250,interval):
        print('runPennAction({},{})'.format(i, i+ interval))
        # runPennAction(2,10)
