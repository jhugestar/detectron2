import glob
import os
import submitit
from apply_net import denseposeRunner


executor = submitit.AutoExecutor(folder="3dpwImg2")  
executor.update_parameters(timeout_min=4320, gpus_per_node=1, cpus_per_task=8, partition="learnfair", comment= 'CVPR 11/15', name='3dpwImg2')  # timeout in min

# mocapRootDir = '/run/media/hjoo/disk/data/Penn_Action/labels'
g_bIsDevfair = False
if os.path.exists('/private/home/hjoo'):
    g_bIsDevfair = True

if g_bIsDevfair:
    inputDir_root = '/private/home/hjoo/data/3dpw/imageFiles'
    img_outputDir_root = '/private/home/hjoo/data/3dpw/densepose_img'
    json_outputDir_root = '/private/home/hjoo/data/3dpw/densepose'
else:
    inputDir_root = '/run/media/hjoo/disk/data/3dpw/imageFiles'
    img_outputDir_root = '/run/media/hjoo/disk/data/3dpw/densepose_img'
    json_outputDir_root = '/run/media/hjoo/disk/data/3dpw/densepose'


if not os.path.exists(img_outputDir_root):
    os.mkdir(img_outputDir_root)

if not os.path.exists(json_outputDir_root):
    os.mkdir(json_outputDir_root)

seqList = sorted(glob.glob('{0}/*'.format(inputDir_root)) )

for i, inputPath in enumerate(seqList):
    
    seqName = os.path.basename(inputPath)

    outputFolder_img = os.path.join(img_outputDir_root,seqName)
    outputFolder_pkl = os.path.join(json_outputDir_root,seqName+'.pkl')  

    # if not os.path.exists(outputFolder_pkl):
    #     params = ['dump','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'--output',outputFolder_pkl,'-v']
    #     print(">>> Submitting:{}".format(outputFolder_pkl))
    #     # denseposeRunner(params)
    #     job = executor.submit(denseposeRunner,params)  
    # else:
    #     print(">>> Already exists:{}".format(outputFolder_pkl))

    if not os.path.exists(outputFolder_img):
        params = ['show','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'dp_contour,bbox','--output','{}/output.jpg'.format(outputFolder_img),'-v']
        print(">>> Submitting:{}".format(outputFolder_img))
        # denseposeRunner(params)
        job = executor.submit(denseposeRunner,params)  
    else:
        print(">>> Already exists:{}".format(outputFolder_img))

    # if not os.path.exists(outputFolder_img):
    #     os.mkdir(outputFolder_img)
    #     # denseposeRunner(params)
    #     job = executor.submit(denseposeRunner,params)  


    # params = ['show','configs/densepose_rcnn_R_50_FPN_s1x.yaml','model_final_5f3d7f.pkl','{}/*.jpg'.format(inputPath),'dp_contour,bbox','--output','{}/output.jpg'.format(outputFolder_img),'-v']
    # caller(params)
    # job = executor.submit(trainerWrapper,['--bRandOcc', '--skelType','coco_noeyeear','--w_angleLoss','1e4','--w_3dJ_smpl_Loss','0.1', '--w_3dJ_coco_Loss','0.1', '--bPredAnkle','--data_dir','dataset/data_amass_fbbox_noShape/', '--train_batch','20000','--test_batch','2048','--job','3','--train_db','All', '--load', '/private/home/hjoo/dropbox_checkpoint/10-17-44257-bMini_0-WShp_0.0-WAng_10000.0-W3JSM_0.1-W3JCO_0.1-db_All-rCrop_0-ocT_all-skeT_coco_noeyeear-ranOc_1-pAkl_1-bLo_0_best_epoch153/ckpt_last.pth.tar'])  




