import submitit

executor = submitit.AutoExecutor(folder="pennImg")  
executor.update_parameters(timeout_min=4320, gpus_per_node=1, cpus_per_task=8, partition="learnfair", comment= 'CVPR 11/15', name='pennImg')  # timeout in min

from run_penn import runPennAction, runPennAction_img

interval = 100
for i in range(0,2200,interval):
    print('>> runPennAction({},{})'.format(i, i+ interval))
    # job = executor.submit(runPennAction,i,i+interval)
    job = executor.submit(runPennAction_img,i,i+interval)
    # runPennAction(i,+ interval)
    # runPennAction(2,10)

