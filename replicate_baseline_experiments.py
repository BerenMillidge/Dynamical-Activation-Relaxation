import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 1
datasets = ["mnist","fashion"]
for dataset in datasets:
    bcall = base_call + " --dataset " + str(dataset)
    condition = dataset
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = bcall + " --logdir " + str(lpath) + " --savedir " + str(spath)
        print(final_call)
        print(final_call, file=output_file)
