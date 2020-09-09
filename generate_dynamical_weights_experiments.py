import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python main.py"
output_file = open(generated_name, "w")
seeds = 1
datasets = ["mnist","fashion"]

lrs = [0.8,0.5,0.1]
num_inference_steps = [100,50,20,10]
for dataset in datasets:
    base_call = bcall + " --dataset " + str(dataset)
    condition=dataset+"_inference_lrs"
    for lr in lrs:
        base_call += " --inference_learning_rate " + str(lr)
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --dynamical_weight_update True"
            print(final_call)
            print(final_call, file=output_file)

    condition=dataset+"_inference_steps"
    for inference_steps in num_inference_steps:
        base_call += " --n_inference_steps " + str(inference_steps)
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --dynamical_weight_update True"
            print(final_call)
            print(final_call, file=output_file)

