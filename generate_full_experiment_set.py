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

for dataset in datasets:
    base_call = bcall + " --dataset " + str(dataset)
    for s in range(seeds):
        condition=dataset+"_use_current_x_update"
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_current_x_update True"
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_use_current_x_weights"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_current_x_weights True"
        print(final_call)
        print(final_call, file=output_file)
        
    condition=dataset+"_use_current_x_both"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_current_x_update True --use_current_x_weights True"
        print(final_call)
        print(final_call, file=output_file)
        
    condition=dataset+"_use_fderiv_x_update"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_fderiv_x_update True "
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_use_fderiv_weight_update"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_fderiv_weight_update True"
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_use_fderiv_both"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_fderiv_x_update True --use_fderiv_weight_update True"
        print(final_call)
        print(final_call, file=output_file)

        condition=dataset+"_use_xnext_fderiv_both"
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_fderiv_x_update True --use_fderiv_weight_update True --xnext_fderiv True"
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_fderiv_x_update True --use_fderiv_weight_update True --xnext_fderiv True"
        print(final_call)
        print(final_call, file=output_file)
