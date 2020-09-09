import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python cnn.py"
output_file = open(generated_name, "w")
seeds = 1
datasets = ["cifar","cifar100","svhn"]

for dataset in datasets:
    base_call = bcall + " --dataset " + str(dataset)
    for s in range(seeds):
        condition=dataset+"_use_FC_backwards_weights"
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_FC_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_use_FC_backwards_nonlinearity"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_FC_backwards_nonlinearity True"
        print(final_call)
        print(final_call, file=output_file)
        
    condition=dataset+"_use_conv_backwards_weights"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)
    
    condition=dataset+"_use_conv_backwards_nonlinearity"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_nonlinearity True"
        print(final_call)
        print(final_call, file=output_file)
        
    condition=dataset+"use_FC_both"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_FC_backwards_weights True --use_FC_backwards_nonlinearity True"
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_use_all_nonlinearity"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_nonlinearity True --use_FC_backwards_nonlinearity True"
        print(final_call)
        print(final_call, file=output_file)
        
        
        
    