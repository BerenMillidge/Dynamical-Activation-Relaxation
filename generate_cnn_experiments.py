import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python cnn.py"
output_file = open(generated_name, "w")
seeds = 5
datasets = ["cifar","cifar100","svhn"]
act_fns = ["tanh"]#,"relu"]

for dataset in datasets:
    bbcall = bcall + " --dataset " + str(dataset)
    cond = dataset + "_"

    for act_fn in act_fns:
        base_call = bbcall + " --act_fn " + str(act_fn)
        
        condition= cond + act_fn+"_default"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
            print(final_call)
            print(final_call, file=output_file)

        condition= cond + act_fn+"_use_everything"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_nonlinearity False --use_FC_backwards_nonlinearity False --use_FC_backwards_weights True"
            print(final_call)
            print(final_call, file=output_file)
            
        condition= cond + act_fn+"_use_FC_backwards_weights"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_FC_backwards_weights True"
            print(final_call)
            print(final_call, file=output_file)

        condition= cond + act_fn+"_use_FC_backwards_nonlinearity"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_FC_backwards_nonlinearity False"
            print(final_call)
            print(final_call, file=output_file)
        
            
        condition= cond + act_fn+"_use_conv_backwards_weights"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_weights True"
            print(final_call)
            print(final_call, file=output_file)
        
        
        condition= cond + act_fn+"_use_conv_backwards_nonlinearity"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_nonlinearity False"
            print(final_call)
            print(final_call, file=output_file)
            
        condition= cond + act_fn+"_use_FC_both"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_FC_backwards_weights True --use_FC_backwards_nonlinearity False"
            print(final_call)
            print(final_call, file=output_file)

        condition= cond + act_fn+"_use_all_nonlinearity"
        for s in range(7,12):
            lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
            spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_conv_backwards_nonlinearity False --use_FC_backwards_nonlinearity False"
            print(final_call)
            print(final_call, file=output_file)
        
