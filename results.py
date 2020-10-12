import numpy as np 
import matplotlib.pyplot as plt
import os 
import sys
import seaborn as sns 
EPOCH_NUM=20000
def get_results(basepath,cnn=True,merged=False):
    ### Loads results losses and accuracies files ###
    dirs = os.listdir(basepath)
    print(dirs)
    acclist = []
    losslist = []
    test_acclist = []
    dirs.sort()
    for i in range(len(dirs)):
        p = basepath + "/" + str(dirs[i]) + "/"
        acclist.append(np.load(p + "accs.npy")[0:EPOCH_NUM])
        losslist.append(np.load(p + "losses.npy")[0:EPOCH_NUM])
        test_acclist.append(np.load(p+"test_accs.npy")[0:EPOCH_NUM])
    print("enumerating through results")
    for i,(acc, l) in enumerate(zip(acclist, losslist)):
        print("acc: ", acc.shape)
        print("l: ", l.shape)
    else:
        return np.array(acclist), np.array(losslist),np.array(test_acclist)


def plot_results(pc_path, backprop_path,title,label1,label2,path3="",label3=""):
    ### Plots initial results and accuracies ###
    acclist, losslist, test_acclist = get_results(pc_path)
    backprop_acclist, backprop_losslist, backprop_test_acclist = get_results(backprop_path)
    titles = ["accuracies", "losses", "test accuracies"]
    if path3 != "":
        p3_acclist, p3_losslist, p3_test_accslist = get_results(path3)
        p3_list = [p3_acclist,p3_losslist,p3_test_accslist]
    pc_list = [acclist, losslist, test_acclist]
    backprop_list = [backprop_acclist, backprop_losslist, backprop_test_acclist]
    print(acclist.shape)
    print(losslist.shape)
    print(test_acclist.shape)
    print(backprop_acclist.shape)
    print(backprop_losslist.shape)
    print(backprop_test_acclist.shape)
    for i,(pc, backprop) in enumerate(zip(pc_list, backprop_list)):
        if titles[i] == "test accuracies":
            xs = np.arange(0,len(pc[0,:]))
            mean_pc = np.mean(pc, axis=0)
            std_pc = np.std(pc,axis=0)
            mean_backprop = np.mean(backprop,axis=0)
            std_backprop = np.std(backprop,axis=0)
            print("mean_pc: ",mean_pc.shape)
            print("std_pc: ", std_pc.shape)
            fig,ax = plt.subplots(1,1)
            ax.fill_between(xs, mean_pc - std_pc, mean_pc+ std_pc, alpha=0.5,color='#228B22')
            plt.plot(mean_pc,label=label1,color='#228B22')
            ax.fill_between(xs, mean_backprop - std_backprop, mean_backprop+ std_backprop, alpha=0.5,color='#B22222')
            plt.plot(mean_backprop,label=label2,color='#B22222')
            if path3 != "":
                p3 = p3_list[i]
                mean_p3 = np.mean(p3, axis=0)
                std_p3 = np.std(p3,axis=0)
                ax.fill_between(xs, mean_p3 - std_p3, mean_p3+ std_p3, alpha=0.5,color='#228B22')
                plt.plot(mean_p3,label=label3)


            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title(de_underscore(title) + " " + de_underscore(str(titles[i])),fontsize=18)
            ax.tick_params(axis='both',which='major',labelsize=12)
            ax.tick_params(axis='both',which='minor',labelsize=12)
            if titles[i] in ["accuracies", "test accuracies"]:
                plt.ylabel("Accuracy",fontsize=18)
            else:
                plt.ylabel("Loss")
            plt.xlabel("Iterations",fontsize=18)
            legend = plt.legend(prop={"size":14})
            legend.fontsize=18
            legend.style="oblique"
            frame  = legend.get_frame()
            frame.set_facecolor("1.0")
            frame.set_edgecolor("1.0")
            fig.tight_layout()
            fig.savefig("./figures/"+underscore(title) +"_"+underscore(titles[i])+"_prelim_2.jpg")
            plt.show()

def underscore(s):
    return s.replace(" ", "_")

def de_underscore(s):
    return s.replace("_", " ")

if __name__ == "__main__":
    basepath = "dynamical_ar_experiments/dynamical_"
    standard = "inference_lrs_0.1"
    datasets = ["mnist","fashion"]
    act_fns = ["tanh"]#, "relu"]
    inference_lrs = [0.1,0.5,0.8]
    inference_steps = [300,500]
    #for dataset in datasets:
        #for inference_lr in inference_lrs:
         #   default = basepath + str(dataset) + "_" + standard
          #  test = basepath + str(dataset) + "_inference_lrs_" + str(inference_lr)
           # plot_results(test, default,dataset + "_Inference LR","inference_lr_" + str(inference_lr), "default")

        #for inference_step in inference_steps:
          #  default = basepath + str(dataset) + "_" + standard
         #   test = basepath + str(dataset) + "_inference_steps_" + str(inference_step)
          #  plot_results(test, default,dataset + "_Inference Steps","inference_steps_" + str(inference_step), "default")


    default_mnist = "dynamical_ar_experiments/baseline_mnist"
    default_fashion = "dynamical_ar_experiments/baseline_fashion"
    basepath = "dynamical_ar_experiments/real_proper__"
    for dataset in datasets:
        bppath = basepath + dataset + "_"
        if dataset == "mnist":
            default = default_mnist
        else:
            default = default_fashion
        for act_fn in act_fns:
            bpath = bppath + act_fn  + "_"
            current_x_weights = bpath + "use_current_x_weights"
            fderiv_both = bpath + "use_fderiv_both"
            fderiv_weight_update = bpath + "use_fderiv_weight_update"
            fderiv_x_update = bpath + "use_fderiv_x_update"
            xnext_fderiv_both = bpath + "use_xnext_fderiv_both"
            xnext_fderiv_weights = bpath + "use_xnext_fderiv_weights" 
            xnext_fderiv_x = bpath + "use_xnext_fderiv_x" 
            print(bpath)
            ppath = dataset + "_" + act_fn + "_"
            
            plot_results(current_x_weights, default, ppath + "Current x weights","current_x_weights", "default")
            plot_results(fderiv_both, default, ppath + "Fderiv both","fderiv_both", "default")
            plot_results(fderiv_weight_update, default, ppath + "Fderiv weight update","fderiv_weight_update", "default")
            plot_results(fderiv_x_update, default, ppath + "Fderiv_x_update","fderiv_x_update", "default")
            #plot_results(xnext_fderiv_both, default, ppath + "xnext_fderiv_both","xnext_fderiv_both", "default")
            #plot_results(xnext_fderiv_weights, default, ppath + "xnext_fderiv_weights","xnext_fderiv_weights", "default")
            #plot_results(xnext_fderiv_x, default, ppath + "xnext_fderiv_x","xnext_fderiv_x", "default")
