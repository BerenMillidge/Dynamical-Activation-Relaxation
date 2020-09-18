
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
        plt.title(title + " " + str(titles[i]),fontsize=18)
        ax.tick_params(axis='both',which='major',labelsize=12)
        ax.tick_params(axis='both',which='minor',labelsize=10)
        if titles[i] in ["accuracies", "test accuracies"]:
            plt.ylabel("Accuracy",fontsize=16)
        else:
            plt.ylabel("Loss")
        plt.xlabel("Iterations",fontsize=16)
        legend = plt.legend()
        legend.fontsize=14
        legend.style="oblique"
        frame  = legend.get_frame()
        frame.set_facecolor("1.0")
        frame.set_edgecolor("1.0")
        fig.tight_layout()
        #fig.savefig("./figures/"+title +"_"+titles[i]+"_prelim_2.jpg")
        plt.show()



if __name__ == "__main__":
    basepath = "dynamical_ar_experiments/cnn__"
    cifar_use_all_nonlinearity = basepath + "cifar_use_all_nonlinearity"
    cifar_conv_backwards_nonlinearity = basepath + "cifar_use_conv_backwards_nonlinearity"
    cifar_conv_backwards_weights = basepath + "cifar_use_conv_backwards_weights"
    cifar_FC_backwards_nonlinearity = basepath + "cifar_use_FC_backwards_nonlinearity"
    cifar_FC_backwards_weights = basepath + "cifar_use_FC_backwards_weights"
    cifar_FC_both = basepath + "cifaruse_FC_both"
    cifar_default = basepath + "cifar_default"
    svhn_use_all_nonlinearity = basepath + "svhn_use_all_nonlinearity"
    svhn_conv_backwards_nonlinearity = basepath + "svhn_use_conv_backwards_nonlinearity"
    svhn_conv_backwards_weights = basepath + "svhn_use_conv_backwards_weights"
    svhn_FC_backwards_nonlinearity = basepath + "svhn_use_FC_backwards_nonlinearity"
    svhn_FC_backwards_weights = basepath + "svhn_use_FC_backwards_weights"
    svhn_FC_both = basepath + "svhnuse_FC_both"
    svhn_default = basepath + "svhn_default"

    plot_results(cifar_conv_backwards_weights, cifar_default, "Cifar Backwards Weights", "Conv backwards weights", "Baseline Conv")
    plot_results(cifar_conv_backwards_nonlinearity, cifar_default, "Cifar Backwards Nonlinearity", "Conv nonlinear", "Baseline Conv")
    plot_results(cifar_FC_backwards_nonlinearity, cifar_default, "Cifar FC Backwards Nonlinearity", "FC nonlinear", "Baseline Conv")
    plot_results(cifar_FC_backwards_weights, cifar_default, "cifar FC backwards weights", "FC weights", "Baseline Conv")
    plot_results(cifar_FC_both, cifar_default, "Cifar FC both", "FC both", "Baseline Conv")
    plot_results(cifar_use_all_nonlinearity, cifar_default, "Cifar FC use all nonlinearity", "Both nonlinear", "Baseline Conv")

    plot_results(svhn_conv_backwards_weights, svhn_default, "svhn backwards weights", "Conv nonlinear", "Baseline Conv")
    plot_results(svhn_conv_backwards_nonlinearity, svhn_default, "svhn conv backwards nonlinearity", "Conv nonlinear", "Baseline Conv")
    plot_results(svhn_FC_backwards_nonlinearity, svhn_default, "svhn fc backwards nonlinearity", "FC nonlinear", "Baseline Conv")
    plot_results(svhn_FC_backwards_weights, svhn_default, "svhn FC backwards weights", "FC weights", "Baseline Conv")
    plot_results(svhn_FC_both, svhn_default, "svhn fc both", "FC both", "Baseline Conv")
    plot_results(svhn_use_all_nonlinearity, svhn_default, "svhn use all nonlinearity", "Both nonlinear", "Baseline Conv")

    #plot_results(cifar_conv_backwards_nonlinearity, cifar_FC_backwards_nonlinearity, "Conv vs FC", "Conv nonlinear", "FC nonlinear")
    #plot_results(cifar_FC_backwards_nonlinearity, cifar_FC_both, "both", "FC", "both")

    #plot_results(svhn_conv_backwards_nonlinearity, svhn_FC_backwards_nonlinearity, "Conv vs FC", "Conv nonlinear", "FC nonlinear")
    #plot_results(svhn_FC_backwards_nonlinearity, svhn_FC_both, "both", "FC", "both")
