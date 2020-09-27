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


#print("loading...")
#pc_path = sys.argv[1]
#backprop_path = sys.argv[2]
#title = str(sys.argv[3])
#EPOCH_NUM = 5000

if __name__ == "__main__":
    basepath = "dynamical_ar_experiments/real_proper_"
    standard = "inference_lrs_0.1"
    datasets = ["mnist","fashion"]
    act_fns = ["tanh", "relu"]
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
    basepath = "dynamical_ar_experiments/prelim_ar_experiments_"
    for dataset in datasets:
        bppath = basepath + dataset + "_"
        if dataset == "mnist":
            default = default_mnist
        else:
            default = default_fashion
        for act_fn in act_fns:
            bpath = bppath + act_fn  + "_"
            current_x_both = bpath + "use_current_x_both" #completely fails
            current_x_update = bpath + "use_current_x_update" #works perfectly
            current_x_weights = bpath + "use_current_x_weights" # completely fails
            fderiv_both = bpath + "use_fderiv_both" # about 0.88, only final layer? # but somehow their combination fails which is weird
            fderiv_weight_update = bpath + "use_fderiv_weight_update" # works perfectly
            fderiv_x_update = bpath + "use_fderiv_x_update" # works perfectly # so this is a good results right so this is good. we only need to store the weights for the final thing.
            xnext_fderiv_both = bpath + "use_xnext_fderiv_both" # about 0.88 -- final layer

            
            plot_results(current_x_both, default, "Current x both","current_x_both", "default")
            plot_results(current_x_update, default, "Current x update","current_x_update", "default")
            plot_results(current_x_weights, default, "Current x weights","current_x_weights", "default")
            plot_results(fderiv_both, default, "Fderiv both","fderiv_both", "default")
            plot_results(fderiv_weight_update, default, "Fderiv weight update","fderiv_weight_update", "default")
            plot_results(fderiv_x_update, default, "Fderiv_x_update","fderiv_x_update", "default")
            plot_results(xnext_fderiv_both, default, "xnext_fderiv_both","xnext_fderiv_both", "default")


        

















    
    #basepath = "activation_relaxation_experiments/full_run_"
    #default_path = basepath + "initial_run1_default"
    #backprop_path = basepath+"backprop_backprop"
    #fa_path = basepath + "initial_run1_feedback_alignment"
    #fa_path_nonlinearity = basepath + "initial_run1_feedback_alignment_no_nonlinearity"
    #learnt_weights = basepath + "initial_run1_backwards_weights_with_update"
    #nonlinearities_path = basepath + "initial_run1_no_nonlinearities"
    #combined_path = basepath + "initial_run1_full_construct"
    #fashion_ar = basepath +"datasets_fashion_AR"
    #fashion_bp = basepath + "datasets_fashion_BP"
    #mnist_ar = basepath + "datasets_mnist_AR"
    #mnist_bp = basepath +"datasets_mnist_BP"
    #svhn_ar = basepath + "datasets_svhn_AR"
    #svhn_bp = basepath + "datasets_svhn_BP"
    #fashion_backward_weight = basepath + "datasets_relaxation_backwards_weights_with_update"
    #fashion_default = basepath + "datasets_relaxation_default"
    #fashion_feedback_alignment = basepath + "datasets_relaxation_feedback_alignment"
    #fashion_feedback_alignment_no_nonlinearity = basepath + "datasets_relaxation_feedback_alignment_no_nonlinearity"
    #fashion_full_construct = basepath + "datasets_relaxation_full_construct"
    #fashion_no_nonlinearities = basepath + "datasets_relaxation_no_nonlinearities"
    """mnist_default = basepath + "mnist_default"
    mnist_bp = basepath + "mnist_bp"
    mnist_fa = basepath + "mnist_feedback_alignment"
    mnist_fa_nonlin = basepath + "mnist_feedback_alignment_no_nonlinearity"
    mnist_backwards_weights = basepath + "mnist_backwards_weights_with_update"
    mnist_nonlin = basepath + "mnist_no_nonlinearities"
    mnist_full_construct = basepath + "mnist_full_construct"
    fashion_default = basepath + "fashion_default"
    fashion_bp = basepath + "fashion_bp"
    fashion_fa = basepath + "fashion_feedback_alignment"
    fashion_fa_nonlin = basepath + "fashion_feedback_alignment_no_nonlinearity"
    fashion_backwards_weights = basepath + "fashion_backwards_weights_with_update"
    fashion_nonlin = basepath + "fashion_no_nonlinearities"
    fashion_full_construct = basepath + "fashion_full_construct"
    #MNIST
    # ar vs backprop
    plot_results(mnist_default,mnist_bp,"Activation Relaxation vs Backprop", "Activation Relaxation", "Backprop")
    #feedback alignment
    plot_results(mnist_backwards_weights, mnist_default,"Backwards Weights", "Default AR", "Learnt Backwards Weights")#,path3=fa_path,label3="Feedback Alignment")
    # nonlinearities
    plot_results(mnist_nonlin, mnist_default,"No Nonlinear Derivative", "Default AR", "No Backwards Derivative")
    # Combined
    plot_results(mnist_full_construct, mnist_default,"Combined Algorithm","Default AR", "Combined Relaxations")
    
    #FASHION
    # ar vs backprop
    plot_results(fashion_default,fashion_bp,"Activation Relaxation vs Backprop", "Activation Relaxation", "Backprop")
    #feedback alignment
    plot_results(fashion_backwards_weights, fashion_default,"Backwards Weights", "Default AR", "Learnt Backwards Weights")#,path3=fa_path,label3="Feedback Alignment")
    # nonlinearities
    plot_results(fashion_nonlin, fashion_default,"No Nonlinear Derivative", "Default AR", "No Backwards Derivative")
    # Combined
    plot_results(fashion_full_construct, fashion_default,"Combined Algorithm","Default AR", "Combined Relaxations")"""
    


    #ar vs backprop MNIST
    #plot_results(mnist_ar,mnist_bp,"Activation Relaxation vs Backprop on MNIST", "Activation Relaxation", "Backprop")
    # ar vs backprop Fashion MNIST
    #plot_results(fashion_ar,fashion_bp,"Activation Relaxation vs Backprop on Fashion MNIST", "Activation Relaxation", "Backprop")
    # ar vs backprop Fashion MNIST
    #plot_results(svhn_ar,svhn_bp,"Activation Relaxation vs Backprop on SVHN", "Activation Relaxation", "Backprop")

    #relaxations on the fashionMNIST dataset
    #feedback alignment
    #plot_results(fashion_default, fashion_backward_weight,"Backwards Weights", "Default AR", "Learnt Backwards Weights")#,path3=fa_path,label3="Feedback Alignment")
    # nonlinearities
    #plot_results(fashion_default, fashion_no_nonlinearities,"No Nonlinear Derivative", "Default AR", "No Backwards Derivative")
    # Combined
    #plot_results(fashion_default, fashion_full_construct,"Combined Algorithm","Default AR", "Combined Relaxations")
    