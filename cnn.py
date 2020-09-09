# Experiments with extending AR to get rid of the need for discrete phases, and the storing of information 
#across phases which is problematic from a biological perspective.

import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os 
import subprocess
import time 
from datetime import datetime
import argparse
import scipy
from cnn_layers import *

def get_cnn_dataset(dataset, batch_size,normalize=True):
    if normalize:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "cifar":
        trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False,
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "svhn":
        trainset = torchvision.datasets.SVHN(root='./svhn_data', split='train',
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.SVHN(root='./svhn_data', split='test',
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "mnist":
        mnist_transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                                download=False, transform=mnist_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                               download=False, transform=mnist_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ",len(train_data))
    print("Test: ", len(test_data))
    return train_data, test_data


def onehot(x):
    z = torch.zeros([len(x),10])
    for i in range(len(x)):
      z[i,x[i]] = 1
    return z.float().to(DEVICE)


### functions ###
def set_tensor(xs):
  return xs.float().to(DEVICE)

def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones_like((x)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel 

def softmax(xs):
  return torch.nn.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))

def edge_zero_pad(img,d):
  N,C, h,w = img.shape 
  x = torch.zeros((N,C,h+(d*2),w+(d*2))).to(DEVICE)
  x[:,:,d:h+d,d:w+d] = img
  return x 

def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B


def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]

class ARNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate, weight_learning_rate,continual_weight_update=False,update_dilation_factor=None,numerical_check=False,device='cpu',use_FC_backwards_weights=False,use_FC_backward_nonlinearity=False,update_backwards_weights = True, use_conv_backwards_weights = False, use_conv_backwards_nonlinearity = False):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.weight_learning_rate = weight_learning_rate
    self.device = device
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.continual_weight_update = continual_weight_update
    self.numerical_check = numerical_check
    self.use_FC_backward_nonlinearity = use_FC_backward_nonlinearity
    self.use_FC_backwards_weights = use_FC_backwards_weights
    self.update_backwards_weights = update_backwards_weights
    self.use_conv_backwards_nonlinearity = use_conv_backwards_nonlinearity
    self.use_conv_backwards_weights = use_conv_backwards_weights
    self.update_dilation_factor = update_dilation_factor if update_dilation_factor is not None else self.n_inference_steps_train
    if self.continual_weight_update:
      for l in self.layers:
        if hasattr(l, "learning_rate"):
          l.learning_rate = l.learning_rate / self.update_dilation_factor

    if self.numerical_check:
      for l in self.layers:
        l.init_numerical_test()

    #apply conditions to layers
    for l in self.layers:
        l.use_FC_backward_nonlinearity = self.use_FC_backward_nonlinearity
        l.use_FC_backwards_weights = self.use_FC_backwards_weights
        l.update_backwards_weights = self.update_backwards_weights
        l.use_conv_backwards_nonlinearity = self.use_conv_backwards_nonlinearity
        l.use_conv_backwards_weights = self.use_conv_backwards_weights
        if hasattr(l, "init_backwards_weights"):
            l.init_backwards_weights()


  def update_weights(self,print_weight_grads=False,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      if i !=1:
        dW = l.update_weights(self.backs[i+1],update_weights=True)
        if print_weight_grads:
          print("weight diffs: ",(dW*2) + true_weight_grad)
          diff = torch.sum((dW -true_dW)**2)
          weight_diffs.append(diff)
    return weight_diffs

  def forward(self,x):
    xs = [[] for i in range(len(self.layers)+1)]
    xs[0] = x
    for i,l in enumerate(self.layers):
      xs[i+1] = l.forward(xs[i])
    return xs[-1]

  def forward_grad(self,x):
    for i,l in enumerate(self.layers):
        x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def learn_batch(self,inps,labels,num_inference_steps,update_weights=True):
    #print("learn batch update weights: ", update_weights)
    xs = [[] for i in range(len(self.layers)+1)]
    xs[0] = inps
    #forward pass
    for i,l in enumerate(self.layers):
      xs[i+1] = l.forward(xs[i])
    #inference
    out_error = 2 * (xs[-1] - labels)
    backs = [[] for i in range(len(self.layers)+1)]
    backs[-1] = out_error
    for n in range(num_inference_steps):
      #backward inference
      for j in reversed(range(len(self.layers))):
        backs[j] = self.layers[j].backward(backs[j+1])
    # weight updates
    for i,l in enumerate(self.layers):
      l.update_weights(backs[i+1],update_weights=update_weights)
      
  def train(self, trainset, testset,logdir,savedir, num_epochs,num_inference_steps,test=True):
    self.unset_numerical_test()
    self.uncheck_numerical_test()
    with torch.no_grad():
      losses = []
      accs = []
      test_accs = []
      #begin training loop
      for n_epoch in range(num_epochs):
        print("Beginning epoch ",n_epoch)
        for n,(img,label) in enumerate(trainset):
          img = img.to(self.device)
          label = onehot(label).to(self.device)
          self.learn_batch(img, label,num_inference_steps,update_weights=True)
          pred_outs = self.forward(img)
          L = torch.sum((pred_outs - label)**2)
          acc = accuracy(pred_outs,label)
          print("epoch: " + str(n_epoch) + " loss batch " + str(n) + "  " + str(L))
          print("acc batch " + str(n) + "  " + str(acc))
          losses.append(L.item())
          accs.append(acc)
        if test:
          for tn, (test_img, test_label) in enumerate(testset):
            test_img = test_img.to(self.device)
            labels = onehot(test_label).to(self.device)
            pred_outs = self.forward(test_img)
            test_acc = accuracy(pred_outs, labels)
            test_accs.append(test_acc)
        self.save_model(logdir,savedir,losses,accs,test_accs)
      if test:
        return losses, accs, test_accs
      else:
        return losses, accs

  def set_numerical_test(self):
    for l in self.layers:
      l.init_numerical_test()
      l.numerical_test=True

  def unset_numerical_test(self):
    for l in self.layers:
      l.unset_numerical_test()
      #l.numerical_test=False

  def uncheck_numerical_test(self):
    for l in self.layers:
      l.numerical_test=False


  def run_numerical_check(self,inp,label):
    self.set_numerical_test()
    out = self.forward_grad(inp)
    L = torch.sum((out - label)**2)
    L.backward()
    print("Backprop gradients")
    for l in self.layers:
      if hasattr(l,"weights"):
        print("BP weight grad: ", l.weights.grad.shape)
        print(l.weights.grad[0:10,0])
      if hasattr(l,"bias"):
        print("BP bias grad: ", l.bias.grad.shape)
        print(l.bias.grad[0:10,0])
    print("AR gradients")
    self.unset_numerical_test()
    with torch.no_grad():
      self.learn_batch(inp,label,self.n_inference_steps_train,update_weights=False)

  def save_model(self,logdir,savedir,losses,accs,test_accs):
    np.save(logdir +"/losses.npy",np.array(losses))
    np.save(logdir+"/accs.npy",np.array(accs))
    np.save(logdir+"/test_accs.npy",np.array(test_accs))
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])

  def accuracy(self, inp, label):
    _,ypred, _ = self.infer(inp, label,test=True)
    print("ACC: ", ypred[0,:])
    print("ACC LABEL: ", label[0,:])
    return accuracy(ypred, label)

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

def numerical_check(net,dataset):
    images,label = dataset[0]
    lr=0.0001
    inference_lr = 0.1
    inp = nn.Parameter(images.to(DEVICE))
    l1 = ConvLayer(32,3,6,64,5,lr,relu,relu_deriv,inference_lr,device=DEVICE)
    l2 = MaxPool(2,device=DEVICE)
    l3 = ConvLayer(14,6,16,64,5,lr,relu,relu_deriv,inference_lr,device=DEVICE)
    l4 = ProjectionLayer((64,16,10,10),120,relu,relu_deriv,lr,inference_lr,device=DEVICE)
    l5 = FCLayer(120,84,64,lr,inference_lr,relu,relu_deriv,device=DEVICE)
    l6 = FCLayer(84,10,64,lr,inference_lr,linear,linear_deriv,device=DEVICE)
    layers =[l1,l2,l3,l4,l5,l6]
    net = ARNet(layers,500,0.05,0.001,device=DEVICE,with_amortisation=False)
    inference_steps = [1,50,100,200,500,1000,2000,3000]
    total_errs = []
    label = onehot(label)
    print(label.shape)
    net.run_numerical_check(inp,label.to(DEVICE))


class Backprop_CNN(object):
  def __init__(self, layers):
    self.layers = layers 
    self.xs = [[] for i in range(len(self.layers)+1)]
    self.e_ys = [[] for i in range(len(self.layers)+1)]
    for l in self.layers:
      l.set_weight_parameters()

  def forward(self, inp):
    self.xs[0] = inp
    for i,l in enumerate(self.layers):
      self.xs[i+1] = l.forward(self.xs[i])
    return self.xs[-1]

  def backward(self,e_y):
    self.e_ys[-1] = e_y
    for (i,l) in reversed(list(enumerate(self.layers))):
      self.e_ys[i] = l.backward(self.e_ys[i+1])
    return self.e_ys[0]

  def update_weights(self,print_weight_grads=False,update_weight=False):
    for (i,l) in enumerate(self.layers):
      dW = l.update_weights(self.e_ys[i+1],update_weights=update_weight)
      if print_weight_grads:
        print("weight grads : ", i)
        print("dW: ", dW*2)
        print("weight grad: ",l.get_true_weight_grad())

  def train(self, dataset,n_epochs):
    with torch.no_grad():
      for n in range(n_epochs):
        print("Epoch: ",n)
        for (inp,label) in dataset:
          out = self.forward(inp.to(DEVICE))
          label = onehot(label).to(DEVICE)
          e_y = out - label
          self.backward(e_y)
          print("out: ",out[0,:])
          print("label: ",label[0,:])
          self.update_weights(update_weight=True)
          print("Loss: ", torch.sum(e_y**2))
          print("Accuracy: ", accuracy(out,label))





if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--normalize_data",type=boolcheck, default=True)
    parser.add_argument("--learning_rate",type=float,default=0.0005)
    parser.add_argument("--N_epochs",type=int, default=30)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--n_inference_steps",type=int,default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--dataset",type=str,default="cifar")
    parser.add_argument("--use_FC_backwards_weights",type=boolcheck, default=False)
    parser.add_argument("--use_FC_backwards_nonlinearity",type=boolcheck, default=False)
    parser.add_argument("--update_backwards_weights",type=boolcheck,default=True)
    parser.add_argument("--use_conv_backwards_weights",type=boolcheck,default=False)
    parser.add_argument("--use_conv_backwards_nonlinearity",type=boolcheck,default=False)
    parser.add_argument("--network_type",type=str,default="ar")

    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    if args.dataset in ["cifar","cifar100","svhn","mnist"]:
        trainset,testset = get_cnn_dataset(args.dataset,args.batch_size,args.normalize_data)
    else:
        raise ValueError("Dataset name not supported. Must be in [cifar,cifar100,svhn,mnist]")
    
    l1 = ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,args.inference_learning_rate,device=DEVICE)
    l2 = MaxPool(2,device=DEVICE)
    l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,args.inference_learning_rate,device=DEVICE)
    l4 = ProjectionLayer((64,16,10,10),120,relu,relu_deriv,args.learning_rate,args.inference_learning_rate,device=DEVICE)
    l5 = FCLayer(120,84,64,args.learning_rate,args.inference_learning_rate,relu,relu_deriv,device=DEVICE)
    l6 = FCLayer(84,10,64,args.learning_rate,args.inference_learning_rate,linear,linear_deriv,device=DEVICE)
    layers =[l1,l2,l3,l4,l5,l6]
    net = ARNet(layers,500,0.05,0.001,device=DEVICE)
    layers =[l1,l2,l3,l4]
    if args.network_type == "ar":
        net = ARNet(layers,args.n_inference_steps,use_FC_backwards_weights=args.use_FC_backwards_weights,
         update_backwards_weights = args.update_backwards_weights, 
         use_FC_backward_nonlinearity = args.use_FC_backward_nonlinearity,
         use_conv_backwards_weights = args.use_conv_backwards_weights,
         use_conv_backwards_nonlinearity = args.use_conv_backwards_nonlinearity,
         device=DEVICE)
    elif args.network_type == "bp":
        net = BackpropNet(layers,device=DEVICE)
    else:
        raise ValueError("Network type not recognised: must be either ar (activation relaxation) or bp (backprop)")
    net.train(trainset[0:-2],testset[0:-2],args.logdir,args.savedir,args.N_epochs, args.n_inference_steps)