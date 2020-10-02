#Layers for the cnn
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvLayer(object):
  def __init__(self,input_size,num_channels,num_filters,batch_size,kernel_size,learning_rate,f,df,inference_lr,padding=0,stride=1,device="cpu",numerical_test=False):
    self.input_size = input_size
    self.num_channels = num_channels
    self.num_filters = num_filters
    self.batch_size = batch_size
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size)/self.stride) +1
    print(self.output_size)
    self.learning_rate = learning_rate
    self.inference_lr = inference_lr
    self.f = f
    self.df = df
    self.device = device
    self.weights= torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
    self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
    self.fold = nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
    self.numerical_test=numerical_test
    self.use_conv_backwards_weights = False
    self.use_conv_backwards_nonlinearity = False
    self.update_backwards_weights = True
    if self.use_conv_backwards_weights:
      self.backwards_weights = torch.empty(self.weights.reshape(self.num_filters,-1).T.shape).normal_(mean=0,std=0.05).to(self.device)

  def init_numerical_test(self):
      self.weights = nn.Parameter(self.weights)

  def unset_numerical_test(self):
    self.weights = self.weights.detach()

  def init_backwards_weights(self):
    pass
    #TODO backwards weights for CNN

  def forward(self,inp):
    self.x = inp.clone()
    self.old_x = self.x.clone()
    self.X_col = self.unfold(inp.clone())
    self.old_X_col = self.X_col.clone()
    self.flat_weights = self.weights.reshape(self.num_filters,-1)
    out = self.flat_weights @ self.X_col
    self.activations = out.reshape(self.batch_size, self.num_filters, self.output_size, self.output_size)
    return self.f(self.activations)

  def update_weights(self,xnext,update_weights=False):
    if self.use_conv_backwards_nonlinearity:
        fn_deriv = self.df(self.activations)
        e = xnext * fn_deriv
    else:
        e = xnext
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    self.dW = self.dout @ self.old_X_col.permute(0,2,1)
    self.dW = torch.sum(self.dW,dim=0)
    dW = self.dW.reshape((self.num_filters,self.num_channels,self.kernel_size,self.kernel_size))
    if update_weights:
      self.weights -= self.learning_rate * torch.clamp(dW * 2,-50,50)
    if self.use_conv_backwards_weights and self.update_backwards_weights:
      self.backwards_weights -= self.learning_rate * torch.clamp(dW.T *2, -50,50)
    if self.numerical_test:
      print("conv weight grad: ", (dW)[0:10,0])
    return self.dW

  def backward(self,xnext):
    if self.use_conv_backwards_nonlinearity:
        fn_deriv = self.df(self.activations)
        e = xnext * fn_deriv
    else:
        e = xnext
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    if self.use_conv_backwards_weights:
      dX_col = self.backwards_weights @ self.dout
    else:
      dX_col = self.flat_weights.T @ self.dout
    dX = self.fold(dX_col)
    xgrad = self.x - dX
    self.x -= self.inference_lr * torch.clamp(xgrad,-50,50)
    return self.x#torch.clamp(dX,-50,50)

  def get_true_weight_grad(self):
    return self.kernel.grad

  def set_weight_parameters(self):
    self.kernel = nn.Parameter(self.kernel)

class MaxPool(object):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)

  def forward(self,x):
    out, self.idxs = F.max_pool2d(x, self.kernel_size,return_indices=True)
    #print(out.shape)
    return out
  
  def backward(self, y):
    return F.max_unpool2d(y,self.idxs, self.kernel_size)

  def init_numerical_test(self):
    pass

  def unset_numerical_test(self):
    pass

  def update_weights(self,e,update_weights=False):
    return 0

  def get_true_weight_grad(self):
    return None

  def set_weight_parameters(self):
    pass

class AvgPool(object):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)
  
  def forward(self, x):
    self.B_in,self.C_in,self.H_in,self.W_in = x.shape
    return F.avg_pool2d(x,self.kernel_size)

  def backward(self, y):
    N,C,H,W = y.shape
    print("in backward: ", y.shape)
    return F.interpolate(y,scale_factor=(1,1,self.kernel_size,self.kernel_size))

  def update_weights(self,x):
    return 0

  def init_numerical_test(self):
    pass

  def unset_numerical_test(self):
    pass


class ProjectionLayer(object):
  def __init__(self,input_size, output_size,f,df,learning_rate,inference_lr,device='cpu',numerical_test=False):
    self.input_size = input_size
    self.B, self.C, self.H, self.W = self.input_size
    self.output_size =output_size
    self.learning_rate = learning_rate
    self.inference_lr = inference_lr
    self.f = f
    self.df = df
    self.device = device
    self.bias = torch.zeros((self.B, self.output_size)).float().to(self.device)
    self.Hid = self.C * self.H * self.W
    self.weights = torch.empty((self.Hid, self.output_size)).normal_(mean=0.0, std=0.05).to(self.device)
    self.numerical_test = numerical_test
    self.use_FC_backwards_weights =False
    self.use_FC_backwards_nonlinearity = False
    self.update_backwards_weights = True

  def init_backwards_weights(self):
      if self.use_FC_backwards_weights:
          self.backwards_weights = torch.empty([self.output_size,self.Hid]).normal_(mean=0.0,std=0.05).to(self.device)

  def forward(self, x):
    self.x = x.detach().clone()
    self.old_x = self.x.clone()
    out = x.reshape((len(x), -1))
    self.activations = torch.matmul(out,self.weights) + self.bias
    return self.f(self.activations)

  def backward(self, xnext):
    if not self.use_FC_backwards_weights:
        if self.use_FC_backwards_nonlinearity:
            fn_deriv = self.df(self.activations)
            out = torch.matmul(xnext * fn_deriv, self.weights.T)
        else:
            out = torch.matmul(xnext, self.weights.T)
    else:
        if self.use_FC_backwards_nonlinearity:
            fn_deriv = self.df(self.activations)
            out = torch.matmul(xnext * fn_deriv, self.backwards_weights)
        else:
            out = torch.matmul(xnext, self.backwards_weights)
    out = out.reshape((len(xnext), self.C, self.H, self.W))
    xgrad = self.x - out
    self.x -= self.inference_lr * torch.clamp(xgrad,-50,50)
    return self.x

  def update_weights(self, xnext,update_weights=False):
    out = self.old_x.reshape((len(self.old_x), -1))
    fn_deriv = self.df(self.activations)
    if self.use_FC_backwards_nonlinearity:
        dw = torch.matmul(out.T, xnext * fn_deriv)
    else:
        dw = torch.matmul(out.T, xnext)
    biasgrad = xnext * fn_deriv
    if update_weights:
      self.weights -= self.learning_rate * torch.clamp(dw,-50,50)
      self.bias -= self.learning_rate * torch.clamp(biasgrad,-50,50)
      if self.use_FC_backwards_weights and self.update_backwards_weights:
        self.backwards_weights -= self.learning_rate * torch.clamp(dw,-50,50).T
    if self.numerical_test:
      print("Player weight grad: ", (dw)[0:10,0])
      print("Player bias grad: ", (biasgrad)[0:10,0])
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def init_numerical_test(self):
      self.weights = nn.Parameter(self.weights)
      self.bias = nn.Parameter(self.bias)

  def unset_numerical_test(self):
    self.weights = self.weights.detach()
    self.bias = self.bias.detach()


class FCLayer(object):
  def __init__(self, input_size,output_size,batch_size, learning_rate,inference_lr,f,df,device="cpu",numerical_test=False):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.inference_lr = inference_lr
    self.f = f 
    self.df = df
    self.device = device
    self.bias = torch.zeros([self.batch_size, self.output_size]).float().to(self.device)
    self.weights = torch.empty([self.input_size,self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
    self.numerical_test= numerical_test
    self.use_FC_backwards_weights =False
    self.use_FC_backwards_nonlinearity = False
    self.update_backwards_weights = True

  def init_backwards_weights(self):
      if self.use_FC_backwards_weights:
          self.backwards_weights = torch.empty([self.output_size,self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)

  def forward(self,x):
    #self.inp = x.detach()
    self.x = x.clone()
    self.old_x = self.x.clone()
    self.activations = torch.matmul(self.x, self.weights) + self.bias
    return self.f(self.activations)

  def backward(self,xnext):
    if not self.use_FC_backwards_weights:
        if self.use_FC_backwards_nonlinearity:
            self.fn_deriv = self.df(self.activations)
            out = torch.matmul(xnext * self.fn_deriv, self.weights.T)
        else:
            out = torch.matmul(xnext, self.weights.T)
    else:
        if self.use_FC_backwards_nonlinearity:
            self.fn_deriv = self.df(self.activations)
            out = torch.matmul(xnext * self.fn_deriv, self.backwards_weights)
        else:
            out = torch.matmul(xnext, self.backwards_weights)
    xgrad = self.x - out
    self.x -= self.inference_lr * torch.clamp(xgrad,-50,50)
    return self.x

  def update_weights(self,xnext,update_weights=False):
    self.fn_deriv = self.df(self.activations)
    if self.use_FC_backwards_nonlinearity:
        dw = torch.matmul(self.old_x.T, xnext * self.fn_deriv)
    else:
        dw = torch.matmul(self.old_x.T, xnext)
    biasgrad = xnext * self.fn_deriv
    if update_weights:
      self.weights -= self.learning_rate * torch.clamp(dw,-50,50)
      self.bias -= self.learning_rate * torch.clamp(biasgrad,-50,50)
      if self.use_FC_backwards_weights and self.update_backwards_weights:
          self.backwards_weights -= self.learning_rate * torch.clamp(dw,-50,50).T
    if self.numerical_test:
      print("FC weight grad: ", (dw)[0:10,0])
      print("FC bias grad: ", (biasgrad)[0:10,0])
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def init_numerical_test(self):
    #print("INIT NUMERICAL TEST: ", self.weights.shape)
    self.weights = nn.Parameter(self.weights)
    #print(self.weights)
    self.bias = nn.Parameter(self.bias)

  def unset_numerical_test(self):
    self.weights = self.weights.detach()
    self.bias = self.bias.detach()

