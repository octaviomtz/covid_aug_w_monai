# import cv2 
import torch
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from IPython.display import Image, HTML, clear_output
import matplotlib
import io
import sys

def correct_label_in_plot(model):
    '''get a string with the network architecture to print in the figure'''
    # https://www.kite.com/python/answers/how-to-redirect-print-output-to-a-variable-in-python
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    print(model);
    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    model_str = [i.split(', k')[0] for i in output.split('\n')]
    model_str_layers = [i.split(':')[-1] for i in model_str[2:-3]]
    model_str = [model_str[0]]+model_str_layers
    model_str = str(model_str).replace("', '",'\n')
    return model_str

def create_sobel_and_identity(device='cuda'):
  ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).to(device)
  sobel_x = (torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0).to(device)
  lap = (torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0).to(device)
  return ident, sobel_x, lap

def prepare_seed(target, this_seed, device, num_channels = 16, pool_size = 1024):
# prepare seed
  height, width, _ = np.shape(target)
  seed = np.zeros([1, height, width, num_channels], np.float32)
  for i in range(num_channels-1):
    seed[:,..., i+1] = this_seed
  # Preparing the seed pool
  seed_tensor = torch.tensor(seed).permute(0,-1,1,2).to(device)
  seed_pool = torch.repeat_interleave(seed_tensor, repeats = pool_size, dim = 0)
  return seed, seed_tensor, seed_pool

def epochs_in_inner_loop(i, inner_iter_aux=0, inner_iter=0, thresh_do_nothing=100, thresh_do_something=200, increase=10, inner_iter_max=100):
  if i < thresh_do_nothing:
    inner_iter = 100
  elif i % thresh_do_something == 0: 
    inner_iter_aux = inner_iter_aux + increase
    inner_iter = np.min([inner_iter_aux, inner_iter_max])
  else:
    inner_iter=inner_iter
  return inner_iter, inner_iter_aux

def plot_loss_and_lesion_synthesis(losses, optimizer, model_str, i, loss, sample_size, out, no_plot=False):
  if no_plot:
    lr_info = f'\nlr_init={optimizer.param_groups[0]["initial_lr"]:.1E}\nlr_last={optimizer.param_groups[0]["lr"]:.1E}'
    model_str_final = model_str+lr_info
    return model_str_final
  clear_output(True)
  f, (ax0, ax1) = plt.subplots(2, 1, figsize=(12,10), gridspec_kw={'height_ratios': [4, 1]})
  lr_info = f'\nlr_init={optimizer.param_groups[0]["initial_lr"]:.1E}\nlr_last={optimizer.param_groups[0]["lr"]:.1E}'
  model_str_final = model_str+lr_info
  ax0.plot(losses, label=model_str_final)
  ax0.set_yscale('log')
  ax0.legend(loc='upper right', fontsize=16)

  stack = []
  for z in range(sample_size):
      stack.append(to_rgb(out[z].permute(-2, -1,0).cpu().detach().numpy()))
  ax1.imshow(np.clip(np.hstack(np.squeeze(stack)), 0,1))
  ax1.axis('off')
  plt.show()
  print(i, loss.item(), flush = True)
  return model_str_final

def to_rgb(img, channel=1):
    '''return visible channel'''
    # rgb, a = img[:,:,:1], img[:,:,1:2]
    rgb, a = img[:,:,:channel], img[:,:,channel:channel+1]
    return 1.0-a+rgb

class CeA_BASE(nn.Module):
    def __init__(self, checkpoint = None, seq_layers = None, device = 'cuda', grow_on_k_iter=1, background_intensity=.19, step_size=1, scale_mask=1, pretrain_thres=100, ch0_1=1, ch1_16=16, alive_thresh=0.1):
        '''
        Kind of a modular class for a CA model
        args:
            checkpoint = 'path/to/model.pt'
            seq_layers = nn.Sequential(your, pytorch, layers)
            device = 'cuda' or 'cpu'
        '''
        super(CeA_BASE, self).__init__()

        self.ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).to(device)
        self.sobel_x = (torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0).to(device)
        self.lap = (torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0).to(device)
        self.grow_on_k_iter = grow_on_k_iter
        self.background_intensity = background_intensity
        self.step_size = step_size
        self.scale_mask = scale_mask
        self.pretrain_thres = pretrain_thres 
        self.ch0_1 = ch0_1
        self.ch1_16 = ch1_16
        self.alive_thresh = alive_thresh
        
        if seq_layers is not None:
            self.model = seq_layers
        else:
            self.model = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 16, kernel_size =  1, bias = True),
            )
            
        '''
        initial condition for "do nothing" behaviour:
            * all biases should be zero
            * the weights of the last layer should be zero
        '''
        for l in range(len(self.model)):
            if isinstance(self.model[l], nn.Conv2d):
                self.model[l].bias.data.fill_(0)
                if l == len(self.model) -1:
                    self.model[l].weight.data.fill_(0)

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint))

        self.to(device= device)
    
    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:,None])
        return y.reshape(b, -1, h, w)

    def perception(self, x):
        filters = torch.stack([self.ident, self.sobel_x, self.sobel_x.T, self.lap])
        return self.perchannel_conv(x, filters)
        
    def normalize_grads(self):
        '''
        gradient normalization for constant step size and to avoid spikes 
        '''
        for p in self.parameters():
            p.grad.data = p.grad.data/(p.grad.data.norm()+1e-8)    
            
            
    def get_alive_mask(self, x):
        '''
        looks for cells that have values over 0.1, 
        and allows only their adjacent cells to participate in growth
        '''
        alpha = x[:,1:2,:,:]
        pooled = (F.max_pool2d(alpha, 3,1, padding =1 ) > 0.1).float()
        return pooled
    
    def train_step(self, seed, target, target_loss_func, epochs_inside, epoch_outside = 1000, masked_loss=False):
        '''
        a single training step for the model,
        feel free to play around with different loss functions like L1 loss 

        the loss is calculated for only the first 4 channels of the output
        '''
        x = seed 
        for epoch_in in range(epochs_inside):
            x, alive_mask, other =  self.forward(x, epoch_in, epoch_outside)

        if masked_loss == True:
            alive_mask_dilated = (F.max_pool2d(alive_mask[0], 3,1, padding =1 ) > 0.1).float()
            target_loss  =  target_loss_func(x[:,:1, :,:] * alive_mask_dilated, target * alive_mask_dilated)  
        else:
            # target_loss  =  target_loss_func(x[:,:2, :,:] * target[:,1:,...], target * target[:,1:,...]) # used to synthesize almost all nodules 
            target_loss  =  target_loss_func(x[:,:2, :,:], target) # ORIGINAL

        loss = target_loss 
            
        return loss, x, alive_mask.cpu().numpy(), other.detach().cpu().numpy() #batch_mean_rmse_per_pixel.detach().cpu().numpy()

    def forward(self, x, epoch_in, epoch_outside):
        '''
        nice little forward function for the model
        1. fetches an alive mask 
        2. generates another random mask of 0's and 1's 
        3. updates the input 
        4. applies alive mask 
        '''
        mask_previous = alive_mask = (x[:,1:2,:,:] > 0.1).float()
        # self_pretraining
        if epoch_outside < self.pretrain_thres: 
          alive_mask = self.get_alive_mask(x)
        else:
          if epoch_in % self.grow_on_k_iter == 0:
            alive_mask = self.get_alive_mask(x)
          else:
            alive_mask = (x[:,1:2,:,:] > 0.1).float()
            mask_previous = torch.zeros_like(alive_mask)#OMM added in CeA

        # MASK CLAMP     
        # | = self.background_intensity  
        # X = self.step_size
        # S = self.scale_mask
        # ch0            ch1           ch2           ...
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # ||XX||||||XX|| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS

        mask_diff = alive_mask - mask_previous
        mask_clamp_ch0 = torch.clip((1-mask_diff) + self.background_intensity,0,self.step_size) #make sure this is only applied to the first channel
        mask_clamp = torch.repeat_interleave(mask_clamp_ch0,16,1)
        mask_clamp_ones = torch.ones_like(torch.squeeze(mask_clamp_ch0))*self.scale_mask
        for idx_channel in np.arange(1,16,1):
          mask_clamp[:,idx_channel,:,:] = mask_clamp_ones

        
        mask = torch.clamp(torch.round(torch.rand_like(x[:,:1,:,:])) , 0,1)        
        P = self.perception(x)
        Y = self.model(P)
        # out = x + (Y * mask * mask_clamp)  
        out = x + (Y * mask) #original
        out *= alive_mask
        
        return out, alive_mask, mask_clamp

class CeA_00(nn.Module):
    def __init__(self, checkpoint = None, seq_layers = None, device = 'cuda', grow_on_k_iter=3, background_intensity=.19, step_size=1, scale_mask=1, pretrain_thres=100):
        '''
        Kind of a modular class for a CA model
        args:
            checkpoint = 'path/to/model.pt'
            seq_layers = nn.Sequential(your, pytorch, layers)
            device = 'cuda' or 'cpu'
        '''
        super(CeA_00, self).__init__()

        self.ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).to(device)
        self.sobel_x = (torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0).to(device)
        self.lap = (torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0).to(device)
        self.grow_on_k_iter = grow_on_k_iter
        self.background_intensity = background_intensity
        self.step_size = step_size
        self.scale_mask = scale_mask
        self.pretrain_thres = pretrain_thres 
        
        if seq_layers is not None:
            self.model = seq_layers
        else:
            self.model = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 16, kernel_size =  1, bias = True),
            )
            
        '''
        initial condition for "do nothing" behaviour:
            * all biases should be zero
            * the weights of the last layer should be zero
        '''
        for l in range(len(self.model)):
            if isinstance(self.model[l], nn.Conv2d):
                self.model[l].bias.data.fill_(0)
                if l == len(self.model) -1:
                    self.model[l].weight.data.fill_(0)

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint))

        self.to(device= device)
    
    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:,None])
        return y.reshape(b, -1, h, w)

    def perception(self, x):
        filters = torch.stack([self.ident, self.sobel_x, self.sobel_x.T, self.lap])
        return self.perchannel_conv(x, filters)
        
    def normalize_grads(self):
        '''
        gradient normalization for constant step size and to avoid spikes 
        '''
        for p in self.parameters():
            p.grad.data = p.grad.data/(p.grad.data.norm()+1e-8)    
            
            
    def get_alive_mask(self, x):
        '''
        looks for cells that have values over 0.1, 
        and allows only their adjacent cells to participate in growth
        '''
        alpha = x[:,1:2,:,:]
        pooled = (F.max_pool2d(alpha, 3,1, padding =1 ) > 0.1).float()
        return pooled
    
    def train_step(self, seed, target, target_loss_func, epochs_inside, epoch_outside = 1000, masked_loss=False):
        '''
        a single training step for the model,
        feel free to play around with different loss functions like L1 loss 

        the loss is calculated for only the first 4 channels of the output
        '''
        x = seed 
        for epoch_in in range(epochs_inside):
            x, alive_mask, other =  self.forward(x, epoch_in, epoch_outside)

        if masked_loss == True:
            alive_mask_dilated = (F.max_pool2d(alive_mask[0], 3,1, padding =1 ) > 0.1).float()
            target_loss  =  target_loss_func(x[:,:1, :,:] * alive_mask_dilated, target * alive_mask_dilated)  
        else:
            # target_loss  =  target_loss_func(x[:,:2, :,:] * target[:,1:,...], target * target[:,1:,...]) # used to synthesize almost all nodules 
            target_loss  =  target_loss_func(x[:,:2, :,:], target) # ORIGINAL

        loss = target_loss 
            
        return loss, x, alive_mask.cpu().numpy(), other.detach().cpu().numpy() #batch_mean_rmse_per_pixel.detach().cpu().numpy()

    def forward(self, x, epoch_in, epoch_outside):
        '''
        nice little forward function for the model
        1. fetches an alive mask 
        2. generates another random mask of 0's and 1's 
        3. updates the input 
        4. applies alive mask 
        '''
        mask_previous = alive_mask = (x[:,1:2,:,:] > 0.1).float()
        # self_pretraining
        if epoch_outside < self.pretrain_thres: 
          alive_mask = self.get_alive_mask(x)
        else:
          if epoch_in % self.grow_on_k_iter == 0:
            alive_mask = self.get_alive_mask(x)
          else:
            alive_mask = (x[:,1:2,:,:] > 0.1).float()
            mask_previous = torch.zeros_like(alive_mask)#OMM added in CeA

        # MASK CLAMP     
        # | = self.background_intensity  
        # X = self.step_size
        # S = self.scale_mask
        # ch0            ch1           ch2           ...
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # ||XX||||||XX|| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS

        mask_diff = alive_mask - mask_previous
        mask_clamp_ch0 = torch.clip((1-mask_diff) + self.background_intensity,0,self.step_size) #make sure this is only applied to the first channel
        mask_clamp = torch.repeat_interleave(mask_clamp_ch0,16,1)
        mask_clamp_ones = torch.ones_like(torch.squeeze(mask_clamp_ch0))*self.scale_mask
        for idx_channel in np.arange(1,16,1):
          mask_clamp[:,idx_channel,:,:] = mask_clamp_ones

        
        mask = torch.clamp(torch.round(torch.rand_like(x[:,:1,:,:])) , 0,1)        
        P = self.perception(x)
        Y = self.model(P)
        out = x + (Y * mask * mask_clamp)  
        out *= alive_mask
        
        return out, alive_mask, mask_clamp

    
class CeA_0x(nn.Module):
    def __init__(self, checkpoint = None, seq_layers = None, device = 'cuda', grow_on_k_iter=3, background_intensity=.19, step_size=1, scale_mask=1, pretrain_thres=100, ch0_1=1, ch1_16=16, alive_thresh=0.1):
        '''
        Kind of a modular class for a CA model
        args:
            checkpoint = 'path/to/model.pt'
            seq_layers = nn.Sequential(your, pytorch, layers)
            device = 'cuda' or 'cpu'
        '''
        super(CeA_0x, self).__init__()

        self.ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).to(device)
        self.sobel_x = (torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0).to(device)
        self.lap = (torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0).to(device)
        self.grow_on_k_iter = grow_on_k_iter
        self.background_intensity = background_intensity
        self.step_size = step_size
        self.scale_mask = scale_mask
        self.pretrain_thres = pretrain_thres 
        self.ch0_1 = ch0_1
        self.ch1_16 = ch1_16
        self.alive_thresh = alive_thresh
        
        if seq_layers is not None:
            self.model = seq_layers
        else:
            self.model = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 16, kernel_size =  1, bias = True),
            )
            
        '''
        initial condition for "do nothing" behaviour:
            * all biases should be zero
            * the weights of the last layer should be zero
        '''
        for l in range(len(self.model)):
            if isinstance(self.model[l], nn.Conv2d):
                self.model[l].bias.data.fill_(0)
                if l == len(self.model) -1:
                    self.model[l].weight.data.fill_(0)

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint))

        self.to(device= device)
    
    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:,None])
        return y.reshape(b, -1, h, w)

    def perception(self, x):
        filters = torch.stack([self.ident, self.sobel_x, self.sobel_x.T, self.lap])
        return self.perchannel_conv(x, filters)
        
    def normalize_grads(self):
        '''
        gradient normalization for constant step size and to avoid spikes 
        '''
        for p in self.parameters():
            p.grad.data = p.grad.data/(p.grad.data.norm()+1e-8)    
            
            
    def get_alive_mask(self, x):
        '''
        looks for cells that have values over 0.1, 
        and allows only their adjacent cells to participate in growth
        '''
        alpha = x[:,1:2,:,:]
        pooled = (F.max_pool2d(alpha, 3,1, padding =1 ) > self.alive_thresh).float()
        return pooled
    
    def train_step(self, seed, target, target_loss_func, epochs_inside, epoch_outside = 1000, masked_loss=False):
        '''
        a single training step for the model,
        feel free to play around with different loss functions like L1 loss 

        the loss is calculated for only the first 4 channels of the output
        '''
        x = seed 
        for epoch_in in range(epochs_inside):
            x, alive_mask, other =  self.forward(x, epoch_in, epoch_outside)

        if masked_loss == True:
            alive_mask_dilated = (F.max_pool2d(alive_mask[0], 3,1, padding =1 ) > 0.1).float()
            target_loss  =  target_loss_func(x[:,:1, :,:] * alive_mask_dilated, target * alive_mask_dilated)  
        else:
            # target_loss  =  target_loss_func(x[:,:2, :,:] * target[:,1:,...], target * target[:,1:,...]) # used to synthesize almost all nodules 
            target_loss  =  target_loss_func(x[:,:2, :,:], target) # ORIGINAL

        loss = target_loss 
            
        return loss, x, alive_mask.cpu().numpy(), other.detach().cpu().numpy() #batch_mean_rmse_per_pixel.detach().cpu().numpy()

    def forward(self, x, epoch_in, epoch_outside):
        '''
        nice little forward function for the model
        1. fetches an alive mask 
        2. generates another random mask of 0's and 1's 
        3. updates the input 
        4. applies alive mask 
        '''
        mask_previous = alive_mask = (x[:,1:2,:,:] > self.alive_thresh).float()
        # self_pretraining
        if epoch_outside < self.pretrain_thres: 
          alive_mask = self.get_alive_mask(x)
        else:
          if epoch_in % self.grow_on_k_iter == 0:
            alive_mask = self.get_alive_mask(x)
          else:
            alive_mask = (x[:,1:2,:,:] > self.alive_thresh).float()
            mask_previous = torch.zeros_like(alive_mask)#OMM added in CeA

        # MASK CLAMP     
        # | = self.background_intensity  
        # X = self.step_size
        # S = self.scale_mask
        # ch0            ch1           ch2           ...
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # ||XX||||||XX|| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS

        mask_diff = alive_mask - mask_previous
        mask_clamp_ch0 = torch.clip((1-mask_diff) + self.background_intensity,0,self.step_size) #make sure this is only applied to the first channel
        mask_clamp = torch.repeat_interleave(mask_clamp_ch0,16,1)
        mask_clamp_ones = torch.ones_like(torch.squeeze(mask_clamp_ch0))*self.scale_mask
        for idx_channel in np.arange(self.ch0_1,self.ch1_16,1):
          mask_clamp[:,idx_channel,:,:] = mask_clamp_ones

        
        mask = torch.clamp(torch.round(torch.rand_like(x[:,:1,:,:])) , 0,1)        
        P = self.perception(x)
        Y = self.model(P)
        out = x + (Y * mask * mask_clamp)  
        out *= alive_mask
        
        return out, alive_mask, mask_clamp

class CeA_BASE_1CNN(nn.Module):
    def __init__(self, checkpoint = None, seq_layers = None, device = 'cuda', grow_on_k_iter=1, background_intensity=.19, step_size=1, scale_mask=1, pretrain_thres=100, ch0_1=1, ch1_16=16, alive_thresh=0.1):
        '''
        Kind of a modular class for a CA model
        args:
            checkpoint = 'path/to/model.pt'
            seq_layers = nn.Sequential(your, pytorch, layers)
            device = 'cuda' or 'cpu'
        '''
        super(CeA_BASE_1CNN, self).__init__()

        self.ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).to(device)
        self.sobel_x = (torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0).to(device)
        self.lap = (torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0).to(device)
        self.grow_on_k_iter = grow_on_k_iter
        self.background_intensity = background_intensity
        self.step_size = step_size
        self.scale_mask = scale_mask
        self.pretrain_thres = pretrain_thres 
        self.ch0_1 = ch0_1
        self.ch1_16 = ch1_16
        self.alive_thresh = alive_thresh
        
        if seq_layers is not None:
            self.model = seq_layers
        else:
            self.model = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size = 3,padding =1,  bias = True),  
                nn.ReLU(),
                nn.Conv2d(256, 16, kernel_size =  1, bias = True),
            )
            
        '''
        initial condition for "do nothing" behaviour:
            * all biases should be zero
            * the weights of the last layer should be zero
        '''
        for l in range(len(self.model)):
            if isinstance(self.model[l], nn.Conv2d):
                self.model[l].bias.data.fill_(0)
                if l == len(self.model) -1:
                    self.model[l].weight.data.fill_(0)

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint))

        self.to(device= device)
    
    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:,None])
        return y.reshape(b, -1, h, w)

    def perception(self, x):
        filters = torch.stack([self.ident, self.sobel_x, self.sobel_x.T, self.lap])
        return self.perchannel_conv(x, filters)
        
    def normalize_grads(self):
        '''
        gradient normalization for constant step size and to avoid spikes 
        '''
        for p in self.parameters():
            p.grad.data = p.grad.data/(p.grad.data.norm()+1e-8)    
            
            
    def get_alive_mask(self, x):
        '''
        looks for cells that have values over 0.1, 
        and allows only their adjacent cells to participate in growth
        '''
        alpha = x[:,1:2,:,:]
        pooled = (F.max_pool2d(alpha, 3,1, padding =1 ) > 0.1).float()
        return pooled
    
    def train_step(self, seed, target, target_loss_func, epochs_inside, epoch_outside = 1000, masked_loss=False):
        '''
        a single training step for the model,
        feel free to play around with different loss functions like L1 loss 

        the loss is calculated for only the first 4 channels of the output
        '''
        x = seed 
        for epoch_in in range(epochs_inside):
            x, alive_mask, other =  self.forward(x, epoch_in, epoch_outside)

        if masked_loss == True:
            alive_mask_dilated = (F.max_pool2d(alive_mask[0], 3,1, padding =1 ) > 0.1).float()
            target_loss  =  target_loss_func(x[:,:1, :,:] * alive_mask_dilated, target * alive_mask_dilated)  
        else:
            # target_loss  =  target_loss_func(x[:,:2, :,:] * target[:,1:,...], target * target[:,1:,...]) # used to synthesize almost all nodules 
            target_loss  =  target_loss_func(x[:,:2, :,:], target) # ORIGINAL

        loss = target_loss 
            
        return loss, x, alive_mask.cpu().numpy(), other.detach().cpu().numpy() #batch_mean_rmse_per_pixel.detach().cpu().numpy()

    def forward(self, x, epoch_in, epoch_outside):
        '''
        nice little forward function for the model
        1. fetches an alive mask 
        2. generates another random mask of 0's and 1's 
        3. updates the input 
        4. applies alive mask 
        '''
        mask_previous = alive_mask = (x[:,1:2,:,:] > 0.1).float()
        # self_pretraining
        if epoch_outside < self.pretrain_thres: 
          alive_mask = self.get_alive_mask(x)
        else:
          if epoch_in % self.grow_on_k_iter == 0:
            alive_mask = self.get_alive_mask(x)
          else:
            alive_mask = (x[:,1:2,:,:] > 0.1).float()
            mask_previous = torch.zeros_like(alive_mask)#OMM added in CeA

        # MASK CLAMP     
        # | = self.background_intensity  
        # X = self.step_size
        # S = self.scale_mask
        # ch0            ch1           ch2           ...
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # ||XX||||||XX|| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||XX||||XX||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||XXXX||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS
        # |||||||||||||| SSSSSSSSSSSSS SSSSSSSSSSSSS SSSSSSSSSSSSS

        mask_diff = alive_mask - mask_previous
        mask_clamp_ch0 = torch.clip((1-mask_diff) + self.background_intensity,0,self.step_size) #make sure this is only applied to the first channel
        mask_clamp = torch.repeat_interleave(mask_clamp_ch0,16,1)
        mask_clamp_ones = torch.ones_like(torch.squeeze(mask_clamp_ch0))*self.scale_mask
        for idx_channel in np.arange(1,16,1):
          mask_clamp[:,idx_channel,:,:] = mask_clamp_ones

        
        mask = torch.clamp(torch.round(torch.rand_like(x[:,:1,:,:])) , 0,1)        
        P = self.perception(x)
        Y = self.model(P)
        # out = x + (Y * mask * mask_clamp)  
        out = x + (Y * mask) #original
        out *= alive_mask
        
        return out, alive_mask, mask_clamp