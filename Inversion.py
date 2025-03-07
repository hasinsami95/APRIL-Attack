import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
#from .metrics import total_variation as TV
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device=torch.device("cpu")

def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def inversion_attack(im_size, model, index, target_grad, grad_pos_target, batch_size, label,iter,dm,ds):
    criterion=nn.CrossEntropyLoss()
    alpha=10**(-6)
    pos_coeff=0.1
    x=torch.randn(batch_size,1, 3, im_size, im_size, requires_grad=True)
    #x=x.reshape(1,1,3,32,32)
    optimizer=optim.Adam([x], lr=0.1)
    #print(x.parameters)
    scheduler=StepLR(optimizer, step_size=500, gamma=0.1)
    dummy_gradient=model.state_dict()
    for i in range(iter):
        print("iter:", i)
        if iter%100==0:
            for param_group in optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
        model.train()
        output=model(x.to(device))
        #print(output)
        loss=criterion(output.to(device),label.to(device))
        model.zero_grad()
        gradient=torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True, only_inputs=True)
        #loss.backward()
        grad_pos_dummy=gradient[index].view(-1)
        dummy_grad_list=[]
        for param in gradient:
            #if param.grad is not None:
            #dummy_gradient[name]=param #.grad
            dummy_grad_list.append(param.view(-1))
        dummy_grad_list=torch.cat(dummy_grad_list)
        
        recon_loss_pos=-F.cosine_similarity(grad_pos_dummy, grad_pos_target, dim=0)
        # recon_loss+= alpha*total_variation(x)
        recon_loss=torch.sum((dummy_grad_list-target_grad).pow(2)) +pos_coeff*recon_loss_pos
        optimizer.zero_grad()
        model.zero_grad()
        #x.zero_grad()
        #x.grad= None #zero_grad()
        #x.grad.zero_()
        #recon_loss.backward(retain_graph=True)
        if x.grad is not None:
           x.grad.zero_()
        gradient=torch.autograd.grad(recon_loss, x, retain_graph=True, create_graph=True, only_inputs=True)
        #x.grad.sign_()
        x.grad=gradient[0] #This updates x and prevents error
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
             x.data = torch.max(torch.min(x, (1 - dm) / ds), -dm / ds)
        print("recon loss")
        print(recon_loss)
        del dummy_grad_list
        del gradient
        del grad_pos_dummy
        torch.cuda.empty_cache()
    return x.detach(), recon_loss
        
        
                
        
    
    
    