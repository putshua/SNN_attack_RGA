import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self, X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_output * tmp
        return grad_input, None


class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem*tau + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau = ctx.saved_tensors
        x = x.mean(0, keepdim=True)
        gamma = 0.2
        ext = 1 #
        des = 1
        grad = (x>=1-tau).float()*(x<=1+ext).float()*(des-gamma+gamma*tau)/(tau+ext) + (x<=1-tau).float()*(x>=0).float()*gamma
        grad_input = grad_output * grad
        return grad_input, None


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

floor = STE.apply

class qcfs(nn.Module):
    def __init__(self, up=8., t=8):
        super().__init__()
        # self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        # x = x / self.up
        x = torch.clamp(x, 0, 1)
        x = floor(x*self.t+0.5)/self.t
        # x = x * self.up
        return x

class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = qcfs()
        self.ratebp = RateBp.apply
        self.mode = 'bptt'
        self.T = T
        self.up = 1

    def forward(self, x):
        # x = x/self.up
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x, self.tau)
            x = self.merge(x)
        elif self.T > 0:
            x = self.expand(x)
            mem = 0.
            spike_pot = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                spike = self.act(mem - self.thresh, 1)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            # random.shuffle(spike_pot)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = self.relu(x)
        # x = x*self.up
        return x

def add_dimention(x, T):
    x = x.unsqueeze(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

class Poi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rand = torch.rand_like(x)
        out = (x>=rand).float()
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        # x = x.mean(0, keepdim=True)
        # out = out.mean(0, keepdim=True)
        return grad_output

poi = Poi.apply

class Poisson(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = poi(x)
        return out
