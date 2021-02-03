import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoMap:
    def __init__(self):
        pass
    def map_size(self):
        return 2
    def map(self, X):
        return X

class FFM:
    def __init__(self, B):
        def proj(x, B):
            x_proj = torch.matmul(x, B.T)
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        self.B = torch.Tensor(B)
        self.map_f = lambda x: proj(x, self.B)

    def map_size(self):
        return 2*self.B.shape[0]

    def map(self, X):
        if self.B.device != X.device:
            self.B = self.B.to(X.device)
        return self.map_f(X)

class RFF:
    '''
    Generate Random Fourier Features (RFF) corresponding to the kernel:
        k(x, y) = k_a(x_a, y_a)*k_b(x_b, y_b)
    where 
        k_a(x_a, y_a) = exp(-\norm(x_a-y_a)/gamma_1),
        k_b(x_b, y_b) = <x_b, y_b>^gamma_2.
    '''
    def __init__(self, W, b):
        self.W = torch.Tensor(W)
        self.b = torch.Tensor(b)
        self.N = self.W.shape[0]

    def map_size(self):
        return self.N

    def map(self, X):
        if self.W.device != X.device:
            self.W = self.W.to(X.device)
            self.b = self.b.to(X.device)
        Z = torch.cos(X @ self.W.T + self.b)
        return Z

class MLP(nn.Module):
    def __init__(self, D, W, map):
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.map = map
        self.input_size = map.map_size()
        
        self.linears = nn.ModuleList(
            [nn.Linear(self.input_size, W)] + [nn.Linear(W, W) for i in range(D)])
        self.output_linear = nn.Linear(W, 3)

    def forward(self, x):
        h = self.map.map(x)
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        outputs = torch.sigmoid(self.output_linear(h))
        return outputs

class SIREN(MLP):
    ### Taken from official SIREN repo
    class Sine(nn.Module):
        def __init(self):
            super().__init__()

        def forward(self, input):
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            return torch.sin(30 * input)

    def __init__(self, D, W, map):
        super().__init__(D, W, map)
        self.nl = SIREN.Sine()

        # SIREN initialization
        for linear in self.linears:
            linear.apply(self.sine_init)
        self.output_linear.apply(SIREN.sine_init)

        # Apply special initialization to first layer
        self.linears[0].apply(SIREN.first_layer_sine_init)

    def forward(self, x):
        h = self.map.map(x)
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = self.nl(h)
        outputs = torch.sigmoid(self.output_linear(h))
        return outputs

    @staticmethod
    def sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

    @staticmethod
    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)
    
def make_ffm_network(D, W, B=None, use_siren=False):
    map = FFM(B)
    if not use_siren:
        return MLP(D, W, map).float()
    else:
        return SIREN(D, W, map).float()

def make_rff_network(D, W, We, b, use_siren=False):
    map = RFF(We, b)
    if not use_siren:
        return MLP(D, W, map).float()
    else:
        return SIREN(D, W, map).float()

def make_relu_network(D, W, use_siren=False):
    map = NoMap()
    if not use_siren:
        return MLP(D, W, map).float()
    else:
        return SIREN(D, W, map).float()

model_pred = lambda model, x: model(x)
model_loss = lambda pred, y: torch.mean((pred - y) ** 2)
model_loss2 = lambda model, x, y: torch.mean((model_pred(model, x) - y) ** 2)
model_psnr = lambda loss : -10. * torch.log10(loss)