import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F



# class FixedRankParameter(nn.Module):
#     def __init__(self, U, S, V):
#         super().__init__()

#         self.U = nn.Parameter(U)
#         self.S = nn.Parameter(S)
#         self.V = nn.Parameter(V)
#         self.X = nn.Parameter()
    
#     @property
#     def matrix(self):
#         #self.X.copy_(U @ S @ V.T)
#         return X

class RiemannianLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        pretrained_weights = None,
        bias=True,
        init_scale=1e-4,
    ):
        super().__init__()

        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        # frozen pretrained weight
        if pretrained_weights == None:
            W0 = torch.empty(out_features, in_features) 
            nn.init.kaiming_uniform_(W0)
        else:
            W0 = pretrained_weights

        self.weight = nn.Parameter(
            W0,
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # rank-r initialization

        
        U, _ = torch.linalg.qr(torch.randn(out_features, rank))
        V, _ = torch.linalg.qr(torch.randn(in_features, rank))
        
        self.B = nn.Parameter(U)
        self.A = nn.Parameter(V.T)
        
    

    def forward(self, x):
        return F.linear(
            x,
            self.weight + (self.B @ self.A)
        )



def set_submodule(model, path, new_module):
    if "." in path:
        parent_path, child_name = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        #set_submodule(parent, child_name, new_module) don't think I need the recursion
    else:
        parent = model
        child_name = path

    parent._modules[child_name] = new_module
    

def riemannize(model, rank, exclusions = []):
    linear_layers = []
    for name, module in model.named_modules():
        if not module in exclusions:
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

    for layer in linear_layers:
        name, module = layer
        new_layer = RiemannianLinear(module.in_features, module.out_features, rank, module.weight).to("cuda:0")
        set_submodule(model, name, new_layer)



    return model