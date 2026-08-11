import torch
import torch.nn as nn
import torch.nn.functional as F





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

        #Make sure the update isn't actually higher rank than the original weights
        self.rank = min(out_features // 2, rank)
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

     
        # U, _ = torch.linalg.qr(torch.randn(out_features, rank)) 
        U = torch.zeros(out_features, rank)
        nn.init.normal_(U, std=1e-6)

        V, _ = torch.linalg.qr(torch.randn(in_features, rank))
        
        self.B = nn.Parameter(U)
        self.A = nn.Parameter(V.T)
        
    

    def forward(self, x):
        return F.linear(
            x,
            self.weight + (self.B @ self.A)
        )


class RiemannianLinear_USVh(nn.Module):
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

        #Make sure the update isn't actually higher rank than the original weights
        self.rank = min(out_features // 2, rank)
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

        U, _ = torch.linalg.qr(torch.randn(out_features, rank))
        V, _ = torch.linalg.qr(torch.randn(in_features, rank))
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(torch.diag(torch.ones(rank)))
        self.Vh = nn.Parameter(V.T)
        
    

    def forward(self, x):
        return F.linear(
            x,
            self.weight + (self.U @ self.S @ self.Vh),
            bias = self.bias
        )


# class RiemannianEmbedding(nn.Module):
#     def __init__(
#         self,
#         num_embeddings,
#         embedding_dim,
#         rank,
#         pretrained_weights=None,
#     ):
#         super().__init__()

#         self.rank = rank

#         if pretrained_weights is None:
#             W0 = torch.empty(num_embeddings, embedding_dim)
#             nn.init.normal_(W0, mean=0.0, std=0.02)
#         else:
#             W0 = pretrained_weights

#         self.weight = nn.Parameter(W0, requires_grad=False)

#         # U, _ = torch.linalg.qr(torch.randn(num_embeddings, rank)) 
#         U = torch.zeros(num_embeddings, rank)
#         nn.init.normal_(U, std=1e-6)
        
#         V, _ = torch.linalg.qr(torch.randn(embedding_dim, rank))
        
#         self.B = nn.Parameter(U)
#         self.A = nn.Parameter(V.T)

#     def forward(self, input_ids):
#         base = F.embedding(input_ids, self.weight)

#         B_rows = F.embedding(input_ids, self.B)
#         update = B_rows @ self.A

#         return base + update

class RiemannConv2d(nn.Module):
    def __init__(self, conv, rank=8, alpha=1.0):
        super().__init__()

        self.weight = nn.Parameter(conv.weight, requires_grad = False)
        

        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups


        if not (conv.bias is None):
            self.bias = nn.Parameter(conv.bias)
        else:
            self.register_parameter("bias", None)


        out_c, in_c, kh, kw = self.weight.shape
        d = in_c * kh * kw

        # A starts on the Stiefel manifold (orthonormal rows)
        Q, _ = torch.linalg.qr(torch.randn(d, rank))
        self.A = nn.Parameter(Q.T)

        self.B = nn.Parameter(torch.zeros(out_c, rank))
        nn.init.normal_(self.B, std=1e-6)




    def forward(self, x):
        delta = (self.B @ self.A).view_as(self.weight)

        return F.conv2d(
            x,
            self.weight + delta,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class RiemannConv2d_USVh(nn.Module):
    def __init__(self, conv, rank=8, alpha=1.0):
        super().__init__()

        self.weight = nn.Parameter(conv.weight, requires_grad = False)
        

        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups


        if not (conv.bias is None):
            self.bias = nn.Parameter(conv.bias)
        else:
            self.register_parameter("bias", None)


        out_c, in_c, kh, kw = self.weight.shape
        d = in_c * kh * kw
        U, _ = torch.linalg.qr(torch.randn(out_c, rank))
        V, _ = torch.linalg.qr(torch.randn(d, rank))


        self.U = nn.Parameter(U)
        self.S = nn.Parameter(torch.diag(torch.ones(rank)))
        self.Vh = nn.Parameter(V.T)


  

    def forward(self, x):
        delta = (self.U @ self.S @ self.Vh).view_as(self.weight)
       

        return F.conv2d(
            x,
            self.weight + delta,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
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




from torch import nn

from transformers.pytorch_utils import Conv1D
from torch.nn.modules.sparse import Embedding

@torch.no_grad()
def riemannize(model, rank, exclusions=None, mode = "BA"):

    if mode == "BA":
        linear_layer = RiemannianLinear
        conv_layer = RiemannConv2d
    elif mode == "USVh":
        linear_layer = RiemannianLinear_USVh
        conv_layer = RiemannConv2d_USVh

    if exclusions is None:
        exclusions = []

    layers_to_replace = []

    # Find eligible layers
    for name, module in model.named_modules():
        if module in exclusions:
            continue

        if isinstance(module, nn.Linear):
            layers_to_replace.append((name, module, "linear"))

        elif isinstance(module, Conv1D):
            layers_to_replace.append((name, module, "conv1d"))

        elif isinstance(module, nn.Conv2d):
            layers_to_replace.append((name, module, "conv2d"))

    # Replace them
    for name, module, layer_type in layers_to_replace:

        if layer_type == "linear":
            new_layer = linear_layer(
                module.in_features,
                module.out_features,
                rank,
                module.weight,
                bias = (hasattr(module, "bias") and module.bias != None)
            ).to(module.weight.device)

            if hasattr(module, "bias") and module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)

        elif layer_type =="conv1d":  # GPT-2 Conv1D
            # Conv1D weight shape: (in_features, out_features)
            weight = module.weight.T

            new_layer = linear_layer(
                weight.shape[1],  # in_features
                weight.shape[0],  # out_features
                rank,
                weight,
                bias = (hasattr(module, "bias") and module.bias != None)
            ).to(weight.device)

            if hasattr(module, "bias") and module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)

        elif layer_type =="conv2d":  
            new_layer = conv_layer(conv = module, rank = rank).to(module.weight.device)

        set_submodule(model, name, new_layer)

    torch.cuda.empty_cache()
    return model



def split_parameters(model):
    flattened = [m for m in model.modules() if len(list(m.children())) == 0]
    used_params = []
    riemann_params = []
    other_params = []

    for module in flattened:
        if isinstance(module, RiemannianLinear) or isinstance(module, RiemannConv2d):
            riemann_params.extend([module.A, module.B])
            used_params.extend([id(module.A), id(module.B)])
        else:
            current_parameters = [param for param in module.parameters() if not (id(param) in used_params)]
            used_params.extend([id(param) for param in current_parameters])
            other_params.extend(current_parameters)

    return riemann_params, other_params
    

