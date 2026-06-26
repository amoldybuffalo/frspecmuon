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


class RiemannianEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        rank,
        pretrained_weights=None,
    ):
        super().__init__()

        self.rank = rank

        if pretrained_weights is None:
            W0 = torch.empty(num_embeddings, embedding_dim)
            nn.init.normal_(W0, mean=0.0, std=0.02)
        else:
            W0 = pretrained_weights

        self.weight = nn.Parameter(W0, requires_grad=False)

        # U, _ = torch.linalg.qr(torch.randn(num_embeddings, rank)) 
        U = torch.zeros(num_embeddings, rank)
        nn.init.normal_(U, std=1e-6)
        
        V, _ = torch.linalg.qr(torch.randn(embedding_dim, rank))
        
        self.B = nn.Parameter(U)
        self.A = nn.Parameter(V.T)

    def forward(self, input_ids):
        base = F.embedding(input_ids, self.weight)

        B_rows = F.embedding(input_ids, self.B)
        update = B_rows @ self.A

        return base + update


def set_submodule(model, path, new_module):
    if "." in path:
        parent_path, child_name = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        #set_submodule(parent, child_name, new_module) don't think I need the recursion
    else:
        parent = model
        child_name = path

    parent._modules[child_name] = new_module
    
@torch.no_grad
def riemannize(model, rank, exclusions = []):
    linear_layers = []
    #Get all non-excluded linear layers
    for name, module in model.named_modules():
        if not module in exclusions:
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

    # Replace them with LoRAed versions
    for layer in linear_layers:
        name, module = layer
        new_layer = RiemannianLinear(module.in_features, module.out_features, rank, module.weight).to(module.weight.device)
        if layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias.data)

        set_submodule(model, name, new_layer)


    torch.cuda.empty_cache()
    return model



from torch import nn
from transformers.pytorch_utils import Conv1D
from torch.nn.modules.sparse import Embedding

@torch.no_grad()
def riemannize_experimental(model, rank, exclusions=None):
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

        elif isinstance(module, Embedding):
            layers_to_replace.append((name, module, "embedding"))

    # Replace them
    for name, module, layer_type in layers_to_replace:

        if layer_type == "linear":
            new_layer = RiemannianLinear(
                module.in_features,
                module.out_features,
                rank,
                module.weight
            ).to(module.weight.device)

            # if module.bias is not None:
            #     new_layer.bias.data.copy_(module.bias.data)

        elif layer_type =="conv1d":  # GPT-2 Conv1D
            # Conv1D weight shape: (in_features, out_features)
            weight = module.weight.T

            new_layer = RiemannianLinear(
                weight.shape[1],  # in_features
                weight.shape[0],  # out_features
                rank,
                weight
            ).to(weight.device)

            if module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)

        elif layer_type =="embedding":  # GPT-2 Conv1D
            # Conv1D weight shape: (in_features, out_features)
            weight = module.weight

            new_layer = RiemannianEmbedding(
                module.num_embeddings,  # in_features
                module.embedding_dim,  # out_features
                rank,
                weight
            ).to(weight.device)

        set_submodule(model, name, new_layer)

    torch.cuda.empty_cache()
    return model