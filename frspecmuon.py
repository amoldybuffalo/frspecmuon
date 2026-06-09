import torch
from torch.optim import Optimizer


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class FrSpecMuon(Optimizer):

    def tangent_space_projection(U, V, G):
        U, S, Vh = torch.linalg.svd(grad, full_matrices=False) 

        UU = U @ U.T
        VV = V @ V.T

        return (
            UU @ G
            + G @ VV
            - UU @ G @ VV
        )

    def __init__(self, model, param_groups, device, lr=1e-3):
        self.device = device
        self.r = [None] * sum([len(param_group) for param_group in param_groups])
        self.r_tilde = [None] * sum([len(param_group) for param_group in param_groups])
        self.relaxation_tolerance = 0.95
        self.top_k_num = 6
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())


    def evolve_discrete_energy(self, r, lr, singular_values, energy):
        r_tilde = torch.zeros(len(singular_values)).to(self.device)
        for i in range(len(singular_values)):
            r_tilde[i] = r[i]/(1+ (lr/2)*(singular_values[i]/energy**2))
        return r_tilde

    def find_relaxation_coefficient(self, old_grad, new_grad, r_tilde, E, loss, lr):
        D = (1/lr) * (new_grad - old_grad).norm()**2
        a = (r_tilde - E)**2
        b = 2*E*(r_tilde - E)
        c = loss - r_tilde**2 - self.relaxation_tolerance * D

        return min(0, (-b - torch.sqrt(b**2 - 4*a*c))/2*a) 



    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

       
        for group in self.param_groups:
            lr = group["lr"]
            if group["use_muon"]:
                for k, p in enumerate(group["params"]):
                   
                    if p.grad is None:
                        continue
                
                    grad = p.grad / p.grad.norm()
                    
                    U, S, Vh = torch.linalg.svd(grad, full_matrices=False) 
                    singular_count = min(self.top_k_num, len(S)) 

                    values, indices = torch.topk(S, singular_count)

                    E = torch.sqrt(loss)
                    if self.r[k] is None:
                        self.r[k] = torch.full_like(S, E)

                    self.r_tilde[k] = self.evolve_discrete_energy(self.r[k], lr, S, E)


                    update = sum(
                        self.r_tilde[k][i] * torch.outer(U[:, i], Vh[i, :])
                        for i in indices
                    )

                    old_p = p
                    p -= (lr/E) * update

                    for i in range(len(self.r[k])):
                        zeta = self.find_relaxation_coefficient(old_p, p, self.r_tilde[k][i], E, loss, lr)
                        self.r[k][i] = self.r_tilde[k][i] * zeta + (1 - zeta) * E

                    
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])


        return loss