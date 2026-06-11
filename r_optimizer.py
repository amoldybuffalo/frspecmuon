import torch
from torch.optim import Optimizer
from riemann_layers import RiemannianLinear

#stolen from the muon repo
def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

#debugging function from earlier
def assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} became non-finite")

class FrSpecMuon(Optimizer):
    def __init__(self, model, lr=1e-3, beta = 0): #no momentum by default


        param_groups = []

        flattened = [m for m in model.modules() if len(list(m.children())) == 0]

        remaining_params = []
        used_params = []

        #decide whether the parameter gets a low rank muon update or a normal update based on whether it's a riemann layer or not (only riemannian linear layers for now)
        for module in flattened:
            if isinstance(module, RiemannianLinear):
                param_groups.append(dict(params = [module.A, module.B], riemann = True, lr = lr))
                used_params.extend([id(module.A), id(module.B)])
            else:
                current_parameters = [param for param in module.parameters() if not (id(param) in used_params)]
                used_params.extend([id(param) for param in current_parameters])
                remaining_params.extend(current_parameters)

        param_groups.append(dict(params=remaining_params, riemann=False, lr=lr, betas=(0.9, 0.95), weight_decay=0.01))


        for group in param_groups:
            assert "riemann" in group
            if group["riemann"]:
                group["lr"] = group.get("lr", 0.02)
                group["betas"] = (beta, 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "riemann"])
               
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "riemann"])

        
        super().__init__(param_groups, dict())
        
        riemann_param_count =  len([param_group for param_group in param_groups if param_group["riemann"]])
        self.r = [None] * riemann_param_count
        self.r_tilde = [None] * riemann_param_count
        # self.momentum = [None] * riemann_param_count
        # self.velocity = [None] * riemann_param_count
        self.momentum = [None] * riemann_param_count
        self.velocity = [None] * riemann_param_count
        self.relaxation_tolerance = 0.95 #recommended value 
        

    def right_multiply_by_Rinv(self, G, R):
        return torch.linalg.solve_triangular(R.T, G.T, upper=False).T

    def evolve_discrete_energy(self, r, lr, singular_values, energy):
        return r / (1+ (lr/2)*(singular_values/energy**2))
       

    def find_relaxation_coefficient(self, old_grad, new_grad, r, r_tilde, E, lr):
        D = (1/lr) * (new_grad - old_grad).norm()**2 
        a = ((r_tilde - E)**2).clamp(min=1e-5)          
        b = 2 * E * (r_tilde - E)                      
        c = E**2 - r_tilde**2 - (r_tilde - r)**2 - self.relaxation_tolerance * D 

        discriminant = (b**2 - 4*a*c).clamp(min=0)   
        roots = (-b - discriminant.sqrt()) / (2 * a)   
        return roots.clamp(min=0)

    def tangent_core_svd(self, A, B, beta, beta2, k):
        """
        Exact small-core SVD from Section 4.1.
        """

        #Apply momentum
        if self.momentum[k] is None:
            self.momentum[k] = [torch.zeros_like(A.grad), torch.zeros_like(B.grad)]
            self.velocity[k] = [torch.zeros_like(A.grad), torch.zeros_like(B.grad)]
    
        self.momentum[k][0] = (
                    beta * self.momentum[k][0] 
                    + (1 - beta) * A.grad
                )

        self.momentum[k][1] = (
                    beta * self.momentum[k][1] 
                    + (1 - beta) * B.grad
                )
        
        self.velocity[k][0] = (
                     beta2 * self.velocity[k][0]
                     + (1 - beta2) * A.grad.square()
                 )
        self.velocity[k][1] = (
                     beta2 * self.velocity[k][1]
                     + (1 - beta2) * B.grad.square()
                 )

        A_grad_modified = (self.momentum[k][0] / (torch.sqrt(self.velocity[k][0]) + 1e-8))
        B_grad_modified = (self.momentum[k][1] / (torch.sqrt(self.velocity[k][1]) + 1e-8))

        U, Rb = torch.linalg.qr(B, mode="reduced")

        r = U.shape[1]

        #A should alread be orthogonal so we don't do a Qr to save time
        V = A.T
        S = Rb

        # GV = self.right_multiply_by_Rinv(B.grad, Ra)
        #Since Ra should just be the identity we don't bother with dividing by it
        GV = B_grad_modified
        GTU = self.right_multiply_by_Rinv((A_grad_modified).T, Rb)


        K = U.T @ GV

        Y = GV - U @ K 

        QU, RU = torch.linalg.qr(Y, mode="reduced")

        Z = GTU - V@V.T @ GTU 

        QV, RV = torch.linalg.qr(Z, mode="reduced")

        # the small core
        C = torch.vstack([
            torch.hstack([K, RV.T]),
            torch.hstack([RU, torch.zeros(RU.shape[0], RV.shape[1],
                device=K.device,
                dtype=K.dtype
            )])
        ])
       

   
        return C, S, U, V, QU, QV, r


    @torch.no_grad()    
    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                #I choose 1 as my kappa because it seems to work and choosing zero has lead to nan loss in the past (no idea why, the loss should never be negative)
                E = torch.sqrt(loss + 1).item()

                #some debugging steps from earlier
                if torch.isnan(loss):
                    print("NAN LOSS")
                    print(self.r)
                    exit(-1)

        k = -1 #I just wanna increment k first, okay?
        for group in self.param_groups:
            lr = group["lr"]
            if group["riemann"]:
                k += 1
                params = group["params"]
                A, B = params     
                old_X = B @ A          
    
                rank = A.size()[0]
                device = A.device

                beta, beta2 = group["betas"]
                C, S, U, V, QU, QV, rank = self.tangent_core_svd(A, B, beta, beta2, k)  
                
                    
                Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)

                #truncate the core SVD
                U_r = Uc[:, :rank]
                S_r = Sc[:rank]
                Vh_r = Vhc[:rank, :]


                if self.r[k] is None:
                    self.r[k] = torch.full((rank,), E).to(device)

                self.r_tilde[k] = self.evolve_discrete_energy(self.r[k], lr, S_r, E)

                # Hk = U_r @ torch.diag(self.r[k]) @ Vh_r
                Hk = U_r @ Vh_r

                # if self.momentum[k] is None:
                #     self.momentum[k] = torch.zeros_like(C)
                #     self.velocity[k] = torch.zeros_like(C)

                # beta, beta2 = group["betas"]

                # self.momentum[k] = (
                #     beta * self.momentum[k]
                #     + (1 - beta) * Hk
                # )

                # self.velocity[k] = (
                #     beta2 * self.velocity[k]
                #     + (1 - beta2) * (Hk * Hk)
                # )

                

                S_pad = torch.zeros_like(Hk)
                S_pad[:rank, :rank] = S  # current weight core

                #I believe this is more or less the actual update step
                # Ak = S_pad - (lr/E) * (self.momentum[k] / (torch.sqrt(self.velocity[k]) + 1e-8))
                Ak = S_pad - (lr/E) * Hk

                # SVD back into the right basis
                Ua, Sa, Vha = torch.linalg.svd(Ak, full_matrices=False)

                # retract back
                U_new = torch.cat([U, QU], dim=1) @ Ua[:, :rank]
                S_new = torch.diag(Sa[:rank])
                V_new = torch.cat([V, QV], dim=1) @ Vha[:rank, :].T

                B.copy_(U_new @ S_new) # B gets the singular values 
                A.copy_(V_new.T) # A remains orthogonal
                X = B @ A 

                # relaxation step
                zeta = self.find_relaxation_coefficient(old_X, X, self.r_tilde[k], self.r[k], E, lr) 
                self.r[k] = self.r_tilde[k] * zeta + (1 - zeta) * E 

            else:
                # otherwise just do a normal adam update as usual
                for p in group["params"]:
                    if p.grad is None:
                        continue
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






# Ignore this for now. Just an unfinished template to implement specmuon 
class SpecMuon(Optimizer):
    def __init__(self, param_groups, lr=1e-2, beta = 0.2):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
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
        
        riemann_param_count =  len([param_group for param_group in param_groups if param_group["use_muon"]])
        self.r = [None] * riemann_param_count
        self.r_tilde = [None] * riemann_param_count
        self.momentum = [None] * riemann_param_count

        self.beta = beta
        self.relaxation_tolerance = 0.95
        


    
    def evolve_discrete_energy(self, r, lr, singular_values, energy):
        return r / (1+ (lr/2)*(singular_values/energy**2))
       

    def find_relaxation_coefficient(self, old_grad, new_grad, r, r_tilde, E, lr):
        D = (1/lr) * (new_grad - old_grad).norm()**2  
        a = ((r_tilde - E)**2).clamp(min=1e-5)          
        b = 2 * E * (r_tilde - E)                      
        c = E**2 - r_tilde**2 - (r_tilde - r)**2 - self.relaxation_tolerance * D  

        discriminant = (b**2 - 4*a*c).clamp(min=0)    # guard sqrt domain
        roots = (-b - discriminant.sqrt()) / (2 * a)   
        return roots.clamp(min=0)



    @torch.no_grad()    
    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                E = torch.sqrt(loss + 1).item()
                if torch.isnan(loss):
                    print("NAN LOSS")
                    print(self.r)
                    exit(-1)

        k = -1 #I just wanna increment k first, okay?
        for group in self.param_groups:
            lr = group["lr"]
            if group["use_muon"]:
                pass

            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
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