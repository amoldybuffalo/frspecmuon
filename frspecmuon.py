import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from riemann_layers import RiemannianLinear, RiemannConv2d, RiemannConv2d_USVh, RiemannianLinear_USVh
import copy



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

#Old version using X = BA
# class FrSpecMuon(Optimizer):
#     def __init__(self, model, **kwargs): #no momentum by default

#         self.relaxation_tolerance = kwargs.get("relaxation_tolerance", 0.95) #recommended value 
#         lr = kwargs.get("lr", 0.02)
#         weight_decay = kwargs.get("weight_decay", 0.00)
#         self.q_multiplier = kwargs.get("q_multiplier", 2)
#         self.debug = kwargs.get("debug", False)

#         param_groups = []

#         flattened = [m for m in model.modules() if len(list(m.children())) == 0]

#         remaining_params = []
#         used_params = []

#         #decide whether the parameter gets a low rank muon update or a normal update based on whether it's a riemann layer or not (only riemannian linear layers for now)
#         for module in flattened:
#             if isinstance(module, RiemannianLinear) or isinstance(module, RiemannConv2d):
#                 param_groups.append(dict(params = [module.A, module.B], riemann = True, lr = lr))
#                 used_params.extend([id(module.A), id(module.B)])
#                 if hasattr(module, "bias") and module.bias != None:
#                     remaining_params.append(module.bias)
#                     used_params.append(id(module.bias))
                    
#             else:
#                 current_parameters = [param for param in module.parameters() if not (id(param) in used_params)]
#                 used_params.extend([id(param) for param in current_parameters])
#                 remaining_params.extend(current_parameters)

#         param_groups.append(dict(params=remaining_params, riemann=False))


#         for group in param_groups:
#             assert "riemann" in group
#             if group["riemann"]:
#                 group["lr"] = group.get("lr", lr)
#                 group["betas"] = group.get("betas", (0,0))
#                 group["weight_decay"] = group.get("weight_decay", weight_decay)
#                 assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "riemann"])
               
#             else:
#                 # defaults
#                 group["lr"] = group.get("lr", 3e-4)
#                 group["betas"] = group.get("betas", (0.9, 0.999))
#                 group["eps"] = group.get("eps", 1e-10)
#                 group["weight_decay"] = group.get("weight_decay", 0.01)
#                 assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "riemann"])

        
#         super().__init__(param_groups, dict())
        
#         riemann_param_count =  len([param_group for param_group in param_groups if param_group["riemann"]])
#         self.r = [None] * riemann_param_count
#         self.r_tilde = [None] * riemann_param_count
#         self.prev_U = [None] * riemann_param_count
#         self.prev_V = [None] * riemann_param_count
#         self.momentum_core = [None] * riemann_param_count
#         self.t = 0

#     def right_multiply_by_Rinv(self, G, R):
#         return torch.linalg.solve_triangular(R.T, G.T, upper=False).T

#     def evolve_discrete_energy(self, r, lr, singular_values, energy):
#         return r / (1+ (lr/2)*(singular_values/energy**2))
       

#     def find_relaxation_coefficient(self, old_grad, new_grad, r, r_tilde, E, lr):
#         D = (1/lr) * (new_grad - old_grad).norm()**2 
#         a = ((r_tilde - E)**2).clamp(min=1e-5)          
#         b = 2 * E * (r_tilde - E)                      
#         c = E**2 - r_tilde**2 - (r_tilde - r)**2 - self.relaxation_tolerance * D 

#         discriminant = (b**2 - 4*a*c).clamp(min=0)   
#         roots = (-b - discriminant.sqrt()) / (2 * a)   
#         return roots.clamp(min=0)


#     def tangent_core_svd(self, A, B, beta1, beta2, k, t):
#         """
#         Exact small-core SVD from Section 4.1.
#         """

#         A_grad_modified = A.grad
#         B_grad_modified = B.grad

#         U, Rb = torch.linalg.qr(B, mode="reduced")

#         r = U.shape[1]

     

#         #A should alread be orthogonal so we don't do a Qr to save time
#         V = A.T
#         S = Rb

#         # GV = self.right_multiply_by_Rinv(B.grad, Ra)
#         #Since Ra should just be the identity we don't bother with dividing by it
#         GV = B_grad_modified
#         GTU = self.right_multiply_by_Rinv((A_grad_modified).T, Rb)

#         K = U.T @ GV

#         Y = GV - U @ K 

#         Z = GTU - V@V.T @ GTU

#         QU, RU = torch.linalg.qr(Y, mode="reduced")        

#         QV, RV = torch.linalg.qr(Z, mode="reduced")


       

#         # the small core
#         C = torch.vstack([
#             torch.hstack([K, RV.T]),
#             torch.hstack([RU, torch.zeros(RU.shape[0], RV.shape[1],
#                 device=K.device,
#                 dtype=K.dtype
#             )])
#         ])
       

   
#         return C, S, U, V, QU, QV, r


#     @torch.no_grad()    
#     def step(self, closure=None):
#         self.t += 1
#         loss = None
        
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#                 #I choose 1 as my kappa because it seems to work and choosing zero has lead to nan loss in the past (no idea why, the loss should never be negative)
#                 E = torch.sqrt(loss+1).item()

#                 #some debugging steps from earlier
#                 if torch.isnan(loss):
#                     print("NAN LOSS")
#                     print(self.r)
#                     exit(-1)

#         k = -1 #I just wanna increment k first, okay?
#         for group in self.param_groups:
#             lr = group["lr"]
#             if group["riemann"]:
#                 k += 1
#                 params = group["params"]
#                 A, B = params     
#                 old_X = B @ A          
    
#                 rank = A.size()[0]
#                 device = A.device

#                 beta1, beta2 = group["betas"]
#                 C, S, U, V, QU, QV, rank = self.tangent_core_svd(A, B, beta1, beta2, k, self.t)  
           
#                 Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)

#                 q = int(rank * self.q_multiplier)
#                 #truncate the core SVD
#                 U_r = Uc[:, :q]
#                 S_r = Sc[:q]
#                 Vh_r = Vhc[:q, :]


#                 if self.r[k] is None:
#                     self.r[k] = torch.full((q,), E).to(device)

#                 self.r_tilde[k] = self.evolve_discrete_energy(self.r[k], lr, S_r, E)


#                 Hk = (1 / E) * U_r @ torch.diag(self.r_tilde[k]) @ Vh_r
                 

#                 S_pad = torch.zeros_like(Hk)
#                 S_pad[:rank, :rank] = S  # current weight core

#                 #I believe this is more or less the actual update step
#                 Ak = S_pad * (1 - lr * group["weight_decay"]) - (lr) * Hk #We apply weight decay here 
             
#                 # SVD back into the right basis
#                 Ua, Sa, Vha = torch.linalg.svd(Ak, full_matrices=False)
               
    
#                 # retract back
#                 U_new = torch.cat([U, QU], dim=1) @ Ua[:, :rank]
#                 S_new = torch.diag(Sa[:rank])
#                 V_new = torch.cat([V, QV], dim=1) @ Vha[:rank, :].T

#                 B.copy_(U_new @ S_new) # B gets the singular values 
#                 A.copy_(V_new.T) # A remains orthogonal


            
#                 X = B @ A 

#                 # relaxation step
#                 zeta = self.find_relaxation_coefficient(old_X, X, self.r[k], self.r_tilde[k], E, lr) 
#                 self.r[k] = self.r_tilde[k] * zeta + (1 - zeta) * E 

#             else:
#                 # otherwise just do a normal adam update as usual
#                 for p in group["params"]:
#                     if p.grad is None:
#                         continue
#                     state = self.state[p]
#                     if len(state) == 0:
#                         state["exp_avg"] = torch.zeros_like(p)
#                         state["exp_avg_sq"] = torch.zeros_like(p)
#                         state["step"] = 0
#                     state["step"] += 1
#                     update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
#                                          state["step"], group["betas"], group["eps"])
#                     p.mul_(1 - group["lr"] * group["weight_decay"])
#                     p.add_(update, alpha=-group["lr"])

#         if self.debug:
#             mean_r = sum([torch.mean(r_k) for r_k in self.r]) / len(self.r)
#             return loss, mean_r

#         else:
#             return loss


class FrSpecMuon(Optimizer):
    def __init__(self, model, **kwargs):

        self.relaxation_tolerance = kwargs.get("relaxation_tolerance", 0.95) #recommended value 
        lr = kwargs.get("lr", 0.02)
        weight_decay = kwargs.get("weight_decay", 0.00)
        self.q_multiplier = kwargs.get("q_multiplier", 2)
        self.debug = kwargs.get("debug", False)
        
        param_groups = []

        flattened = [m for m in model.modules() if len(list(m.children())) == 0]

        remaining_params = []
        used_params = []

        #decide whether the parameter gets a low rank muon update or a normal update based on whether it's a riemann layer or not (only riemannian linear layers for now)
        for module in flattened:
            if isinstance(module, RiemannianLinear_USVh) or isinstance(module, RiemannConv2d_USVh):
                param_groups.append(dict(params = [module.U, module.S, module.Vh], riemann = True, lr = lr))
                used_params.extend([id(module.U), id(module.S), id(module.Vh)])
                if hasattr(module, "bias") and module.bias != None:
                    remaining_params.append(module.bias)
                    used_params.append(id(module.bias))
                    
            else:
                current_parameters = [param for param in module.parameters() if not (id(param) in used_params)]
                used_params.extend([id(param) for param in current_parameters])
                remaining_params.extend(current_parameters)

        param_groups.append(dict(params=remaining_params, riemann=False))


        for group in param_groups:
            assert "riemann" in group
            if group["riemann"]:
                group["lr"] = group.get("lr", lr)
                group["betas"] = group.get("betas", (0,0))
                group["weight_decay"] = group.get("weight_decay", weight_decay)
                assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "riemann"])
               
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "riemann"])

        
        super().__init__(param_groups, dict())
        
        riemann_param_count =  len([param_group for param_group in param_groups if param_group["riemann"]])
        self.r = [None] * riemann_param_count
        self.r_tilde = [None] * riemann_param_count
        self.D = [None] * riemann_param_count

        self.t = 0
        

    def evolve_discrete_energy(self, r, lr, singular_values, energy):
        return r / (1+ (lr/2)*(singular_values/energy**2))
       

    def find_relaxation_coefficient(self, D, r, r_tilde, E, lr):
        D = (1/lr) * (new_grad - old_grad).norm()**2 
        a = ((r_tilde - E)**2).clamp(min=1e-5)          
        b = 2 * E * (r_tilde - E)                      
        c = E**2 - r_tilde**2 - (r_tilde - r)**2 - self.relaxation_tolerance * D 

        discriminant = (b**2 - 4*a*c).clamp(min=0)   
        roots = (-b - discriminant.sqrt()) / (2 * a)   
        return roots.clamp(min=0)

    def relax_r_values():
        pass

    def tangent_core_svd(self, U, S, Vh):
        """
        Exact small-core SVD from Section 4.1.
        """

        r = U.shape[1]     
        V = Vh.T

    
        GV = U.grad
        GTU = Vh.grad.T

        K = U.T @ GV

        Y = GV - U @ K 

        Z = GTU - V@V.T @ GTU

        QU, RU = torch.linalg.qr(Y, mode="reduced")        

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
        self.t += 1
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                #I choose 1 as my kappa because it seems to work and choosing zero has lead to nan loss in the past (no idea why, the loss should never be negative)
                E = torch.sqrt(loss+1).item()

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
                U, S, Vh = params     
                 = U @ S @ Vh          
    
                rank = Vh.size()[0]

                device = Vh.device

                beta1, beta2 = group["betas"]
                C, S, U, V, QU, QV, rank = self.tangent_core_svd(U, S, Vh)  
           
                Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)

                q = int(rank * self.q_multiplier)
                #truncate the core SVD
                U_r = Uc[:, :q]
                S_r = Sc[:q]
                Vh_r = Vhc[:q, :]


                if self.r[k] is None:
                    self.r[k] = torch.full((q,), E).to(device)

                self.r_tilde[k] = self.evolve_discrete_energy(self.r[k], lr, S_r, E)


                Hk = (1 / E) * U_r @ torch.diag(self.r_tilde[k]) @ Vh_r
                 

                S_pad = torch.zeros_like(Hk)
                S_pad[:rank, :rank] = S  # current weight core

                #I believe this is more or less the actual update step
                Ak = S_pad * (1 - lr * group["weight_decay"]) - (lr) * Hk #We apply weight decay here 
             
                # SVD back into the right basis
                Ua, Sa, Vha = torch.linalg.svd(Ak, full_matrices=False)
               
    
                # retract back
                U_new = torch.cat([U, QU], dim=1) @ Ua[:, :rank]
                S_new = torch.diag(Sa[:rank])
                V_new = torch.cat([V, QV], dim=1) @ Vha[:rank, :].T


                U.copy_(U_new) 
                S.copy_(S_new)
                Vh.copy_(V_new.T)


            
                X = U @ S @ Vh

                
                # relaxation step
                
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



        else:
            return loss



class FrSpecMuon_with_momentum(Optimizer):
    def __init__(self, model, **kwargs):

        self.relaxation_tolerance = kwargs.get("relaxation_tolerance", 0.95) #recommended value 
        lr = kwargs.get("lr", 0.02)
        betas = kwargs.get("betas", (0.9, 0.999))
        weight_decay = kwargs.get("weight_decay", 0.00)
        self.q_multiplier = kwargs.get("q_multiplier", 2)
        self.debug = kwargs.get("debug", False)

        param_groups = []

        flattened = [m for m in model.modules() if len(list(m.children())) == 0]

        remaining_params = []
        used_params = []

        #decide whether the parameter gets a low rank muon update or a normal update based on whether it's a riemann layer or not (only riemannian linear layers for now)
        for module in flattened:
            if isinstance(module, RiemannianLinear_USVh) or isinstance(module, RiemannConv2d_USVh):
                param_groups.append(dict(params = [module.U, module.S, module.Vh], riemann = True, lr = lr))
                used_params.extend([id(module.U), id(module.S), id(module.Vh)])
                if hasattr(module, "bias") and module.bias != None:
                    remaining_params.append(module.bias)
                    used_params.append(id(module.bias))
                    
            else:
                current_parameters = [param for param in module.parameters() if not (id(param) in used_params)]
                used_params.extend([id(param) for param in current_parameters])
                remaining_params.extend(current_parameters)

        param_groups.append(dict(params=remaining_params, riemann=False))


        for group in param_groups:
            assert "riemann" in group
            if group["riemann"]:
                group["lr"] = group.get("lr", lr)
                group["betas"] = group.get("betas", betas)
                group["weight_decay"] = group.get("weight_decay", weight_decay)
                assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "riemann"])
               
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "riemann"])

        
        super().__init__(param_groups, dict())
        
        riemann_param_count =  len([param_group for param_group in param_groups if param_group["riemann"]])
        self.r = [None] * riemann_param_count
        self.r_tilde = [None] * riemann_param_count
        self.t = 0


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


    def tangent_core_svd(self, U, S, Vh):
        """
        Exact small-core SVD from Section 4.1.
        """
        r = U.shape[1]     
        V = Vh.T
    
        GV = U.grad
        GTU = Vh.grad.T

        K = U.T @ GV
        Y = GV - U @ K 
        Z = GTU - V@V.T @ GTU

        QU, RU = torch.linalg.qr(Y, mode="reduced")        
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
        self.t += 1
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                #I choose 1 as my kappa because it seems to work and choosing zero has lead to nan loss in the past (no idea why, the loss should never be negative)
                E = torch.sqrt(loss+1).item()

        k = -1 
        for group in self.param_groups:
            lr = group["lr"]
            if group["riemann"]:

                k += 1
                params = group["params"]
                beta1, beta2 = group["betas"]
                


                U, S, Vh = params
                state = self.state[S]

                old_X = U @ S @ Vh          
    
                rank = Vh.size()[0]

                device = Vh.device

                beta1, beta2 = group["betas"]

                C, S, U, V, QU, QV, rank = self.tangent_core_svd(U, S, Vh)  
           
                Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)

                q = int(rank * self.q_multiplier)
                #truncate the core SVD
                U_r = Uc[:, :q]
                S_r = Sc[:q]
                Vh_r = Vhc[:q, :]

             

                

                Q = U_r @ Vh_r
                g = S_r / E
                a = 1/S_r


                if self.r[k] is None:
                    self.r[k] = torch.full((q,), E).to(device)

                if not "momentum_buffer" in state:
                    state["momentum_buffer"] = torch.zeros(q, device=device)
                

                Z = state["momentum_buffer"].clone()

                z = torch.tensor([torch.inner(Z, col) for col in Q.T], device=device)

    
         
                p_hat =  (1/torch.sqrt(a)) * z

            

                denom = S_r + (lr / 2) * g.square()
                num = beta1 * p_hat - lr * self.r[k] * g 

                s = num / denom 

               
                p = S_r * s

                d = 0.5 * g * s

                state["momentum_buffer"] = torch.sqrt(a) * p * Q.sum(dim=1) 

                tau = (torch.norm(Z)**2 - torch.norm(Z.product(dim=0))**2)/2 * lr

                self.r_tilde[k] = self.r[k] + d
                
                D_1 = (torch.norm(Z) - torch.norm(state["momentum_buffer"]))/ 2 * lr + (torch.norm(self.r[k])**2 - torch.norm(self.r_tilde[k]))
              
                
                H = U_r @ torch.diag(s) @ Vh_r


                S_pad = torch.zeros_like(H)
              
                S_pad[:rank, :rank] = S  # current weight core

                #I believe this is more or less the actual update step
                A = S_pad * (1 - lr * group["weight_decay"]) + H #We apply weight decay here 

        

                zeta = self.find_relaxation_coefficient(old_X, X, self.r[k], self.r_tilde[k], E, lr) 
                self.r[k] = self.r_tilde[k] + zeta * (E - self.r_tilde[k]) 

                # SVD back into the right basis
                Ua, Sa, Vha = torch.linalg.svd(A, full_matrices=False)
               
    
                # retract back
                U_new = torch.cat([U, QU], dim=1) @ Ua[:, :rank]
                S_new = torch.diag(Sa[:rank])
                V_new = torch.cat([V, QV], dim=1) @ Vha[:rank, :].T


                U.copy_(U_new) 
                S.copy_(S_new)
                Vh.copy_(V_new.T)


            
                X = U @ S @ Vh

                # relaxation step
                

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

        if self.debug:
            mean_r = sum([torch.mean(r_k) for r_k in self.r]) / len(self.r)
            return loss, mean_r

        else:
            return loss