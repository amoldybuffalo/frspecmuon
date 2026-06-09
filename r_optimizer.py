import torch
from torch.optim import Optimizer
from riemann_layers import RiemannianLinear

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


def assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} became non-finite")

class FrSpecMuon(Optimizer):
    def __init__(self, model, lr=1e-2, beta = 0.2):


        param_groups = []

        flattened = [m for m in model.modules() if len(list(m.children())) == 0]

        remaining_params = []
        used_params = []
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
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "riemann"])
               
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
        self.momentum = [None] * riemann_param_count

        self.beta = beta
        self.relaxation_tolerance = 0.95
        

    def right_multiply_by_Rinv(self, G, R):
        return torch.linalg.solve_triangular(R.T, G.T, upper=False).T

    def evolve_discrete_energy(self, r, lr, singular_values, energy):
        return r / (1+ (lr/2)*(singular_values/energy**2))
       

    def find_relaxation_coefficient(self, old_grad, new_grad, r, r_tilde, E, lr):
        D = (1/lr) * (new_grad - old_grad).norm()**2  # scalar
        a = ((r_tilde - E)**2).clamp(min=1e-5)          # [rank]
        b = 2 * E * (r_tilde - E)                      # [rank]
        c = E**2 - r_tilde**2 - (r_tilde - r)**2 - self.relaxation_tolerance * D  # [rank]

        discriminant = (b**2 - 4*a*c).clamp(min=0)    # guard sqrt domain
        roots = (-b - discriminant.sqrt()) / (2 * a)   # [rank]
        return roots.clamp(min=0)

    def tangent_core_svd(self, A, B):
        """
        Exact small-core SVD from Section 4.1.
        """

        U, Rb = torch.linalg.qr(B, mode="reduced")
        # V, Ra = torch.linalg.qr(A.T, mode="reduced")
        # S = Rb * Ra
        V = A.T
        S = Rb

        # GV = self.right_multiply_by_Rinv(B.grad, Ra)
        GV = B.grad
        GTU = self.right_multiply_by_Rinv((A.grad).T, Rb)


        K = U.T @ GV

        Y = GV - U @ K 

        QU, RU = torch.linalg.qr(Y, mode="reduced")

        Z = GTU - V@V.T @ GTU 

        QV, RV = torch.linalg.qr(Z, mode="reduced")

        r = U.shape[1]

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
                E = torch.sqrt(loss + 1).item()
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

                C, S, U, V, QU, QV, rank = self.tangent_core_svd(A, B)  
                

                Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)

                U_r = Uc[:, :rank]
                S_r = Sc[:rank]
                Vh_r = Vhc[:rank, :]


                if self.r[k] is None:
                    self.r[k] = torch.full((rank,), E).to(device)

                self.r_tilde[k] = self.evolve_discrete_energy(self.r[k], lr, S_r, E)
                step_sv = -(lr / E) * self.r_tilde[k]
            

                #Hk = -lr * C

            

                Hk = U_r @ torch.diag(step_sv) @ Vh_r  # [2r x 2r]

                if self.momentum[k] is None:
                    self.momentum[k] = torch.zeros_like(C)

                self.momentum[k] = self.beta * self.momentum[k] + Hk

                S_pad = torch.zeros_like(Hk)
                S_pad[:rank, :rank] = S  # current weight core

                Ak = S_pad + self.momentum[k]

                Ua, Sa, Vha = torch.linalg.svd(Ak, full_matrices=False)

                U_new = torch.cat([U, QU], dim=1) @ Ua[:, :rank]
                S_new = torch.diag(Sa[:rank])
                V_new = torch.cat([V, QV], dim=1) @ Vha[:rank, :].T

                B.copy_(U_new @ S_new)
                A.copy_(V_new.T)
                X = B @ A

                zeta = self.find_relaxation_coefficient(old_X, X, self.r_tilde[k], self.r[k], E, lr)
                self.r[k] = self.r_tilde[k] * zeta + (1 - zeta) * E 

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