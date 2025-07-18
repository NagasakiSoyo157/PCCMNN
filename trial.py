import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pccmnn as pc

# ------------------ 你已有的部分 ------------------
csf_dict = pc.load_csf_data()

keys_to_delete = []
for key in csf_dict:
    sample = csf_dict[key]
    sample = sample[~np.isnan(sample).any(axis=1)]   # 删缺失
    csf_dict[key] = sample
    if sample.shape[0] < 2 :
        keys_to_delete.append(key)
for key in keys_to_delete:
    del csf_dict[key]
# ---------------------------------------------------

# ★ 把 numpy 数组转成 patient_data ★
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()            # 年龄
    y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

import torch.nn as nn

class PopulationODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(21) * 1e-3)

    def f(self, y):
        A, T, N, C = y
        w = self.w
        dA = w[0] + w[1]*A + w[2]*A**2
        dT = (w[3] + w[4]*T + w[5]*T**2 +
              w[6]*A + w[7]*A**2 + w[8]*A*T)
        dN = (w[9] + w[10]*N + w[11]*N**2 +
              w[12]*T + w[13]*T**2 + w[14]*T*N)
        dC = (w[15] + w[16]*C + w[17]*C**2 +
              w[18]*N + w[19]*N**2 + w[20]*N*C)
        return torch.stack([dA, dT, dN, dC])

    def forward(self, s_grid, y0):
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            ys.append(ys[-1] + h * self.f(ys[-1]))
        return torch.stack(ys)          # (len_s, 4)
    
from torch import optim

def fit_population(patient_data, n_outer=15, lr_w=5e-3, lr_ab=5e-3):
    max_alpha = 4.0  # 最大的 α 值
    model = PopulationODE()

    # 每个受试者自己的 θ=(θ_α, β)
    ab = {pid: {'theta': torch.randn(2, requires_grad=True)}
          for pid in patient_data}

    for it in range(n_outer):
        # ---- (A) 固定 αβ，优化 w ----------------------------------
        opt_w = optim.LBFGS(model.parameters(), lr=lr_w, max_iter=20, tolerance_grad=0, tolerance_change=0)

        def closure_w():
            opt_w.zero_grad()
            loss = 0.
            for pid, dat in patient_data.items():
                alpha = max_alpha * torch.sigmoid(ab[pid]['theta'][0]) + 1e-4
                beta  = ab[pid]['theta'][1]
                s = alpha * dat['t'] + beta
                y_pred = model(s, dat['y0'])           # ★ 用各自 y0 ★
                loss += torch.mean((y_pred - dat['y'])**2)
            loss.backward()
            return loss
        opt_w.step(closure_w)

        # ---- 估计残差方差 σk (可选，不想加权就跳过这一步) ------------
        with torch.no_grad():
            res2 = torch.zeros(4)
            cnt  = torch.zeros(4)
            for pid, dat in patient_data.items():
                alpha = max_alpha * torch.sigmoid(ab[pid]['theta'][0]) + 1e-4
                beta  = ab[pid]['theta'][1]
                s = alpha * dat['t'] + beta
                y_pred = model(s, dat['y0'])
                res2 += torch.sum((y_pred - dat['y'])**2, dim=0)
                cnt  += torch.tensor(dat['y'].shape[0])
            sigma = res2 / cnt.clip(min=1.0)

        # ---- (B) 固定 w，分别更新每个 αβ ----------------------------
        for pid, dat in patient_data.items():
            opt_ab = optim.LBFGS([ab[pid]['theta']], lr=lr_ab, max_iter=15)

            def closure_ab():
                opt_ab.zero_grad()
                alpha = max_alpha * torch.sigmoid(ab[pid]['theta'][0]) + 1e-4
                beta  = ab[pid]['theta'][1]
                s = alpha * dat['t'] + beta
                y_pred = model(s, dat['y0'])
                loss = torch.mean(((y_pred - dat['y'])**2) / sigma)  # σ 加权
                loss.backward()
                return loss
            opt_ab.step(closure_ab)

        print(f"iter {it:02d} | loss = {closure_w().item():.4f}")

    # --------- 结果汇总 -------------------------------------------
    pop_w = model.w.detach().clone()
    alpha_beta = {pid: (float(max_alpha * torch.sigmoid(v['theta'][0]) + 1e-4),
                        float(v['theta'][1]))
                  for pid, v in ab.items()}
    return pop_w, alpha_beta

pop_param, ab_dict = fit_population(patient_data,
                                    n_outer=15,)

print("learned w shape:", pop_param.shape)   # (21,)
print("first 3 α,β:", list(ab_dict.items())[:3])