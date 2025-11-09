import torch
import matplotlib.pyplot as plt
import numpy as np
import math

D = 1000
N = 768
M = 32
Nc = 256
temp = 0.001

u_e = torch.randn(D, M)
u = torch.cat([u_e, torch.zeros(D, N - M)], dim=1)

codebook = torch.nn.Parameter(torch.randn(Nc, M))
Pu = torch.nn.Parameter(torch.randn(M, N)/math.sqrt(N))
# Pc = torch.nn.Parameter(torch.randn(N, M))
optim = torch.optim.Adam([codebook, Pu], lr=1e-2)

N_iter = 1000
resample_interval = 100
resample_keep_thresh = 0.1/Nc
noise_scale = 0.01

for i in range(N_iter):
    optim.zero_grad()

    c_proj = codebook @ Pu
    u_proj = u # @ Pu.T

    similarity = torch.sum(u_proj ** 2, dim=1, keepdim=True) + torch.sum(c_proj ** 2, dim=1) - 2 * u_proj @ c_proj.T

    code_responsibilities = torch.nn.functional.softmax(-similarity/temp, dim=1)
    min_inds = torch.argmax(code_responsibilities, dim=1)
    loss = torch.sum(code_responsibilities * similarity)

    """
    min_inds = torch.argmin(similarity, dim=1)
    codes_proj = c_proj[min_inds]

    loss = torch.sum((codes_proj - u_proj) ** 2)
    """

    loss.backward()
    optim.step()

    if i % resample_interval == 0:
        e_mean = torch.mean(code_responsibilities, dim=0)
        most_used = torch.nonzero(e_mean/e_mean.sum() > resample_keep_thresh).squeeze()

        remainder = Nc % most_used.shape[0]
        n_repeats = Nc // most_used.shape[0]

        new_codes = torch.cat([codebook[most_used].repeat(n_repeats, 1),
                               codebook[most_used[:remainder]]], dim=0)
        new_codes = new_codes + noise_scale * torch.randn_like(new_codes)
        codebook = new_codes.detach()

    if i % 25 == 0:

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        print(f"Iter {i}, loss: {loss.item()}")

        """
        min_encodings = torch.zeros(
            min_inds.shape[0], Nc)
        min_encodings.scatter_(1, min_inds.unsqueeze(1), 1)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        """
        e_mean = torch.mean(code_responsibilities, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        print(f"perplexity: {perplexity}")

        ax.set_title("In Manifold Dimensions")
        ax.plot(c_proj[:, 0].detach().numpy(), c_proj[:, 1].detach().numpy(), "rx", label="Codebook")
        ax.scatter(u_proj[:, 0].detach().numpy(), u_proj[:, 1].detach().numpy(), c=min_inds.detach().numpy(), label="Data")
        ax.legend()

        plt.show()

print(Pu)









