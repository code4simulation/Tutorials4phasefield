# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Simulation parameters
N = 128                      # The number of grid points
num_grains = 5               # The number of grains
dx = 1.0                     # Size between grid points
dt = 0.01                    # Time step
nsteps = 2500                # The number of steps
grad_coeff = 0.1             # Gradient coefficient (ε²)
p_coeff = 2.0                # Penalty term coefficient
mobility = 5.0               # Mobility
radius = 5                   # Seed radius
interval = 100               # Update interval for plotting

# PBC options (Periodic Boundary Conditions)
pbc_x = True
pbc_y = True

# phi (N x N x num_grains)
phi = np.zeros((N, N, num_grains))
X, Y = np.meshgrid(np.arange(N), np.arange(N))

# Initialization: set the initial seed for each grain
@njit(nogil=True, parallel=True)
def laplacian(phi, pbc_x, pbc_y):
    Nx, Ny, Ng = phi.shape
    lap = np.zeros_like(phi)
    for i in prange(Ng):  # parallelization
        for x in prange(Nx):
            for y in prange(Ny):
                # X direction calculation using periodic boundary conditions if enabled
                if pbc_x:
                    xp = (x + 1) % Nx
                    xm = (x - 1) % Nx
                else:
                    xp = x + 1 if x < Nx-1 else x
                    xm = x - 1 if x > 0 else x
                # Y direction calculation using periodic boundary conditions if enabled
                if pbc_y:
                    yp = (y + 1) % Ny
                    ym = (y - 1) % Ny
                else:
                    yp = y + 1 if y < Ny-1 else y
                    ym = y - 1 if y > 0 else y
                lap[x,y,i] = (phi[xp,y,i] + phi[xm,y,i] 
                             + phi[x,yp,i] + phi[x,ym,i] 
                             - 4*phi[x,y,i]) / (dx*dx)
    return lap

@njit(nogil=True, parallel=True)
def update_phi(phi, lap, total_phi_sq, grad_coeff, mobility, dt, p_coeff):
    Ng = phi.shape[2]
    for i in prange(Ng):  # parallelization
        phi_slice = phi[:,:,i]
        penalty = 2 * p_coeff * phi_slice * (total_phi_sq - phi_slice**2)
        phi[:,:,i] += dt * mobility * (
            grad_coeff * lap[:,:,i] 
            - (phi_slice**3 - phi_slice) 
            - penalty
        )
    return phi

# Plotting settings (pcolormesh, colorbar range [-1, 1])
# (2x3 subplots: 5 grains plots + combined view)
plt.ion()
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
cmap_combined = plt.get_cmap('tab10', num_grains)  # Grain coloring

for i in range(num_grains):
    ax = axs.flatten()[i]
    c = ax.pcolormesh(phi[:, :, i], cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(f'Grain {i+1}')
    plt.colorbar(c, ax=ax)

# Initializing combined view
ax_combined = axs[1, 2]
cmap = plt.get_cmap('tab10', num_grains)
grain_colors = [np.array(cmap(i)[:3]) for i in range(num_grains)]
combined_rgb = np.ones((N, N, 3))  # Initialize with white background (RGB 3-channel, float)
for i in range(num_grains):
    combined_rgb += np.expand_dims(phi[:, :, i], axis=2) * grain_colors[i]
im_combined = ax_combined.imshow(combined_rgb, 
                                 interpolation='nearest', 
                                 extent=[0, N, 0, N])
ax_combined.set_title('Combined Grains')

for step in range(nsteps):
    lap = laplacian(phi, pbc_x, pbc_y)
    total_phi_sq = np.sum(phi**2, axis=2)
    phi = update_phi(phi, lap, total_phi_sq, grad_coeff, mobility, dt, p_coeff)

    # Update every {interval}
    if step % interval == 0:
        # Update each grain plot
        for i in range(num_grains):
            ax = axs.flatten()[i]
            ax.clear()
            c = ax.pcolormesh(phi[:, :, i], cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f'Grain {i+1} (Step {step})')

        # Update combined view 
        # (Combine each grain's φ field weighted by its respective color)
        combined_rgb = np.ones((N, N, 3))
        for i in range(num_grains):
            combined_rgb -= np.expand_dims(phi[:, :, i], axis=2) * grain_colors[i]
        combined_rgb = np.clip(combined_rgb, 0, 1)
        
        im_combined.set_data(combined_rgb)
        ax_combined.set_title(f'Combined Grains (Step {step})')
        plt.pause(0.01)

plt.ioff()
plt.show()
