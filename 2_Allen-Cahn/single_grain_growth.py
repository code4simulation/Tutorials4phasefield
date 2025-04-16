import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
N = 128                      # The number of grid points
dx = 1.0                     # Size between grid points
dt = 0.01                    # Time step
nsteps = 2500                # The number of steps
grad_coeff = 0.1             # Gradient coefficient (ε²)
mobility = 5.0               # Mobility
radius = 5                   # Seed radius
interval = 100               # Plot update interval

# PBC options (Periodic Boundary Conditions)
pbc_x = True
pbc_y = True

# Initialization
phi = np.zeros((N, N))
x = np.arange(N)
y = np.arange(N)
X, Y = np.meshgrid(x, y)
center = N // 2
mask = (X - center)**2 + (Y - center)**2 <= radius**2
phi[mask] = 1.0

def laplacian(phi):
    # X direction calculation using periodic boundary conditions if enabled
    if pbc_x:
        phi_xp = np.roll(phi, -1, axis=0)
        phi_xm = np.roll(phi, 1, axis=0)
    else:
        phi_xp = np.zeros_like(phi)
        phi_xm = np.zeros_like(phi)
        phi_xp[:-1, :] = phi[1:, :]
        phi_xm[1:, :] = phi[:-1, :]
    # Y direction calculation using periodic boundary conditions if enabled
    if pbc_y:
        phi_yp = np.roll(phi, -1, axis=1)
        phi_ym = np.roll(phi, 1, axis=1)
    else:
        phi_yp = np.zeros_like(phi)
        phi_ym = np.zeros_like(phi)
        phi_yp[:, :-1] = phi[:, 1:]
        phi_ym[:, 1:] = phi[:, :-1]
    return (phi_xp + phi_xm + phi_yp + phi_ym - 4 * phi) / (dx * dx)

# Plotting settings (pcolormesh, colorbar range [-1, 1])
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
c = ax.pcolormesh(phi, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(c, ax=ax)

for step in range(nsteps+1):
    lap = laplacian(phi)
    # Allen-Cahn Model: ∂φ/∂t = mobility*(grad_coeff*Δφ - (φ³ - φ))
    phi += dt * mobility * (grad_coeff * lap - (phi**3 - phi))
    
    # Update the plot every {interval} steps
    if step % interval == 0:
        ax.clear()
        c = ax.pcolormesh(phi, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f"Time: {step} steps")
        plt.pause(0.01)

plt.ioff()
plt.show()
