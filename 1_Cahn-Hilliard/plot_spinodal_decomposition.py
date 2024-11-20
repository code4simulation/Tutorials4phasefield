import pyvista as pv
import numpy as np

fps = 16

def grad(arr):
    return (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0))/2

def mul(arr):
    res = 1
    for x in arr: res *= x
    return res

pvd = pv.get_reader("simulation_results.pvd")
sargs = dict(height=0.40, vertical=True, position_x= 0.8, position_y=0.4, n_labels=5, title='psi', fmt='%.2f')
mesh_kwargs = {'scalars':'psi',
              'scalar_bar_args':sargs,
              'clim':[0,1],
              'cmap':'bwr'}

p = pv.Plotter()
p.open_movie("res.mp4", framerate=fps)

nframe = len(pvd.time_values)

free_energy = []
conc = []
for frame in range(nframe):
    data = pvd.set_active_time_point(frame)
    grid = pvd.read()[0]
    arr = grid.point_data.get_array(mesh_kwargs['scalars'])
    
    grad_x = grad(arr)
    grad_y = grad(arr.T).T
    grad_mag_sq = grad_x**2 + grad_y**2
    free_energy.append(np.sum( arr**2 * (1-arr)**2 + 0.5/2 * grad_mag_sq))
    conc.append(np.sum(arr)/(mul(grid.dimensions)))

    p.add_mesh(pvd.read()[0], show_scalar_bar=True, **mesh_kwargs)
    p.camera_position = 'xy'
    p.write_frame()

p.close()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fs = 12
fig,ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
x = np.linspace(0, 1, len(free_energy))
ax[0].plot(x[0], free_energy[0], c='C3')

ax[1].plot(x[0], conc[0], c='C3')

def update(frame):
    ax[0].clear()
    ax[1].clear()

    ticks = np.arange(0.0, 1.1, 0.25)
    for i in range(2):
        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels([f"{x:.1f}" for x in ticks], fontsize=fs)
        ax[i].set_xlim([0, 1])
        ax[i].set_xlabel("Time", fontsize=fs)

    for i in range(2):
        if i==0:
            ticks = np.arange(0.0, 1001, 200)
            ax[i].set_yticks(ticks)
            ax[i].set_yticklabels([f"{x:.0f}" for x in ticks], fontsize=fs)
            ax[i].set_ylim([0, 1000])
            ax[i].set_ylabel("Free energy", fontsize=fs)
        elif i==1:
            ticks = np.arange(0.48, 0.521, 0.02)
            ax[i].set_yticks(ticks)
            ax[i].set_yticklabels([f"{x:.2f}" for x in ticks], fontsize=fs)
            ax[i].set_ylim([0.48, 0.52])
            ax[i].set_ylabel("Average psi per grid point", fontsize=fs)

    ax[0].plot(x[:frame+1], free_energy[:frame+1], c='C3')
    ax[1].plot(x[:frame+1], conc[:frame+1], c='C3')
    ax[0].plot(x[frame], free_energy[frame], marker='o', c='C3')
    ax[1].plot(x[frame], conc[frame], marker='o', c='C3')
    return ax[0], ax[1]

ani = FuncAnimation(fig, update, frames=len(x), blit=False)

FFwriter = matplotlib.animation.FFMpegWriter(fps=fps)
ani.save('animation.mp4', writer=FFwriter)

plt.show()
