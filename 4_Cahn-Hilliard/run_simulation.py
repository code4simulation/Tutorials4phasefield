import os
import sys
import glob
import platform
import subprocess
import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
import argparse
import uuid

# --- Constants ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(CURRENT_DIR, "cpp_src")

# Clean old locked DLLs from previous aborted runs
for old_lib in glob.glob(os.path.join(CPP_DIR, "cahn_hilliard_*.dll")) + glob.glob(os.path.join(CPP_DIR, "libcahn_hilliard_*.so")):
    try:
        os.remove(old_lib)
    except OSError:
        pass # Ignore locked files

unique_id = uuid.uuid4().hex[:8]
LIB_NAME = f"cahn_hilliard_{unique_id}.dll" if platform.system() == "Windows" else f"libcahn_hilliard_{unique_id}.so"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Compilation ---
def compile_cpp_lib():
    cpp_source = os.path.join(CPP_DIR, "cahn_hilliard_simulation.cpp")
    lib_output = os.path.join(CPP_DIR, LIB_NAME)
    
    print(f"[*] Compiling Cahn-Hilliard Library: {cpp_source}...")
    
    # Flags similar to run_simulation.py but simplified since we don't have complex deps
    flags = ["-O3", "-march=native", "-fopenmp", "-ffast-math"]
    
    if platform.system() == "Windows":
        cmd = ["g++", "-shared", "-o", lib_output, cpp_source] + flags + ["-static"]
    else:
        cmd = ["g++", "-shared", "-fPIC", "-o", lib_output, cpp_source] + flags
        
    try:
        subprocess.check_call(cmd)
        print("[+] Compilation successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Compilation failed: {e}")
        return False

# --- Wrapper Class ---
class CHWrapper:
    def __init__(self):
        self.lib_path = os.path.join(CPP_DIR, LIB_NAME)
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Library not found at {self.lib_path}. Compile first.")
            
        try:
            if platform.system() == "Windows":
                self.lib = ctypes.CDLL(self.lib_path, winmode=0)
            else:
                self.lib = ctypes.CDLL(self.lib_path)
        except OSError as e:
            print(f"[ERROR] Failed to load lib: {e}")
            raise

    def run(self, c_grid, config, M, kappa, W):
        class SimConfigCH(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_double), ("dt", ctypes.c_double),
                ("M", ctypes.c_double),
                ("kappa", ctypes.c_double),
                ("W", ctypes.c_double),
                ("time_total", ctypes.c_double),
                ("c_ref", ctypes.c_double),
                ("geom_type", ctypes.c_int),
                ("R_in", ctypes.c_double),
                ("R_out", ctypes.c_double),
                ("gamma_in", ctypes.c_double),
                ("gamma_out", ctypes.c_double),
                ("Nx", ctypes.c_int), ("Ny", ctypes.c_int), ("Nz", ctypes.c_int),
                ("output_interval", ctypes.c_int)
            ]
        
        sim = config['simulation']
        geom = sim.get('geometry', {})
        geom_type_str = geom.get('type', 'cube')
        geom_type_map = {'cube': 0, 'solid_cylinder': 1, 'hollow_cylinder': 2}
        
        cfg = SimConfigCH(
            float(sim['dx_m']), float(sim['dt_s']),
            float(M), float(kappa), float(W),
            float(sim['total_time_s']),
            float(config['initialization']['c_ref_atoms_cm3']),
            int(geom_type_map.get(geom_type_str, 0)),
            float(geom.get('R_inner_m', 0.0)),
            float(geom.get('R_outer_m', 1e9)), # large default for cube
            float(geom.get('gamma_in_J_m2', 0.0)),
            float(geom.get('gamma_out_J_m2', 0.0)),
            int(sim['Nx']), int(sim['Ny']), int(sim.get('Nz', 1)),
            int(sim['output_interval_steps'])
        )
        
        c_flat = c_grid.flatten().astype(np.float64)
        c_ptr = c_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        dump_dir = os.path.join(CURRENT_DIR, sim['dump_dir'])
        ensure_dir(dump_dir)
        dump_dir_bytes = dump_dir.encode('utf-8')
        
        # Setup argtypes
        self.lib.run_ch_simulation.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(SimConfigCH),
            ctypes.c_char_p,
            ctypes.c_int
        ]
        
        print(f"[*] Starting 3D Simulation in C++...")
        start_t = time.time()
        
        self.lib.run_ch_simulation(c_ptr, ctypes.byref(cfg), dump_dir_bytes, 0)
        
        elapsed = time.time() - start_t
        print(f"[+] Simulation completed in {elapsed:.2f} seconds.")
        
        return c_flat.reshape(sim['Nx'], sim['Ny'], sim.get('Nz', 1))

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_ch.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 1. Compile
    if not compile_cpp_lib():
        sys.exit(1)
        
    # 2. Config & Parameter Mode
    config = load_config(args.config)
    sim = config['simulation']
    init = config['initialization']
    
    if sim.get('parameter_mode', 'direct') == 'physical':
        phys = sim['physical_params']
        gamma_int = float(phys['interfacial_energy_J_m2'])
        Lc = float(phys['interfacial_thickness_m'])
        D = float(phys['diffusivity_m2_s'])
        
        # Physical to CH parameter conversion
        # f(c) = W c^2 (1-c)^2 assumption
        kappa = (3.0 * gamma_int * Lc) / np.sqrt(2.0)
        W = (3.0 * gamma_int) / (np.sqrt(2.0) * Lc)
        M = D / (2.0 * W)
        print(f"[*] Computed Physical Parameters: M={M:.3e}, kappa={kappa:.3e}, W={W:.3e}")
    else:
        dirs = sim['direct_params']
        M, kappa, W = float(dirs['M']), float(dirs['kappa']), float(dirs['W'])
        print(f"[*] Parsed Direct Parameters: M={M:.3e}, kappa={kappa:.3e}, W={W:.3e}")
    
    # Check Stability
    dx = float(sim['dx_m'])
    max_dt = (dx**4) / (8.0 * M * kappa)
    if float(sim['dt_s']) > max_dt:
        print(f"[WARNING] dt_s ({sim['dt_s']}) > Stability Limit ({max_dt:.3e}). Exploding risk!")
    
    # 3. Setup 3D Geometry and Initialize Field
    Nx, Ny, Nz = int(sim['Nx']), int(sim['Ny']), int(sim.get('Nz', 1))
    geom = sim.get('geometry', {})
    geom_type = geom.get('type', 'cube')
    
    # Construct base coordinate grids (centered around Nx/2, Ny/2) for distance calculations
    # Assuming physical dimensions for masks
    X, Y = np.meshgrid((np.arange(Nx) - Nx / 2.0 + 0.5) * dx, 
                       (np.arange(Ny) - Ny / 2.0 + 0.5) * dx, indexing='ij')
    R_grid = np.sqrt(X**2 + Y**2)
    
    # Create mask mapping (3D extrusion of 2D cross section)
    mask_2d = np.ones((Nx, Ny), dtype=bool)
    if geom_type == 'hollow_cylinder':
        R_in = float(geom.get('R_inner_m', 0))
        R_out = float(geom.get('R_outer_m', 1e9))
        mask_2d = (R_grid >= R_in) & (R_grid <= R_out)
    elif geom_type == 'solid_cylinder':
        R_out = float(geom.get('R_outer_m', 1e9))
        mask_2d = (R_grid <= R_out)
        
    mask_3d = np.repeat(mask_2d[:, :, np.newaxis], Nz, axis=2)
    
    # Spinodal Decomposition: Convert from physical to dimensionless CH parameter
    np.random.seed(42)
    c_ref = float(init['c_ref_atoms_cm3'])
    c0_sim = float(init['c0_atoms_cm3']) / c_ref
    noise_sim = float(init['noise_amplitude_atoms_cm3']) / c_ref
    
    c_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    active_points = np.random.normal(c0_sim, noise_sim, size=np.sum(mask_3d))
    active_points = np.clip(active_points, 0.0, 1.0)
    c_grid[mask_3d] = active_points
    
    # 4. Run
    wrapper = CHWrapper()
    final_c = wrapper.run(c_grid, config, M, kappa, W)
    
    # 5. Visualize Final State (Middle Z-slice)
    mid_z = Nz // 2
    final_c_phys = final_c[:, :, mid_z] * c_ref
    mask_slice = mask_3d[:, :, mid_z]
    final_c_phys[~mask_slice] = np.nan # Transparent outside geometry
    
    plt.figure(figsize=(8, 8))
    plt.imshow(final_c_phys.T, cmap='viridis', origin='lower', interpolation='nearest') 
    plt.colorbar(label='Concentration (atoms/cm³)')
    plt.title(f"Final State Z-slice={mid_z} ({geom_type})")
    plt.title("Final Cahn-Hilliard State")
    output_png = os.path.join(CURRENT_DIR, sim['dump_dir'], "final_state.png")
    plt.savefig(output_png)
    print(f"[+] Saved final state image to {output_png}")
    
    # Plot Energy
    log_file = os.path.join(CURRENT_DIR, sim['dump_dir'], "sim_log.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(df['time'], df['total_mass'])
        plt.title('Total Mass Conservation')
        plt.xlabel('Time')
        
        plt.subplot(1, 2, 2)
        plt.plot(df['time'], df['free_energy'], color='orange')
        plt.title('Free Energy Minimization')
        plt.xlabel('Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(CURRENT_DIR, sim['dump_dir'], "diagnostics.png"))
        print(f"[+] Saved diagnostics to {os.path.join(CURRENT_DIR, sim['dump_dir'], 'diagnostics.png')}")
