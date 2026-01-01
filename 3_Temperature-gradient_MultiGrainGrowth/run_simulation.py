"""
Main orchestration script for Phase-Field Simulation.
Architecture: Hybrid Python (Manager) + C++ (Worker)
"""

import os
import sys
import glob
import platform
import subprocess
import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Module Import Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(CURRENT_DIR, "python_modules")
if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

try:
    from materials import SiliconProperties
except ImportError:
    print(f"[ERROR] Could not import 'materials'. Ensure 'python_modules/materials.py' exists.")
    sys.exit(1)

# --- Configuration Constants ---
CPP_DIR = os.path.join(CURRENT_DIR, "cpp_src")
DATA_DIR = os.path.join(CURRENT_DIR, "data")
CALIB_LOG_DIR = os.path.join(DATA_DIR, "calibration_logs")

# Phase 1: Beta Calibration Config
BETA_CPP_SOURCE = "beta_calibration.cpp"
BETA_EXE_NAME = "beta_calibration.exe" if platform.system() == "Windows" else "beta_calibration"
BETA_TABLE_FILE = os.path.join(DATA_DIR, "beta_table.csv")

# Phase 2: KMC Nucleation Config
KMC_CPP_SOURCE = "kmc_nucleation_schedule.cpp"
if platform.system() == "Windows":
    KMC_LIB_NAME = "libkmc.dll"
else:
    KMC_LIB_NAME = "libkmc.so"
KMC_EVENTS_FILE = os.path.join(DATA_DIR, "nucleation_events.csv")


# --- Helper Classes ---

class DimensionlessSystem:
    def __init__(self, dx_real_meters: float, V_ref_meters_per_sec: float, sigma: float):
        self.l0 = dx_real_meters
        self.E0 = sigma / self.l0
        if V_ref_meters_per_sec < 1e-12:
            V_ref_meters_per_sec = 1e-12
        self.t0 = self.l0 / V_ref_meters_per_sec

    def to_sim_energy(self, J_per_m3: float) -> float:
        return J_per_m3 / self.E0

    def to_real_time(self, t_sim: float) -> float:
        return t_sim * self.t0
        
    def to_real_length(self, x_sim: float) -> float:
        return x_sim * self.l0


# --- Compilation Helpers ---

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def compile_cpp_exe(source_file: str, exe_name: str) -> bool:
    ensure_dir(CPP_DIR)
    cpp_path = os.path.join(CPP_DIR, source_file)
    exe_output = os.path.join(CPP_DIR, exe_name)
    print(f"[*] Compiling Executable: {cpp_path}...")
    cmd = ["g++", "-O3", "-o", exe_output, cpp_path]
    if platform.system() == "Windows":
        cmd.append("-static")
    try:
        subprocess.check_call(cmd)
        print("[+] Compilation successful.")
        return True
    except subprocess.CalledProcessError:
        print("[-] Compilation failed.")
        return False

def compile_shared_lib(source_file: str, lib_name: str) -> bool:
    ensure_dir(CPP_DIR)
    cpp_path = os.path.join(CPP_DIR, source_file)
    lib_output = os.path.join(CPP_DIR, lib_name)
    print(f"[*] Compiling Shared Library: {cpp_path}...")
    if platform.system() == "Windows":
        cmd = ["g++", "-shared", "-o", lib_output, cpp_path, "-O3", "-static", "-fopenmp"]
    else:
        cmd = ["g++", "-shared", "-fPIC", "-o", lib_output, cpp_path, "-O3", "-fopenmp"]
    try:
        subprocess.check_call(cmd)
        print("[+] Library compilation successful.")
        return True
    except subprocess.CalledProcessError:
        print("[-] Library compilation failed.")
        return False


# --- Phase 1: Beta Calibration Logic ---

def run_cpp_calibration_step(config: dict, dG_sim: float, V_sim_target: float, beta: float, temp_k: float) -> tuple[float, float]:
    exe_path = os.path.join(CPP_DIR, BETA_EXE_NAME)
    if not os.path.exists(exe_path):
        if platform.system() != "Windows" and os.path.exists("./" + exe_path):
             exe_path = "./" + exe_path
        elif not os.path.exists(exe_path):
             return 0.0, 0.0

    ensure_dir(CALIB_LOG_DIR)
    history_file = os.path.join(CALIB_LOG_DIR, f"history_T{int(temp_k)}.csv")
    profile_file = os.path.join(CALIB_LOG_DIR, f"profile_T{int(temp_k)}.csv")

    args = [
        str(config['Nx']), str(config['dx']), str(config['xi_sim']),
        str(dG_sim), str(V_sim_target), str(beta),
        str(config['max_steps']), str(config['run_dist_sim']),
        history_file, profile_file
    ]

    try:
        output = subprocess.check_output([exe_path] + args).decode('utf-8').strip()
        parts = output.split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
        return 0.0, 0.0
    except Exception as e:
        print(f"[-] C++ Runtime Error: {e}")
        return 0.0, 0.0

def visualize_calibration_step(temp_k, beta, r_squared, config, scale_obj):
    """
    Beta Calibration 결과를 시각화합니다.
    - 좌측: 최종 Phase Profile (계면 형태 확인)
    - 우측: 시간에 따른 계면 위치 변화 및 선형 회귀 (속도 확인)
    """
    hist_file = os.path.join(CALIB_LOG_DIR, f"history_T{int(temp_k)}.csv")
    prof_file = os.path.join(CALIB_LOG_DIR, f"profile_T{int(temp_k)}.csv")
    
    # 파일 존재 여부 확인
    if not os.path.exists(hist_file):
        # history 파일이 없으면 그릴 게 없으므로 리턴
        return 
    
    # profile 파일은 선택적 (없을 수도 있음)
    has_profile = os.path.exists(prof_file)

    try:
        # 데이터 로드
        df_hist = pd.read_csv(hist_file)
        df_prof = pd.read_csv(prof_file) if has_profile else None
        
        # --- History 데이터 처리 (우측 그래프용) ---
        # 필수 컬럼: 'time', 'position' (혹은 'radius', 'r')
        time_col = next((c for c in df_hist.columns if 'time' in c.lower()), None)
        pos_col = next((c for c in df_hist.columns if c.lower() in ['position', 'radius', 'r']), None)
        
        if not time_col or not pos_col:
            print(f"[WARN] Missing columns in history file for T={temp_k}. Found: {df_hist.columns.tolist()}")
            return

        # 물리적 단위 변환 (Dimensionless -> Real)
        real_time_ns = scale_obj.to_real_time(df_hist[time_col].values) * 1e9
        real_pos_nm = scale_obj.to_real_length(df_hist[pos_col].values) * 1e9
        
        # Figure 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- 좌측: Phase Profile Plot ---
        if has_profile and df_prof is not None:
            # phi 데이터 찾기
            phi_col = next((c for c in df_prof.columns if 'phi' in c.lower()), None)
            
            if phi_col:
                # X축 데이터 결정 (인덱스 vs 별도 컬럼)
                prof_x_col = next((c for c in df_prof.columns if c.lower() in ['index', 'x', 'grid']), None)
                if prof_x_col:
                    x_data = df_prof[prof_x_col]
                    xlabel = prof_x_col
                else:
                    x_data = df_prof.index
                    xlabel = "Grid Index"
                
                ax1.plot(x_data, df_prof[phi_col], 'k-', linewidth=1.5, label='Phase field')
                ax1.set_title(f"Phase Profile (T={int(temp_k)}K)")
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(phi_col)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, "No 'phi' column found", ha='center', va='center')
        else:
            ax1.text(0.5, 0.5, "Profile data not available", ha='center', va='center')
            ax1.set_title(f"Phase Profile (T={int(temp_k)}K)")

        # --- 우측: Interface Motion Plot ---
        ax2.scatter(real_time_ns, real_pos_nm, s=15, c='blue', alpha=0.6, label='Simulation Data')
        
        # 선형 회귀 및 추세선
        if len(real_time_ns) > 1:
            try:
                res = linregress(real_time_ns, real_pos_nm)
                # 데이터 범위에 맞춰 선 그리기
                fit_line = res.slope * real_time_ns + res.intercept
                ax2.plot(real_time_ns, fit_line, 'r--', linewidth=2, 
                         label=f'Fit V={res.slope:.2f} m/s')
            except Exception as reg_err:
                print(f"[WARN] Regression failed: {reg_err}")

        ax2.set_title(rf"Interface Motion (Beta={beta:.4f}, $R^2$={r_squared:.4f})")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Position (nm)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 저장 및 정리
        plt.tight_layout()
        save_path = os.path.join(CALIB_LOG_DIR, f"calib_plot_T{int(temp_k)}.png")
        plt.savefig(save_path, dpi=100)
        # print(f"  -> Calibration plot saved: {save_path}") # 필요시 주석 해제
        
        plt.close(fig) # 메모리 누수 방지
        
    except Exception as e:
        print(f"[WARN] Visualization failed for T={temp_k}: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all') # 안전장치


def calibrate_beta(temps: np.ndarray, config: dict) -> pd.DataFrame:
    ensure_dir(CALIB_LOG_DIR)
    results = []
    mat = SiliconProperties()
    
    print(f"[*] Starting Calibration for {len(temps)} temperatures...")

    for T in temps:
        print(f"\n=== Calibrating for T={T} K ===")
        V_phys = mat.get_physical_velocity(T)
        
        if V_phys < 1e-6:
            results.append({'Temperature': T, 'Beta': 1.0, 'V_sim': 0.0, 'V_phys': V_phys})
            continue
            
        scale = DimensionlessSystem(config['dx_real_meters'], V_phys, mat.sigma)
        dG_sim = scale.to_sim_energy(mat.get_driving_force(T))
        V_sim_target = 1.0
        
        beta_curr = 1.0
        beta_prev, err_prev = None, None
        
        config['run_dist_sim'] = (config['run_dist_nm'] * 1e-9) / scale.l0
        
        final_r_sq = 0.0
        
        for i in range(config['max_iter']):
            V_sim, r_sq = run_cpp_calibration_step(config, dG_sim, V_sim_target, beta_curr, T)
            final_r_sq = r_sq
            
            ratio = V_sim / V_sim_target if V_sim_target > 0 else 0
            error = V_sim - V_sim_target
            
            print(f"  [Iter {i+1}] Beta={beta_curr:.4f} | Ratio={ratio:.4f} | R2={r_sq:.4f}")
            
            if 0.995 <= ratio <= 1.005:
                break
            
            if beta_prev is not None and abs(error - err_prev) > 1e-12:
                new_beta = beta_curr - error * (beta_curr - beta_prev) / (error - err_prev)
            else:
                new_beta = beta_curr * (1.0 / ratio) if ratio > 0 else beta_curr
            
            new_beta = max(beta_curr * 0.2, min(beta_curr * 5.0, new_beta))
            beta_prev, err_prev = beta_curr, error
            beta_curr = new_beta

        visualize_calibration_step(T, beta_curr, final_r_sq, config, scale)

        results.append({
            'Temperature': T,
            'Beta': beta_curr,
            'V_phys': V_phys,
            'V_sim_real': V_sim * (scale.l0 / scale.t0)
        })

    return pd.DataFrame(results)

def run_phase1_calibration(force_rerun: bool = False):
    if not force_rerun and os.path.exists(BETA_TABLE_FILE):
        print(f"[INFO] Phase 1: Calibration file found. Skipping.")
        return

    print("[INFO] Phase 1: Starting Beta Calibration...")
    if not compile_cpp_exe(BETA_CPP_SOURCE, BETA_EXE_NAME):
        return

    sim_config = {
        'dx': 1.0, 'xi_sim': 5.0, 'Nx': 2000,
        'max_steps': 500000, 'run_dist_nm': 200.0,
        'dx_real_meters': 1.0e-9, 'max_iter': 15
    }
    
    temps = np.arange(400, 1601, 50) 
    
    df = calibrate_beta(temps, sim_config)
    df.to_csv(BETA_TABLE_FILE, index=False)
    print(f"[+] Phase 1 Completed. Data saved.")
    
    plt.figure(figsize=(6, 4))
    plt.plot(df['Temperature'], df['Beta'], 'o-')
    plt.title("Beta Calibration Summary")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Beta")
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, "phase1_beta_summary.png"))

def verify_and_plot_environment(Nx, dx, T_profile, df_beta, df_dG, config, output_dir):
    """
    시뮬레이션 환경 변수(온도, 구동력, Mobility)를 검증하고 독립적인 subplot으로 시각화합니다.
    """
    print("[*] Verifying simulation environment (T, dG, Mobility)...")
    
    # 1. 데이터 준비 (컬럼명 안전 처리 포함)
    beta_col = 'beta' if 'beta' in df_beta.columns else ('Beta' if 'Beta' in df_beta.columns else df_beta.columns[1])
    dG_col = 'dG_sim' if 'dG_sim' in df_dG.columns else df_dG.columns[1]
    
    beta_T = df_beta['Temperature'].values
    beta_val = df_beta[beta_col].values
    dG_T = df_dG['Temperature'].values
    dG_val = df_dG[dG_col].values

    L_profile = np.zeros(Nx)
    dG_profile = np.zeros(Nx)
    beta_profile = np.zeros(Nx)
    
    # Stability Limit (L_max) 계산
    inv_dx2 = 1.0 / (config['dx'] * config['dx'])
    L_max_limit = 0.2 * inv_dx2 / (config['dt'] * config['kappa'])

    # 데이터 보간 및 계산
    for i in range(Nx):
        T = T_profile[i]
        b = np.interp(T, beta_T, beta_val)
        g = np.interp(T, dG_T, dG_val)
        
        beta_profile[i] = b
        dG_profile[i] = g
        
        # Mobility 계산 (Clipping 적용)
        L_val = 0.0
        if g > 1e-12:
            L_val = (1.0 / g) / b
        if L_val > L_max_limit:
            L_val = L_max_limit  # Clipping
            
        L_profile[i] = L_val

    # 2. CSV 저장
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "environment_profile.csv")
    pd.DataFrame({
        'Grid_Index': np.arange(Nx),
        'Temperature': T_profile,
        'dG_sim': dG_profile,
        'beta': beta_profile,
        'Mobility_L': L_profile,
        'Mobility_Limit': [L_max_limit] * Nx
    }).to_csv(csv_path, index=False)
    print(f" -> Environment data saved to: {csv_path}")

    # 3. 그래프 그리기 (3개의 Subplot)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    x_axis = np.arange(Nx)

    # (1) Temperature Plot
    ax1 = axes[0]
    ax1.plot(x_axis, T_profile, color='tab:red', linewidth=2)
    ax1.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax1.set_title('1. Temperature Profile', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # (2) Driving Force Plot
    ax2 = axes[1]
    ax2.plot(x_axis, dG_profile, color='tab:green', linewidth=2)
    ax2.set_ylabel('Driving Force (dG)', fontsize=12, fontweight='bold')
    ax2.set_title('2. Driving Force Profile', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # (3) Mobility Plot (Limit Line 포함)
    ax3 = axes[2]
    ax3.plot(x_axis, L_profile, color='tab:blue', linewidth=2, label='Mobility (L)')
    ax3.axhline(y=L_max_limit, color='black', linestyle='-.', linewidth=1.5, label='Stability Limit')
    ax3.set_ylabel('Mobility (L)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Grid Position (x)', fontsize=12)
    ax3.set_title('3. Mobility Profile with Stability Limit', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_yscale('log') 

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    png_path = os.path.join(output_dir, "environment_check.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f" -> Verification plot saved to: {png_path}")


# --- Phase 2: Nucleation Scheduling & Validation Logic ---

class KMCWrapper:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self._load_lib()

    def _load_lib(self):
        try:
            if platform.system() == "Windows" and sys.version_info >= (3, 8):
                self.lib = ctypes.CDLL(self.lib_path, winmode=0)
            else:
                self.lib = ctypes.CDLL(self.lib_path)

            self.lib.run_kmc.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int
            ]
            self.lib.run_kmc.restype = ctypes.c_int
        except OSError as e:
            print(f"[ERROR] Failed to load KMC library: {e}")
            raise

    def run(self, rates: np.ndarray, nx: int, ny: int, time_limit: float, max_events: int = 200000):
        rates_flat = rates.astype(np.float64)
        rates_ptr = rates_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result_buffer = (ctypes.c_double * (max_events * 3))()
        
        count = self.lib.run_kmc(rates_ptr, nx, ny, time_limit, result_buffer, max_events)
        if count == 0: return np.empty((0, 3))
        return np.frombuffer(result_buffer, dtype=np.float64, count=count * 3).reshape((count, 3))

def get_nucleation_profile(Nx, T_left, T_right, mat: SiliconProperties):
    kB = mat.kB
    elementary_charge = 1.60217663e-19
    T = np.linspace(T_left, T_right, Nx)
    I = np.zeros_like(T)
    mask = (T > 0) & (T < mat.Tm)
    T_valid = T[mask]
    
    dGv = (mat.Hf * (mat.Tm - T_valid)) / (mat.Tm * mat.Vm)
    dG_star = (16 * np.pi * mat.sigma**3) / (3 * dGv**2)
    QJ = mat.Qdiff_eV * elementary_charge
    I[mask] = mat.I0 * np.exp(-QJ / (kB * T_valid)) * np.exp(-dG_star / (kB * T_valid))
    return T, I

def run_phase2_nucleation(force_rerun: bool = False):
    if not force_rerun and os.path.exists(KMC_EVENTS_FILE):
        print(f"[INFO] Phase 2: Event file found. Skipping.")
        return

    print("[INFO] Phase 2: Starting Nucleation Scheduling...")
    if not compile_shared_lib(KMC_CPP_SOURCE, KMC_LIB_NAME):
        return

    # Grid Setup
    Nx, Ny = 500, 100
    T_left, T_right = 500.0, 1400.0
    cell_size = 10e-9 
    cell_vol = cell_size**3
    mat = SiliconProperties()
    
    T_profile, I_profile = get_nucleation_profile(Nx, T_left, T_right, mat)
    peak_rate = np.max(I_profile)
    
    if peak_rate == 0: return

    # Run KMC
    kmc = KMCWrapper(os.path.join(CPP_DIR, KMC_LIB_NAME))
    cell_rates = I_profile * cell_vol
    
    sim_duration = 50.0 / np.max(cell_rates) 
    print(f" -> Simulating for {sim_duration:.2e} seconds...")
    
    events = kmc.run(cell_rates, Nx, Ny, sim_duration, max_events=500000)
    print(f" -> Generated {len(events)} nucleation events.")

    df_events = pd.DataFrame(events, columns=['time', 'x_idx', 'y_idx'])
    df_events['x_um'] = df_events['x_idx'] * cell_size * 1e6
    df_events['y_um'] = df_events['y_idx'] * cell_size * 1e6
    df_events.to_csv(KMC_EVENTS_FILE, index=False)

    # --- Updated Visualization Code ---
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])
    
    # 1. Temperature Environment
    ax1 = fig.add_subplot(gs[0])
    x_axis = np.linspace(0, Nx*cell_size*1e6, Nx)
    ax1.plot(x_axis, T_profile, 'r--', label='Temperature')
    ax1.set_ylabel("Temperature (K)", color='r')
    ax1.set_title("Nucleation Environment")
    ax1.grid(True)
    ax1.set_xticklabels([])  # Hide x labels for top plot

    # 2. Validation: Log-Scale Histogram vs CNT Theory
    ax2 = fig.add_subplot(gs[1])
    
    # Histogram Setup
    num_bins = 50
    counts, bins = np.histogram(df_events['x_um'], bins=num_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width_m = (bins[1] - bins[0]) * 1e-6 # um to m
    
    # Convert Counts to Rate (events / m^3 / s)
    # Rate = Count / (Volume * Time)
    domain_vol_bin = (bin_width_m * (Ny * cell_size) * cell_size) # Using cell_size as thickness (2D assumption) or Volume if 3D
    # Actually, Nucleation Rate I is per UNIT VOLUME.
    # Here simulation is 2D grid representing a thin slice? 
    # Usually I is [m^-3 s^-1]. Our cell_vol is [m^3]. 
    # Total Volume of a bin = (Ny * cell_size) * (bin_width_m) * (thickness=cell_size implicitly for 2D->3D mapping?)
    # Let's assume thickness = cell_size for consistency with cell_vol = cell_size^3.
    vol_per_bin = (Ny * cell_size) * bin_width_m * cell_size
    
    measured_rate = counts / (vol_per_bin * sim_duration)
    
    # Plot KMC Results (Histogram Step Plot)
    ax2.step(bin_centers, measured_rate, where='mid', color='black', linewidth=1.5, label='KMC Simulation')
    # Fill for better visibility
    ax2.fill_between(bin_centers, measured_rate, step='mid', color='gray', alpha=0.3)
    
    # Plot CNT Theory (Solid Line)
    ax2.plot(x_axis, I_profile, 'C0-', linewidth=2.5, alpha=0.8, label='Classical Nucleation Theory')
    
    # Configuration
    ax2.set_yscale('log')
    ax2.set_xlabel("Position X (um)")
    ax2.set_ylabel("Nucleation Rate ($m^{-3}s^{-1}$)")
    ax2.set_title("Validation: KMC vs CNT Theory")
    ax2.legend(loc='upper right')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_ylim(bottom=max(1e25, np.min(I_profile[I_profile>0])*0.1)) # Auto-scale y-min

    # Save
    save_path = os.path.join(DATA_DIR, "nucleation_validation_log.png")
    plt.savefig(save_path)
    print(f"[+] Detailed validation plot saved to {save_path}")

class MultiGrainWrapper:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        try:
            if platform.system() == "Windows":
                self.lib = ctypes.CDLL(self.lib_path, winmode=0)
            else:
                self.lib = ctypes.CDLL(self.lib_path)
            print(f"[+] MultiGrain Library loaded: {self.lib_path}")
        except OSError as e:
            print(f"[ERROR] Failed to load MultiGrain lib: {e}")
            raise

    def run(self, 
            phi_grid: np.ndarray, 
            temp_profile: np.ndarray,
            beta_table: pd.DataFrame,
            dG_table: pd.DataFrame, 
            nucleation_events: pd.DataFrame,
            config: dict,
            dump_dir: str):
        
        class SimConfig(ctypes.Structure):
            _fields_ = [
                ("Nx", ctypes.c_int), ("Ny", ctypes.c_int),
                ("dx", ctypes.c_double), ("dt", ctypes.c_double),
                ("xi", ctypes.c_double), ("epsilon_penalty", ctypes.c_double),
                ("kappa", ctypes.c_double), ("W", ctypes.c_double),
                ("max_grains", ctypes.c_int), ("time_total", ctypes.c_double),
                ("output_interval", ctypes.c_int),  ("use_active_list", ctypes.c_int)
            ]

        cfg = SimConfig(
            config['Nx'], config['Ny'], 
            config['dx'], config['dt'],
            config['xi'], config['epsilon_penalty'],
            config['kappa'], config['W'],
            config['max_grains'], config['time_total'], 
            config['output_interval'], config['use_active_list']
        )

        phi_flat = phi_grid.flatten().astype(np.float64)
        phi_ptr = phi_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        temp_ptr = temp_profile.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        beta_T = beta_table['Temperature'].values.astype(np.float64)
        beta_val = beta_table['Beta'].values.astype(np.float64)
        
        dG_T = dG_table['Temperature'].values.astype(np.float64)
        dG_val = dG_table['dG_sim'].values.astype(np.float64)

        nucl_data = nucleation_events[['time', 'x_idx', 'y_idx']].values.flatten().astype(np.float64)
        nucl_ptr = nucl_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n_events = len(nucleation_events)

        # Convert dump_dir to bytes for C compatibility
        dump_dir_bytes = dump_dir.encode('utf-8')

        self.lib.run_mpf_simulation.argtypes = [
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
            SimConfig,
            ctypes.c_char_p  # <-- NEW: dump_dir path
        ]

        print(f"[*] Launching C++ MPF Kernel...")
        print(f"    - Time Total: {config['time_total']:.2f} (dimensionless)")
        print(f"    - Output Interval: {config['output_interval']} steps")
        print(f"    - Dump Directory: {dump_dir}")
        
        self.lib.run_mpf_simulation(
            phi_ptr, temp_ptr,
            beta_T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            beta_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(beta_T),
            dG_T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dG_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(dG_T),
            nucl_ptr, n_events,
            cfg,
            dump_dir_bytes  # <-- Pass dump directory
        )
        print("[+] C++ Kernel Finished.")

        return phi_flat.reshape(config['max_grains'], config['Nx'], config['Ny'])


def create_animation(dump_dir, frames_dir, output_file, nx, ny, max_grains, fps=30, scale_obj=None, dt_sim=None):
    """
    scale_obj와 dt_sim을 받아 타이틀에 실제 시간을 표시하는 기능 추가
    """
    print("[*] Creating animation from dump files...")
    
    dump_files = sorted(
        glob.glob(os.path.join(dump_dir, "*.bin")),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )
    
    if not dump_files:
        print("[WARN] No dump files found to create animation.")
        return
        
    print(f"[*] Found {len(dump_files)} dump files.")
    ensure_dir(frames_dir)
    
    total_elements = max_grains * nx * ny
    for i, file_path in enumerate(dump_files):
        try:
            with open(file_path, 'rb') as f:
                phi_flat = np.fromfile(f, dtype=np.float64, count=total_elements)
                
            if phi_flat.size != total_elements: 
                print(f"\n[WARN] Skipping {file_path}: size mismatch")
                continue
            
            phi_grid = phi_flat.reshape(max_grains, nx, ny)
            
            max_vals = np.max(phi_grid, axis=0)
            argmax_ids = np.argmax(phi_grid, axis=0)
            grain_map = np.where(max_vals > 0.5, argmax_ids + 1, 0)
            
            plt.figure(figsize=(10, 4))
            plt.imshow(grain_map.T, cmap='nipy_spectral', origin='lower', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=max_grains)
            
            step_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            title_str = f"Microstructure (Step: {step_num})"
            if scale_obj and dt_sim:
                sim_time = step_num * dt_sim
                real_time_sec = scale_obj.to_real_time(sim_time)
                real_time_us = real_time_sec * 1e6
                title_str = rf"Microstructure (Time: {real_time_us:.2f} $\mu$s)"

            plt.title(title_str, fontsize=12)
            plt.xlabel("X (Temperature Gradient)")
            plt.ylabel("Y")
            plt.colorbar(label='Grain ID')
            
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path, dpi=100)
            plt.close()
            
            print(f"  -> Generated frame {i+1}/{len(dump_files)}", end='\r')
            
        except Exception as e:
            print(f"\n[WARN] Failed to process {file_path}: {e}")
            
    print("\n[+] Frame generation complete.")
    
    print("[*] Encoding video with ffmpeg...")
    frame_pattern = os.path.join(frames_dir, "frame_%04d.png")
    
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] ffmpeg not found. Install ffmpeg to create animation.")
        return

    cmd = [
        'ffmpeg', '-y', '-r', str(fps), '-i', frame_pattern,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] Animation saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[-] ffmpeg Error: {e.stderr}")

def create_sigma_animation(dump_dir, frames_dir, output_file, nx, ny, max_grains, fps=30, scale_obj=None, dt_sim=None):
    """
    Creates an MP4 animation visualizing the sum of squares (Sigma phi^2).
    Useful for detecting numerical instability (values > 1.0).
    """
    print(f"[*] Creating Sigma-Sq animation to: {output_file}")
    
    # Clean up old frames specifically for sigma
    ensure_dir(frames_dir)
    for f in glob.glob(os.path.join(frames_dir, "sigma_frame_*.png")):
        os.remove(f)

    dump_files = sorted(
        glob.glob(os.path.join(dump_dir, "*.bin")),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )

    if not dump_files:
        print("[WARN] No dump files found.")
        return

    total_elements = max_grains * nx * ny

    for i, file_path in enumerate(dump_files):
        try:
            with open(file_path, 'rb') as f:
                phi_flat = np.fromfile(f, dtype=np.float64, count=total_elements)
            
            if phi_flat.size != total_elements:
                continue

            # Reshape and Calculate Sigma Squared
            phi_grid = phi_flat.reshape(max_grains, nx, ny)
            sigma_sq_map = np.sum(phi_grid**2, axis=0) # Sum across all grains (axis 0)

            # Statistics for checking instability
            max_val = np.max(sigma_sq_map)
            min_val = np.min(sigma_sq_map)

            # Plotting
            plt.figure(figsize=(10, 4))
            
            # vmin=0, vmax=1.2로 설정하여 1.0을 넘는 부분을 강조
            im = plt.imshow(sigma_sq_map.T, cmap='inferno', origin='lower', aspect='auto',
                       interpolation='nearest', vmin=0.0, vmax=1.2)
            
            step_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            
            # Title shows the Maximum value in this frame (Critical for debugging)
            title_color = 'red' if max_val > 1.01 else 'black'
            title_str = rf"$\Sigma$($\phi^2$) (Step: {step_num})"
            if scale_obj and dt_sim:
                sim_time = step_num * dt_sim
                real_time_us = scale_obj.to_real_time(sim_time) * 1e6
                title_str = rf"$\Sigma$($\phi^2$) (Time: {real_time_us:.3f} $\mu$s)"
            
            title_str += f" | Max: {max_val:.4f}"
            
            plt.title(title_str, color=title_color, fontweight='bold')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.colorbar(im, label='$\\sum \\phi_i^2$')

            frame_path = os.path.join(frames_dir, f"sigma_frame_{i:04d}.png")
            plt.savefig(frame_path, dpi=100)
            plt.close()

            print(f" -> Processing Sigma Frame {i+1}/{len(dump_files)} (Max={max_val:.3f})", end='\r')

        except Exception as e:
            print(f"\n[WARN] Error on {file_path}: {e}")

    print("\n[+] Sigma frames generated. Encoding video...")

    # FFmpeg encoding
    try:
        cmd = [
            'ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(frames_dir, "sigma_frame_%04d.png"),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[SUCCESS] Sigma animation saved to {output_file}")
    except Exception as e:
        print(f"[-] FFmpeg failed: {e}")


def run_phase3_simulation():
    print("\n[INFO] Phase 3: Multi-Grain Simulation Starting...")
    
    DUMP_DIR = os.path.join(DATA_DIR, "dump")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")
    
    # Ensure directories exist BEFORE C++ execution
    ensure_dir(DUMP_DIR)
    ensure_dir(FRAMES_DIR)
    
    # Clean old files
    for f in glob.glob(os.path.join(DUMP_DIR, "*.bin")): os.remove(f)
    for f in glob.glob(os.path.join(FRAMES_DIR, "*.png")): os.remove(f)

    MPF_SOURCE = "multi_grain_simulation.cpp"
    MPF_LIB = "libmpf.dll" if platform.system() == "Windows" else "libmpf.so"
    if not compile_shared_lib(MPF_SOURCE, MPF_LIB): return

    if not os.path.exists(BETA_TABLE_FILE) or not os.path.exists(KMC_EVENTS_FILE):
        print("[ERROR] Missing calibration/nucleation data.")
        return
        
    df_beta = pd.read_csv(BETA_TABLE_FILE)
    df_nucl = pd.read_csv(KMC_EVENTS_FILE)
    
    mat = SiliconProperties()
    Nx, Ny = 500, 100
    #dx_real = 10e-9 
    dx_real = 1e-6
    xi_sim = 10.0    
    
    V_ref = mat.get_physical_velocity(500.0) 
    scale = DimensionlessSystem(dx_real, V_ref, mat.sigma)
    
    T_left, T_right = 500.0, 1650.0
    T_profile = np.linspace(T_left, T_right, Nx)
    
    T_lookup = np.linspace(500, 1650, 1151)
    dG_lookup = [scale.to_sim_energy(mat.get_driving_force(t)) for t in T_lookup]
    df_dG = pd.DataFrame({'Temperature': T_lookup, 'dG_sim': dG_lookup})
    
    df_nucl_sim = df_nucl.copy()
    df_nucl_sim['time'] = df_nucl['time'] / scale.t0
    
    physical_duration_sec = 10000.0e-6 
    time_total_sim = physical_duration_sec / scale.t0

    use_optimization = 1 # 1: 활성 Grain만 계산 (빠름), 0: 전체 Grain 계산 (느림, 비교용)
    config = {
        'Nx': Nx, 'Ny': Ny,
        'dx': 1.0, 
        'dt': 0.01, 
        'xi': xi_sim,
        'epsilon_penalty': 20.0,
        'kappa': 1.0 * xi_sim / 4.0,
        'W': 12.0 / xi_sim,
        'max_grains': 100, 
        'time_total': time_total_sim,
        'output_interval': 50,  # Save every 10 steps for ~100 frames
        'use_active_list': use_optimization
    }

    verify_and_plot_environment(
        Nx=Nx, 
        dx=config['dx'], 
        T_profile=T_profile, 
        df_beta=df_beta, 
        df_dG=df_dG, 
        config=config, 
        output_dir=DATA_DIR  # 또는 DUMP_DIR
    )

    phi_grid = np.zeros((config['max_grains'], Nx, Ny), dtype=np.float64)
    
    mpf = MultiGrainWrapper(os.path.join(CPP_DIR, MPF_LIB))
    mpf.run(phi_grid, T_profile, df_beta, df_dG, df_nucl_sim, config, DUMP_DIR)
    
    print("\n--- Generating Microstructure Video ---")
    animation_output_file = os.path.join(DATA_DIR, "microstructure_evolution.mp4")
    create_animation(DUMP_DIR, FRAMES_DIR, animation_output_file, Nx, Ny, config['max_grains'], fps=30,scale_obj=scale, dt_sim=config['dt'])

    print("\n--- Generating Validation Video (Sigma Squared) ---")
    sigma_anim_file = os.path.join(DATA_DIR, "validation_sigma_sq.mp4")
    create_sigma_animation(DUMP_DIR, FRAMES_DIR, sigma_anim_file, Nx, Ny, config['max_grains'], fps=30,scale_obj=scale, dt_sim=config['dt'])

def run_phase3_simulation_vis():
    print("\n[INFO] Phase 3: Multi-Grain Simulation Starting...")
    
    DUMP_DIR = os.path.join(DATA_DIR, "dump")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")
    
    # Ensure directories exist BEFORE C++ execution
    ensure_dir(DUMP_DIR)
    ensure_dir(FRAMES_DIR)
    
    df_beta = pd.read_csv(BETA_TABLE_FILE)
    df_nucl = pd.read_csv(KMC_EVENTS_FILE)
    
    mat = SiliconProperties()
    Nx, Ny = 500, 100
    dx_real = 10e-9 
    xi_sim = 3.0    
    
    V_ref = mat.get_physical_velocity(500.0) 
    scale = DimensionlessSystem(dx_real, V_ref, mat.sigma)
    
    T_left, T_right = 500.0, 1650.0
    T_profile = np.linspace(T_left, T_right, Nx)
    
    T_lookup = np.linspace(500, 1650, 1151)
    dG_lookup = [scale.to_sim_energy(mat.get_driving_force(t)) for t in T_lookup]
    df_dG = pd.DataFrame({'Temperature': T_lookup, 'dG_sim': dG_lookup})
    
    df_nucl_sim = df_nucl.copy()
    df_nucl_sim['time'] = df_nucl['time'] / scale.t0
    
    physical_duration_sec = 200.0e-6 
    time_total_sim = physical_duration_sec / scale.t0

    use_optimization = 1 # 1: 활성 Grain만 계산 (빠름), 0: 전체 Grain 계산 (느림, 비교용)
    config = {
        'Nx': Nx, 'Ny': Ny,
        'dx': 1.0, 
        'dt': 0.01, 
        'xi': xi_sim,
        'epsilon_penalty': 5.0,
        'kappa': 1.0 * xi_sim / 4.0,
        'W': 12.0 / xi_sim,
        'max_grains': 1000, 
        'time_total': time_total_sim,
        'output_interval': 50,  # Save every 10 steps for ~100 frames
        'use_active_list': use_optimization
    }

    phi_grid = np.zeros((config['max_grains'], Nx, Ny), dtype=np.float64)
    
    #mpf = MultiGrainWrapper(os.path.join(CPP_DIR, MPF_LIB))
    #mpf.run(phi_grid, T_profile, df_beta, df_dG, df_nucl_sim, config, DUMP_DIR)
    
    print("\n--- Generating Microstructure Video ---")
    animation_output_file = os.path.join(DATA_DIR, "microstructure_evolution.mp4")
    create_animation(DUMP_DIR, FRAMES_DIR, animation_output_file, Nx, Ny, config['max_grains'], fps=30,scale_obj=scale, dt_sim=config['dt'])

    print("\n--- Generating Validation Video (Sigma Squared) ---")
    sigma_anim_file = os.path.join(DATA_DIR, "validation_sigma_sq.mp4")
    create_sigma_animation(DUMP_DIR, FRAMES_DIR, sigma_anim_file, Nx, Ny, config['max_grains'], fps=30,scale_obj=scale, dt_sim=config['dt'])

def analyze_line_profile_video(dump_dir, output_video_path, nx, ny, max_grains, target_y=80, x_range=(300, 340), fps=10, scale_obj=None, dt_sim=None):
    """
    특정 라인(Line Profile)의 Phi 값 변화를 추적하여 그래프를 그리고, 이를 MP4 영상으로 저장합니다.
    """
    print(f"[*] Analyzing line profile video (Y={target_y}, X={x_range}) -> {output_video_path}")
    
    # 임시 프레임 저장 폴더
    frames_dir = os.path.join(os.path.dirname(output_video_path), "temp_profile_frames")
    ensure_dir(frames_dir)
    
    # 기존 프레임 청소
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)
    
    dump_files = sorted(
        glob.glob(os.path.join(dump_dir, "*.bin")),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )
    
    if not dump_files:
        print("[WARN] No dump files found.")
        return

    x_start, x_end = x_range
    x_coords = np.arange(x_start, x_end)
    total_elements = max_grains * nx * ny
    
    # --- 1. 프레임 생성 루프 ---
    for i, file_path in enumerate(dump_files):
        try:
            step_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            
            with open(file_path, 'rb') as f:
                phi_flat = np.fromfile(f, dtype=np.float64, count=total_elements)
            
            if phi_flat.size != total_elements:
                continue

            phi_grid = phi_flat.reshape(max_grains, nx, ny)
            slice_data = phi_grid[:, x_start:x_end, target_y] # (Grains, X_len)
            
            # 활성 Grain 필터링 (최대값이 0.01보다 큰 녀석들만)
            active_indices = np.where(np.max(slice_data, axis=1) > 0.01)[0]
            
            # Plotting
            plt.figure(figsize=(10, 6))
            
            # 아무것도 없어도 빈 그래프는 그립니다 (축 유지)
            if len(active_indices) > 0:
                for gid in active_indices:
                    plt.plot(x_coords, slice_data[gid, :], marker='.', markersize=4, linewidth=1.5, label=f'G{gid+1}')
            
            # Title Generation (Unicode Safe)
            title_str = f"Line Profile @ Y={target_y} (Step: {step_num})"
            if scale_obj and dt_sim:
                real_time_us = scale_obj.to_real_time(step_num * dt_sim) * 1e6
                title_str += f" | Time: {real_time_us:.3f} μs"
            
            plt.title(title_str, fontsize=14)
            plt.xlabel(f"X Grid Index (Range: {x_start}~{x_end})", fontsize=12)
            plt.ylabel("Order Parameter (φ)", fontsize=12)
            plt.ylim(-0.05, 1.1) # 0~1 범위를 조금 여유있게
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # 범례가 너무 많으면 가려지므로 최대 10개까지만 표시하거나 밖으로 뺌
            if len(active_indices) > 0:
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
            
            plt.tight_layout()
            
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path, dpi=100)
            plt.close()
            
            print(f" -> Generating profile frame {i+1}/{len(dump_files)}", end='\r')

        except Exception as e:
            print(f"[WARN] Error processing {file_path}: {e}")
            
    print("\n[+] Frames generated. Encoding video...")

    # --- 2. FFmpeg 인코딩 ---
    try:
        # ffmpeg -y -r 10 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
        cmd = [
            'ffmpeg', '-y', 
            '-r', str(fps), 
            '-i', os.path.join(frames_dir, "frame_%04d.png"),
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '-crf', '23',
            output_video_path
        ]
        
        # Windows 환경에서 subprocess 실행 시 shell=True가 필요할 수도 있음 (보통은 불필요)
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[SUCCESS] Profile video saved to: {output_video_path}")
        
        # (선택) 임시 폴더 삭제하려면 아래 주석 해제
        # import shutil
        # shutil.rmtree(frames_dir)
        
    except FileNotFoundError:
        print("[ERROR] FFmpeg not found. Please install FFmpeg to create video.")
    except subprocess.CalledProcessError as e:
        print(f"[-] FFmpeg encoding failed: {e}")


def run_phase3_simulation_line_analysis():
    print("\n[INFO] Phase 3: Multi-Grain Simulation Starting...")
    
    DUMP_DIR = os.path.join(DATA_DIR, "dump")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")
    
    # Ensure directories exist BEFORE C++ execution
    ensure_dir(DUMP_DIR)
    ensure_dir(FRAMES_DIR)
    
    df_beta = pd.read_csv(BETA_TABLE_FILE)
    df_nucl = pd.read_csv(KMC_EVENTS_FILE)
    
    mat = SiliconProperties()
    Nx, Ny = 500, 100
    #dx_real = 10e-9 
    dx_real = 1e-6
    xi_sim = 10.0    
    
    V_ref = mat.get_physical_velocity(500.0) 
    scale = DimensionlessSystem(dx_real, V_ref, mat.sigma)
    
    T_left, T_right = 500.0, 1650.0
    T_profile = np.linspace(T_left, T_right, Nx)
    
    T_lookup = np.linspace(500, 1650, 1151)
    dG_lookup = [scale.to_sim_energy(mat.get_driving_force(t)) for t in T_lookup]
    df_dG = pd.DataFrame({'Temperature': T_lookup, 'dG_sim': dG_lookup})
    
    df_nucl_sim = df_nucl.copy()
    df_nucl_sim['time'] = df_nucl['time'] / scale.t0
    
    physical_duration_sec = 10000.0e-6 
    time_total_sim = physical_duration_sec / scale.t0

    use_optimization = 1 # 1: 활성 Grain만 계산 (빠름), 0: 전체 Grain 계산 (느림, 비교용)
    config = {
        'Nx': Nx, 'Ny': Ny,
        'dx': 1.0, 
        'dt': 0.01, 
        'xi': xi_sim,
        'epsilon_penalty': 20.0,
        'kappa': 1.0 * xi_sim / 4.0,
        'W': 12.0 / xi_sim,
        'max_grains': 100, 
        'time_total': time_total_sim,
        'output_interval': 50,  # Save every 10 steps for ~100 frames
        'use_active_list': use_optimization
    }
    phi_grid = np.zeros((config['max_grains'], Nx, Ny), dtype=np.float64)
    
    # 3. Line Profile Video (Debugging Specific Area)
    print("\n--- Generating Line Profile Debug Video ---")
    debug_video_path = os.path.join(DATA_DIR, "debug_profile_Y80.mp4")
    
    analyze_line_profile_video(
        DUMP_DIR, 
        debug_video_path, 
        Nx, Ny, 
        config['max_grains'], 
        target_y=75,        # 문제의 Y 좌표
        x_range=(230, 270), # 문제의 X 구간
        fps=10,             # 그래프 변화는 빠르므로 FPS를 낮춰서 천천히 관찰
        scale_obj=scale, 
        dt_sim=config['dt']
    )

def main():
    ensure_dir(DATA_DIR)
    ensure_dir(CPP_DIR)
    
    run_phase1_calibration(force_rerun=True)
    run_phase2_nucleation(force_rerun=True)
    run_phase3_simulation()
    #run_phase3_simulation_vis()
    run_phase3_simulation_line_analysis()

if __name__ == "__main__":
    main()
