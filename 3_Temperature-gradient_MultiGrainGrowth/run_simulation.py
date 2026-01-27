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
import matplotlib.cm as cm
from scipy.stats import linregress
import threading
import time
import argparse
from sim_utils import DimensionlessSystem, determine_required_epsilon, resume_from_vtk
from config_loader import load_sim_config

# --- Module Import Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(CURRENT_DIR, "python_modules")
if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

try:
    from sim_utils import MaterialProperties
except ImportError:
    print(f"[ERROR] Could not import 'MaterialProperties' from sim_utils.")
    sys.exit(1)

# --- Configuration Constants ---
CPP_DIR = os.path.join(CURRENT_DIR, "cpp_src")
DATA_DIR = os.path.join(CURRENT_DIR, "data")
CALIB_LOG_DIR = os.path.join(DATA_DIR, "calibration_logs")

# Phase 1: Beta Calibration Config
BETA_CPP_SOURCE = "beta_calibration.cpp"
BETA_EXE_NAME = "beta_calibration.exe" if platform.system() == "Windows" else "beta_calibration"
BETA_TABLE_FILE = os.path.join(DATA_DIR, "beta_table.csv")

# --- Compilation Helpers ---
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def compile_cpp_exe(source_file: str, exe_name: str) -> bool:
    ensure_dir(CPP_DIR)
    cpp_path = os.path.join(CPP_DIR, source_file)
    exe_output = os.path.join(CPP_DIR, exe_name)
    print(f"[*] Compiling Executable: {cpp_path}...")
    
    # AVX512 and SIMD Flags
    flags = ["-O3", "-march=native", "-mavx512f", "-mavx512dq", "-mfma", "-fopenmp", "-ffast-math"]
    
    cmd = ["g++", "-o", exe_output, cpp_path] + flags
    if platform.system() == "Windows":
        cmd.append("-static")
        
    try:
        subprocess.check_call(cmd)
        print("[+] Compilation successful with AVX512/SIMD flags.")
        return True
    except subprocess.CalledProcessError:
        print("[-] Compilation failed with AVX512. Retrying with fallback...")
        fallback_flags = ["-O3", "-march=native", "-fopenmp"]
        cmd = ["g++", "-o", exe_output, cpp_path] + fallback_flags
        if platform.system() == "Windows":
             cmd.append("-static")
             
        try:
             subprocess.check_call(cmd)
             print("[+] Fallback compilation successful.")
             return True
        except subprocess.CalledProcessError:
             print("[-] Compilation failed.")
             return False

def compile_shared_lib(source_file: str, lib_name: str) -> bool:
    ensure_dir(CPP_DIR)
    cpp_path = os.path.join(CPP_DIR, source_file)
    lib_output = os.path.join(CPP_DIR, lib_name)
    print(f"[*] Compiling Shared Library: {cpp_path}...")
    
    # AVX512 and SIMD Flags
    # -march=native: Optimizes for the local machine (enables AVX512 if available)
    # Explicit flags added as per user request to ensure AVX512 is attempted
    flags = ["-O3", "-march=native", "-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mavx512cd", "-mfma", "-fopenmp", "-ffast-math"]
    
    if platform.system() == "Windows":
        cmd = ["g++", "-shared", "-o", lib_output, cpp_path] + flags + ["-static"]
    else:
        cmd = ["g++", "-shared", "-fPIC", "-o", lib_output, cpp_path] + flags
        
    try:
        subprocess.check_call(cmd)
        print("[+] Library compilation successful with AVX512/SIMD flags.")
        return True
    except subprocess.CalledProcessError:
        print("[-] Library compilation failed. Retrying without explicit AVX512 flags (fallback)...")
        # Fallback without explicit AVX512 (in case user machine doesn't support it or compiler is old)
        fallback_flags = ["-O3", "-march=native", "-fopenmp"]
        if platform.system() == "Windows":
             cmd = ["g++", "-shared", "-o", lib_output, cpp_path] + fallback_flags + ["-static"]
        else:
             cmd = ["g++", "-shared", "-fPIC", "-o", lib_output, cpp_path] + fallback_flags
        
        try:
            subprocess.check_call(cmd)
            print("[+] Fallback compilation successful.")
            return True
        except subprocess.CalledProcessError:
             print("[-] Fallback compilation also failed.")
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
            # Check if phi is present as a column (header case)
            phi_col = next((c for c in df_prof.columns if 'phi' in str(c).lower()), None)
            
            # Fallback: No header or single column -> Treat as phi
            if not phi_col and len(df_prof.columns) == 1:
                # Reload with header=None to capture the first row correctly
                try:
                    df_prof = pd.read_csv(prof_file, header=None, names=['phi'])
                    phi_col = 'phi'
                except Exception:
                    pass
            
            if phi_col:
                # X축 데이터 결정 (인덱스 vs 별도 컬럼)
                prof_x_col = next((c for c in df_prof.columns if str(c).lower() in ['index', 'x', 'grid']), None)
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
        if len(real_time_ns) > 2: # Need at least 2 points AFTER skipping first
            try:
                # User request: Skip the very first point (initial state)
                fit_t = real_time_ns[1:]
                fit_x = real_pos_nm[1:]
                
                res = linregress(fit_t, fit_x)
                
                # 데이터 범위에 맞춰 선 그리기 (전체 범위에 대해 그리되 fit은 부분 데이터로)
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
    mat = MaterialProperties()
    
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

def run_phase1_calibration(temps: np.ndarray, dx_real: float, force_rerun: bool = False):
    if not force_rerun and os.path.exists(BETA_TABLE_FILE):
        print(f"[INFO] Phase 1: Calibration file found. Skipping.")
        return

    print("[INFO] Phase 1: Starting Beta Calibration...")
    if not compile_cpp_exe(BETA_CPP_SOURCE, BETA_EXE_NAME):
        return

    sim_config = {
        'dx': 1.0, 'xi_sim': 5.0, 'Nx': 2000,
        'max_steps': 500000, 'run_dist_nm': 200.0,
        'dx_real_meters': dx_real, 'max_iter': 15
    }
    
    # temps is passed as argument
    
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
            config: dict,
            dump_dir: str,
            start_step: int = 0,
            monitor: bool = True):
        
        class SimConfig(ctypes.Structure):
            _fields_ = [
                # Doubles (8 bytes)
                ("dx", ctypes.c_double), ("dt", ctypes.c_double),
                ("xi", ctypes.c_double), ("epsilon_penalty", ctypes.c_double),
                ("kappa", ctypes.c_double), ("W", ctypes.c_double),
                ("time_total", ctypes.c_double),
                ("I0", ctypes.c_double),
                ("Qdiff_J", ctypes.c_double),
                ("sigma", ctypes.c_double),
                ("Tm", ctypes.c_double),
                ("Vm", ctypes.c_double),
                ("Hf", ctypes.c_double),
                ("cell_vol", ctypes.c_double),
                ("t_scale", ctypes.c_double),
                
                # Ints (4 bytes)
                ("Nx", ctypes.c_int), ("Ny", ctypes.c_int),
                ("max_grains", ctypes.c_int),
                ("output_interval", ctypes.c_int),  ("use_active_list", ctypes.c_int),
                ("z_nucleation", ctypes.c_int)
            ]

        # Nucleation Params Extraction
        z_nucleation = config.get('z_nucleation', 0)
        nucl_params = config.get('nucl_params', {})
        
        cfg = SimConfig(
            config['dx'], config['dt'],
            config['xi'], config['epsilon_penalty'],
            config['kappa'], config['W'],
            config['time_total'],
            nucl_params.get('I0', 1.0e35), 
            nucl_params.get('Qdiff_J', 0.0),
            nucl_params.get('sigma', 0.3),
            nucl_params.get('Tm', 1687.0),
            nucl_params.get('Vm', 1.2e-5),
            nucl_params.get('Hf', 2.35e9),
            nucl_params.get('cell_vol', 1.0e-27),
            config.get('t_scale', 1.0),
            
            config['Nx'], config['Ny'], 
            config['max_grains'], 
            config['output_interval'], config['use_active_list'],
            z_nucleation
        )

        phi_flat = phi_grid.flatten().astype(np.float64)
        phi_ptr = phi_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Temperature Schedule Arrays
        sched_time_arr = np.array(config.get('sched_time', [0.0]), dtype=np.float64)
        sched_TL_arr = np.array(config.get('sched_TL', [1687.0]), dtype=np.float64)
        sched_TR_arr = np.array(config.get('sched_TR', [1687.0]), dtype=np.float64)
        
        sched_t_ptr = sched_time_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sched_TL_ptr = sched_TL_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sched_TR_ptr = sched_TR_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sched_len = len(sched_time_arr)

        beta_T = np.array(beta_table['Temperature'], dtype=np.float64)
        beta_val = np.array(beta_table['Beta'], dtype=np.float64)
        dG_T = np.array(dG_table['Temperature'], dtype=np.float64)
        dG_val = np.array(dG_table['dG_sim'], dtype=np.float64)

        # Convert dump_dir to bytes for C compatibility
        dump_dir_bytes = dump_dir.encode('utf-8')

        self.lib.run_mpf_simulation.argtypes = [
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
            ctypes.POINTER(SimConfig),
            ctypes.c_char_p,
            ctypes.c_int 
        ]

        print(f"[*] Launching C++ MPF Kernel...")
        print(f"    - Time Total: {config['time_total']:.2f} (dimensionless)")
        print(f"    - Output Interval: {config['output_interval']} steps")
        print(f"    - Dump Directory: {dump_dir}")
        
        # Thread Target Function
        def run_simulation_thread():
             self.lib.run_mpf_simulation(
                phi_ptr, 
                sched_t_ptr, sched_TL_ptr, sched_TR_ptr, sched_len,
                beta_T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                beta_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(beta_T),
                dG_T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                dG_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(dG_T),
                ctypes.byref(cfg),
                dump_dir_bytes,
                start_step
            )

        # Start Thread
        t_start = time.time()
        sim_thread = threading.Thread(target=run_simulation_thread)
        sim_thread.start()

        if monitor:
             log_path = os.path.join(dump_dir, "energy_log.csv")
             time.sleep(1.0) # Wait for file creation
             
             last_step = 0
             
             import re
             
             print("\n[PROGRESS] Monitoring simulation progress...")
             while sim_thread.is_alive():
                 if os.path.exists(log_path):
                     try:
                        # Tail the last few lines to find the latest step
                        with open(log_path, 'r') as f:
                            # Read reasonably large chunk from end if possible, 
                            # but simpler to just readlines if file isn't massive (log is small text)
                            # Better: seek to end?
                            lines = f.readlines()
                            if len(lines) > 1:
                                last_line = lines[-1].strip()
                                parts = last_line.split(',')
                                if len(parts) >= 2 and parts[0].isdigit():
                                    step = int(parts[0])
                                    sim_time = float(parts[1])
                                    
                                    progress = min(1.0, sim_time / config['time_total'])
                                    pct = progress * 100.0
                                    
                                    elapsed = time.time() - t_start
                                    if elapsed > 0 and progress > 0.001:
                                        eta_sec = (elapsed / progress) - elapsed
                                    else:
                                        eta_sec = 0.0
                                    
                                    # Format time
                                    elapsed_str = f"{int(elapsed)//60}m {int(elapsed)%60}s"
                                    eta_str = f"{int(eta_sec)//60}m {int(eta_sec)%60}s"
                                    
                                    # Progress Bar
                                    bar_len = 30
                                    filled_len = int(bar_len * progress)
                                    bar = '=' * filled_len + '-' * (bar_len - filled_len)
                                    
                                    sys.stdout.write(f"\r[{bar}] {pct:5.1f}% | t={sim_time:.2f}/{config['time_total']:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str}   ")
                                    sys.stdout.flush()
                     except Exception:
                         pass
                 time.sleep(0.5)
             print("") # Newline after loop

        sim_thread.join()
        
        elapsed = time.time() - t_start
        sys.stdout.write(f"\r[{'='*30}] 100.0% | t={config['time_total']:.2f}/{config['time_total']:.2f} | Elapsed: {int(elapsed)//60}m {int(elapsed)%60}s | ETA: 0m 0s   \n")
        sys.stdout.flush()
        
        print("[+] C++ Kernel Finished.")

        return phi_flat.reshape(config['max_grains'], config['Nx'], config['Ny'])

    def run_benchmark(self, nx=512, ny=512, steps=1000):
        print(f"[*] Running AVX-512 Benchmark (Nx={nx}, Ny={ny}, Steps={steps})...")
        
        self.lib.run_benchmark_kernel.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double)
        ]
        
        results = (ctypes.c_double * 4)()
        self.lib.run_benchmark_kernel(nx, ny, steps, results)
        
        t_scalar = results[0]
        t_avx = results[1]
        
        print("-" * 40)
        print(f"Scalar Time: {t_scalar:.2f} ms")
        if t_avx > 0:
            print(f"AVX-512 Time: {t_avx:.2f} ms")
            speedup = t_scalar / t_avx
            print(f"Speedup:     {speedup:.2f}x")
        else:
            print("AVX-512 Time: N/A (Not Supported or Disabled)")
        print("-" * 40)


def visualize_energy_log(log_path, output_dir, T_val, epsilon):
    """
    Plots the tracked maximum energy terms over time.
    """
    if not os.path.exists(log_path):
        print(f"[WARN] Energy log not found: {log_path}")
        return

    try:
        df = pd.read_csv(log_path)
        if df.empty: return

        plt.figure(figsize=(10, 6))
        
        plt.plot(df['time'], df['max_drive'], label='Max Drive Force', linewidth=2)
        plt.plot(df['time'], df['max_penalty'], label='Max Penalty Force', linewidth=2, linestyle='--')
        plt.plot(df['time'], df['max_grad'], label='Max Grad (Curvature)', alpha=0.6)
        plt.plot(df['time'], df['max_dw'], label='Max Double-Well', alpha=0.6)
        
        plt.yscale('log')
        plt.title(f"Energy Term Evolution (T={T_val}K, Eps={epsilon:.1f})")
        plt.xlabel("Simulation Time")
        plt.ylabel("Max Force Term Magnitude (Log Scale)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        save_path = os.path.join(output_dir, f"energy_plot_T{int(T_val)}_Eps{int(epsilon)}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[RESULT] Energy plot saved to {save_path}")
        
    except Exception as e:
        print(f"[WARN] Failed to plot energy log: {e}")


def create_animation(dump_dir, frames_dir, output_file, nx_arg, ny_arg, max_grains, fps=30, scale_obj=None, dt_sim=None, temp_schedule=None):
    """
    Creates an MP4 animation from Multi-Array VTK files.
    """
    from sim_utils import load_packed_vtk
    print("[*] Creating animation from VTK files...")
    
    vtk_files = sorted(
        glob.glob(os.path.join(dump_dir, "output_*.vtk")),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )
    
    if not vtk_files:
        print("[WARN] No output_*.vtk files found.")
        return
        
    print(f"[*] Found {len(vtk_files)} VTK files.")
    ensure_dir(frames_dir)
    
    for i, file_path in enumerate(vtk_files):
        try:
            step_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            
            # Use updated loader
            data = load_packed_vtk(file_path)
            arrays = data['arrays']
            nx = data['nx']
            ny = data['ny']
            
            # --- Reconstruct Grain ID Map with Threshold 0.75 ---
            # Default to -1 or 0 for "Liquid/None"
            grain_map = np.full((nx, ny), -1, dtype=int)
            max_phi_map = np.zeros((nx, ny))
            
            # Iterate over all "Phi_*" arrays
            for name, arr in arrays.items():
                if name.startswith("Phi_"):
                    gid = int(name.split('_')[1])
                    # Mask where this grain's phi > 0.75
                    mask = arr > 0.75
                    # Update grain map
                    # If overlapping, last one wins (or could do argmax logic if we tracked max)
                    grain_map[mask] = gid
                    
                    # Track max phi for visualization (optional)
                    max_phi_map = np.maximum(max_phi_map, arr)

            # Get Other Scalar Fields
            sigma_sq = arrays.get("SigmaPhiSq", np.zeros((nx, ny)))
            temp_grid = arrays.get("Temperature", np.zeros((nx, ny)))
            
            # --- Visualization ---
            import matplotlib.gridspec as gridspec
            
            fig = plt.figure(figsize=(10, 15), constrained_layout=True)
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.6], figure=fig)
            
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            
            # Common Tick Settings
            def set_ticks(ax, nx, ny):
                # Major ticks at 100 intervals, minor at 50?
                # Using 100 unit steps for visibility
                ax.set_xticks(np.arange(0, nx + 1, 100))
                ax.set_yticks(np.arange(0, ny + 1, 100))
                ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)

            # Ax1: Grain ID
            cmap_grains = plt.cm.get_cmap('nipy_spectral', max_grains + 1)
            masked_grains = np.ma.masked_where(grain_map < 0, grain_map)
            
            ax1.set_facecolor('black')
            im1 = ax1.imshow(masked_grains.T, cmap=cmap_grains, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=max_grains)
            ax1.set_title(f"Microstructure [Phi > 0.75]", fontsize=14)
            ax1.set_ylabel(r"Y ($\mu$m)")
            ax1.set_xlabel(r"X ($\mu$m)")
            set_ticks(ax1, nx, ny)
            plt.colorbar(im1, ax=ax1, orientation='vertical', label='Grain ID')
            
            # Ax2: SigmaPhiSq
            im2 = ax2.imshow(sigma_sq.T, cmap='plasma', origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1.2)
            ax2.set_title(rf"$\Sigma \phi^2$ (Max: {np.max(sigma_sq):.2f})", fontsize=14)
            ax2.set_ylabel(r"Y ($\mu$m)")
            ax2.set_xlabel(r"X ($\mu$m)")
            set_ticks(ax2, nx, ny)
            plt.colorbar(im2, ax=ax2, orientation='vertical', label=r'$\Sigma \phi^2$')
            
            # Ax3: Temperature
            if np.max(temp_grid) > 0:
                T_min, T_max = np.min(temp_grid), np.max(temp_grid)
            else:
                T_min, T_max = 0, 1
            
            im3 = ax3.imshow(temp_grid.T, cmap='bwr', origin='lower', aspect='auto', interpolation='bilinear')
            ax3.set_title(f"Temperature ({T_min:.1f} K - {T_max:.1f} K)", fontsize=14)
            ax3.set_ylabel(r"Y ($\mu$m)")
            ax3.set_xlabel(r"X ($\mu$m)")
            set_ticks(ax3, nx, ny)
            plt.colorbar(im3, ax=ax3, orientation='vertical', label='Temperature (K)')
            
            # Time Annotation
            if scale_obj and dt_sim:
                try: 
                    real_time = scale_obj.to_real_time(step_num * dt_sim)
                except: 
                    real_time = 0.0
                fig.suptitle(f"Time: {real_time*1e6:.2f} us", fontsize=16)
            else:
                fig.suptitle(f"Step: {step_num}", fontsize=16)

            save_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
            
            if i % 10 == 0:
                print(f"  -> Generated frame {i+1}/{len(vtk_files)}")

        except Exception as e:
            print(f"[ERR] Failed {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print("[+] Frame generation complete.")
    
    # FFmpeg (Optional check)
    try:
        print("[*] Encoding video...")
        cmd = ['ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(frames_dir, "frame_%04d.png"), 
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_file]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[SUCCESS] Saved {output_file}")
    except Exception as e:
        print(f"[WARN] ffmpeg failed or not found: {e}")


def save_total_volume_vtk(dump_dir, output_path):
    """
    Stacks all output_*.vtk files into a single 3D Volume VTK (X, Y, Time).
    Useful for visualizing time evolution as a volume in Paraview.
    """
    import glob
    print(f"[*] Generating Total Volume VTK: {output_path}")
    vtk_files = sorted(
        glob.glob(os.path.join(dump_dir, "output_*.vtk")),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )
    
    if not vtk_files:
        print("[WARN] No VTK files found.")
        return

    # Load first file to get dimensions
    from sim_utils import load_packed_vtk
    first_data = load_packed_vtk(vtk_files[0])
    nx, ny = first_data['nx'], first_data['ny']
    num_frames = len(vtk_files)
    
    print(f"    Dimensions: ({nx}, {ny}, {num_frames})")

    # Allocate 3D Arrays (Time is Z-axis)
    # VTK Structure Points: X fastest, then Y, then Z.
    # Our data: (Nx, Ny) per frame.
    packed_vol = np.zeros((num_frames, nx, ny), dtype=np.float32)
    sigma_vol = np.zeros((num_frames, nx, ny), dtype=np.float32)
    temp_vol = np.zeros((num_frames, nx, ny), dtype=np.float64)
    
    for i, fpath in enumerate(vtk_files):
        data = load_packed_vtk(fpath)
        arrays = data['arrays']
        
        # 1. Reconstruct Packed Data (GrainID + Phi)
        phi_keys = [k for k in arrays.keys() if k.startswith("Phi_")]
        if phi_keys:
            # Sort keys to ensure correct ID mapping
            phi_keys.sort(key=lambda x: int(x.split('_')[1]))
            
            # Stack: (NumGrains, Nx, Ny)
            stack = np.stack([arrays[k] for k in phi_keys], axis=0) # (G, Nx, Ny)
            
            # Find dominant grain
            best_gid = np.argmax(stack, axis=0) # (Nx, Ny)
            best_phi = np.max(stack, axis=0)    # (Nx, Ny)
            
            # Thresholding for ID (if phi is too small, treat as liquid/void ID 0 or keep argmax?)
            # Old logic: if max_phi > 0.5 -> use ID.
            # If max_phi < 0.5 -> ID = 0?
            mask = best_phi < 0.5
            best_gid[mask] = 0
            
            # Packed = ID + Phi
            packed_frame = best_gid.astype(np.float32) + best_phi.astype(np.float32)
        else:
            packed_frame = np.zeros((nx, ny), dtype=np.float32)

        # 2. Other Fields
        sigma = arrays.get('SigmaPhiSq', np.zeros((nx, ny)))
        temp = arrays.get('Temperature', np.zeros((nx, ny)))

        packed_vol[i, :, :] = packed_frame
        sigma_vol[i, :, :] = sigma
        temp_vol[i, :, :] = temp
        
        if i % 10 == 0:
            print(f"    Processed frame {i+1}/{num_frames}", end='\r')

    # Write VTK (Big Endian)
    try:
        with open(output_path, 'wb') as f:
            f.write(b"# vtk DataFile Version 3.0\n")
            f.write(b"Total PhaseField Volume\n")
            f.write(b"BINARY\n")
            f.write(b"DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {ny} {nx} {num_frames}\n".encode('ascii'))
            f.write(b"ORIGIN 0 0 0\n")
            f.write(b"SPACING 1 1 1\n")
            
            total_points = nx * ny * num_frames
            f.write(f"POINT_DATA {total_points}\n".encode('ascii'))
            
            # 1. PackedState
            f.write(b"SCALARS PackedState float 1\n")
            f.write(b"LOOKUP_TABLE default\n")
            f.write(packed_vol.astype('>f4').tobytes())
            
            # 2. Sigma
            f.write(b"\nSCALARS SigmaPhiSq float 1\n")
            f.write(b"LOOKUP_TABLE default\n")
            f.write(sigma_vol.astype('>f4').tobytes())
            
            # 3. Temp
            f.write(b"\nSCALARS Temperature double 1\n")
            f.write(b"LOOKUP_TABLE default\n")
            f.write(temp_vol.astype('>f8').tobytes())
            
        print(f"\n[SUCCESS] Total Volume VTK saved: {output_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save total VTK: {e}")


def run_phase3_simulation(sim_params: dict, resume_file: str = None, animate_only: bool = False):
    print("\n[INFO] Phase 3: Multi-Grain Simulation Starting...")
    
    DUMP_DIR = os.path.join(DATA_DIR, "dump")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")
    
    ensure_dir(DUMP_DIR)
    ensure_dir(FRAMES_DIR)
    
    start_step = 0
    
    if not animate_only:
        if resume_file:
            # Resume Mode: Don't clean old files if they are BEFORE the resume point?
            # Actually safer to clean FUTURE files to avoid mix-up.
            pass 
        else:
            # Fresh Start: Clean old files
            for f in glob.glob(os.path.join(DUMP_DIR, "*.bin")): os.remove(f)
            for f in glob.glob(os.path.join(DUMP_DIR, "*.vtk")): os.remove(f)
            for f in glob.glob(os.path.join(FRAMES_DIR, "*.png")): os.remove(f)
            if os.path.exists(os.path.join(DUMP_DIR, "energy_log.csv")): os.remove(os.path.join(DUMP_DIR, "energy_log.csv"))

    MPF_SOURCE = "multi_grain_simulation.cpp"
    MPF_LIB = "libmpf.dll" if platform.system() == "Windows" else "libmpf.so"
    if not animate_only:
        if not compile_shared_lib(MPF_SOURCE, MPF_LIB): return

    if not os.path.exists(BETA_TABLE_FILE):
        print("[ERROR] Missing calibration data.")
        return
        
    df_beta = pd.read_csv(BETA_TABLE_FILE)
    
    mat = sim_params.get('mat') if 'mat' in sim_params else MaterialProperties()
    
    # Extract params
    Nx = sim_params.get('Nx', 500)
    Ny = sim_params.get('Ny', 100)
    dx_real = sim_params.get('dx_real', 10e-9)
    xi_sim = sim_params.get('xi_sim', 5.0)
    
    # Use config-provided T_left/Right if available (e.g. from static config)
    T_left = sim_params.get('T_left_static', sim_params.get('T_left', 500.0))
    T_right = sim_params.get('T_right_static', sim_params.get('T_right', 1650.0))
    
    if 'scale' in sim_params:
        scale = sim_params['scale']
    else:
        V_ref = mat.get_physical_velocity(T_left) 
        scale = DimensionlessSystem(dx_real, V_ref, mat.sigma)
    
    T_profile = np.linspace(T_left, T_right, Nx)
    
    # Load dG table (computed in Phase 1)
    dG_table_path = os.path.join(DATA_DIR, "dG_table.csv")
    if not os.path.exists(dG_table_path):
         print("[WARN] dG_table.csv not found. Re-calculating temporary table.")
         # Quick table gen
         temps_dg = np.linspace(500, 1800, 1301)
         vals_dg = [scale.to_sim_energy(mat.get_driving_force(t)) for t in temps_dg]
         df_dG = pd.DataFrame({'Temperature': temps_dg, 'dG_sim': vals_dg})
    else:
         df_dG = pd.read_csv(dG_table_path)
         
    # Setup Time Schedule (Nucleation window)
    time_total_phys = sim_params.get('physical_duration_sec', 2.0e-6)
    time_total_sim = time_total_phys / scale.t0
    
    # Schedule setup... (Same as before)
    nucleation_enabled = sim_params.get('z_nucleation', 0)
    nucl_params = sim_params.get('nucl_params', [0.0, 0.0, 0.0, 0.0]) # I0, Q, Tm, Vm
    
    sched_times = sim_params.get('sched_time', [])
    sched_TL = sim_params.get('sched_TL', [])
    sched_TR = sim_params.get('sched_TR', [])
    
    if not sched_times:
         # Default simple static
         sched_times = [0.0, time_total_sim]
         sched_TL = [T_left, T_left]
         sched_TR = [T_right, T_right]
    
    # Config Dictionary for C++
    use_optimization = sim_params.get('use_active_list', 1)
    
    # Determine Penalty
    if 'epsilon_penalty' in sim_params:
        eps_auto = sim_params['epsilon_penalty']
    else:
        safety_factor = sim_params.get('safety_factor', 2.0)
        # Fallback estimation
        T_min = np.min(sched_TL) if len(sched_TL) > 0 else T_left
        if len(sched_TR) > 0 and np.min(sched_TR) < T_min: T_min = np.min(sched_TR)
        eps_auto = determine_required_epsilon(T_min, scale, mat, safety_factor=safety_factor)

    config = {
        'Nx': Nx, 'Ny': Ny,
        'dx': 1.0, 
        'dt': sim_params.get('dt_sim', 0.01), 
        'xi': xi_sim,
        'epsilon_penalty': eps_auto, 
        'kappa': 1.0 * xi_sim / 4.0,
        'W': 12.0 / xi_sim,
        'max_grains': sim_params.get('max_grains', 500), 
        'time_total': time_total_sim,
        'output_interval': sim_params.get('output_interval', 100), 
        'use_active_list': use_optimization,
        # New
        'z_nucleation': nucleation_enabled,
        'nucl_params': nucl_params,
        't_scale': scale.t0,
        'sched_time': sched_times,
        'sched_TL': sched_TL,
        'sched_TR': sched_TR
    }

    if not animate_only:
        verify_and_plot_environment(
            Nx=Nx, 
            dx=config['dx'], 
            T_profile=T_profile, 
            df_beta=df_beta, 
            df_dG=df_dG, 
            config=config, 
            output_dir=DATA_DIR 
        )

    phi_grid = None
    
    if resume_file:
         phi_grid, start_step = resume_from_vtk(resume_file, config)
         if phi_grid is None:
             print("[ERROR] Resume failed. Falling back to fresh start.")
             start_step = 0
             phi_grid = np.zeros((config['max_grains'], Nx, Ny), dtype=np.float64)
         else:
             # Adjust time_total? C++ loop runs from t=0. 
             # We need to tell C++ to start from t_start, or just step offset.
             # Current C++ wrapper `run` initializes t=0 and runs loop.
             # We should probably update `MultiGrainWrapper.run` to accept start_step/start_time.
             # Or just accept that C++ sim time will reset to 0 but physical state is advanced?
             # For correct temperature schedule, C++ needs CORRECT t_curr.
             # t_curr = step * dt.
             print(f"[INFO] Adjusting internal start time to step {start_step}")
    
    if phi_grid is None: # Fresh start
        phi_grid = np.zeros((config['max_grains'], Nx, Ny), dtype=np.float64)
    
    # Dummy args for compatibility
    dummy_T_prof = np.zeros(Nx)
    
    mpf = MultiGrainWrapper(os.path.join(CPP_DIR, MPF_LIB))
    
    # Pass start_step info? 
    # Python MultiGrainWrapper calls `run_mpf_simulation` and passes arguments.
    # The current C++ signature doesn't take 'start_step'. 
    # Adding it would require changing C++ signature AND ctypes argtypes.
    # Quick Workaround: Pass start_step via `sched_time` unused slot? Or just add it properly.
    # Adding properly is better.
    
    if not animate_only:
        mpf.run(phi_grid, dummy_T_prof, df_beta, df_dG, config, DUMP_DIR, start_step=start_step)
    else:
        print("[INFO] Skipping C++ Simulation (--animate-only active).")
        print("[INFO] Proceeding directly to visualization using existing data.")
    
    print("\n--- Generating Microstructure Video ---")
    animation_output_file = os.path.join(DATA_DIR, "microstructure_evolution.mp4")
    create_animation(DUMP_DIR, FRAMES_DIR, animation_output_file, Nx, Ny, config['max_grains'], fps=30,scale_obj=scale, dt_sim=config['dt'])
    
    # Total Volume
    save_total_volume_vtk(DUMP_DIR, os.path.join(DATA_DIR, "total_volume.vtk"))

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


def run_phase3_simulation_line_analysis(sim_params: dict):
    print("\n[INFO] Phase 3: Multi-Grain Simulation Starting...")
    
    DUMP_DIR = os.path.join(DATA_DIR, "dump")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")
    
    ensure_dir(DUMP_DIR)
    ensure_dir(FRAMES_DIR)
    
    df_beta = pd.read_csv(BETA_TABLE_FILE)
    df_nucl = pd.read_csv(KMC_EVENTS_FILE)
    
    mat = MaterialProperties()
    Nx = sim_params.get('Nx', 500)
    Ny = sim_params.get('Ny', 100)
    dx_real = sim_params.get('dx_real', 10e-9)
    xi_sim = sim_params.get('xi_sim', 10.0)
    
    V_ref = mat.get_physical_velocity(sim_params.get('T_left', 500.0)) 
    scale = DimensionlessSystem(dx_real, V_ref, mat.sigma)
    
    T_left = sim_params.get('T_left', 500.0)
    T_right = sim_params.get('T_right', 1650.0)
    T_profile = np.linspace(T_left, T_right, Nx)
    
    T_lookup = np.linspace(500, 1650, 1151)
    dG_lookup = [scale.to_sim_energy(mat.get_driving_force(t)) for t in T_lookup]
    df_dG = pd.DataFrame({'Temperature': T_lookup, 'dG_sim': dG_lookup})
    
    physical_duration_sec = sim_params.get('physical_duration_sec', 1.0e-7)
    time_total_sim = physical_duration_sec / scale.t0

    use_optimization = sim_params.get('use_active_list', 1)
    
    if 'epsilon_penalty' in sim_params:
         eps_val = sim_params['epsilon_penalty']
    else:
         eps_val = 20.0
         
    config = {
        'Nx': Nx, 'Ny': Ny,
        'dx': 1.0, 
        'dt': sim_params.get('dt_sim', 0.01), 
        'xi': xi_sim,
        'epsilon_penalty': eps_val,
        'kappa': 1.0 * xi_sim / 4.0,
        'W': 12.0 / xi_sim,
        'max_grains': sim_params.get('max_grains', 100), 
        'time_total': time_total_sim,
        'output_interval': sim_params.get('output_interval', 50),
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

def run_unit_test_collision(config_override=None, T_val=1000.0):
    """
    C++ 커널을 사용하여 1D 상에서 두 Grain의 충돌을 테스트합니다.
    Overlap 여부를 정량적으로 검증합니다.
    T_val: Simulation Temperature (K)
    """
    print(f"\n[UNIT TEST] Starting 1D Two-Grain Collision Test (T={T_val}K)...")
    
    MPF_SOURCE = "multi_grain_simulation.cpp"
    TEST_DIR = os.path.join(DATA_DIR, "unit_test_collision")
    ensure_dir(TEST_DIR)
    
    for f in glob.glob(os.path.join(TEST_DIR, "*.bin")):
        os.remove(f)

    # 1. Physics & Scale Setup
    mat = MaterialProperties()
    dx_real = 10.0e-9
    
    # Calculate physical velocity at T_val for V_ref
    V_phys = mat.get_physical_velocity(T_val)
    if V_phys <= 0: V_phys = 1e-12
    
    
    # Scale System based on T_val specific velocity
    scale = DimensionlessSystem(dx_real, V_phys, mat.sigma)
    
    # [NEW] Auto-determine Epsilon if not provided in override
    eps_val = 20.0 # Default fallback
    if config_override and 'epsilon_penalty' in config_override:
        eps_val = config_override['epsilon_penalty']
    else:
        eps_val = determine_required_epsilon(T_val, scale, mat, safety_factor=2.0)
        
    # 2. 1D 테스트를 위한 설정 (Ny=1로 하여 1D 효과)
    # 기본 설정에서 필요한 부분만 덮어씁니다.
    nx_test = 200
    ny_test = 1
    
    test_config = {
        'Nx': nx_test, 'Ny': ny_test,
        'dx': 1.0, 'dt': 0.05,
        'xi': 5.0, 
        'epsilon_penalty': eps_val, # Auto-determined value
        'kappa': 1.0 * 5.0 / 4.0, # xi=5.0 기준
        'W': 12.0 / 5.0,
        'max_grains': 3, # 0(배경), 1(좌), 2(우)
        'time_total': 500.0,
        'output_interval': 100,
        'use_active_list': 0 # 전체 계산 (안전하게)
    }
    
    # 사용자 오버라이드 적용
    if config_override:
        test_config.update(config_override)
        # kappa, W 등 파생 변수 재계산 필요시 로직 추가
        if 'xi' in config_override:
             xi = config_override['xi']
             test_config['kappa'] = 1.0 * xi / 4.0
             test_config['W'] = 12.0 / xi

    # 3. 초기 조건 생성 (좌우 Grain 배치)
    phi_grid = np.zeros((test_config['max_grains'], nx_test, ny_test), dtype=np.float64)
    
    # Grain 1: 왼쪽 1/4 지점 (x=50)에 위치
    phi_grid[1, 20:60, :] = 1.0
    
    # Grain 2: 오른쪽 3/4 지점 (x=150)에 위치
    phi_grid[2, 140:180, :] = 1.0
    
    # 4. 물성치 기반 데이터 생성 (온도, dG 등)
    # 균일한 온도장 설정 (충돌만 보기 위해)
    T_profile = np.full(nx_test, T_val, dtype=np.float64)
    
    # Lookup Table 생성 (physics based)
    T_lookup = np.linspace(500, 1650, 1151)
    beta_lookup = [1.0 for _ in T_lookup] # Beta는 1.0으로 고정 (Calibration 불필요한 테스트)
    dG_lookup = [scale.to_sim_energy(mat.get_driving_force(t)) for t in T_lookup]
    
    df_beta = pd.DataFrame({'Temperature': T_lookup, 'Beta': beta_lookup}) 
    df_dG = pd.DataFrame({'Temperature': T_lookup, 'dG_sim': dG_lookup})
    
    # Display Driving Force for Debugging
    current_dG = scale.to_sim_energy(mat.get_driving_force(T_val))
    print(f" -> T={T_val}K, V_phys={V_phys:.2e} m/s, dG_sim={current_dG:.4f}, E0={scale.E0:.2e}")

    # 5. C++ 커널 실행
    MPF_LIB = "libmpf.dll" if platform.system() == "Windows" else "libmpf.so"
    
    mpf_lib_path = os.path.join(CPP_DIR, MPF_LIB)
    if not os.path.exists(mpf_lib_path):
        print(f"[*] Library not found. Compiling {MPF_SOURCE}...")
        if not compile_shared_lib(MPF_SOURCE, MPF_LIB):
             return None

    mpf = MultiGrainWrapper(mpf_lib_path)
    
    mpf.run(phi_grid, T_profile, df_beta, df_dG, test_config, TEST_DIR)
    
    # 6. 결과 분석 및 Time-lapse 시각화
    dump_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.bin")))
    
    # [추가] Time-lapse Plot 설정
    plt.figure(figsize=(10, 6))
    colors_g1 = cm.Blues(np.linspace(0.4, 1.0, len(dump_files)))
    colors_g2 = cm.Reds(np.linspace(0.4, 1.0, len(dump_files)))
    
    max_linear_sum = 0.0
    max_sq_sum = 0.0
    
    for i, fpath in enumerate(dump_files):
        with open(fpath, 'rb') as f:
            phi_flat = np.fromfile(f, dtype=np.float64, count=test_config['max_grains']*nx_test*ny_test)
        
        grid = phi_flat.reshape(test_config['max_grains'], nx_test, ny_test)
        phi1 = grid[1, :, 0]
        phi2 = grid[2, :, 0]
        
        # 지표 갱신
        max_linear_sum = max(max_linear_sum, np.max(phi1 + phi2))
        max_sq_sum = max(max_sq_sum, np.max(phi1**2 + phi2**2))
        
        # [추가] 5스텝마다 프로파일 그리기 (너무 많으면 복잡하므로)
        if i % 2 == 0 or i == len(dump_files)-1:
            plt.plot(phi1, color=colors_g1[i], alpha=0.6, linewidth=1)
            plt.plot(phi2, color=colors_g2[i], alpha=0.6, linewidth=1)

    # Time-lapse 그래프 꾸미기
    plt.title(f"Collision Process (T={T_val}K, xi={test_config['xi']}, eps={test_config['epsilon_penalty']})")
    plt.xlabel("Grid Index")
    plt.ylabel("Phi")
    plt.ylim(-0.05, 1.05)
    # 대표 범례
    plt.plot([], [], color='blue', label='Grain 1 (Time ->)')
    plt.plot([], [], color='red', label='Grain 2 (Time ->)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장 (각 조건별 폴더나 이름으로 구분)
    plot_filename = f"profile_T{int(T_val)}_xi{test_config['xi']}_eps{test_config['epsilon_penalty']:.1f}.png"
    plt.savefig(os.path.join(os.path.dirname(TEST_DIR), plot_filename))
    plt.close()

    # [NEW] Visualize Energy Log
    visualize_energy_log(os.path.join(TEST_DIR, "energy_log.csv"), os.path.dirname(TEST_DIR), T_val, test_config['epsilon_penalty'])

    return {
        'xi': test_config['xi'],
        'eps': test_config['epsilon_penalty'],
        'max_linear_sum': max_linear_sum,
        'max_sq_sum': max_sq_sum
    }


def optimize_penalty(T_val=1000.0):
    # Auto-mode test: just run once with auto-determined epsilon
    print("[*] Running Optimization with Auto-Determined Penalty...")
    run_unit_test_collision(config_override=None, T_val=T_val)

def visualize_optimization_results(results_list, output_dir):
    """
    최적화 결과를 히트맵으로 시각화합니다.
    X축: Epsilon (Penalty), Y축: Xi (Interface Width)
    Color: Max Linear Sum (1.0에 가까울수록 좋음)
    """
    if not results_list:
        return

    df = pd.DataFrame(results_list)
    
    # Pivot Table 생성
    pivot_table = df.pivot(index='xi', columns='eps', values='max_linear_sum')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_table, cmap='RdYlGn_r', aspect='auto', origin='lower') # Red-Yellow-Green (Reverse)
    
    # 축 설정
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    plt.xlabel('Epsilon (Penalty Strength)', fontsize=12)
    plt.ylabel('Xi (Interface Width)', fontsize=12)
    plt.title('Collision Test: Max Overlap (Linear Sum)', fontsize=14)
    
    # 값 표시
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            color = 'white' if val > 1.2 else 'black'
            plt.text(j, i, f"{val:.4f}", ha='center', va='center', color=color, fontweight='bold')
            
    plt.colorbar(label='Max Linear Sum (Phi1 + Phi2)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_heatmap.png"))
    plt.close()
    print(f"[RESULT] Optimization heatmap saved to {output_dir}")

def main(animate_only=False):
    ensure_dir(DATA_DIR)
    ensure_dir(CPP_DIR)
    
    # --- Global Simulation Parameters ---
    dx_real = 10e-9
    
    # --- Step 1: Calibration (Phase 1) ---
    print("\n=== Phase 1 Setup ===")
    calibration_temps = np.arange(700, 1201, 50)
    if not animate_only:
        run_phase1_calibration(temps=calibration_temps, dx_real=dx_real, force_rerun=True)
    else:
        print("[INFO] Skipping Phase 1 Calibration (--animate-only active).")
    
    # --- Step 2: Nucleation (Phase 2) ---
    # print("\n=== Phase 2 Setup ===")
    # p2_config = {
    #     'Nx': 500, 'Ny': 100,
    #     'T_left': 700.0, 'T_right': 1200.0,
    #     'dx_real': dx_real
    # }
    # run_phase2_nucleation(config=p2_config, force_rerun=True)
    print("\n=== Phase 2: Skipped (Using Internal Nucleation in Phase 3) ===")
    
    # --- Step 3: Simulation (Phase 3) ---
    print("\n=== Phase 3 Setup ===")
    p3_config = {
        'Nx': 500, 'Ny': 100,
        'dx_real': dx_real,
        'xi_sim': 5.0,
        'T_left': 700.0, 'T_right': 1200.0,
        'dt_sim': 0.01,
        'physical_duration_sec': 5.0e-6, 
        'safety_factor': 1.2,
        'max_grains': 100,
        'use_active_list': 1,
        'output_interval': 100
    }
    
    run_phase3_simulation(p3_config, animate_only=animate_only)
    
    # Optional Diagnostics (Uncomment to run)
    # optimize_penalty(T_val=1200.0)
    # run_phase3_simulation_vis(p3_config)
    # run_phase3_simulation_line_analysis(p3_config)

def main_configured():
    parser = argparse.ArgumentParser(description="Multi-Grain Phase Field Simulation")
    parser.add_argument('--config', type=str, help='Path to config.yaml', default='config.yaml')
    parser.add_argument('--animate-only', action='store_true', help='Skip simulation and only run animation/visualization on existing dump files.')
    parser.add_argument('--benchmark', action='store_true', help='Run AVX-512 Performance Benchmark and exit.')
    args, unknown = parser.parse_known_args()

    if args.benchmark:
        ensure_dir(CPP_DIR)
        MPF_SOURCE = "multi_grain_simulation.cpp"
        MPF_LIB = "libmpf.dll" if platform.system() == "Windows" else "libmpf.so"
        
        if not compile_shared_lib(MPF_SOURCE, MPF_LIB):
            sys.exit(1)
            
        try:
            mpf = MultiGrainWrapper(os.path.join(CPP_DIR, MPF_LIB))
            mpf.run_benchmark(512, 512, 2000)
        except Exception as e:
            print(f"[ERROR] Benchmark failed: {e}")
        return

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"[WARN] Config file {config_path} not found. Falling back to legacy hardcoded defaults.")
        main(animate_only=args.animate_only)
        return

    ensure_dir(DATA_DIR)
    ensure_dir(CPP_DIR)
    print(f"\n=== Running in Config Mode: {config_path} ===")
    sim_config = load_sim_config(config_path)
    
    # Phase 1: Ensure Calibration Exists
    # calibration_temps = np.arange(500, 1651, 50) 
    # run_phase1_calibration(temps=calibration_temps, dx_real=sim_config['dx_real'], force_rerun=True)
    
    # Phase 3: Run
    resume_file = None
    for arg in sys.argv:
        if arg.startswith("--resume="):
            resume_file = arg.split("=", 1)[1]
            break
            
    run_phase3_simulation(sim_config, resume_file=resume_file, animate_only=args.animate_only)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare_nucleation":
        # Run Validation Mode
        validation_config = {
            'Nx': 500, 'Ny': 100,
            'T_left': 700.0, 'T_right': 1400.0,
            'dx_real': 10e-9,
            'dt_real': 1e-11 
        }
        run_nucleation_comparison(validation_config)
    else:
        main_configured()
