import yaml
import os
import sys
import numpy as np
from sim_utils import DimensionlessSystem, determine_required_epsilon, MaterialProperties

def load_sim_config(yaml_path):
    """
    Loads configuration from YAML, calculates derived parameters,
    and returns a structured dictionary ready for the simulation.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)

    # 1. Initialize Material & Physics (Defaults)
    mat = MaterialProperties()
    
    # Override material properties if provided in YAML
    if 'material' in raw_cfg:
        m_cfg = raw_cfg['material']
        mat.Tm = float(m_cfg.get('Tm_K', mat.Tm))
        mat.Vm = float(m_cfg.get('Vm_m3_mol', mat.Vm))
        mat.sigma = float(m_cfg.get('sigma_J_m2', mat.sigma))
        mat.Hf = float(m_cfg.get('Hf_J_mol', mat.Hf))
        
        if 'kinetics' in m_cfg:
            k_cfg = m_cfg['kinetics']
            mat.V0 = float(k_cfg.get('V0_m_s', mat.V0))
            mat.Qdiff_eV = float(k_cfg.get('Q_eV', mat.Qdiff_eV))
            mat.I0 = float(k_cfg.get('I0', mat.I0))
        # Support legacy "diffusivity" key if present
        elif 'diffusivity' in m_cfg:
             d_cfg = m_cfg['diffusivity']
             mat.I0 = float(d_cfg.get('I0', mat.I0))
             mat.Qdiff_eV = float(d_cfg.get('Q_eV', mat.Qdiff_eV))

    # 2. Domain & Geometry
    dom = raw_cfg['domain']
    nx = dom['nx']
    ny = dom['ny']
    dx_real = float(dom['dx_real_m'])
    
    # 3. Temperature (Needed for V_ref)
    temp_cfg = raw_cfg['temperature']
    method = temp_cfg.get('method', 'static') # Default to static if not specified
    
    if method == 'static':
        t_left = float(temp_cfg['static']['T_left_K'])
        t_right = float(temp_cfg['static']['T_right_K'])
        temp_type_val = 'static_gradient'
        csv_path = ''
    elif method == 'dynamic':
        temp_type_val = 'dynamic_csv'
        csv_path = temp_cfg['dynamic']['csv_path']
        
        if os.path.exists(csv_path):
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                # Assume 2nd col is T_left
                t_left = df.iloc[0, 1]
                t_right = df.iloc[0, 2] # just for info
            except Exception as e:
                print(f"[WARN] Failed to read CSV {csv_path}: {e}")
                t_left = 1680.0
                t_right = 1690.0
        else:
            print(f"[WARN] CSV {csv_path} not found. Using default T=1680K for scaling.")
            t_left = 1680.0
            t_right = 1690.0
    else:
        print(f"[WARN] Unknown temperature method '{method}', defaulting to static.")
        t_left = 1680.0
        t_right = 1690.0
        temp_type_val = 'static_gradient'
        csv_path = ''

    # 4. Dimensionless Scaling
    v_ref = mat.get_physical_velocity(t_left)
    scale = DimensionlessSystem(dx_real, v_ref, mat.sigma)
    
    # 5. Time Integration
    time_cfg = raw_cfg['time_integration']
    dt_sim = time_cfg['dt_sim']
    phys_duration = float(time_cfg['physical_duration_sec'])
    time_total_sim = phys_duration / scale.t0
    
    # 6. Penalty Coefficient (Auto-Calculation)
    pf_cfg = raw_cfg['phase_field']
    eps_input = pf_cfg['epsilon_penalty']
    
    if eps_input == 'auto':
        t_min_prox = 500.0 # Conservative check
        safety = pf_cfg.get('safety_factor', 2.0)
        epsilon = determine_required_epsilon(t_min_prox, scale, mat, safety_factor=safety)
    else:
        epsilon = float(eps_input)

    # 7. Construct Final Config Dictionary
    nucl_cfg = raw_cfg['nucleation']
    
    # Unit Conversions for Nucleation Params
    Q_J = mat.Qdiff_eV * 1.60217663e-19
    
    processed_config = {
        'Nx': nx, 'Ny': ny,
        'dx_real': dx_real,
        'dt_sim': dt_sim,
        'time_total_sim': time_total_sim,
        'physical_duration_sec': phys_duration,
        
        # Phase Field
        'xi_sim': pf_cfg['xi_sim'],
        'epsilon_penalty': epsilon,
        'kappa': 1.0 * pf_cfg['xi_sim'] / 4.0, 
        
        'max_grains': dom['max_grains'],
        'output_interval': time_cfg['output_interval_steps'],
        
        # Temperature
        'temp_type': temp_type_val,
        'temp_csv': csv_path,
        'T_left_static': t_left if temp_type_val == 'static_gradient' else 0,
        'T_right_static': t_right if temp_type_val == 'static_gradient' else 0,
        
        # Nucleation
        'z_nucleation': 1 if nucl_cfg['enabled'] else 0,
        'nucl_params': {
            'I0': mat.I0,
            'Qdiff_J': Q_J,
            'sigma': mat.sigma,
            'Tm': mat.Tm,
            'Vm': mat.Vm,
            'Hf': mat.Hf, # Passing J/mol 
            'cell_vol': dx_real**3
        },
        
        # Paths
        'paths': raw_cfg['paths'],
        
        # Objects
        'scale': scale,
        'mat': mat
    }
    
    return processed_config
