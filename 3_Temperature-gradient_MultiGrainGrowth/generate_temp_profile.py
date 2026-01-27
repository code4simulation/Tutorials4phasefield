import numpy as np
import pandas as pd
import os

def generate_profile():
    # Configuration
    duration = 5.0e-6 # match run_simulation.py default
    steps = 50
    
    # Time points
    times = np.linspace(0, duration, steps + 1)
    
    # Linear interpolation
    # T_left: 500 -> 800
    t_left = np.linspace(500, 800, steps + 1)
    
    # T_right: 500 -> 1400
    t_right = np.linspace(500, 1400, steps + 1)
    
    # Save
    df = pd.DataFrame({
        'Time': times,
        'T_left': t_left,
        'T_right': t_right
    })
    
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "temperature_profile.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}")
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    generate_profile()
