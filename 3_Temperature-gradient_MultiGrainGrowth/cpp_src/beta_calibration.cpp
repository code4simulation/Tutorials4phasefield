#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>

// --- Configuration Structure ---
struct Config {
    int Nx;
    double dx, dt;        // Dimensionless
    double beta_correction;
    double dG_sim;        // Dimensionless Driving Force
    double V_sim_target;  // Dimensionless Target Velocity
    double xi_sim;
    int max_steps;
    double run_dist_sim;
    int interface_pos;
    std::string history_file; // Output file for time-pos history
    std::string profile_file; // Output file for final phi profile
};

// --- Statistics Helper ---
struct VelocityResult {
    double slope;
    double r_squared;
    double final_dist;
};

VelocityResult calculate_velocity_robust(const std::vector<double>& times, 
                                         const std::vector<double>& positions, 
                                         int min_points = 5) {
    size_t n = times.size();
    if (n < 2) return {0.0, 0.0, 0.0};

    // Use the last 15 points or all if less (increased window for robustness)
    size_t window_size = (n < 15) ? n : 15;
    if (window_size < (size_t)min_points) window_size = n;

    double sum_t = 0.0, sum_x = 0.0, sum_tx = 0.0, sum_tt = 0.0;
    double mean_x = 0.0;

    for (size_t i = n - window_size; i < n; ++i) {
        double t = times[i];
        double x = positions[i];
        sum_t += t;
        sum_x += x;
        sum_tx += t * x;
        sum_tt += t * t;
        mean_x += x;
    }
    mean_x /= window_size;

    double denominator = window_size * sum_tt - sum_t * sum_t;
    double slope = 0.0;
    double intercept = 0.0;

    if (std::abs(denominator) > 1e-12) {
        slope = (window_size * sum_tx - sum_t * sum_x) / denominator;
        intercept = (sum_x - slope * sum_t) / window_size;
    }

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (size_t i = n - window_size; i < n; ++i) {
        double fit = slope * times[i] + intercept;
        ss_res += (positions[i] - fit) * (positions[i] - fit);
        ss_tot += (positions[i] - mean_x) * (positions[i] - mean_x);
    }
    double r2 = (ss_tot > 0) ? (1.0 - ss_res / ss_tot) : 0.0;

    return {slope, r2, positions.back() - positions[0]};
}

// --- Main Simulation Logic ---
int main(int argc, char* argv[]) {
    // Expected args: Nx dx xi_sim dG_sim V_target beta max_steps run_dist_sim history_file profile_file
    if (argc < 11) {
        std::cerr << "Usage: ./beta_calibration Nx dx xi_sim dG_sim V_target beta max_steps run_dist_sim history_file profile_file" << std::endl;
        return 1;
    }

    Config c;
    try {
        c.Nx = std::stoi(argv[1]);
        c.dx = std::stod(argv[2]);
        c.xi_sim = std::stod(argv[3]);
        c.dG_sim = std::stod(argv[4]);
        c.V_sim_target = std::stod(argv[5]);
        c.beta_correction = std::stod(argv[6]);
        c.max_steps = std::stoi(argv[7]);
        c.run_dist_sim = std::stod(argv[8]);
        c.history_file = argv[9];
        c.profile_file = argv[10];
        c.interface_pos = 100;
    } catch (...) {
        std::cerr << "Error parsing arguments." << std::endl;
        return 1;
    }

    // --- Derived Parameters for f_doublewell = W * p^2 * (p - 1)^2 ---
    double W_sim = 12.0 / c.xi_sim;
    double kappa_sim = 3.0 * c.xi_sim / 4.0; 

    double L_sim = 0.0;
    if (c.dG_sim > 1e-9) {
        L_sim = (c.V_sim_target / c.dG_sim) / c.beta_correction;
    }

    double dt_diff = (c.dx * c.dx) / (4.0 * L_sim * kappa_sim);
    double V_est = L_sim * c.dG_sim * c.beta_correction;
    if (V_est < 1e-9) V_est = 1e-9;
    double dt_adv = 0.2 * c.dx / V_est;
    c.dt = std::min(dt_diff, dt_adv);

    // --- Initialization ---
    std::vector<double> phi(c.Nx);
    std::vector<double> new_phi(c.Nx);

    for (int i = 0; i < c.Nx; ++i) {
        if (i < c.interface_pos - 10) phi[i] = 1.0;
        else if (i > c.interface_pos + 10) phi[i] = 0.0;
        else {
            phi[i] = 0.5 * (1.0 - std::tanh((i - c.interface_pos) / (c.xi_sim / 2.0)));
        }
    }

    // --- Tracking setup ---
    double start_pos = c.interface_pos * c.dx;
    double target_dist = c.run_dist_sim;
    double current_pos = start_pos;
    double last_sample_pos = start_pos;
    double sampling_interval = target_dist / 40.0; // More frequent sampling for better plots

    std::vector<double> times;
    std::vector<double> positions;

    times.push_back(0.0);
    positions.push_back(start_pos);

    // --- Time Stepping Loop ---
    double t_curr = 0.0;
    double inv_dx2 = 1.0 / (c.dx * c.dx);

    for (int step = 0; step < c.max_steps; ++step) {
        // Update Field
        for (int i = 0; i < c.Nx; ++i) {
            int ip = (i == c.Nx - 1) ? i : i + 1;
            int im = (i == 0) ? i : i - 1;

            double val = phi[i];
            double lap = (phi[ip] - 2.0 * val + phi[im]) * inv_dx2;

            // --- MODIFIED HERE ---
            // Old: double term1 = 2.0 * W_sim * val * (1.0 - val) * (1.0 - 2.0 * val);
            // New: p^3 - p form (Standard symmetric double-well derivative)
            // Note: Depending on specific W definition, a factor of 2 or 4 might be implicit.
            // Assuming direct replacement as per request: term1 = W_sim * (p^3 - p)
            double term1 = 2.0 * W_sim * val * (1.0 - val) * (1.0 - 2.0 * val);
            //double term1 = W_sim * (val * val * val - val); 
            // ---------------------

            double term2 = 30.0 * (val * val) * ((1.0 - val) * (1.0 - val)) * c.dG_sim;

            // Equation: dphi/dt = L * (kappa * lap - term1 + term2)
            double rhs = L_sim * (kappa_sim * lap - term1 + term2);
            new_phi[i] = val + c.dt * rhs;

            if (new_phi[i] > 1.0) new_phi[i] = 1.0;
            if (new_phi[i] < 0.0) new_phi[i] = 0.0;
        }

        std::swap(phi, new_phi);
        t_curr += c.dt;

        // Sample Position
        if (step % 50 == 0) {
            int idx = -1;
            for (int i=0; i < c.Nx-1; ++i){
                if (phi[i] >= 0.5 && phi[i+1] < 0.5) {
                    idx = i;
                    break;
                }
            }
            if (idx != -1) {
                double y1 = phi[idx];
                double y2 = phi[idx+1];
                double fraction = (0.5 - y1) / (y2 - y1 + 1e-12);
                current_pos = (idx + fraction) * c.dx;

                if ((current_pos - last_sample_pos) >= sampling_interval) {
                    times.push_back(t_curr);
                    positions.push_back(current_pos);
                    last_sample_pos = current_pos;

                    if ((current_pos - start_pos) >= target_dist) {
                        break;
                    }
                }
            }
        }
    }

    times.push_back(t_curr);
    positions.push_back(current_pos);

    // --- Save Detailed Data ---
    // 1. History File (csv)
    std::ofstream f_hist(c.history_file);
    if (f_hist.is_open()) {
        f_hist << "time,position\n";
        for (size_t i=0; i<times.size(); ++i) {
            f_hist << times[i] << "," << positions[i] << "\n";
        }
        f_hist.close();
    }

    // 2. Profile File (last step)
    std::ofstream f_prof(c.profile_file);
    if (f_prof.is_open()) {
        for (int i=0; i<c.Nx; ++i) {
            f_prof << phi[i] << "\n";
        }
        f_prof.close();
    }

    // --- Calculate Final Stats ---
    VelocityResult res = calculate_velocity_robust(times, positions);

    // Output strictly in this format for Python parsing
    // Format: SLOPE R2 FINAL_DIST
    std::cout << std::fixed << std::setprecision(6) 
              << res.slope << " " << res.r_squared << " " << res.final_dist << std::endl;

    return 0;
}
