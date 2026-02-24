#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Memory Alignment Helpers ---
void* aligned_alloc_wrapper(size_t index_alignment, size_t size) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, index_alignment);
#else
    return std::aligned_alloc(index_alignment, size);
#endif
}

void aligned_free_wrapper(void* ptr) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// Branchless clamp
static inline double clamp01(double x) {
    return std::fmin(1.0, std::fmax(0.0, x));
}

// --- Configuration Structures ---
struct SimConfigCH {
    double dx, dt;
    double M;          // Mobility
    double kappa;      // Gradient energy coefficient
    double W;          // Double-well potential height (Barrier)
    double time_total;
    double c_ref;      // Reference physical concentration scale
    
    int geom_type;     // 0: cube, 1: solid_cylinder, 2: hollow_cylinder
    double R_in;
    double R_out;
    double gamma_in;
    double gamma_out;
    
    // Ints
    int Nx, Ny, Nz;
    int output_interval;
};

template <typename T>
void SwapEndian(T& val) {
    char* valPtr = reinterpret_cast<char*>(&val);
    std::reverse(valPtr, valPtr + sizeof(T));
}

static void write_vtk_scalar(int step, int Nx, int Ny, int Nz, 
                             double dx, double c_ref,
                             const double* c_flat, 
                             const char* dump_dir) {
    if (dump_dir == nullptr) return;
    std::ostringstream fn;
    fn << dump_dir << "/output_" << std::setw(6) << std::setfill('0') << step << ".vtk";
    
    std::ofstream out(fn.str(), std::ios::binary);
    if (!out) {
        std::cerr << "[ERROR] Failed to open VTK file: " << fn.str() << "\n";
        return;
    }

    // 1. VTK Header
    out << "# vtk DataFile Version 3.0\n";
    out << "Cahn-Hilliard Sim Data\n";
    out << "BINARY\n";
    // Paraview expects points ordered with X varying fastest, then Y, then Z.
    // In our tensor, i is X, j is Y, k is Z.
    // DIMENSIONS usually take Nx Ny Nz
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n"; 
    out << "ORIGIN 0 0 0\n";
    out << "SPACING " << dx << " " << dx << " " << dx << "\n"; 
    out << "POINT_DATA " << (size_t)Nx * (size_t)Ny * (size_t)Nz << "\n";

    // 2. Scalar Field c
    out << "SCALARS Concentration float 1\n";
    out << "LOOKUP_TABLE default\n";

    std::vector<float> buffer((size_t)Nx * (size_t)Ny * (size_t)Nz);
    
    // VTK expects points in order:
    // for Z, for Y, for X.
    // Our raw array `c_flat` is flattened from Python as shape (Nx, Ny, Nz) -> idx = i*(Ny*Nz) + j*Nz + k.
    // Wait, Python's c_grid.flatten() on shape (Nx, Ny, Nz) lays out Z fastest.
    // So c_flat[i, j, k] = i*(Ny*Nz) + j*Nz + k.
    // Let's rewrite the layout everywhere to be standard contiguous.
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                // VTK point index
                size_t p_vtk = (size_t)k * (Nx * Ny) + (size_t)j * Nx + (size_t)i;
                // Python/C++ array index
                size_t p_arr = (size_t)i * (Ny * Nz) + (size_t)j * Nz + (size_t)k;
                buffer[p_vtk] = (float)(c_flat[p_arr] * c_ref);
            }
        }
    }

    // Endian swap
    for (size_t k = 0; k < buffer.size(); ++k) {
        SwapEndian(buffer[k]);
    }
    out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
    out << "\n";

    out.close();
}

static void write_log(int step, double t, double total_mass, double free_energy, const char* dump_dir) {
    if (dump_dir == nullptr) return;
    std::string log_path = std::string(dump_dir) + "/sim_log.csv";
    bool new_file = false;
    std::ifstream check(log_path);
    if (!check.good()) new_file = true;
    check.close();

    std::ofstream out(log_path, std::ios::app);
    if (!out) return;

    if (new_file) {
        out << "step,time,total_mass,free_energy\n";
    }
    out << step << "," << t << "," << total_mass << "," << free_energy << "\n";
    out.close();
}

// --- Main Simulation Kernel ---
extern "C" {
    void run_ch_simulation(
        double* c_flat,       // Initial Concentration Field (Nx * Ny)
        SimConfigCH* cfg_ptr, // Configuration
        const char* dump_dir,
        int start_step
    ) {
        if (!c_flat || !cfg_ptr) return;
        SimConfigCH cfg = *cfg_ptr;
        
        const int Nx = cfg.Nx;
        const int Ny = cfg.Ny;
        const int Nz = cfg.Nz;
        const size_t total_size = (size_t)Nx * (size_t)Ny * (size_t)Nz;
        
        // Characteristic scales for Non-dimensionalization
        double Lc = std::sqrt(cfg.kappa / cfg.W);
        double tc = cfg.kappa / (cfg.M * cfg.W * cfg.W);
        
        // Dimensionless spacing and time step
        double dx_dl = cfg.dx / Lc;
        double dt_dl = cfg.dt / tc;
        double inv_dx2_dl = 1.0 / (dx_dl * dx_dl);
        
        // Chemical Potential mu_dl = 2*c*(1-c)(1-2c) - lap_dl(c)
        // dc/dt_dl = lap_dl(mu_dl)
        
        // Allocate buffers
        double* mu = (double*)aligned_alloc_wrapper(64, total_size * sizeof(double));
        double* new_c = (double*)aligned_alloc_wrapper(64, total_size * sizeof(double));
        
        #ifdef _OPENMP
        #pragma omp parallel
        #pragma omp single
        std::cout << "[OMP] threads=" << omp_get_num_threads() << "\n";
        #endif
        
        std::cout << "[INFO] Starting 3D Cahn-Hilliard Simulation...\n";
        std::cout << "       Nx=" << Nx << ", Ny=" << Ny << ", Nz=" << Nz << ", dt=" << cfg.dt << " (dl: " << dt_dl << "), dx=" << cfg.dx << " (dl: " << dx_dl << ")\n";
        std::cout << "       M=" << cfg.M << ", kappa=" << cfg.kappa << ", W=" << cfg.W << "\n";
        std::cout << "       Characteristic Length Lc=" << Lc << ", Characteristic Time tc=" << tc << "\n";

        double t_curr = start_step * cfg.dt;
        int step = start_step;

        // Write initial state
        if (step == 0) {
            write_vtk_scalar(0, Nx, Ny, Nz, cfg.dx, cfg.c_ref, c_flat, dump_dir);
        }
        
        // Shape from python is (Nx, Ny, Nz) contiguous in memory (C-style)
        // idx = i * (Ny * Nz) + j * Nz + k
        auto idx = [&](int i, int j, int k) -> size_t {
            return (size_t)i * (Ny * Nz) + (size_t)j * Nz + (size_t)k;
        };

        // Helper to check if a physical coordinate is inside the domain
        auto is_inside = [&](int i, int j, int k) -> bool {
            if (cfg.geom_type == 0) return true; // Cube (all)
            double x = (i - Nx/2.0 + 0.5) * cfg.dx;
            double y = (j - Ny/2.0 + 0.5) * cfg.dx;
            double r = std::sqrt(x*x + y*y);
            if (cfg.geom_type == 1) return r <= cfg.R_out; // Solid
            if (cfg.geom_type == 2) return r >= cfg.R_in && r <= cfg.R_out; // Hollow
            return true;
        };

        // Helper to determine gamma for a specific boundary
        auto get_gamma = [&](int i, int j, int k) -> double {
            if (cfg.geom_type == 0) return cfg.gamma_out; // Cube
            double x = (i - Nx/2.0 + 0.5) * cfg.dx;
            double y = (j - Ny/2.0 + 0.5) * cfg.dx;
            double r = std::sqrt(x*x + y*y);
            if (cfg.geom_type == 1) return cfg.gamma_out; // Solid
            // Hollow: distinguish inner vs outer boundary by radius
            double mid_r = (cfg.R_in + cfg.R_out) / 2.0;
            return (r < mid_r) ? cfg.gamma_in : cfg.gamma_out; 
        };

        while (t_curr < cfg.time_total) {
            
            // 1. Compute Chemical Potential mu
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int k = 0; k < Nz; ++k) {
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        if (!is_inside(i, j, k)) continue;
                        
                        size_t curr = idx(i, j, k);
                        double c = c_flat[curr];
                        
                        // Z is periodic, X and Y use Ghost nodes if they touch boundary
                        int im = (i == 0) ? (Nx - 1) : (i - 1);
                        int ip = (i == Nx - 1) ? 0 : (i + 1);
                        int jm = (j == 0) ? (Ny - 1) : (j - 1);
                        int jp = (j == Ny - 1) ? 0 : (j + 1);
                        int km = (k == 0) ? (Nz - 1) : (k - 1);
                        int kp = (k == Nz - 1) ? 0 : (k + 1);
                        
                        double c_ip = is_inside(ip, j, k) ? c_flat[idx(ip, j, k)] : c - (get_gamma(i,j,k) / cfg.kappa) * cfg.dx;
                        double c_im = is_inside(im, j, k) ? c_flat[idx(im, j, k)] : c - (get_gamma(i,j,k) / cfg.kappa) * cfg.dx;
                        double c_jp = is_inside(i, jp, k) ? c_flat[idx(i, jp, k)] : c - (get_gamma(i,j,k) / cfg.kappa) * cfg.dx;
                        double c_jm = is_inside(i, jm, k) ? c_flat[idx(i, jm, k)] : c - (get_gamma(i,j,k) / cfg.kappa) * cfg.dx;
                        double c_kp = is_inside(i, j, kp) ? c_flat[idx(i, j, kp)] : c - (get_gamma(i,j,k) / cfg.kappa) * cfg.dx;
                        double c_km = is_inside(i, j, km) ? c_flat[idx(i, j, km)] : c - (get_gamma(i,j,k) / cfg.kappa) * cfg.dx;
                        
                        double lap_c_dl = (c_ip + c_im + c_jp + c_jm + c_kp + c_km - 6.0 * c) * inv_dx2_dl;
                                        
                        double dfdc_dl = 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c);
                        mu[curr] = dfdc_dl - lap_c_dl;
                    }
                }
            }
            
            // 2. Update Concentration c
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int k = 0; k < Nz; ++k) {
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        if (!is_inside(i, j, k)) continue;
                        
                        size_t curr = idx(i, j, k);
                        double c = c_flat[curr];
                        double mu_c = mu[curr];
                        
                        int im = (i == 0) ? (Nx - 1) : (i - 1);
                        int ip = (i == Nx - 1) ? 0 : (i + 1);
                        int jm = (j == 0) ? (Ny - 1) : (j - 1);
                        int jp = (j == Ny - 1) ? 0 : (j + 1);
                        int km = (k == 0) ? (Nz - 1) : (k - 1);
                        int kp = (k == Nz - 1) ? 0 : (k + 1);
                        
                        // Mass conservation at boundary: grad mu = 0 => mu_ghost = mu_curr
                        double mu_ip = is_inside(ip, j, k) ? mu[idx(ip, j, k)] : mu_c;
                        double mu_im = is_inside(im, j, k) ? mu[idx(im, j, k)] : mu_c;
                        double mu_jp = is_inside(i, jp, k) ? mu[idx(i, jp, k)] : mu_c;
                        double mu_jm = is_inside(i, jm, k) ? mu[idx(i, jm, k)] : mu_c;
                        double mu_kp = is_inside(i, j, kp) ? mu[idx(i, j, kp)] : mu_c;
                        double mu_km = is_inside(i, j, km) ? mu[idx(i, j, km)] : mu_c;
                        
                        double lap_mu_dl = (mu_ip + mu_im + mu_jp + mu_jm + mu_kp + mu_km - 6.0 * mu_c) * inv_dx2_dl;
                        new_c[curr] = c + dt_dl * lap_mu_dl;
                    }
                }
            }
            
            // 3. Commit
            // Update points inside geometry, keeping points outside unchanged
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int k = 0; k < Nz; ++k) {
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        if (is_inside(i, j, k)) {
                            size_t curr = idx(i, j, k);
                            c_flat[curr] = new_c[curr];
                        }
                    }
                }
            }
            
            t_curr += cfg.dt;
            step++;
            
            // 4. Output & Logging
            if (cfg.output_interval > 0 && (step % cfg.output_interval == 0)) {
                double total_mass = 0.0;
                double free_energy = 0.0;
                
                #ifdef _OPENMP
                #pragma omp parallel for reduction(+:total_mass, free_energy)
                #endif
                for (int k = 0; k < Nz; ++k) {
                    for (int i = 0; i < Nx; ++i) {
                        for (int j = 0; j < Ny; ++j) {
                            if (!is_inside(i, j, k)) continue;
                            size_t curr = idx(i, j, k);
                            double c = c_flat[curr];
                            total_mass += (c * cfg.c_ref) * cfg.dx * cfg.dx * cfg.dx; // 3D volume
                            
                            double bulk = cfg.W * c * c * (1.0 - c) * (1.0 - c);
                            free_energy += bulk * cfg.dx * cfg.dx * cfg.dx;  
                        }
                    }
                }
                
                write_log(step, t_curr, total_mass, free_energy, dump_dir);
                write_vtk_scalar(step, Nx, Ny, Nz, cfg.dx, cfg.c_ref, c_flat, dump_dir);
            }
        }
        
        aligned_free_wrapper(mu);
        aligned_free_wrapper(new_c);
    }
}
