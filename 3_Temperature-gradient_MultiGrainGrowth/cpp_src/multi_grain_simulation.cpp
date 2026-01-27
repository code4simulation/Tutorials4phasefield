// cpp_src/multi_grain_simulation.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <random>
#include <immintrin.h> // AVX intrinsics
#include <chrono>

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

// Global flag for AVX-512
static bool g_use_avx512 = false;

// AVX/SIMD Check
void check_cpu_features() {
    #if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    bool support_avx2 = __builtin_cpu_supports("avx2");
    bool support_avx512f = __builtin_cpu_supports("avx512f");
    
    std::cout << "[CPU] AVX2 Supported: " << (support_avx2 ? "YES" : "NO") << "\n";
    std::cout << "[CPU] AVX512F Supported: " << (support_avx512f ? "YES" : "NO") << "\n";
    
    if (support_avx512f) {
        g_use_avx512 = true;
        std::cout << "[CPU] Using Explicit AVX-512 optimization path.\n";
    }
    #else
    // Windows/MSVC specific check could be added here, assuming AVX2 min if compiling with flags
    std::cout << "[CPU] Feature detection skipped (non-GCC compiler). Assuming compiled flags apply.\n";
    #endif
}

// Branchless clamp
static inline double clamp01(double x) {
    return std::fmin(1.0, std::fmax(0.0, x));
}

// --- Configuration Structures ---
struct SimConfig {
    // Doubles (8 bytes)
    double dx, dt;
    double xi;
    double epsilon_penalty;
    double kappa;
    double W;
    double time_total;
    double I0;
    double Qdiff_J; 
    double sigma;
    double Tm;
    double Vm;
    double Hf;   
    double cell_vol; 
    double t_scale; 

    // Ints (4 bytes)
    int Nx, Ny;
    int max_grains;
    int output_interval;
    int use_active_list;
    int z_nucleation; 
};

struct TemperatureSchedule {
    std::vector<double> times;
    std::vector<double> T_left;
    std::vector<double> T_right;

    void get_temperatures(double t, double& out_TL, double& out_TR) const {
        if (times.empty()) {
            out_TL = 1687.0; out_TR = 1687.0; return;
        }
        if (t <= times.front()) {
            out_TL = T_left.front();
            out_TR = T_right.front();
            return;
        }
        if (t >= times.back()) {
            out_TL = T_left.back();
            out_TR = T_right.back();
            return;
        }
        
        auto it = std::lower_bound(times.begin(), times.end(), t);
        int idx = (int)std::distance(times.begin(), it);
        double t1 = times[idx-1];
        double t2 = times[idx];
        double ratio = (t - t1) / (t2 - t1);
        
        out_TL = T_left[idx-1] + ratio * (T_left[idx] - T_left[idx-1]);
        out_TR = T_right[idx-1] + ratio * (T_right[idx] - T_right[idx-1]);
    }
};

struct LookUpTable {
    std::vector<double> T;
    std::vector<double> val;

    double interpolate(double tq) const {
        if (T.empty()) return 1.0;
        if (tq <= T.front()) return val.front();
        if (tq >= T.back()) return val.back();
        auto it = std::lower_bound(T.begin(), T.end(), tq);
        int idx = (int)std::distance(T.begin(), it);
        double t1 = T[idx - 1], t2 = T[idx];
        double v1 = val[idx - 1], v2 = val[idx];
        double a = (tq - t1) / (t2 - t1);
        return v1 + (v2 - v1) * a;
    }
};

template <typename T>
    void SwapEndian(T& val) {
        char* valPtr = reinterpret_cast<char*>(&val);
        std::reverse(valPtr, valPtr + sizeof(T));
    }

    static void write_vtk_packed(int step, int Nx, int Ny, int max_grains, 
                                 const double* phi_flat, 
                                 const std::vector<double>& T_curr, 
                                 const std::vector<double>& sigma_sq,
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
        out << "PhaseField Sim Data\n";
        out << "BINARY\n";
        out << "DATASET STRUCTURED_POINTS\n";
        out << "DIMENSIONS " << Ny << " " << Nx << " 1\n"; 
        out << "ORIGIN 0 0 0\n";
        out << "SPACING 1 1 1\n"; 
        out << "POINT_DATA " << Nx * Ny << "\n";

        // 2. Individual Phi Fields using raw loops
        std::vector<float> buffer(Nx * Ny);

        for (int g = 0; g < max_grains; ++g) {
            out << "SCALARS Phi_" << g << " float 1\n";
            out << "LOOKUP_TABLE default\n";

            size_t base_offset = (size_t)g * (size_t)Nx * (size_t)Ny;
            
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    buffer[(size_t)i * (size_t)Ny + (size_t)j] = (float)phi_flat[base_offset + (size_t)i * (size_t)Ny + (size_t)j];
                }
            }

            // Endian swap
            for (size_t k = 0; k < buffer.size(); ++k) {
                SwapEndian(buffer[k]);
            }
            out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
            out << "\n"; // Optional newline for safety between arrays
        }

        // 3. SigmaPhiSq -> FLOAT
        out << "SCALARS SigmaPhiSq float 1\n";
        out << "LOOKUP_TABLE default\n";
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                 buffer[(size_t)i * (size_t)Ny + (size_t)j] = (float)sigma_sq[(size_t)i * (size_t)Ny + (size_t)j];
            }
        }

        for (size_t k = 0; k < buffer.size(); ++k) {
             SwapEndian(buffer[k]);
        }
        out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
        out << "\n";

        // 4. Temperature Field
        out << "SCALARS Temperature float 1\n";
        out << "LOOKUP_TABLE default\n";
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < Nx; ++i) {
            double T_val = T_curr[i]; 
            for (int j = 0; j < Ny; ++j) {
                buffer[(size_t)i * (size_t)Ny + (size_t)j] = (float)T_val;
            }
        }

        for (size_t k = 0; k < buffer.size(); ++k) {
             SwapEndian(buffer[k]);
        }
        out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
        out << "\n";

        out.close();
    }

    static void write_energy_log(int step, double t, double max_grad, double max_dw, double max_drive, double max_penalty, const char* dump_dir) {
    if (dump_dir == nullptr) return;
    std::string log_path = std::string(dump_dir) + "/energy_log.csv";
    bool new_file = false;
    std::ifstream check(log_path);
    if (!check.good()) new_file = true;
    check.close();

    std::ofstream out(log_path, std::ios::app);
    if (!out) return;

    if (new_file) {
        out << "step,time,max_grad,max_dw,max_drive,max_penalty\n";
    }
    out << step << "," << t << "," << max_grad << "," << max_dw << "," << max_drive << "," << max_penalty << "\n";
    out.close();
}

// --- AVX-512 Optimized Kernel ---
#if defined(__AVX512F__)
static void update_grain_avx512(
    int Nx, int Ny,
    const double* phi,      // Current Grain Phi (Read)
    double* new_phi,        // Next Grain Phi (Write)
    const double* sigma_sq, // Sigma Squared Field (Read)
    const double* dGx,      // Driving Force per Row
    const double* Lx,       // Mobility per Row
    const SimConfig& cfg,
    double inv_dx2
) {
    // Constants Broadcast
    __m512d v_inv_dx2 = _mm512_set1_pd(inv_dx2);
    __m512d v_kappa = _mm512_set1_pd(cfg.kappa);
    __m512d v_W = _mm512_set1_pd(2.0 * cfg.W);
    __m512d v_epsilon_penalty = _mm512_set1_pd(2.0 * cfg.epsilon_penalty);
    __m512d v_drive_pre = _mm512_set1_pd(30.0);
    __m512d v_dt = _mm512_set1_pd(cfg.dt);
    __m512d v_one = _mm512_set1_pd(1.0);
    __m512d v_two = _mm512_set1_pd(2.0);
    __m512d v_zero = _mm512_setzero_pd();
    
    // Rows
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nx; ++i) {
        double L = Lx[i];
        if (L < 1e-12) {
            // Optimization: If L is 0, just copy old to new, or skip if assuming initialized
            // We must copy because 'new_phi' needs to be valid state
            // Use AVX copy for speed if aligned
            const double* src_row = phi + (size_t)i * Ny;
            double* dst_row = new_phi + (size_t)i * Ny;
            std::memcpy(dst_row, src_row, Ny * sizeof(double));
            continue;
        }

        double dG = dGx[i];
        __m512d v_L = _mm512_set1_pd(L);
        __m512d v_dG = _mm512_set1_pd(dG);

        int im = (i == 0) ? 0 : (i - 1);
        int ip = (i == Nx - 1) ? (Nx - 1) : (i + 1);

        const double* ptr_im = phi + (size_t)im * Ny;
        const double* ptr_ip = phi + (size_t)ip * Ny;
        const double* ptr_curr = phi + (size_t)i * Ny;
        const double* ptr_sig = sigma_sq + (size_t)i * Ny;
        double* ptr_out = new_phi + (size_t)i * Ny;

        int j = 0;
        // Main Vector Loop
        for (; j <= Ny - 8; j += 8) {
            // Load Neighbors
            // i-1, i+1 (Vertical)
            __m512d v_im = _mm512_loadu_pd(&ptr_im[j]);
            __m512d v_ip = _mm512_loadu_pd(&ptr_ip[j]);
            
            // i, j (Center)
            __m512d v_p = _mm512_loadu_pd(&ptr_curr[j]);
            
            // Horizontal Neighbors (j-1, j+1)
            // Handling boundary for vector load is tricky.
            // Approach: Load unaligned offset -1 and +1?
            // Only valid if j > 0 and j < Ny-8.
            // For Safety in periodic/boundary, let's construct explicit or safe loads.
            // Optimized: Load [j-1...j+6] and [j+1...j+8]
            
            __m512d v_jm, v_jp;
            
            if (j > 0 && j < Ny - 8) {
                // Safe to load unaligned offsets
                v_jm = _mm512_loadu_pd(&ptr_curr[j - 1]);
                v_jp = _mm512_loadu_pd(&ptr_curr[j + 1]);
            } else {
                 // Boundary fallback for first/last block
                 // Element 0 needs j-1 -> Ny-1 (wrapped) or clamp
                 // We will just do manual gathering for edge vectors or simpler:
                 // Construct array of 8 doubles for jm and jp
                 double tmp_jm[8], tmp_jp[8];
                 for(int k=0; k<8; ++k) {
                     int jk = j + k;
                     int j_minus = (jk == 0) ? (Ny - 1) : (jk - 1);
                     int j_plus = (jk == Ny - 1) ? 0 : (jk + 1);
                     tmp_jm[k] = ptr_curr[j_minus];
                     tmp_jp[k] = ptr_curr[j_plus];
                 }
                 v_jm = _mm512_loadu_pd(tmp_jm);
                 v_jp = _mm512_loadu_pd(tmp_jp);
            }

            // Laplacian
            // (ip + im + jp + jm - 4*p) * inv_dx2
            __m512d v_sum = _mm512_add_pd(v_im, v_ip);
            v_sum = _mm512_add_pd(v_sum, v_jm);
            v_sum = _mm512_add_pd(v_sum, v_jp);
            __m512d v_lap = _mm512_mul_pd(_mm512_sub_pd(v_sum, _mm512_mul_pd(_mm512_set1_pd(4.0), v_p)), v_inv_dx2);
            
            // Term Grad
            __m512d v_term_grad = _mm512_mul_pd(v_kappa, v_lap);

            // Term DW
            // 2 * W * p * (1-p) * (1-2p)
            __m512d v_1_sub_p = _mm512_sub_pd(v_one, v_p);
            __m512d v_1_sub_2p = _mm512_sub_pd(v_one, _mm512_mul_pd(v_two, v_p));
            __m512d v_term_dw = _mm512_mul_pd(v_W, _mm512_mul_pd(v_p, _mm512_mul_pd(v_1_sub_p, v_1_sub_2p)));

            // Term Drive
            // 30 * p^2 * (1-p)^2 * dG
            __m512d v_p2 = _mm512_mul_pd(v_p, v_p);
            __m512d v_1_sub_p_2 = _mm512_mul_pd(v_1_sub_p, v_1_sub_p);
            __m512d v_term_drive = _mm512_mul_pd(v_drive_pre, _mm512_mul_pd(v_p2, _mm512_mul_pd(v_1_sub_p_2, v_dG)));

            // Penalty
            __m512d v_sig = _mm512_loadu_pd(&ptr_sig[j]);
            __m512d v_sig_others = _mm512_sub_pd(v_sig, v_p2);
            v_sig_others = _mm512_max_pd(v_zero, v_sig_others); // max(0, sig - p^2)
            __m512d v_term_pen = _mm512_mul_pd(v_epsilon_penalty, v_sig_others);
            
            // B = L * term_pen
            __m512d v_B = _mm512_mul_pd(v_L, v_term_pen);
            
            // Explicit Force = L * (grad - dw + drive)
            __m512d v_force = _mm512_mul_pd(v_L, _mm512_add_pd(_mm512_sub_pd(v_term_grad, v_term_dw), v_term_drive));

            // Update
            // numer = p + dt * force
            __m512d v_numer = _mm512_add_pd(v_p, _mm512_mul_pd(v_dt, v_force));
            // denom = 1 + dt * B
            __m512d v_denom = _mm512_add_pd(v_one, _mm512_mul_pd(v_dt, v_B));
            
            __m512d v_new = _mm512_div_pd(v_numer, v_denom);
            
            // Clamp
            v_new = _mm512_min_pd(v_one, _mm512_max_pd(v_zero, v_new));
            
            // Store
            _mm512_storeu_pd(&ptr_out[j], v_new);
        }

        // Tail Loop (Scalar)
        for (; j < Ny; ++j) {
             double p = ptr_curr[j];
             int jm = (j == 0) ? (Ny - 1) : (j - 1);
             int jp = (j == Ny - 1) ? 0 : (j + 1);
             
             double val_im = ptr_im[j];
             double val_ip = ptr_ip[j];
             double val_jm = ptr_curr[jm];
             double val_jp = ptr_curr[jp];
             
             double lap = (val_ip + val_im + val_jp + val_jm - 4.0 * p) * inv_dx2;
             double term_grad = cfg.kappa * lap;
             double term_dw = 2.0 * cfg.W * p * (1.0 - p) * (1.0 - 2.0 * p);
             double term_drive = 30.0 * p * p * (1.0 - p) * (1.0 - p) * dG;
             
             double sig = ptr_sig[j];
             double sig_others = std::fmax(0.0, sig - p*p);
             double B = L * (2.0 * cfg.epsilon_penalty * sig_others);
             double explicit_force = L * (term_grad - term_dw + term_drive);
             
             double numer = p + cfg.dt * explicit_force;
             double denom = 1.0 + cfg.dt * B;
             ptr_out[j] = clamp01(numer / denom);
        }
    }
}
#endif

// --- Main Simulation Kernel ---
extern "C" {
    void run_mpf_simulation(
        double* phi_flat,
        // Temperature Schedule Inputs
        const double* sched_time, const double* sched_TL, const double* sched_TR, int sched_len,
        // Material LUTs
        const double* beta_T, const double* beta_val, int beta_len,
        const double* dG_T, const double* dG_val, int dG_len,
        // Config
        SimConfig* cfg_ptr,
        const char* dump_dir,
        int start_step
    ) {
        // Validation
        if (!phi_flat || !sched_time || !cfg_ptr) return;
        SimConfig cfg = *cfg_ptr;
        if (cfg.Nx <= 0 || cfg.Ny <= 0 || cfg.max_grains <= 0) return;

        const int Nx = cfg.Nx;
        const int Ny = cfg.Ny;
        const size_t plane_size = (size_t)Nx * (size_t)Ny;
        const size_t total_elems = (size_t)cfg.max_grains * plane_size;
        const double inv_dx2 = 1.0 / (cfg.dx * cfg.dx);
        
        // Constants for Nucleation
        const double kB = 1.380649e-23;
        const double term_pre = 16.0 * 3.1415926535 * std::pow(cfg.sigma, 3.0) / 3.0;
        const double dt_phys = cfg.dt * cfg.t_scale;

        // Initialize Lookups and Schedule
        TemperatureSchedule ts;
        ts.times.assign(sched_time, sched_time + sched_len);
        ts.T_left.assign(sched_TL, sched_TL + sched_len);
        ts.T_right.assign(sched_TR, sched_TR + sched_len);

        LookUpTable beta_lut, dG_lut;
        beta_lut.T.assign(beta_T, beta_T + beta_len);
        beta_lut.val.assign(beta_val, beta_val + beta_len);
        dG_lut.T.assign(dG_T, dG_T + dG_len);
        dG_lut.val.assign(dG_val, dG_val + dG_len);

        #ifdef _OPENMP
        #pragma omp parallel
        #pragma omp single
        std::cout << "[OMP] threads=" << omp_get_num_threads() << "\n";
        #endif

        check_cpu_features();

        if (cfg.z_nucleation) {
            std::cout << "[INFO] Stochastic Nucleation ENABLED in C++.\n";
            std::cout << "       dt_phys=" << dt_phys << " s (dt_sim=" << cfg.dt << ", scale=" << cfg.t_scale << ")\n";
        }

        // Feature Vectors
        // Feature Vectors
        std::vector<double> Lx(Nx, 0.0);
        std::vector<double> dGx(Nx, 0.0);
        std::vector<double> T_curr(Nx, 0.0);
        std::vector<double> NuclRate(Nx, 0.0);

        // Aligned Memory Allocation
        double* sigma_sq = (double*)aligned_alloc_wrapper(64, plane_size * sizeof(double));
        double* new_phi = (double*)aligned_alloc_wrapper(64, total_elems * sizeof(double));
        
        // Zero init
        std::memset(sigma_sq, 0, plane_size * sizeof(double));
        // new_phi acts as a swap buffer, init to 0 is good practice but we overwrite usually
        std::memset(new_phi, 0, total_elems * sizeof(double));

        // Active List
        std::vector<int> active_grains;
        active_grains.reserve(std::min(cfg.max_grains, 1024));
        
        // Initial Active Check
        for (int g = 1; g < cfg.max_grains; ++g) {
            bool is_active = false;
            size_t offset = (size_t)g * plane_size;
            for(size_t i=0; i<plane_size; ++i) {
                if (phi_flat[offset + i] > 0.001) {
                    is_active = true;
                    break;
                }
            }
            if(is_active) active_grains.push_back(g);
        }

        auto idx = [&](int g, int i, int j) -> size_t {
            return (size_t)g * plane_size + (size_t)i * (size_t)Ny + (size_t)j;
        };

        // RNG
        std::mt19937 gen(12345u); 
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        int step = start_step;
        double t_curr = step * cfg.dt;
        
        if (start_step > 0) {
            std::cout << "[INFO] Resuming simulation from Step " << step << " (t=" << t_curr << ")\n";
        }

        int grain_counter = 1; 
        if (!active_grains.empty()) {
            grain_counter = *std::max_element(active_grains.begin(), active_grains.end()) + 1;
        }

        // --- Step 0 Initialization ---
        // 1. Temperature Init
        double TL0, TR0;
        ts.get_temperatures(0.0, TL0, TR0);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < Nx; ++i) {
            double x_ratio = (double)i / (double)(Nx - 1);
            T_curr[i] = TL0 + x_ratio * (TR0 - TL0);
        }

        // 2. Sigma Sq Init
        {
             std::fill(sigma_sq, sigma_sq + plane_size, 0.0);
             #ifdef _OPENMP
             #pragma omp parallel
             #endif
             {
                 #ifdef _OPENMP
                 #pragma omp for schedule(static) nowait
                 #endif
                 for (size_t p = 0; p < plane_size; ++p) sigma_sq[p] = 0.0;
                 
                 #ifdef _OPENMP
                 #pragma omp barrier
                 #endif

                 for (int gid : active_grains) {
                     const double* phi_ptr = &phi_flat[(size_t)gid * plane_size];
                     #ifdef _OPENMP
                     #pragma omp for schedule(static)
                     #endif
                     for (size_t p = 0; p < plane_size; ++p) {
                         sigma_sq[p] += phi_ptr[p] * phi_ptr[p];
                     }
                 }
             }
        }

        std::vector<double> sigma_sq_vec(sigma_sq, sigma_sq + plane_size);
        if (start_step == 0) {
            write_vtk_packed(0, Nx, Ny, cfg.max_grains, phi_flat, T_curr, sigma_sq_vec, dump_dir);
        }

        // --- Time Loop ---
        while (t_curr < cfg.time_total) {
            // 1. Update Temperature Profile & Parameters
            double TL, TR;
            ts.get_temperatures(t_curr * cfg.t_scale, TL, TR); 
            
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int i = 0; i < Nx; ++i) {
                double x_ratio = (double)i / (double)(Nx - 1);
                double T_local = TL + x_ratio * (TR - TL);
                T_curr[i] = T_local;

                double beta = beta_lut.interpolate(T_local);
                double dG = dG_lut.interpolate(T_local);
                dGx[i] = dG;
                
                double L = 0.0;
                if (dG > 1e-12) L = (1.0 / dG) / beta;
                Lx[i] = L;

                // Nucleation Rate I(T)
                double I_val = 0.0;
                if (cfg.z_nucleation && T_local < cfg.Tm) {
                    double dGv_p = (cfg.Hf * (cfg.Tm - T_local)) / (cfg.Tm * cfg.Vm);
                    if (dGv_p > 1.0) { 
                        double dG_star = term_pre / (dGv_p * dGv_p);
                        double term_diff = std::exp(-cfg.Qdiff_J / (kB * T_local));
                        double term_thermo = std::exp(-dG_star / (kB * T_local));
                        I_val = cfg.I0 * term_diff * term_thermo;
                    }
                }
                NuclRate[i] = I_val; 
            }

            // 2. Stochastic Nucleation
            if (cfg.z_nucleation && grain_counter < cfg.max_grains) {
                std::vector<std::pair<int,int>> new_nuclei;
                for (int i = 0; i < Nx; ++i) {
                    double rate = NuclRate[i];
                    if (rate < 1e-20) continue;
                    double prob = 1.0 - std::exp(-rate * cfg.cell_vol * dt_phys);
                    
                    for (int j = 0; j < Ny; ++j) {
                        if (dis(gen) < prob) {
                            new_nuclei.push_back({i, j});
                        }
                    }
                }

                for (auto& p : new_nuclei) {
                     int nx = p.first;
                     int ny = p.second;
                     double sum_sq = sigma_sq[(size_t)nx*Ny + ny];
                     
                     if (sum_sq < 0.1 && grain_counter < cfg.max_grains) {
                         int new_gid = grain_counter++;
                         active_grains.push_back(new_gid);
                         const int R = std::max(1, (int)std::lround(cfg.xi));
                         for (int dx = -R; dx <= R; ++dx) {
                             for (int dy = -R; dy <= R; ++dy) {
                                 int cx = nx + dx;
                                 int cy = ny + dy;
                                 if (cx >= 0 && cx < Nx) {
                                     if (cy < 0) cy += Ny;
                                     if (cy >= Ny) cy -= Ny;
                                     if (dx*dx + dy*dy <= R*R) {
                                         phi_flat[idx(new_gid, cx, cy)] = 1.0;
                                     }
                                 }
                             }
                         }
                     }
                }
            }

            // 3. Sigma Sq Update
            const std::vector<int>& compute_grains = active_grains;
            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                #ifdef _OPENMP
                #pragma omp for schedule(static) nowait
                #endif
                for (size_t p = 0; p < plane_size; ++p) sigma_sq[p] = 0.0;

                #ifdef _OPENMP
                #pragma omp barrier
                #endif

                for (int gid : compute_grains) {
                    const double* phi_ptr = &phi_flat[(size_t)gid * plane_size];
                    #ifdef _OPENMP
                    #pragma omp for simd schedule(static)
                    #endif
                    for (size_t p = 0; p < plane_size; ++p) {
                        sigma_sq[p] += phi_ptr[p] * phi_ptr[p];
                    }
                }
            }

            // 4. Update Grains (AVX-512 vs Scalar)
            if (g_use_avx512) {
                #if defined(__AVX512F__)
                for (int gid : compute_grains) {
                    update_grain_avx512(Nx, Ny, 
                        &phi_flat[(size_t)gid * plane_size], 
                        &new_phi[(size_t)gid * plane_size], 
                        sigma_sq, dGx.data(), Lx.data(), cfg, inv_dx2);
                }
                #endif
            } else {
                // Fallback Scalar/OpenMP Loop
                for (int gid : compute_grains) {
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(static)
                    #endif
                    for (int i = 0; i < Nx; ++i) {
                        const double L = Lx[i];
                        if (L < 1e-12) {
                            size_t base_idx = idx(gid, i, 0);
                            for (int j=0; j<Ny; ++j) new_phi[base_idx + j] = phi_flat[base_idx + j];
                            continue;
                        }

                        int im = (i == 0) ? 0 : (i - 1);
                        int ip = (i == Nx - 1) ? (Nx - 1) : (i + 1);
                        
                        // Optimized Inner Loop
                        #pragma omp simd
                        for (int j = 0; j < Ny; ++j) {
                            size_t curr_idx = idx(gid, i, j);
                            double p = phi_flat[curr_idx];
                            int jm = (j == 0) ? (Ny - 1) : (j - 1);
                            int jp = (j == Ny - 1) ? 0 : (j + 1);

                            double val_im = phi_flat[idx(gid, im, j)];
                            double val_ip = phi_flat[idx(gid, ip, j)];
                            double val_jm = phi_flat[idx(gid, i, jm)];
                            double val_jp = phi_flat[idx(gid, i, jp)];

                            double lap = (val_ip + val_im + val_jp + val_jm - 4.0 * p) * inv_dx2;
                            double term_grad = cfg.kappa * lap;
                            double term_dw = 2.0 * cfg.W * p * (1.0 - p) * (1.0 - 2.0 * p);
                            double term_drive = 30.0 * p * p * (1.0 - p) * (1.0 - p) * dGx[i];
                            
                            double sig = sigma_sq[(size_t)i * Ny + j];
                            double sig_others = std::fmax(0.0, sig - p*p);
                            double B = L * (2.0 * cfg.epsilon_penalty * sig_others);
                            double explicit_force = L * (term_grad - term_dw + term_drive);
                            
                            double numer = p + cfg.dt * explicit_force;
                            double denom = 1.0 + cfg.dt * B;
                            new_phi[curr_idx] = clamp01(numer / denom);
                        }
                    }
                }
            }

            // 5. Commit
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int k = 0; k < (int)compute_grains.size(); ++k) {
                int gid = compute_grains[(size_t)k];
                std::memcpy(&phi_flat[(size_t)gid * plane_size],
                            &new_phi[(size_t)gid * plane_size],
                            plane_size * sizeof(double));
            }

            t_curr += cfg.dt;
            step++;
            
            if (cfg.output_interval > 0 && (step % cfg.output_interval == 0)) {
                
                // Compute Stats for Log
                double max_grad = 0.0, max_dw = 0.0, max_drive = 0.0, max_penalty = 0.0;
                
                #ifdef _OPENMP
                #pragma omp parallel for reduction(max:max_grad, max_dw, max_drive, max_penalty)
                #endif
                for (int gx = 0; gx < (int)active_grains.size(); ++gx) {
                    int gid = active_grains[gx];
                    for (int i = 0; i < Nx; ++i) {
                         for(int j=0; j<Ny; ++j) {
                            double p = phi_flat[idx(gid, i, j)];
                            if(p < 0.001) continue;
                         }
                    }
                }
                
                 for (int gx = 0; gx < (int)active_grains.size(); ++gx) {
                    int gid = active_grains[gx];
                    for (int i = 0; i < Nx; ++i) {
                        for (int j = 0; j < Ny; ++j) {
                            const double p = phi_flat[idx(gid, i, j)];
                            if (p < 0.001) continue; 
                            
                            int im = (i == 0) ? 0 : (i - 1);
                            int ip = (i == Nx - 1) ? (Nx - 1) : (i + 1);
                            int jm = (j == 0) ? (Ny - 1) : (j - 1);
                            int jp = (j == Ny - 1) ? 0 : (j + 1);

                            const double lap =
                                (phi_flat[idx(gid, ip, j)] + phi_flat[idx(gid, im, j)] +
                                 phi_flat[idx(gid, i, jp)] + phi_flat[idx(gid, i, jm)] - 4.0 * p) * inv_dx2;
                            
                            double term_grad = std::abs(cfg.kappa * lap);
                            double term_dw = std::abs(2.0 * cfg.W * p * (1.0 - p) * (1.0 - 2.0 * p));
                            double term_drive = std::abs(30.0 * p * p * (1.0 - p) * (1.0 - p) * dGx[i]);
                            
                            const double sig = sigma_sq[(size_t)i*(size_t)Ny + (size_t)j];
                            double sig_others = sig - p*p;
                            if (sig_others < 0) sig_others = 0;
                            double term_pen = std::abs(2.0 * cfg.epsilon_penalty * sig_others);

                            if (term_grad > max_grad) max_grad = term_grad;
                            if (term_dw > max_dw) max_dw = term_dw;
                            if (term_drive > max_drive) max_drive = term_drive;
                            if (term_pen > max_penalty) max_penalty = term_pen;
                        }
                    }
                }
                
                #ifdef _OPENMP
                #pragma omp parallel
                #endif
                {
                    #ifdef _OPENMP
                    #pragma omp for schedule(static) nowait
                    #endif
                    for (size_t p = 0; p < plane_size; ++p) sigma_sq[p] = 0.0;
                    #ifdef _OPENMP
                    #pragma omp barrier
                    #endif
                    for (int gid : active_grains) {
                        const double* phi_ptr = &phi_flat[(size_t)gid * plane_size];
                        #ifdef _OPENMP
                        #pragma omp for simd schedule(static)
                        #endif
                        for (size_t p = 0; p < plane_size; ++p) {
                            sigma_sq[p] += phi_ptr[p] * phi_ptr[p];
                        }
                    }
                }

                std::vector<double> out_sig(sigma_sq, sigma_sq + plane_size);
                write_vtk_packed(step, Nx, Ny, cfg.max_grains, phi_flat, T_curr, out_sig, dump_dir);
                write_energy_log(step, t_curr, max_grad, max_dw, max_drive, max_penalty, dump_dir);
            }
        }
        
        std::vector<double> final_sig(sigma_sq, sigma_sq + plane_size);
        write_vtk_packed(step, Nx, Ny, cfg.max_grains, phi_flat, T_curr, final_sig, dump_dir);
        
        aligned_free_wrapper(sigma_sq);
        aligned_free_wrapper(new_phi);
        
        std::cout << "\n[DONE] Simulation Finished.\n";
    }

    void run_benchmark_kernel(int Nx, int Ny, int steps, double* results) {
        // results[0] = Scalar Time (ms)
        // results[1] = AVX512 Time (ms) (or -1 if not supported)
        
        check_cpu_features(); // Initialize g_use_avx512
        
        std::cout << "[BENCH] Starting Benchmark (Nx=" << Nx << ", Ny=" << Ny << ", Steps=" << steps << ")\n";
        
        size_t plane_size = (size_t)Nx * (size_t)Ny;
        
        // Allocate Memory
        double* phi = (double*)aligned_alloc_wrapper(64, plane_size * sizeof(double));
        double* new_phi = (double*)aligned_alloc_wrapper(64, plane_size * sizeof(double));
        double* sigma_sq = (double*)aligned_alloc_wrapper(64, plane_size * sizeof(double));
        double* dGx = (double*)aligned_alloc_wrapper(64, Nx * sizeof(double));
        double* Lx = (double*)aligned_alloc_wrapper(64, Nx * sizeof(double));

        // Initialize Random
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for(size_t i=0; i<plane_size; ++i) {
            phi[i] = dis(gen);
            sigma_sq[i] = dis(gen) * 0.1;
        }
        for(int i=0; i<Nx; ++i) {
            dGx[i] = 1.0; 
            Lx[i] = 1.0;
        }

        SimConfig cfg;
        cfg.dx = 1.0; cfg.dt = 0.005; cfg.xi = 1.0;
        cfg.epsilon_penalty = 1000.0; cfg.kappa = 2.0; cfg.W = 1.0;
        double inv_dx2 = 1.0;

        // 1. Scalar Benchmark
        std::cout << "[BENCH] Running Scalar...\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        
        for(int s=0; s<steps; ++s) {
            // Unoptimized Scalar Loop (Simulated)
             #pragma omp parallel for schedule(static)
             for (int i = 0; i < Nx; ++i) {
                // Simplified Scalar Logic matching main kernel
                double L = Lx[i];
                if (L < 1e-12) continue;
                
                int im = (i == 0) ? 0 : (i - 1);
                int ip = (i == Nx - 1) ? (Nx - 1) : (i + 1);
                
                for (int j = 0; j < Ny; ++j) {
                    size_t curr = (size_t)i*Ny + j;
                    double p = phi[curr];
                    int jm = (j == 0) ? (Ny - 1) : (j - 1);
                    int jp = (j == Ny - 1) ? 0 : (j + 1);
                    
                    double val_im = phi[(size_t)im*Ny + j];
                    double val_ip = phi[(size_t)ip*Ny + j];
                    double val_jm = phi[(size_t)i*Ny + jm];
                    double val_jp = phi[(size_t)i*Ny + jp];
                    
                    double lap = (val_im + val_ip + val_jm + val_jp - 4.0 * p);
                    double term_drive = 30.0 * p*p * (1.0-p)*(1.0-p) * dGx[i];
                    double sig = sigma_sq[curr];
                    double sig_others = std::fmax(0.0, sig - p*p);
                    double val = p + 0.001 * (lap + term_drive - sig_others); // Simplified
                    new_phi[curr] = val; // Write
                }
             }
             // Swap pointers implied (not done here for speed)
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_scalar = t2 - t1;
        results[0] = ms_scalar.count();
        std::cout << "   -> Scalar Time: " << results[0] << " ms\n";

        // 2. AVX-512 Benchmark
        #if defined(__AVX512F__)
        if (g_use_avx512) {
             std::cout << "[BENCH] Running AVX-512...\n";
             auto t3 = std::chrono::high_resolution_clock::now();
             for(int s=0; s<steps; ++s) {
                 update_grain_avx512(Nx, Ny, phi, new_phi, sigma_sq, dGx, Lx, cfg, inv_dx2);
             }
             auto t4 = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double, std::milli> ms_avx = t4 - t3;
             results[1] = ms_avx.count();
             std::cout << "   -> AVX-512 Time: " << results[1] << " ms\n";
        } else {
             results[1] = -1.0;
        }
        #else
        results[1] = -1.0;
        #endif

        aligned_free_wrapper(phi);
        aligned_free_wrapper(new_phi);
        aligned_free_wrapper(sigma_sq);
        aligned_free_wrapper(dGx);
        aligned_free_wrapper(Lx);
    }
}
