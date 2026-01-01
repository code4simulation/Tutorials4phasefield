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

#ifdef _OPENMP
#include <omp.h>
#endif

struct SimConfig {
    int Nx, Ny;
    double dx, dt;
    double xi;
    double epsilon_penalty;
    double kappa;
    double W;
    int max_grains;
    double time_total;
    int output_interval;
    int use_active_list;
};

struct NucleationEvent {
    double time;
    int x;
    int y;
    int id;
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

static inline double clamp01(double x) {
    return (x < 0.0) ? 0.0 : (x > 1.0 ? 1.0 : x);
}

static void write_dump_file(int step, size_t total_elems, const double* phi_flat, const char* dump_dir) {
    if (dump_dir == nullptr) return;
    std::ostringstream fn;
    fn << dump_dir << "/dump_step_" << std::setw(6) << std::setfill('0') << step << ".bin";
    std::ofstream out(fn.str(), std::ios::binary);
    if (!out) {
        std::cerr << "[ERROR] Failed to open dump file: " << fn.str() << "\n";
        return;
    }
    out.write(reinterpret_cast<const char*>(phi_flat), (std::streamsize)(total_elems * sizeof(double)));
    out.close();
}

extern "C" {
    void run_mpf_simulation(
        double* phi_flat,
        const double* temp_profile,
        const double* beta_T, const double* beta_val, int beta_len,
        const double* dG_T, const double* dG_val, int dG_len,
        const double* nucl_data, int n_events,
        SimConfig cfg,
        const char* dump_dir
    ) {
        if (!phi_flat || !temp_profile) return;
        if (cfg.Nx <= 0 || cfg.Ny <= 0 || cfg.max_grains <= 0) return;

        const int Nx = cfg.Nx;
        const int Ny = cfg.Ny;
        const size_t plane_size = (size_t)Nx * (size_t)Ny;
        const size_t total_elems = (size_t)cfg.max_grains * plane_size;
        const double inv_dx2 = 1.0 / (cfg.dx * cfg.dx);

        // LUTs
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

        if (cfg.use_active_list == 0) {
            std::cout << "[INFO] Running in FULL SCAN mode (Active Grain Optimization DISABLED)\n";
        } else {
            std::cout << "[INFO] Running in ACTIVE LIST mode (Optimization ENABLED)\n";
        }

        // Precompute L(x), dG(x)
        std::vector<double> Lx(Nx, 0.0);
        std::vector<double> dGx(Nx, 0.0);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < Nx; ++i) {
            const double T = temp_profile[i];
            const double beta = beta_lut.interpolate(T);
            const double dG = dG_lut.interpolate(T);
            dGx[i] = dG;
            double L = 0.0;
            if (dG > 1e-12) L = (1.0 / dG) / beta;
            Lx[i] = L;
        }

        // Load nucleation events
        std::vector<NucleationEvent> events;
        events.reserve((size_t)n_events);
        for (int k = 0; k < n_events; ++k) {
            double t = nucl_data[k * 3 + 0];
            int x = (int)std::llround(nucl_data[k * 3 + 1]);
            int y = (int)std::llround(nucl_data[k * 3 + 2]);
            if (x < 0) x = 0; if (x >= Nx) x = Nx - 1;
            if (y < 0) y = 0; if (y >= Ny) y = Ny - 1;
            events.push_back({t, x, y, k + 1});
        }
        std::sort(events.begin(), events.end(),
            [](const NucleationEvent& a, const NucleationEvent& b) { return a.time < b.time; });

        // Buffers
        std::vector<double> sigma_sq(plane_size, 0.0);
        std::vector<double> new_phi(total_elems, 0.0);

        // [수정] Active grains 리스트와 비교군을 위한 All grains 리스트 준비
        std::vector<int> active_grains;
        active_grains.reserve(std::min(cfg.max_grains, 1024));

        std::vector<int> all_grains;
        if (cfg.use_active_list == 0) {
            all_grains.reserve(cfg.max_grains);
            // Grain ID 1부터 max_grains-1까지 (0번은 보통 비어있거나 배경)
            for (int g = 1; g < cfg.max_grains; ++g) {
                all_grains.push_back(g);
            }
        }

        auto idx = [&](int g, int i, int j) -> size_t {
            return (size_t)g * plane_size + (size_t)i * (size_t)Ny + (size_t)j;
        };

        std::mt19937 gen(12345u);
        int event_idx = 0;
        double t_curr = 0.0;
        int step = 0;

        write_dump_file(0, total_elems, phi_flat, dump_dir);

        while (t_curr < cfg.time_total) {

            // ---- A) Nucleation ----
            while (event_idx < n_events && events[event_idx].time <= t_curr) {
                NucleationEvent& ev = events[event_idx];
                if (ev.id >= cfg.max_grains) {
                    event_idx++;
                    continue;
                }

                const std::vector<int>& scan_grains = (cfg.use_active_list) ? active_grains : all_grains;

                std::vector<int> candidates;
                candidates.reserve((size_t)Ny);

                for (int y = 0; y < Ny; ++y) {
                    double solid_sum = 0.0;
                    for (int gid : scan_grains) {
                        solid_sum += phi_flat[idx(gid, ev.x, y)];
                        if (solid_sum > 0.1) break;
                    }
                    if (solid_sum < 0.1) candidates.push_back(y);
                }

                if (!candidates.empty()) {
                    std::uniform_int_distribution<> dis(0, (int)candidates.size() - 1);
                    int chosen_y = candidates[dis(gen)];

                    bool already_active = false;
                    for (int gid : active_grains) {
                        if (gid == ev.id) { already_active = true; break; }
                    }
                    if (!already_active) active_grains.push_back(ev.id);

                    const int R = std::max(1, (int)std::lround(cfg.xi));
                    for (int dx = -R; dx <= R; ++dx) {
                        for (int dy = -R; dy <= R; ++dy) {
                            int nx = ev.x + dx;
                            int ny = chosen_y + dy;
                            if (nx < 0 || nx >= Nx) continue;
                            if (ny < 0) ny += Ny; 
                            if (ny >= Ny) ny -= Ny;
                            if (dx * dx + dy * dy <= R * R) {
                                phi_flat[idx(ev.id, nx, ny)] = 1.0;
                            }
                        }
                    }
                }
                event_idx++;
            }

            const std::vector<int>& compute_grains = (cfg.use_active_list) ? active_grains : all_grains;

            // ---- B) sigma_sq = sum_g phi_g^2 ----
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int p = 0; p < (int)plane_size; ++p) {
                double s = 0.0;
                for (int gid : compute_grains) {
                    double v = phi_flat[(size_t)gid * plane_size + (size_t)p];
                    s += v * v;
                }
                sigma_sq[(size_t)p] = s;
            }


            // ---- C) Update (Semi-Implicit Scheme) ----
            for (int gid : compute_grains) {
                #ifdef _OPENMP
                #pragma omp parallel for collapse(2) schedule(static)
                #endif
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        const double L = Lx[i];
                        const double p = phi_flat[idx(gid, i, j)];
                        
                        // L이 0이면 변화 없음
                        if (L < 1e-12) {
                            new_phi[idx(gid, i, j)] = p;
                            continue;
                        }

                        // Laplacian (Explicit)
                        int im = (i == 0) ? 0 : (i - 1);
                        int ip = (i == Nx - 1) ? (Nx - 1) : (i + 1);
                        int jm = (j == 0) ? (Ny - 1) : (j - 1);
                        int jp = (j == Ny - 1) ? 0 : (j + 1);

                        const double lap =
                            (phi_flat[idx(gid, ip, j)] + phi_flat[idx(gid, im, j)] +
                             phi_flat[idx(gid, i, jp)] + phi_flat[idx(gid, i, jm)] - 4.0 * p) * inv_dx2;
                        
                        const double term_grad = cfg.kappa * lap;

                        // Double Well (Explicit)
                        // f'(p) = 2*W*p*(1-p)*(1-2p)
                        const double term_dw = 2.0 * cfg.W * p * (1.0 - p) * (1.0 - 2.0 * p);

                        // Driving Force (Explicit)
                        const double dG = dGx[i];
                        const double term_drive = 30.0 * p * p * (1.0 - p) * (1.0 - p) * dG;

                        // Penalty Term 분해 (Semi-Implicit 핵심)
                        // Original: - 2 * eps * p * (sig - p^2)
                        // sig_others = sig - p^2 (다른 Grain들의 제곱 합)
                        // Force = - 2 * eps * p * sig_others
                        // 이것을 Implicit으로 처리: - 2 * eps * phi_new * sig_others
                        
                        const double sig = sigma_sq[(size_t)i * (size_t)Ny + (size_t)j];
                        double sig_others = sig - p * p;
                        if (sig_others < 0.0) sig_others = 0.0; // 수치 오차 방지

                        // Implicit Damping Coefficient (B)
                        // 우변에서 - B * phi_new 로 작용할 항의 계수
                        // dphi/dt = ... - L * (2 * eps * sig_others) * phi
                        const double B = L * (2.0 * cfg.epsilon_penalty * sig_others);

                        // Explicit Source Terms (A)
                        // dphi/dt = L * (Grad - DW + Drive) ...
                        // *주의: Penalty 항은 이미 B로 뺐으므로 여기서 제외!
                        const double explicit_force = L * (term_grad - term_dw + term_drive);

                        // Semi-Implicit Update Formula
                        // phi_new = (phi_old + dt * Explicit_Force) / (1 + dt * B)
                        double numer = p + cfg.dt * explicit_force;
                        double denom = 1.0 + cfg.dt * B;

                        double val = numer / denom;
                        
                        // 안전장치 (Clamp)
                        new_phi[idx(gid, i, j)] = clamp01(val);
                    }
                }
            }


            // ---- D) Commit ----
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int k = 0; k < (int)compute_grains.size(); ++k) {
                int gid = compute_grains[(size_t)k];
                std::memcpy(&phi_flat[(size_t)gid * plane_size],
                            &new_phi[(size_t)gid * plane_size],
                            plane_size * sizeof(double));
            }

            // ---- [NEW] E) Projection / Normalization Step ----
            // 모든 격자점을 돌면서 Sum(phi^2) > 1 인 곳을 눌러줍니다.
            // active_grains만 확인하면 되므로 효율적입니다.
            const size_t total_pixels = plane_size; // Nx * Ny
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif

            for (size_t pix = 0; pix < total_pixels; ++pix) {
                double sum_sq = 0.0;
                
                for (int gid : compute_grains) {
                    double val = phi_flat[(size_t)gid * plane_size + pix];
                    sum_sq += val * val;
                }

                if (sum_sq > 1.000001) {
                    double scale = 1.0 / std::sqrt(sum_sq);
                    for (int gid : compute_grains) {
                        phi_flat[(size_t)gid * plane_size + pix] *= scale;
                    }
                }
            }
            // ---- E) Advance ----
            t_curr += cfg.dt;
            step++;
            if (cfg.output_interval > 0 && (step % cfg.output_interval == 0)) {
                write_dump_file(step, total_elems, phi_flat, dump_dir);
            }
        }
        
        // Final dump
        write_dump_file(step, total_elems, phi_flat, dump_dir);
    }
}
