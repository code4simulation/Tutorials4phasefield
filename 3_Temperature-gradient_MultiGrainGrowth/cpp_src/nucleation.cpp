#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

struct Event {
    double time;
    int x;
    int y;
};

extern "C" {

/**
 * @brief Method A: Rejection-free KMC (Existing)
 * Pre-calculates all events based on static rates.
 */
int run_kmc_nucleation(const double* rates_1d_ptr, int nx, int ny, double total_time_limit, double* result_ptr, int max_events) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<Event> all_events;
    all_events.reserve(max_events);

    for (int x = 0; x < nx; ++x) {
        double rate = rates_1d_ptr[x];
        if (rate <= 1e-20) continue; 

        for (int y = 0; y < ny; ++y) {
            double t = 0.0;
            while (true) {
                double u = dis(gen);
                double dt = -std::log(u) / rate;
                t += dt;
                if (t > total_time_limit) break;
                all_events.push_back({t, x, y});
            }
        }
    }

    std::sort(all_events.begin(), all_events.end(), [](const Event& a, const Event& b) {
        return a.time < b.time;
    });

    int count = 0;
    int limit = std::min((int)all_events.size(), max_events);
    for (int i = 0; i < limit; ++i) {
        result_ptr[i * 3 + 0] = all_events[i].time;
        result_ptr[i * 3 + 1] = (double)all_events[i].x;
        result_ptr[i * 3 + 2] = (double)all_events[i].y;
        count++;
    }
    return count;
}

/**
 * @brief Method B: Stochastic Time-Stepping (New)
 * Simulates nucleation "on-the-fly" using discrete time steps dt.
 * This mimics how it will run inside the main MPF simulation loop.
 */
int run_stochastic_nucleation(const double* rates_1d_ptr, int nx, int ny, double dt, double total_time_limit, double* result_ptr, int max_events) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Dynamic buffer
    std::vector<Event> collected_events;
    collected_events.reserve(max_events);

    double t_curr = 0.0;
    
    // We iterate time steps
    while (t_curr < total_time_limit) {
        t_curr += dt;
        
        // For each cell, check probability
        for (int x = 0; x < nx; ++x) {
            double rate = rates_1d_ptr[x];
            if (rate <= 1e-20) continue;

            // Probability to nucleate in this step: P = 1 - exp(-rate * dt)
            // Approx P ~ rate * dt (if P << 1)
            double prob = 1.0 - std::exp(-rate * dt);

            // Iterate over Y column
            for (int y = 0; y < ny; ++y) {
                 if (dis(gen) < prob) {
                     // Event occurred!
                     collected_events.push_back({t_curr, x, y});
                 }
            }
        }
        
        if (collected_events.size() >= (size_t)max_events) break;
    }
    
    // Copy to buffer
    int count = 0;
    int limit = std::min((int)collected_events.size(), max_events);
    for (int i = 0; i < limit; ++i) {
        result_ptr[i * 3 + 0] = collected_events[i].time;
        result_ptr[i * 3 + 1] = (double)collected_events[i].x;
        result_ptr[i * 3 + 2] = (double)collected_events[i].y;
        count++;
    }
    return count;
}

} // extern "C"
