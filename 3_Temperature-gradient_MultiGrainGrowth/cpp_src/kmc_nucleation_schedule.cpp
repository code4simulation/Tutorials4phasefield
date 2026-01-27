#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

// Structure to hold event data
struct Event {
    double time;
    int x;
    int y;
};

extern "C" {

/**
 * @brief Performs rejection-free KMC for nucleation scheduling.
 * 
 * To avoid spatial bias, we generate events for ALL cells freely using a dynamic container,
 * sort them by time, and THEN truncate if the total count exceeds max_events.
 * 
 * @param rates_1d_ptr Pointer to array of cell rates (I[x] * V_cell)
 * @param nx Grid dimension X
 * @param ny Grid dimension Y
 * @param total_time_limit Simulation duration
 * @param result_ptr Output buffer allocated by Python [time, x, y, time, x, y...]
 * @param max_events Capacity of the result buffer
 * @return Number of events written to buffer
 */
int run_kmc(const double* rates_1d_ptr, int nx, int ny, double total_time_limit, double* result_ptr, int max_events) {
    
    // 1. Setup Random Number Generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // 2. Dynamic container for all generated events
    std::vector<Event> all_events;
    
    // Reserve memory to optimize performance (heuristic)
    all_events.reserve(max_events);

    // 3. Generate events for each cell independently
    // Since rates are static, each cell behaves as an independent Poisson process.
    for (int x = 0; x < nx; ++x) {
        double rate = rates_1d_ptr[x];
        
        // Skip cells with zero nucleation rate
        if (rate <= 1e-20) continue; 

        for (int y = 0; y < ny; ++y) {
            double t = 0.0;
            while (true) {
                // Inverse Transform Sampling for Exponential Distribution
                // dt = -ln(U) / lambda
                double u = dis(gen);
                double dt = -std::log(u) / rate;
                t += dt;

                if (t > total_time_limit) {
                    break;
                }
                
                // Store event
                all_events.push_back({t, x, y});
            }
        }
    }

    // 4. Sort all events by time to ensure chronological order
    std::sort(all_events.begin(), all_events.end(), [](const Event& a, const Event& b) {
        return a.time < b.time;
    });

    // 5. Copy to Output Buffer (Truncate if necessary)
    int count = 0;
    int total_generated = (int)all_events.size();
    int limit = (total_generated < max_events) ? total_generated : max_events;

    for (int i = 0; i < limit; ++i) {
        result_ptr[i * 3 + 0] = all_events[i].time;
        result_ptr[i * 3 + 1] = (double)all_events[i].x;
        result_ptr[i * 3 + 2] = (double)all_events[i].y;
        count++;
    }

    return count;
}

} // extern "C"