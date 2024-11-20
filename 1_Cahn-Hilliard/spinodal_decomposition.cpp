#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <random>

using std::string;
using std::vector;
using std::ofstream;

using Grid = vector<vector<double>>; 

class CahnHilliard {
public:
    CahnHilliard(int size, double kappa, double M)
    : size(size), kappa(kappa), M(M) {
        psi = Grid(size, vector<double>(size, 0.5));
    }

    void initializeSimulation() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                psi[i][j] = dist(rng);
            }
        }
    }

    void runSimulation(int totalSteps, int saveInterval) {
        for (int step = 0; step < totalSteps + 1; ++step) {
            updateOrderParameter(true, true);

            if (step % saveInterval == 0) {
                string filename = "output_" + std::to_string(step) + ".vtk";
                saveVTK(filename, psi);
            }
        }
        createPVDFile("simulation_results.pvd", totalSteps, saveInterval);
    }

private:
    int size;
    double kappa; // Interface energy coefficient
    double M; // Mobility
    Grid psi; // Order parameter grid

    void updateOrderParameter(bool isPeriodicX, bool isPeriodicY) {
        double h = 1; // a unit grid spacing 
        double dt = 1E-3; // Time step size
        Grid dFdpsi(size, vector<double>(size));
        Grid psi_new(size, vector<double>(size));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                auto [ip, im] = isPeriodicX ? getPBCIndex(i, size) : std::make_pair(i == size - 1 ? i : i + 1, i == 0 ? i : i - 1);
                auto [jp, jm] = isPeriodicY ? getPBCIndex(j, size) : std::make_pair(j == size - 1 ? j : j + 1, j == 0 ? j : j - 1);
                double laplacian_psi = computeLaplacian(psi, i, j, h, ip, im, jp, jm);
                // F(psi) = integral( f(psi) + kappa/2*gradient^2 )
                // f(psi) = (psi)^2 * (1-psi)^2
                dFdpsi[i][j] = 2 * psi[i][j] * (1 - psi[i][j]) * (1 - 2 * psi[i][j]) - kappa * laplacian_psi; 
            }
        }

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                auto [ip, im] = isPeriodicX ? getPBCIndex(i, size) : std::make_pair(i == size - 1 ? i : i + 1, i == 0 ? i : i - 1);
                auto [jp, jm] = isPeriodicY ? getPBCIndex(j, size) : std::make_pair(j == size - 1 ? j : j + 1, j == 0 ? j : j - 1);
                double laplacian_mu = computeLaplacian(dFdpsi, i, j, h, ip, im, jp, jm);
                psi_new[i][j] = psi[i][j] + dt * M * laplacian_mu;
            }
        }
        psi = std::move(psi_new);
    }

    std::pair<int, int> getPBCIndex(int currentIndex, int gridSize) {
        int nextIndex = (currentIndex + 1) % gridSize;
        int prevIndex = (currentIndex - 1 + gridSize) % gridSize;
        return {nextIndex, prevIndex};
    }

    double computeLaplacian(const Grid& grid, int i, int j, double h, int ip, int im, int jp, int jm) {
        return (grid[ip][j] + grid[im][j] + grid[i][jp] + grid[i][jm] - 4.0*grid[i][j]) / (h*h);
    }

    void saveVTK(const string& filename, const Grid& data) {
        ofstream vtkFile(filename);
        if (!vtkFile.is_open()) {
            std::cerr << "Failed to open file for writing VTK data.\n";
            return;
        }

        vtkFile << "# vtk DataFile Version 3.0\n";
        vtkFile << "Phase field model output\n";
        vtkFile << "ASCII\n";
        vtkFile << "DATASET STRUCTURED_POINTS\n";
        vtkFile << "DIMENSIONS " << data.size() << " " << data[0].size() << " 1\n";
        vtkFile << "ORIGIN 0 0 0\n";
        vtkFile << "SPACING 1 1 1\n";
        vtkFile << "POINT_DATA " << (data.size() * data[0].size()) << "\n";
        vtkFile << "SCALARS psi double\n";
        vtkFile << "LOOKUP_TABLE default\n";

        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                vtkFile << data[i][j] << "\n";
            }
        }
        vtkFile.close();
    }

    void createPVDFile(const string& pvdFilename, int totalSteps, int interval) {
        ofstream pvdFile(pvdFilename);
        if (!pvdFile.is_open()) {
            std::cerr << "Failed to open file for writing PVD data.\n";
            return;
        }

        pvdFile << "<?xml version=\"1.0\"?>\n";
        pvdFile << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvdFile << "  <Collection>\n";

        for (int t = 0; t <= totalSteps; t=t+interval) {
            std::stringstream ss;
            ss << "output_" << t << ".vtk";
            pvdFile << "    <DataSet timestep=\"" << t << "\" group=\"\" part=\"0\" file=\"" << ss.str() << "\"/>\n";
        }

        pvdFile << "  </Collection>\n";
        pvdFile << "</VTKFile>\n";
        pvdFile.close();
    }
};

int main() {
    CahnHilliard model(64, 1, 10);

    model.initializeSimulation();
    // (User-defined total number of simulation steps, save interval)
    model.runSimulation(200000, 500);

    return 0;
}
