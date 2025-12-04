// file: src/main.cpp
#include "INCLUDE/phasefield.hpp"
#include "INCLUDE/inputreader.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Read input
        InputParams params = InputReader::readInputFile("input.dat");
        InputReader::printParams(params);
        
        // Create grid configuration
        gridconfig grid_config = params.toGridConfig();
        
        // Create phase field solver
        PhaseField pf(params, grid_config);
        
        // Time stepping loop
        for (int n = params.nstart; n <= params.nend; n++) {
            // Output at first step and then at intervals
            if (n == params.nstart || n % params.nprint == 0) {
                pf.printDiagnostics();
            }
            
            if (n == params.nstart || n % params.ndump == 0) {
                pf.downloadFromDevice();  // Download fields for output
                std::string filename = "phi_" + std::to_string(n) + ".dat";
                pf.writeTecplot();
            }
            
            // Advance one time step
            if (n < params.nend) {  // Don't advance past final time
                pf.advanceTimeStep();
            }
        }
        
        // Final output
        std::cout << "\nSimulation completed successfully!" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}