// file: src/test_simple.cpp
#include "INCLUDE/include.hpp"
#include "CALCULATORS/calculators.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        std::cout << "=== Simple CalculateFaceVals Test ===\n\n";

        // Create a small test grid
        gridconfig config;
        config.Ngl = 2;
        config.Nx = 8;
        config.Ny = 8;
        config.Nz = 8;
        config.x_start = 0.0;
        config.x_end = 1.0;
        config.y_start = 0.0;
        config.y_end = 1.0;
        config.z_start = 0.0;
        config.z_end = 1.0;
        config.x_option = GRID_UNIFORM;
        config.y_option = GRID_UNIFORM;
        config.z_option = GRID_UNIFORM;
        
        GridInfo grid(config);
        
        std::cout << "Grid created: " << config.Nx << "x" << config.Ny << "x" << config.Nz << "\n";
        std::cout << "Ghost layers: " << config.Ngl << "\n\n";

        // Test: CalculateFaceVals (scalar)
        std::cout << "Test: CalculateFaceVals (scalar field)\n";
        Field phi(&grid, "phi");
        Face phi_f(&grid, "phi_face");
        
        // Initialize phi to a constant value on host
        std::cout << "Initializing phi to 5.0...\n";
        for (int i = 1 - config.Ngl; i <= config.Nx + config.Ngl; i++) {
            for (int j = 1 - config.Ngl; j <= config.Ny + config.Ngl; j++) {
                for (int k = 1 - config.Ngl; k <= config.Nz + config.Ngl; k++) {
                    phi.u_h(i, j, k) = 5.0;
                }
            }
        }
        
        // Upload to device
        phi.upload();
        std::cout << "Uploaded to device\n";
        
        // Call CalculateFaceVals
        std::cout << "Calling CalculateFaceVals...\n";
        CalculateFaceVals(phi, phi_f);
        std::cout << "CalculateFaceVals completed\n";
        
        // Download results
        phi_f.download();
        std::cout << "Downloaded results\n\n";
        
        // Check results at center
        int ic = config.Nx / 2;
        int jc = config.Ny / 2;
        int kc = config.Nz / 2;
        
        std::cout << "Results at center (" << ic << "," << jc << "," << kc << "):\n";
        std::cout << "  Input phi = " << phi.u_h(ic, jc, kc) << "\n";
        std::cout << "  East face (phi_f.E)  = " << phi_f.E.u_h(ic, jc, kc) << " (expected: 5.0)\n";
        std::cout << "  West face (phi_f.W)  = " << phi_f.W.u_h(ic, jc, kc) << " (expected: 5.0)\n";
        std::cout << "  North face (phi_f.N) = " << phi_f.N.u_h(ic, jc, kc) << " (expected: 5.0)\n";
        std::cout << "  South face (phi_f.S) = " << phi_f.S.u_h(ic, jc, kc) << " (expected: 5.0)\n";
#ifndef USE_2D
        std::cout << "  Front face (phi_f.F) = " << phi_f.F.u_h(ic, jc, kc) << " (expected: 5.0)\n";
        std::cout << "  Back face (phi_f.B)  = " << phi_f.B.u_h(ic, jc, kc) << " (expected: 5.0)\n";
#endif
        
        // Check a few more locations
        std::cout << "\nResults at i=1, j=1, k=1:\n";
        std::cout << "  Input phi = " << phi.u_h(1, 1, 1) << "\n";
        std::cout << "  East face  = " << phi_f.E.u_h(1, 1, 1) << "\n";
        std::cout << "  West face  = " << phi_f.W.u_h(1, 1, 1) << "\n";
        std::cout << "  North face = " << phi_f.N.u_h(1, 1, 1) << "\n";
        std::cout << "  South face = " << phi_f.S.u_h(1, 1, 1) << "\n";
        
        // Test with a gradient field
        std::cout << "\n=== Test with linear field phi = x ===\n";
        auto xc = grid.xgrid->c_h;
        for (int i = 1 - config.Ngl; i <= config.Nx + config.Ngl; i++) {
            for (int j = 1 - config.Ngl; j <= config.Ny + config.Ngl; j++) {
                for (int k = 1 - config.Ngl; k <= config.Nz + config.Ngl; k++) {
                    phi.u_h(i, j, k) = xc(i);
                }
            }
        }
        
        phi.upload();
        CalculateFaceVals(phi, phi_f);
        phi_f.download();
        
        std::cout << "Results at center (" << ic << "," << jc << "," << kc << "):\n";
        std::cout << "  Input phi(i) = " << phi.u_h(ic, jc, kc) << " (x = " << xc(ic) << ")\n";
        std::cout << "  Input phi(i+1) = " << phi.u_h(ic+1, jc, kc) << " (x = " << xc(ic+1) << ")\n";
        std::cout << "  East face = " << phi_f.E.u_h(ic, jc, kc) << " (expected: avg of above)\n";
        
        real_t expected_E = 0.5 * (phi.u_h(ic, jc, kc) + phi.u_h(ic+1, jc, kc));
        std::cout << "  Expected East face = " << expected_E << "\n";
        std::cout << "  Error = " << (phi_f.E.u_h(ic, jc, kc) - expected_E) << "\n";
        
        std::cout << "\n=== Test Complete ===\n";
    }
    Kokkos::finalize();
    
    return 0;
}
