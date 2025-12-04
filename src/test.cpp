// file: src/test.cpp
#include "INCLUDE/include.hpp"
#include "CALCULATORS/calculators.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        std::cout << "=== Testing All Calculator Functions ===\n\n";

        // Create a simple test grid
        gridconfig config;
        config.Ngl = 1;
        config.Nx = 32;
        config.Ny = 32;
        config.Nz = 32;
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

        // Test 1: Field initialization and basic operations
        std::cout << "Test 1: Field initialization\n";
        Field phi(&grid, "phi");
        phi.fill(1.0);
        phi.upload();
        phi.download();
        auto xc = grid.xgrid->c;
        auto yc = grid.ygrid->c;
        auto zc = grid.zgrid->c;
        auto u = phi.u;
        int Nx = config.Nx, Ny = config.Ny, Nz = config.Nz, Ngl = config.Ngl;
        
        // Define test point
        #ifndef USE_2D
        int i_test = static_cast<int>(std::floor(Nx / 2.0));
        int j_test = static_cast<int>(std::floor(Ny / 2.0));
        int k_test = static_cast<int>(std::floor(Nz / 2.0));
        #else
        int i_test = static_cast<int>(std::floor(Nx / 2.0));
        int j_test = static_cast<int>(std::floor(Ny / 2.0));
        int k_test = 1;
        #endif
        
        std::cout << "  Test point: (" << i_test << ", " << j_test << ", " << k_test << ")\n";

        std::cout << "  phi initialized to 1.0\n";
        std::cout << "  Sample value at test point: " << phi.u_h(i_test, j_test, k_test) << "\n";
        std::cout << "     phi range check:\n";
        std::cout << "       phi(0, j_test, k_test) = " << phi.u_h(0, j_test, k_test) << "\n";
        std::cout << "       phi(1, j_test, k_test) = " << phi.u_h(1, j_test, k_test) << "\n";
        std::cout << "       phi(Nx, j_test, k_test) = " << phi.u_h(Nx, j_test, k_test) << "\n";
        std::cout << "       phi(Nx+1, j_test, k_test) = " << phi.u_h(Nx+1, j_test, k_test) << "\n";

        // Test 2: CalculateGradients
        std::cout << "\nTest 2: CalculateGradients\n";
        Field phi_x(&grid, "phi_x");
        Field phi_y(&grid, "phi_y");
        Field phi_z(&grid, "phi_z");

        // Create a test field with known gradient: phi = x + 2*y + 3*z
        std::cout << "\n     Initializing phi = x + 2*y + 3*z \n";
        std::cout << "     Grid info: Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << " Ngl=" << Ngl << "\n";
        std::cout << "     Using proper grid bounds from oneDgridinfo:\n";
        std::cout << "       x: bs=" << grid.xgrid->bs << " be=" << grid.xgrid->be << "\n";
        std::cout << "       y: bs=" << grid.ygrid->bs << " be=" << grid.ygrid->be << "\n";
        std::cout << "       z: bs=" << grid.zgrid->bs << " be=" << grid.zgrid->be << "\n";
        // Before the parallel_for, verify coordinates on host
        std::cout << "xc(1) = " << grid.xgrid->c_h(1) << "\n";
        std::cout << "xc(Nx/2) = " << grid.xgrid->c_h(Nx / 2) << "\n";
        std::cout << "yc(1) = " << grid.ygrid->c_h(1) << "\n";
        std::cout << "yc(Ny/2) = " << grid.ygrid->c_h(Ny / 2) << "\n";
        // Use proper bounds from oneDgridinfo
        auto interior_policy = grid.interior_policy;
        auto FullPolicy = grid.full_policy;

        // Check if simple constant fill works
        phi.download();
        std::cout << "     Before any parallel_for: phi(1, Ny/2, 1) = " << phi.u_h(1, Ny / 2, 1) << "\n";

        // Initialize phi with coordinates
        auto xc_device = grid.xgrid->c;
        auto yc_device = grid.ygrid->c;
        auto zc_device = grid.zgrid->c;
        
        std::cout << "     Starting parallel_for with coordinates...\n";
        Kokkos::parallel_for("TEST_Field_Init_Gradient",
        FullPolicy,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
                #ifndef USE_2D
                phi.u(i,j,k) = xc_device(i) + 2.0*yc_device(j) + 3.0*zc_device(k);
                #else
                phi.u(i,j,k) = xc_device(i) + 2.0*yc_device(j);
                #endif
            });
        Kokkos::fence();
        phi.download();
        
        std::cout << "\n     phi = x + 2*y + 3*z initialized \n";
        std::cout << "     phi at test point: " << phi.u_h(i_test, j_test, k_test) << "\n";
        #ifndef USE_2D
        std::cout << "     Expected: " << (grid.xgrid->c_h(i_test) + 2.0*grid.ygrid->c_h(j_test) + 3.0*grid.zgrid->c_h(k_test)) << "\n";
        #else
        std::cout << "     Expected: " << (grid.xgrid->c_h(i_test) + 2.0*grid.ygrid->c_h(j_test)) << "\n";
        #endif

        // Apply boundary conditions to fill ghost cells
        bctype BC;
        BC.W = BC_PERIODIC;
        BC.E = BC_PERIODIC;
        BC.S = BC_PERIODIC;
        BC.N = BC_PERIODIC;
        BC.B = BC_PERIODIC;
        BC.F = BC_PERIODIC;
        BC.W_val = 0.0;
        BC.E_val = 0.0;
        BC.S_val = 0.0;
        BC.N_val = 0.0;
        BC.B_val = 0.0;
        BC.F_val = 0.0;
        CalculateBoundaryValues(phi, BC);
        
        std::cout << "\n     Now calculate grad(phi)=(phi_x, phi_y, phi_z) \n";
        CalculateGradients(phi, phi_x, phi_y, phi_z);
        phi_x.download();
        phi_y.download();
        phi_z.download();

        std::cout << "  Expected gradients: dphi/dx=1.0, dphi/dy=2.0, dphi/dz=3.0\n";
        std::cout << "  Sample values at center:\n";
        std::cout << "    dphi/dx = " << phi_x.u_h(6, 6, 1) << "\n";
        std::cout << "    dphi/dy = " << phi_y.u_h(6, 6, 1) << "\n";
        std::cout << "    dphi/dz = " << phi_z.u_h(6, 6, 1) << "\n";

        // Test 3: CalculateDivergence
        std::cout << "\nTest 3: CalculateDivergence\n";
        Field div(&grid, "divergence");
        CalculateDivergence(phi_x, phi_y, phi_z, div);
        div.download();
        std::cout << "  Expected divergence: 0.0 (constant gradients)\n";
        std::cout << "  Calculated divergence at test point: " << div.u_h(i_test, j_test, k_test) << "\n";

        // Test 4: CalculateFaceVals (scalar)
        std::cout << "\nTest 4: CalculateFaceVals (scalar)\n";
        phi.fill(5.0);
        phi.upload();
        Face phi_f(&grid, "phi_face");
        CalculateFaceVals(phi, phi_f);
        phi_f.download();
        std::cout << "  Input field phi = 5.0 everywhere\n";
        std::cout << "  Face values at test point:\n";
        std::cout << "    East face:  " << phi_f.E.u_h(i_test, j_test, k_test) << "\n";
        std::cout << "    North face: " << phi_f.N.u_h(i_test, j_test, k_test) << "\n";

        // Test 5: CalculateFaceVals (vector)
        std::cout << "\nTest 5: CalculateFaceVals (vector)\n";
        Field vx(&grid, "vx"), vy(&grid, "vy"), vz(&grid, "vz");
        vx.fill(1.0);
        vy.fill(2.0);
        vz.fill(3.0);
        vx.upload();
        vy.upload();
        vz.upload();
        Face vel_f(&grid, "vel_face");
        CalculateFaceVals(vx, vy, vz, vel_f);
        vel_f.download();
        std::cout << "  Input: vx=1.0, vy=2.0, vz=3.0\n";
        std::cout << "  East face velocity at test point: " << vel_f.E.u_h(i_test, j_test, k_test) << "\n";

        // Test 6: CalculateFaceDiffs
        std::cout << "\nTest 6: CalculateFaceDiffs\n";
        Face phi_diff(&grid, "phi_diff");
        CalculateFaceDiffs(phi, phi_diff);
        phi_diff.download();
        std::cout << "  Differences of constant field (should be ~0):\n";
        std::cout << "    East diff:  " << phi_diff.E.u_h(i_test, j_test, k_test) << "\n";
        std::cout << "    North diff: " << phi_diff.N.u_h(i_test, j_test, k_test) << "\n";

        // Test 7: CalculateDivergence (Face version)
        std::cout << "\nTest 7: CalculateDivergence (Face)\n";
        Field div_face(&grid, "div_face");
        CalculateDivergence(vel_f, div_face);
        div_face.download();
        std::cout << "  Divergence of constant velocity field (should be ~0):\n";
        std::cout << "    div at test point: " << div_face.u_h(i_test, j_test, k_test) << "\n";

        // Test 8: CalculateNormals
        std::cout << "\nTest 8: CalculateNormals\n";
        // Create a spherical interface at center
        auto u_phi = phi.u;
        real_t xc_val = 0.5, yc_val = 0.5;
#ifndef USE_2D
        real_t zc_val = 0.5;
#endif
        Kokkos::parallel_for("InitSphere", FullPolicy, 
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
            real_t dx = xc(i) - xc_val;
            real_t dy = yc(j) - yc_val;
#ifndef USE_2D
            real_t dz = zc(k) - zc_val;
            u_phi(i, j, k) = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) - 0.2;
#else
            u_phi(i, j, k) = Kokkos::sqrt(dx * dx + dy * dy) - 0.2;
#endif
        });
        Kokkos::fence();

        Field norm_x(&grid, "norm_x"), norm_y(&grid, "norm_y"), norm_z(&grid, "norm_z");
        CalculateGradients(phi, norm_x, norm_y, norm_z);
        CalculateNormals(norm_x, norm_y, norm_z);
        norm_x.download();
        norm_y.download();
        norm_z.download();
        std::cout << "  Normals for sphere at (0.5, 0.5, 0.5), radius 0.2\n";
        std::cout << "  Normal components at (0.7, 0.5, 0.5) - should point in +x:\n";
        int i_norm = static_cast<int>((0.7 - config.x_start) / (config.x_end - config.x_start) * Nx) + 1;
        std::cout << "    nx = " << norm_x.u_h(i_norm, j_test, k_test) << "\n";
        std::cout << "    ny = " << norm_y.u_h(i_norm, j_test, k_test) << "\n";
        std::cout << "    nz = " << norm_z.u_h(i_norm, j_test, k_test) << "\n";

        // Test 9: CalculateConvection
        std::cout << "\nTest 9: CalculateConvection\n";
        Face psi_f(&grid, "psi_face");
        Field convection(&grid, "convection");
        CalculateFaceVals(phi, psi_f);
        CalculateConvection(psi_f, vel_f, convection);
        convection.download();
        std::cout << "  Convection term at test point: " << convection.u_h(i_test, j_test, k_test) << "\n";

        // Test 10: CalculateDiffusion
        std::cout << "\nTest 10: CalculateDiffusion\n";
        Field diffusion(&grid, "diffusion");
        CalculateDiffusion(psi_f, phi_diff, diffusion);
        diffusion.download();
        std::cout << "  Diffusion term at test point: " << diffusion.u_h(i_test, j_test, k_test) << "\n";

        // Test 11: CalculateSharpening
        std::cout << "\nTest 11: CalculateSharpening\n";
        Face norm_f(&grid, "norm_face");
        Field sharpening(&grid, "sharpening");
        CalculateFaceVals(norm_x, norm_y, norm_z, norm_f);
        CalculateSharpening(psi_f, norm_f, phi_diff, sharpening);
        sharpening.download();
        std::cout << "  Sharpening term at test point: " << sharpening.u_h(i_test, j_test, k_test) << "\n";

        // Test 12: SDF conversion functions
        std::cout << "\nTest 12: SDF conversions\n";
        Field sdf(&grid, "sdf");
        Field phase(&grid, "phase");
        Field epsilon(&grid, "epsilon");
        sdf.fill(0.1);
        epsilon.fill(0.03);
        sdf.upload();
        epsilon.upload();
        CalculateSDF2Phase(sdf, epsilon, phase);
        phase.download();
        std::cout << "  SDF=0.1, epsilon=0.03 -> phase = " << phase.u_h(Nx / 2, Ny / 2, 1) << " (should be ~1.0)\n";

        CalculatePhaseToSdf(phase, epsilon, sdf);
        sdf.download();
        std::cout << "  phase -> SDF = " << sdf.u_h(Nx / 2, Ny / 2, 1) << " (should recover ~0.1)\n";

        // Test 13: CalculateAbsoluteValue (scalar)
        std::cout << "\nTest 13: CalculateAbsoluteValue (scalar)\n";
        Field neg_field(&grid, "negative");
        Field abs_field(&grid, "absolute");
        neg_field.fill(-5.0);
        neg_field.upload();
        CalculateAbsoluteValue(neg_field, abs_field);
        abs_field.download();
        std::cout << "  Input: -5.0, Output: " << abs_field.u_h(i_test, j_test, k_test) << " (should be 5.0)\n";

        // Test 14: CalculateAbsoluteValue (vector)
        std::cout << "\nTest 14: CalculateAbsoluteValue (vector)\n";
        Field mag_field(&grid, "magnitude");
        vx.fill(3.0);
        vy.fill(4.0);
        vz.fill(0.0);
        vx.upload();
        vy.upload();
        vz.upload();
        CalculateAbsoluteValue(vx, vy, vz, mag_field);
        mag_field.download();
        std::cout << "  Input: (3, 4, 0), Magnitude: " << mag_field.u_h(i_test, j_test, k_test) << " (should be 5.0)\n";

        // Test 15: CalculateMaxMagnitude
        std::cout << "\nTest 15: CalculateMaxMagnitude\n";
        real_t max_val;
        CalculateMaxMagnitude(mag_field, max_val);
        std::cout << "  Maximum magnitude in field: " << max_val << " (should be 5.0)\n";

        // Test 16: CalculateBoundaryValues
        std::cout << "\nTest 16: CalculateBoundaryValues\n";
        Field bc_field(&grid, "bc_test");
        bc_field.fill(1.0);
        bc_field.upload();

        BC.W = 2;
        BC.E = 2; // Periodic
        BC.S = 2;
        BC.N = 2;
        BC.B = 2;
        BC.F = 2;
        BC.W_val = 0.0;
        BC.E_val = 0.0;
        BC.S_val = 0.0;
        BC.N_val = 0.0;
        BC.B_val = 0.0;
        BC.F_val = 0.0;

        CalculateBoundaryValues(bc_field, BC);
        bc_field.download();
        std::cout << "  Boundary conditions applied (periodic)\n";
        std::cout << "  Ghost cell value: " << bc_field.u_h(0, j_test, k_test) << "\n";

        std::cout << "\n=== All Calculator Tests Complete ===\n";
    }
    Kokkos::finalize();

    return 0;
}
