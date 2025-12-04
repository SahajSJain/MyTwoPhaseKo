// file: src/INCLUDE/phasefield.cpp
#include "phasefield.hpp"
#include "../CALCULATORS/calculators.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>

// Initialize all fields and parameters
void PhaseField::initialize()
{
    // Allocate all fields
    allocateFields();

    // Compute epsilon field based on grid spacing
    computeEpsilon(epsilon);

    // Set initial condition for psi
    setInitialConditionPsi(psi);

    // Convert to phi
    CalculateSDF2Phase(psi, epsilon, phi);

    // Store initial phi
    phi_old.deepCopy(phi);

    // Set initial velocity field and compute face velocities
    setVelocity(0.0, u, v, w, vel_f);

    // Apply initial boundary conditions
    applyBoundaryConditions(phi);

    // Print initial summary
    printSummary();
}

// Allocate all fields
void PhaseField::allocateFields()
{
    // Primary fields
    phi = Field(grid.get(), "phi");
    phi_old = Field(grid.get(), "phi_old");
    psi = Field(grid.get(), "psi");

    // Velocity fields
    u = Field(grid.get(), "u");
    v = Field(grid.get(), "v");
    w = Field(grid.get(), "w");
    vel_mag = Field(grid.get(), "vel_mag");

    // Parameter field
    epsilon = Field(grid.get(), "epsilon");

    // Normal vector fields
    normx = Field(grid.get(), "normx");
    normy = Field(grid.get(), "normy");
    normz = Field(grid.get(), "normz");

    // Face values
    phi_f = Face(grid.get(), "phi_f");
    phi_diff_f = Face(grid.get(), "phi_diff_f"); // Add this
    psi_f = Face(grid.get(), "psi_f");
    epsilon_f = Face(grid.get(), "epsilon_f");
    norm_f = Face(grid.get(), "norm_f");
    vel_f = Face(grid.get(), "vel_f");

    // Terms
    Conv = Field(grid.get(), "Conv");
    Diff = Field(grid.get(), "Diff");
    Sharp = Field(grid.get(), "Sharp");
    RHS = Field(grid.get(), "RHS");
}

// Upload all fields to device
void PhaseField::uploadToDevice()
{
    phi.upload();
    phi_old.upload();
    psi.upload();
    u.upload();
    v.upload();
    w.upload();
    vel_mag.upload();
    epsilon.upload();
    normx.upload();
    normy.upload();
    normz.upload();
    Conv.upload();
    Diff.upload();
    Sharp.upload();
    RHS.upload();
}

// Download only fields needed for output
void PhaseField::downloadFromDevice()
{
    phi.download();
    psi.download();
    u.download();
    v.download();
    w.download();
}

// Compute epsilon field based on grid spacing
void PhaseField::computeEpsilon(Field &epsilon_field)
{
    auto eps_d = epsilon_field.u;
    auto FullPolicy = grid->full_policy; 

    auto dc_x = grid->xgrid->dc;
    auto dc_y = grid->ygrid->dc;
    auto dc_z = grid->zgrid->dc;

    const real_t eps_fac = epsilon_fac;

    Kokkos::parallel_for("ComputeEpsilon", FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifdef USE_2D
            real_t max_spacing = Kokkos::max(dc_x(i), dc_y(j));
#else
            real_t max_spacing = Kokkos::max(Kokkos::max(dc_x(i), dc_y(j)), dc_z(k));
#endif
            eps_d(i, j, k) = eps_fac * max_spacing; });
}

// Set initial condition for psi
void PhaseField::setInitialConditionPsi(Field &psi_field)
{
    auto psi_d = psi_field.u;
    auto FullPolicy = grid->full_policy;

    auto xc = grid->xgrid->c;
    auto yc = grid->ygrid->c;
    auto zc = grid->zgrid->c;

    const real_t x0_local = x0;
    const real_t y0_local = y0;
    const real_t z0_local = z0;
    const real_t r0_local = r0;
    const int mode_local = init_mode;

    Kokkos::parallel_for("SetInitialPsi", FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            real_t x = xc(i);
            real_t y = yc(j);
            real_t z = zc(k);
            
            // Distance from center
            real_t dist = Kokkos::sqrt((x-x0_local)*(x-x0_local) + 
                                       (y-y0_local)*(y-y0_local) + 
                                       (z-z0_local)*(z-z0_local));
            
            // Signed distance based on mode
            if (mode_local == 0) {
                // Mode 0: positive inside
                psi_d(i, j, k) = r0_local - dist;
            } else {
                // Mode 1: negative inside
                psi_d(i, j, k) = dist - r0_local;
            } });
}

// Set velocity field and compute face velocities
void PhaseField::setVelocity(real_t t, Field &u_field, Field &v_field,
                             Field &w_field, Face &vel_face)
{
    auto u_d = u_field.u;
    auto v_d = v_field.u;
    auto w_d = w_field.u;

    auto FullPolicy = grid->full_policy;

    auto xc = grid->xgrid->c;
    auto yc = grid->ygrid->c;
    auto zc = grid->zgrid->c;

    // Time period for velocity field
    const real_t T = 4.0; // Adjust as needed

    // Deformation velocity field
    Kokkos::parallel_for("SetVelocity", FullPolicy, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            real_t x = xc(i);
            real_t y = yc(j);
#ifndef USE_2D
            real_t z = zc(k);
#endif
            
            // Deformation velocity field
            const real_t pi = M_PI;
            real_t cos_time = cos(pi * t / T);
            
            u_d(i, j, k) = -sin(pi * x) * sin(pi * x) * sin(2.0 * pi * y) * cos_time;
            v_d(i, j, k) = sin(2.0 * pi * x) * sin(pi * y) * sin(pi * y) * cos_time;
            w_d(i, j, k) = 0.0; });

    // Compute face velocities
    CalculateFaceVals(u_field, v_field, w_field, vel_face);
}

// Get maximum velocity magnitude
real_t PhaseField::getMaxVelocity(Field &u_field, Field &v_field,
                                  Field &w_field, Field &vel_mag_field)
{
    // Use CalculateAbsoluteValue for vector field
    CalculateAbsoluteValue(u_field, v_field, w_field, vel_mag_field);

    // Use CalculateMaxMagnitude to get the maximum
    real_t max_vel;
    CalculateMaxMagnitude(vel_mag_field, max_vel);

    return max_vel;
}

// Interpolate fields to faces
void PhaseField::interpolateToFaces(Field &phi_field, Field &psi_field,
                                    Field &epsilon_field,
                                    Field &normx_field, Field &normy_field,
                                    Field &normz_field,
                                    Face &phi_face, Face &psi_face, Face &epsilon_face, Face &norm_face)
{
    // Interpolate phi to faces
    CalculateFaceVals(phi_field, phi_face);

    // Interpolate psi to faces
    CalculateFaceVals(psi_field, psi_face);

    // Interpolate epsilon to faces
    CalculateFaceVals(epsilon_field, epsilon_face);

    // Interpolate normals to faces
    CalculateFaceVals(normx_field, normy_field, normz_field, norm_face);
}

// Compute normal vectors from psi
void PhaseField::computeNormals(Field &psi_field,
                                Field &normx_out, Field &normy_out, Field &normz_out)
{
    // Compute gradients of psi directly into the normal arrays
    CalculateGradients(psi_field, normx_out, normy_out, normz_out);

    // Normalize in-place to get unit normals
    CalculateNormals(normx_out, normy_out, normz_out);
}

// Compute RHS = -Conv + Gamma*Diff - Gamma*Sharp
void PhaseField::computeRHS(Field &phi_field, Face &phi_face, Face &psi_face,
                            Face &epsilon_face, Face &norm_face, Face &vel_face,
                            Field &Conv_out, Field &Diff_out, Field &Sharp_out,
                            Field &RHS_out)
{
    // Compute convection term: div(u*phi)
    CalculateConvection(phi_face, vel_face, Conv_out);

    // Compute face differences for phi (gradients at faces)
    CalculateFaceDiffs(phi_field, phi_diff_f);

    // Compute diffusion term: div(epsilon*grad(phi))
    CalculateDiffusion(phi_diff_f, epsilon_face, Diff_out);

    // Compute sharpening term: div((1/4)*[1-tanh²(psi/2eps)]*normal)
    CalculateSharpening(psi_face, norm_face, epsilon_face, Sharp_out);

    // Combine terms with proper signs and Gamma
    auto grid_local = grid.get();
    auto interior_policy = grid_local->interior_policy;

    auto Conv_d = Conv_out.u;
    auto Diff_d = Diff_out.u;
    auto Sharp_d = Sharp_out.u;
    auto RHS_d = RHS_out.u;
    const real_t Gamma_local = Gamma;

    Kokkos::parallel_for("ComputeRHS", 
        interior_policy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) 
        { 
            RHS_d(i, j, k) =
                     - Conv_d(i, j, k) 
                     + Gamma_local * Diff_d(i, j, k) 
                     - Gamma_local * Sharp_d(i, j, k); 
        });

    Kokkos::fence();
}

// Update phi using Euler explicit
void PhaseField::updatePhi(Field &phi_field, Field &RHS_field, real_t dt_local)
{
    auto phi_d = phi_field.u;
    auto RHS_d = RHS_field.u;

    
    auto interior_policy = grid->interior_policy;
    Kokkos::parallel_for("UpdatePhi", interior_policy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) 
        { phi_d(i, j, k) = phi_d(i, j, k) + dt_local * RHS_d(i, j, k); });

    Kokkos::fence();

    // Apply boundary conditions
    applyBoundaryConditions(phi_field);
}

// Main time stepping function
// Main time stepping function
void PhaseField::advanceTimeStep()
{
    // Store old phi (only deepCopy we need)
    phi_old.deepCopy(phi);

    // Update velocity field for current time
    setVelocity(time, u, v, w, vel_f);

    // Convert phi to psi
    CalculatePhaseToSdf(phi, epsilon, psi);

    // Compute normals directly from psi
    computeNormals(psi, normx, normy, normz);

    // Interpolate fields to faces
    interpolateToFaces(phi, psi, epsilon, normx, normy, normz, phi_f, psi_f, epsilon_f, norm_f);

    // Compute RHS
    computeRHS(phi, phi_f, psi_f, epsilon_f, norm_f, vel_f, Conv, Diff, Sharp, RHS);

    // Update phi
    updatePhi(phi, RHS, dt);

    // Update time and step
    time += dt;
    step++;
}

// Apply boundary conditions
void PhaseField::applyBoundaryConditions(Field &field)
{
    CalculateBoundaryValues(field, BC);
}

// Check if phi is bounded between 0 and 1
void PhaseField::checkBounds(Field &phi_field)
{
    auto phi_d = phi_field.u;

    auto interior_policy = grid->interior_policy;

    real_t min_val, max_val;

    Kokkos::parallel_reduce("CheckBounds", interior_policy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k, real_t &lmin, real_t &lmax) {
            lmin = Kokkos::min(lmin, phi_d(i, j, k));
            lmax = Kokkos::max(lmax, phi_d(i, j, k)); 
        }, Kokkos::Min<real_t>(min_val), Kokkos::Max<real_t>(max_val));

    if (min_val < -0.01 || max_val > 1.01)
    {
        std::cout << "WARNING: Phi out of bounds! Min: " << min_val << ", Max: " << max_val << std::endl;
    }
}

// Compute total mass (integral of phi)
real_t PhaseField::computeMass(Field &phi_field)
{
    auto phi_d = phi_field.u;
    auto Vols = grid->Vols;

    auto interior_policy = grid->interior_policy;

    real_t total_mass = 0.0;

    Kokkos::parallel_reduce("ComputeMass", interior_policy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k, real_t &mass) 
        { 
            mass += phi_d(i, j, k) * Vols(i, j, k); 
        }, 
        Kokkos::Sum<real_t>(total_mass));
        
    return total_mass;
}

// Print summary
void PhaseField::printSummary()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase Field Solver Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Grid: " << grid->xgrid->N << " x " << grid->ygrid->N << " x " << grid->zgrid->N << std::endl;
    std::cout << "Ghost layers: " << grid->xgrid->Ngl << std::endl;
    std::cout << "Time step: " << dt << std::endl;
    std::cout << "Epsilon factor: " << epsilon_fac << std::endl;
    std::cout << "Gamma: " << Gamma << std::endl;
    std::cout << "Initial condition: Circle at (" << x0 << ", " << y0 << ", " << z0 << ") with radius " << r0 << std::endl;
    std::cout << "Mode: " << (init_mode == 0 ? "Positive inside" : "Negative inside") << std::endl;
    std::cout << "========================================\n"
              << std::endl;
}

// Print diagnostics
void PhaseField::printDiagnostics()
{
    psi.download();
    real_t mass = computeMass(phi);
    real_t max_vel = getMaxVelocity(u, v, w, vel_mag);
    std::cout << "Step: " << std::setw(6) << step
              << " | Time: " << std::setw(10) << std::setprecision(6) << time
              << " | Mass: " << std::setw(12) << std::setprecision(8) << mass
              << " | Max vel: " << std::setw(10) << std::setprecision(6) << max_vel
              << "\n";
    checkBounds(phi);
}

// Write Tecplot output with automatic filename
void PhaseField::writeTecplot()
{
    // Format: phase.XXXXXXX.dat where X is zero-padded step number
    std::ostringstream filename;
    filename << "phase." << std::setfill('0') << std::setw(7) << step << ".dat";
    downloadFromDevice();

#ifdef USE_2D
    writeTecplot2D(filename.str());
#else
    writeTecplot3D(filename.str());
#endif
}

// Write 2D Tecplot output
void PhaseField::writeTecplot2D(const std::string &filename)
{
    // Create TECOUT directory if it doesn't exist
    mkdir("TECOUT", 0755);

    std::string fullpath = "TECOUT/" + filename;
    std::ofstream tecout(fullpath);

    if (!tecout.is_open())
    {
        std::cerr << "Error: Cannot open output file '" << fullpath << "'\n";
        return;
    }

    tecout << "Title = \"Phase Field 2D\" \n";
    tecout << "Variables = \"X\", \"Y\", \"Phi\", \"Psi\", \"U\", \"V\" \n";
    tecout << "Zone T=\"Step " << step << "\", I=" << grid->xgrid->N
           << ", J=" << grid->ygrid->N << ", F=POINT \n";

    // Write data
    for (int j = 1; j <= grid->ygrid->N; j++)
    {
        for (int i = 1; i <= grid->xgrid->N; i++)
        {
            tecout << std::scientific << std::setprecision(8)
                   << grid->xgrid->c_h(i) << " "
                   << grid->ygrid->c_h(j) << " "
                   << phi.u_h(i, j, 1) << " "
                   << psi.u_h(i, j, 1) << " "
                   << u.u_h(i, j, 1) << " "
                   << v.u_h(i, j, 1) << "\n";
        }
    }

    tecout.close();
}

// Write 3D Tecplot output
void PhaseField::writeTecplot3D(const std::string &filename)
{
    // Create TECOUT directory if it doesn't exist
    mkdir("TECOUT", 0755);

    std::string fullpath = "TECOUT/" + filename;
    std::ofstream tecout(fullpath);

    if (!tecout.is_open())
    {
        std::cerr << "Error: Cannot open output file '" << fullpath << "'\n";
        return;
    }

    tecout << "Title = \"Phase Field 3D\" \n";
    tecout << "Variables = \"X\", \"Y\", \"Z\", \"Phi\", \"Psi\", \"U\", \"V\", \"W\" \n";
    tecout << "Zone T=\"Step " << step << "\", I=" << grid->xgrid->N
           << ", J=" << grid->ygrid->N
           << ", K=" << grid->zgrid->N << ", F=POINT \n";

    // Write data
    for (int k = 1; k <= grid->zgrid->N; k++)
    {
        for (int j = 1; j <= grid->ygrid->N; j++)
        {
            for (int i = 1; i <= grid->xgrid->N; i++)
            {
                tecout << std::scientific << std::setprecision(8)
                       << grid->xgrid->c_h(i) << " "
                       << grid->ygrid->c_h(j) << " "
                       << grid->zgrid->c_h(k) << " "
                       << phi.u_h(i, j, k) << " "
                       << psi.u_h(i, j, k) << " "
                       << u.u_h(i, j, k) << " "
                       << v.u_h(i, j, k) << " "
                       << w.u_h(i, j, k) << "\n";
            }
        }
    }

    tecout.close();
}