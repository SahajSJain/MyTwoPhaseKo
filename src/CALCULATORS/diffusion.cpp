// file: src/CALCULATORS/diffusion.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"
// Solve diffusion term: Laplacian = div( K * grad(phi) )  
// Should have grad(phi) and K at faces already
// This is done so that we can compute the needed stuff outside this subroutine 
// And can reuse those face values if needed readily.

void CalculateDiffusion( Face &phi_diff_f,
                         Face &K_f,
                         Field &Laplacian) 
{
    // calculate diffusion term: Laplacian = div( K_diff * grad(phi) ) 
    // In this mode we have already computed grad(phi) at faces (phi_diff_f) 
    const GridInfo* grid = phi_diff_f.grid;
    // Get grid dimensions
    auto interior_policy = grid->interior_policy;
    auto xg = grid->xgrid->c;
    auto yg = grid->ygrid->c;
    auto zg = grid->zgrid->c;
    auto delx = grid->xgrid->dc;
    auto dely = grid->ygrid->dc;
    auto delz = grid->zgrid->dc;

    // Store device views for capture
    auto laplacian_u = Laplacian.u;
    auto phi_diff_f_E = phi_diff_f.E.u;
    auto phi_diff_f_W = phi_diff_f.W.u;
    auto phi_diff_f_N = phi_diff_f.N.u;
    auto phi_diff_f_S = phi_diff_f.S.u;
    #ifndef USE_2D
    auto phi_diff_f_F = phi_diff_f.F.u;
    auto phi_diff_f_B = phi_diff_f.B.u;
    #endif
    auto K_f_E = K_f.E.u;
    auto K_f_W = K_f.W.u;
    auto K_f_N = K_f.N.u;
    auto K_f_S = K_f.S.u;
    #ifndef USE_2D
    auto K_f_F = K_f.F.u;
    auto K_f_B = K_f.B.u;
    #endif
    // Interpolate to East and West faces (x-direction)
    Kokkos::parallel_for("Diffusion_Field", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        real_t Laplacian_x = (K_f_E(i,j,k) * phi_diff_f_E(i,j,k) - 
                                K_f_W(i,j,k) * phi_diff_f_W(i,j,k)) / delx(i); 
        real_t Laplacian_y = (K_f_N(i,j,k) * phi_diff_f_N(i,j,k) - 
                                K_f_S(i,j,k) * phi_diff_f_S(i,j,k)) / dely(j);
        #ifndef USE_2D
        real_t Laplacian_z = (K_f_F(i,j,k) * phi_diff_f_F(i,j,k) - 
                                K_f_B(i,j,k) * phi_diff_f_B(i,j,k)) / delz(k);
        laplacian_u(i,j,k) = Laplacian_x + Laplacian_y + Laplacian_z;
        #else
        laplacian_u(i,j,k) = Laplacian_x + Laplacian_y;
        #endif
    });
    Kokkos::fence();
}





