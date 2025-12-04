// file: src/CALCULATORS/sharpening.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"
// Assume we have psi at faces (psi_f),
// normals at faces (norm_f) and epsilon at faces (epsilon_f) 

// Calculate sharpening term: Sharp = Div(Sharp_flux)
// Where Sharp_flux = (1/4)*(1 - tanh^2(0.5*psi/epsilon))*norm_f 
// Where norm_f is the unit normal vector at faces = grad(psi)/|grad(psi)| 
void CalculateSharpening( Face &psi_f, 
                          Face &norm_f,
                          Face &epsilon_f,
                          Field &Sharp)
{
    // calculate convection term: Conv = div( U_f * phi_f ) 
    //    For x direction: Conv_x = ( U_f.E * phi_f.E - U_f.W * phi_f.W ) / dc_x(i) 
    //                       Conv = Conv_x + Conv_y + Conv_z 
    const GridInfo* grid = psi_f.grid;
    // Get grid dimensions
    auto Nx = grid->config.Nx;
    auto Ny = grid->config.Ny; 
    auto Nz = grid->config.Nz;
    auto xg = grid->xgrid->c;
    auto yg = grid->ygrid->c;
    auto zg = grid->zgrid->c;
    auto delx = grid->xgrid->dc;
    auto dely = grid->ygrid->dc;
    auto delz = grid->zgrid->dc;
    // Capture the device views!
    auto Sharp_u = Sharp.u;
    auto epsilon_f_E = epsilon_f.E.u;
    auto epsilon_f_W = epsilon_f.W.u;
    auto epsilon_f_N = epsilon_f.N.u;
    auto epsilon_f_S = epsilon_f.S.u;
    #ifndef USE_2D
    auto epsilon_f_F = epsilon_f.F.u;
    auto epsilon_f_B = epsilon_f.B.u;
    #endif
    auto psi_f_E = psi_f.E.u;
    auto psi_f_W = psi_f.W.u;
    auto psi_f_N = psi_f.N.u;
    auto psi_f_S = psi_f.S.u;
    #ifndef USE_2D
    auto psi_f_F = psi_f.F.u;
    auto psi_f_B = psi_f.B.u;
    #endif
    auto norm_f_E = norm_f.E.u;
    auto norm_f_W = norm_f.W.u;
    auto norm_f_N = norm_f.N.u;
    auto norm_f_S = norm_f.S.u;
    #ifndef USE_2D
    auto norm_f_F = norm_f.F.u;
    auto norm_f_B = norm_f.B.u;
    #endif
    // First compute (phi) at faces (phi_diff_f) 
    // Interpolate to East and West faces (x-direction)
    Kokkos::parallel_for("Diffusion_Field", 
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {1 , 1 , 1 }, 
        {Nx + 1, Ny + 1, Nz + 1}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        real_t Sharp_E = 0.25*(1.0-Kokkos::pow(Kokkos::tanh(0.5*psi_f_E(i,j,k)/epsilon_f_E(i,j,k)), 2))*
                        norm_f_E(i,j,k);
        real_t Sharp_W = 0.25*(1.0-Kokkos::pow(Kokkos::tanh(0.5*psi_f_W(i,j,k)/epsilon_f_W(i,j,k)), 2))*
                        norm_f_W(i,j,k);
        real_t Sharp_x = (Sharp_E - Sharp_W) / delx(i);
        real_t Sharp_N = 0.25*(1.0-Kokkos::pow(Kokkos::tanh(0.5*psi_f_N(i,j,k)/epsilon_f_N(i,j,k)), 2))*
                        norm_f_N(i,j,k);
        real_t Sharp_S = 0.25*(1.0-Kokkos::pow(Kokkos::tanh(0.5*psi_f_S(i,j,k)/epsilon_f_S(i,j,k)), 2))*
                        norm_f_S(i,j,k);
        real_t Sharp_y = (Sharp_N - Sharp_S) / dely(j);
        #ifndef USE_2D
        real_t Sharp_F = 0.25*(1.0-Kokkos::pow(Kokkos::tanh(0.5*psi_f_F(i,j,k)/epsilon_f_F(i,j,k)), 2))*
                        norm_f_F(i,j,k);
        real_t Sharp_B = 0.25*(1.0-Kokkos::pow(Kokkos::tanh(0.5*psi_f_B(i,j,k)/epsilon_f_B(i,j,k)), 2))*
                        norm_f_B(i,j,k);
        real_t Sharp_z = (Sharp_F - Sharp_B) / delz(k);
        Sharp_u(i,j,k) = Sharp_x + Sharp_y + Sharp_z;
        #else
        Sharp_u(i,j,k) = Sharp_x + Sharp_y;
        #endif
    });
    Kokkos::fence();
}