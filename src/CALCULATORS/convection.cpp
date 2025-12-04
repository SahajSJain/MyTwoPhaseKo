// file: src/CALCULATORS/convection.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"
// Should have phi and U at faces already 
// This is done so that we can compute the needed stuff outside this subroutine
// And can reuse those face values if needed readily. 
void CalculateConvection( Face &phi_f, 
                          Face &U_f,
                          Field &Conv) 
{
    // calculate convection term: Conv = div( U_f * phi_f ) 
    //    For x direction: Conv_x = ( U_f.E * phi_f.E - U_f.W * phi_f.W ) / dc_x(i) 
    //                       Conv = Conv_x + Conv_y + Conv_z 
    const GridInfo* grid = phi_f.grid;
    // Get grid dimensions
    auto interior_policy = grid->interior_policy;
    auto xg = grid->xgrid->c;
    auto yg = grid->ygrid->c;
    auto zg = grid->zgrid->c;
    auto delx = grid->xgrid->dc;
    auto dely = grid->ygrid->dc;
    auto delz = grid->zgrid->dc;
    // Capture the device views!
    auto conv_u = Conv.u;
    auto U_f_E = U_f.E.u;
    auto U_f_W = U_f.W.u;
    auto U_f_N = U_f.N.u;
    auto U_f_S = U_f.S.u;
    #ifndef USE_2D
    auto U_f_F = U_f.F.u;
    auto U_f_B = U_f.B.u;
    #endif
    auto phi_f_E = phi_f.E.u;
    auto phi_f_W = phi_f.W.u;
    auto phi_f_N = phi_f.N.u;
    auto phi_f_S = phi_f.S.u;
    #ifndef USE_2D
    auto phi_f_F = phi_f.F.u;
    auto phi_f_B = phi_f.B.u;
    #endif
    // First compute (phi) at faces (phi_diff_f) 
    // Interpolate to East and West faces (x-direction)

    Kokkos::parallel_for("Convection_Field", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // use captured views
        real_t Conv_x = (U_f_E(i,j,k) * phi_f_E(i,j,k) - 
                         U_f_W(i,j,k) * phi_f_W(i,j,k)) / delx(i); 
        real_t Conv_y = (U_f_N(i,j,k) * phi_f_N(i,j,k) - 
                         U_f_S(i,j,k) * phi_f_S(i,j,k)) / dely(j);
        #ifndef USE_2D
        real_t Conv_z = (U_f_F(i,j,k) * phi_f_F(i,j,k) - 
                         U_f_B(i,j,k) * phi_f_B(i,j,k)) / delz(k);
        conv_u(i,j,k) = Conv_x + Conv_y + Conv_z;
        #else
        conv_u(i,j,k) = Conv_x + Conv_y;
        #endif 
    });
    Kokkos::fence();
}


