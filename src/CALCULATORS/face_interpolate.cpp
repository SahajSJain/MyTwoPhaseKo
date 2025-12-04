// file: src/CALCULATORS/face_interpolate.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"
// interpolate field values to faces using central differencing
void CalculateFaceVals( Field &phi, 
                          Face &phi_f ) 
{
    const GridInfo* grid = phi.grid;
    // Get grid dimensions
    auto interior_policy = grid->interior_policy;
    auto inp_x = grid->xgrid->inp; 
    auto inp_y = grid->ygrid->inp;
    auto inp_z = grid->zgrid->inp;
    // Store device views for capture
    auto phi_u = phi.u;
    auto phi_f_E = phi_f.E.u;
    auto phi_f_W = phi_f.W.u;
    auto phi_f_N = phi_f.N.u;
    auto phi_f_S = phi_f.S.u;
    #ifndef USE_2D
    auto phi_f_F = phi_f.F.u;
    auto phi_f_B = phi_f.B.u;
    #endif
    // Interpolate to East and West faces (x-direction)
    Kokkos::parallel_for("CD2_Field_interpolate", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Interpolation factor: inp(i) = (f(i) - c(i-1)) / (c(i) - c(i-1))
        // That means, for central difference, 
        //   east face i+1/2 uses cell i and i+1 
        //   then phi(i+1/2) = (f(i+1) - c(i)) / (c(i+1) - c(i)) * phi(i+1) + ...
        //                     (c(i+1) - f(i+1)) / (c(i+1) - c(i)) * phi(i) 
        //   i.e. phi(i+1/2) = inp(i+1) * phi(i+1) + (1 - inp(i+1)) * phi(i)
        // similaryly for west face i-1/2 uses cell i-1 and i 
        //   then phi(i-1/2) = inp(i) * phi(i) + (1 - inp(i)) * phi(i-1) 
        phi_f_E(i,j,k) = inp_x(i+1) * phi_u(i+1,j,k) + (1 - inp_x(i+1)) * phi_u(i,j,k);
        phi_f_W(i,j,k) = inp_x(i) * phi_u(i,j,k) + (1 - inp_x(i)) * phi_u(i-1,j,k);

        phi_f_N(i,j,k) = inp_y(j+1) * phi_u(i,j+1,k) + (1 - inp_y(j+1)) * phi_u(i,j,k);
        phi_f_S(i,j,k) = inp_y(j) * phi_u(i,j,k) + (1 - inp_y(j)) * phi_u(i,j-1,k);
        // for debugging, use simple average instead of interpolation factors
        // phi_f.E.u(i,j,k) = 0.5*(phi.u(i+1,j,k) + phi.u(i,j,k));
        // phi_f.W.u(i,j,k) = 0.5*(phi.u(i,j,k) + phi.u(i-1,j,k));
        // phi_f.N.u(i,j,k) = 0.5*(phi.u(i,j+1,k) + phi.u(i,j,k));
        // phi_f.S.u(i,j,k) = 0.5*(phi.u(i,j,k) + phi.u(i,j-1,k));
        #ifndef USE_2D
        phi_f_F(i,j,k) = inp_z(k+1) * phi_u(i,j,k+1) + (1 - inp_z(k+1)) * phi_u(i,j,k);
        phi_f_B(i,j,k) = inp_z(k) * phi_u(i,j,k) + (1 - inp_z(k)) * phi_u(i,j,k-1);
        #endif
    });
    Kokkos::fence();
}
// interpolate vector field values to faces using central differencing 
// phi_x -> East and West faces
// phi_y -> North and South faces
// phi_z -> Front and Back faces
void CalculateFaceVals( Field &phi_x, 
                            Field &phi_y,
                            Field &phi_z,
                            Face &phi_f ) 
{
    // Same as before but now we have a vector field (phi_x, phi_y, phi_z)
    // phi_x -> East and West faces
    // phi_y -> North and South faces
    // phi_z -> Front and Back faces 
    const GridInfo* grid = phi_x.grid;
    // Get grid dimensions
    auto interior_policy = grid->interior_policy;
    auto inp_x = grid->xgrid->inp; 
    auto inp_y = grid->ygrid->inp;
    auto inp_z = grid->zgrid->inp;
    // Store device views for capture
    auto phi_f_E = phi_f.E.u;
    auto phi_f_W = phi_f.W.u;
    auto phi_f_N = phi_f.N.u;
    auto phi_f_S = phi_f.S.u;
    #ifndef USE_2D
    auto phi_f_F = phi_f.F.u;
    auto phi_f_B = phi_f.B.u;
    #endif
    auto phi_x_u = phi_x.u;
    auto phi_y_u = phi_y.u;
    auto phi_z_u = phi_z.u;

    // Interpolate to East and West faces (x-direction)
    Kokkos::parallel_for("CD2_Vector_interpolate", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Interpolation factor: inp(i) = (f(i) - c(i-1)) / (c(i) - c(i-1))
        // That means, for central difference, 
        //   east face i+1/2 uses cell i and i+1 
        //   then phi(i+1/2) = (f(i+1) - c(i)) / (c(i+1) - c(i)) * phi(i+1) + ...
        //                     (c(i+1) - f(i+1)) / (c(i+1) - c(i)) * phi(i) 
        //   i.e. phi(i+1/2) = inp(i+1) * phi(i+1) + (1 - inp(i+1)) * phi(i)
        // similaryly for west face i-1/2 uses cell i-1 and i 
        //   then phi(i-1/2) = inp(i) * phi(i) + (1 - inp(i)) * phi(i-1) 
        phi_f_E(i,j,k) = inp_x(i+1) * phi_x_u(i+1,j,k) + (1 - inp_x(i+1)) * phi_x_u(i,j,k);
        phi_f_W(i,j,k) = inp_x(i) * phi_x_u(i,j,k) + (1 - inp_x(i)) * phi_x_u(i-1,j,k);

        phi_f_N(i,j,k) = inp_y(j+1) * phi_y_u(i,j+1,k) + (1 - inp_y(j+1)) * phi_y_u(i,j,k);
        phi_f_S(i,j,k) = inp_y(j) * phi_y_u(i,j,k) + (1 - inp_y(j)) * phi_y_u(i,j-1,k);
        // phi_f.E.u(i,j,k) = 0.5*(phi_x.u(i+1,j,k) + phi_x.u(i,j,k));
        // phi_f.W.u(i,j,k) = 0.5*(phi_x.u(i,j,k) + phi_x.u(i-1,j,k));
        // phi_f.N.u(i,j,k) = 0.5*(phi_y.u(i,j+1,k) + phi_y.u(i,j,k));
        // phi_f.S.u(i,j,k) = 0.5*(phi_y.u(i,j,k) + phi_y.u(i,j-1,k));
        #ifndef USE_2D
        // phi_f.F.u(i,j,k) = 0.5*(phi_z.u(i,j,k+1) + phi_z.u(i,j,k));
        // phi_f.B.u(i,j,k) = 0.5*(phi_z.u(i,j,k) + phi_z.u(i,j,k-1));
        phi_f_F(i,j,k) = inp_z(k+1) * phi_z_u(i,j,k+1) + (1 - inp_z(k+1)) * phi_z_u(i,j,k);
        phi_f_B(i,j,k) = inp_z(k) * phi_z_u(i,j,k) + (1 - inp_z(k)) * phi_z_u(i,j,k-1);
        #endif
    });
    Kokkos::fence();
}


