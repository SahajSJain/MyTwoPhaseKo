// file: src/CALCULATORS/divergence.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

void CalculateDivergence( Field &phi_x, 
                        Field &phi_y,
                        Field &phi_z, 
                        Field &Div) 
{
    // calculate divergence from a vector field (phi_x, phi_y, phi_z)  
    // Same operation as gradient but summing up the three components 
    const GridInfo* grid = phi_x.grid;
    // Get grid dimensions using proper bounds
    auto interior_policy = grid->interior_policy;
    auto dxf = grid->xgrid->df;
    auto dyf = grid->ygrid->df;
    auto dzf = grid->zgrid->df;
    // Store device views for capture
    auto phi_x_u = phi_x.u;
    auto phi_y_u = phi_y.u;
    auto phi_z_u = phi_z.u;
    auto div_u   = Div.u; 
    // Gradient in x-direction
    // Interpolate to East and West faces (x-direction)
    Kokkos::parallel_for("Gradient_Field", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        real_t dxp = dxf(i+1); // c(i+1) - c(i)
        real_t dxm = dxf(i);   // c(i)   - c(i-1) 
        real_t dphi_x_dx = (phi_x_u(i+1,j,k)*dxm*dxm - phi_x_u(i-1,j,k)*dxp*dxp + 
                            phi_x_u(i,j,k)*(dxp*dxp - dxm*dxm)) / (dxp*dxm*(dxp + dxm));
        real_t dyp = dyf(j+1); // c(j+1) - c(j)
        real_t dym = dyf(j);   // c(j)   - c(j-1)
        real_t dphi_y_dy = (phi_y_u(i,j+1,k)*dym*dym - phi_y_u(i,j-1,k)*dyp*dyp + 
                            phi_y_u(i,j,k)*(dyp*dyp - dym*dym)) / (dyp*dym*(dyp + dym));
        #ifndef USE_2D
        real_t dzp = dzf(k+1); // c(k+1) -
        real_t dzm = dzf(k);   // c(k)   - c(k-1)
        real_t dphi_z_dz = (phi_z_u(i,j,k+1)*dzm*dzm - phi_z_u(i,j,k-1)*dzp*dzp + 
                            phi_z_u(i,j,k)*(dzp*dzp - dzm*dzm)) / (dzp*dzm*(dzp + dzm));
        div_u(i,j,k) = dphi_x_dx + dphi_y_dy + dphi_z_dz;
        #else 
        div_u(i,j,k) = dphi_x_dx + dphi_y_dy;
        #endif
    });
    Kokkos::fence();
}

void CalculateDivergence( Face &phi_f, 
                        Field &Div) 
{
    // calculate divergence from face values (phi_f_e, phi_f_w .... )
    // dphi_x/dx = (phi_f_e - phi_f_w)/dx etc. 
    // where dx = dc_x(i) = f(i+1) - f(i);   
    const GridInfo* grid = phi_f.grid;
    // Get grid dimensions using proper bounds
    auto interior_policy = grid->interior_policy;
    auto dx  = grid->xgrid->dc;
    auto dy  = grid->ygrid->dc;
    auto dz  = grid->zgrid->dc; 
    // Store device views for capture
    auto div_u   = Div.u;
    auto phi_f_E = phi_f.E.u;
    auto phi_f_W = phi_f.W.u;
    auto phi_f_N = phi_f.N.u;
    auto phi_f_S = phi_f.S.u;
    #ifndef USE_2D
    auto phi_f_F = phi_f.F.u;
    auto phi_f_B = phi_f.B.u; 
    #endif
    // Gradient in x-direction
    // Interpolate to East and West faces (x-direction)
    Kokkos::parallel_for("Gradient_Field", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        real_t dphi_x_dx = (phi_f_E(i,j,k) - phi_f_W(i,j,k)) / dx(i);
        real_t dphi_y_dy = (phi_f_N(i,j,k) - phi_f_S(i,j,k)) / dy(j);
        #ifndef USE_2D
        real_t dphi_z_dz = (phi_f_F(i,j,k) - phi_f_B(i,j,k)) / dz(k);
        div_u(i,j,k) = dphi_x_dx + dphi_y_dy + dphi_z_dz;
        #else
        div_u(i,j,k) = dphi_x_dx + dphi_y_dy;
        #endif
    });
    Kokkos::fence();
}

