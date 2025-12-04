// file: src/CALCULATORS/gradient.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"
#include <iostream>

// calculate grad(phi) = (phi_x, phi_y, phi_z) 
void CalculateGradients( Field &phi, 
                         Field &phi_x, 
                         Field &phi_y,
                         Field &phi_z) 
{
    const GridInfo* grid = phi.grid;
    auto interior_policy = grid->interior_policy;
    auto dxf = grid->xgrid->df;
    auto dyf = grid->ygrid->df;
    auto dzf = grid->zgrid->df;
    
    // Capture the device views!
    auto phi_u = phi.u;
    auto phi_x_u = phi_x.u;
    auto phi_y_u = phi_y.u;
    auto phi_z_u = phi_z.u;
    
    Kokkos::parallel_for("Gradient_Field", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        real_t dxp = dxf(i+1);
        real_t dxm = dxf(i);
        phi_x_u(i,j,k) = (phi_u(i+1,j,k)*dxm*dxm - phi_u(i-1,j,k)*dxp*dxp + 
                          phi_u(i,j,k)*(dxp*dxp - dxm*dxm)) / (dxp*dxm*(dxp + dxm));
        
        real_t dyp = dyf(j+1);
        real_t dym = dyf(j);
        phi_y_u(i,j,k) = (phi_u(i,j+1,k)*dym*dym - phi_u(i,j-1,k)*dyp*dyp + 
                          phi_u(i,j,k)*(dyp*dyp - dym*dym)) / (dyp*dym*(dyp + dym));
        
        #ifndef USE_2D
        real_t dzp = dzf(k+1);
        real_t dzm = dzf(k);
        phi_z_u(i,j,k) = (phi_u(i,j,k+1)*dzm*dzm - phi_u(i,j,k-1)*dzp*dzp + 
                          phi_u(i,j,k)*(dzp*dzp - dzm*dzm)) / (dzp*dzm*(dzp + dzm));
        #endif
    });
    Kokkos::fence();
}

