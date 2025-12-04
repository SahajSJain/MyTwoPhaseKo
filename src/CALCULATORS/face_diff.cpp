// file: src/CALCULATORS/face_diff.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

void CalculateFaceDiffs( Field &phi, 
                          Face &phi_f ) 
{
    const GridInfo* grid = phi.grid;
    // Get grid dimensions
    auto interior_policy = grid->interior_policy;
    auto xg = grid->xgrid->c;
    auto yg = grid->ygrid->c;
    auto zg = grid->zgrid->c;
    auto delx = grid->xgrid->df;
    auto dely = grid->ygrid->df;
    auto delz = grid->zgrid->df;
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
    Kokkos::parallel_for("CD2_Face_Diff", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        //  phi_f.E.u(i,j,k) = (phi.u(i+1,j,k) - phi.u(i,j,k)) / (xg(i+1) - xg(i));
        //  phi_f.W.u(i,j,k) = (phi.u(i,j,k) - phi.u(i-1,j,k)) / (xg(i) - xg(i-1));
        //  note: we have : df(i) = c(i) - c(i-1); and xg == c 
        //  therefore:
        phi_f_E(i,j,k) = (phi_u(i+1,j,k) - phi_u(i,j,k)) / delx(i+1);
        phi_f_W(i,j,k) = (phi_u(i,j,k) - phi_u(i-1,j,k)) / delx(i);

        phi_f_N(i,j,k) = (phi_u(i,j+1,k) - phi_u(i,j,k)) / dely(j+1);
        phi_f_S(i,j,k) = (phi_u(i,j,k) - phi_u(i,j-1,k)) / dely(j);
        #ifndef USE_2D
        phi_f_F(i,j,k) = (phi_u(i,j,k+1) - phi_u(i,j,k)) / delz(k+1);
        phi_f_B(i,j,k) = (phi_u(i,j,k) - phi_u(i,j,k-1)) / delz(k);
        #endif
    });
    Kokkos::fence();
}



