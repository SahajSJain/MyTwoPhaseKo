// file: src/CALCULATORS/boundary_conditions.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

void CalculateBoundaryValues(Field &phi, bctype BC)
{
    const GridInfo* grid = phi.grid;
    
    // Get grid limits for abstraction
    const int x_cs = grid->xgrid->cs;
    const int x_ce = grid->xgrid->ce;
    const int x_bs = grid->xgrid->bs;
    const int x_be = grid->xgrid->be;
    
    const int y_cs = grid->ygrid->cs;
    const int y_ce = grid->ygrid->ce;
    const int y_bs = grid->ygrid->bs;
    const int y_be = grid->ygrid->be;
    
    const int z_cs = grid->zgrid->cs;
    const int z_ce = grid->zgrid->ce;
    const int z_bs = grid->zgrid->bs;
    const int z_be = grid->zgrid->be;
    
    // Extract BC info for device-safe capture
    int W_type = BC.W;
    int E_type = BC.E;
    int S_type = BC.S;
    int N_type = BC.N;
    int B_type = BC.B;
    int F_type = BC.F;
    real_t W_val = BC.W_val;
    real_t E_val = BC.E_val;
    real_t S_val = BC.S_val;
    real_t N_val = BC.N_val;
    real_t B_val = BC.B_val;
    real_t F_val = BC.F_val;
    
    // Device view for safe capture
    auto u = phi.u;
    
    // Get boundary policies
    auto xBoundary_policy = grid->xBoundary_policy;
    auto yBoundary_policy = grid->yBoundary_policy;
    auto zBoundary_policy = grid->zBoundary_policy;
    
    // West Boundary
    Kokkos::parallel_for("Boundary_Conditions_West",
    xBoundary_policy,
    KOKKOS_LAMBDA(const int j, const int k) {
        // Set first ghost cell
        if (W_type == BC_DIRICHLET) {
            u(x_cs-1,j,k) = static_cast<real_t>(2.0)*W_val - u(x_cs,j,k);
        } 
        else if (W_type == BC_NEUMANN) {
            u(x_cs-1,j,k) = u(x_cs,j,k);
        } 
        else if (W_type == BC_PERIODIC) {
            u(x_cs-1,j,k) = u(x_ce,j,k);
        }
        
        // Fill remaining ghost cells
        for (int ig = x_cs-2; ig >= x_bs; ig--) {
            if(W_type == BC_DIRICHLET || W_type == BC_NEUMANN) {
                u(ig,j,k) = u(ig+1,j,k);
            } else if (W_type == BC_PERIODIC) {
                u(ig,j,k) = u(x_ce + (ig - x_cs + 1),j,k);
            }
        }
    });
    Kokkos::fence();
    
    // East Boundary
    Kokkos::parallel_for("Boundary_Conditions_East",
    xBoundary_policy,
    KOKKOS_LAMBDA(const int j, const int k) {
        // Set first ghost cell
        if (E_type == BC_DIRICHLET) {
            u(x_ce+1,j,k) = static_cast<real_t>(2.0)*E_val - u(x_ce,j,k);
        } 
        else if (E_type == BC_NEUMANN) {
            u(x_ce+1,j,k) = u(x_ce,j,k);
        } 
        else if (E_type == BC_PERIODIC) {
            u(x_ce+1,j,k) = u(x_cs,j,k);
        }
        
        // Fill remaining ghost cells
        for (int ig = x_ce+2; ig <= x_be; ig++) {
            if(E_type == BC_DIRICHLET || E_type == BC_NEUMANN) {
                u(ig,j,k) = u(ig-1,j,k);
            } else if (E_type == BC_PERIODIC) {
                u(ig,j,k) = u(x_cs + (ig - x_ce - 1),j,k);
            }
        }
    });
    Kokkos::fence();
    
    // South Boundary
    Kokkos::parallel_for("Boundary_Conditions_South",
    yBoundary_policy,
    KOKKOS_LAMBDA(const int i, const int k) {
        // Set first ghost cell
        if (S_type == BC_DIRICHLET) {
            u(i,y_cs-1,k) = static_cast<real_t>(2.0)*S_val - u(i,y_cs,k);
        } 
        else if (S_type == BC_NEUMANN) {
            u(i,y_cs-1,k) = u(i,y_cs,k);
        } 
        else if (S_type == BC_PERIODIC) {
            u(i,y_cs-1,k) = u(i,y_ce,k);
        }
        
        // Fill remaining ghost cells
        for (int jg = y_cs-2; jg >= y_bs; jg--) {
            if(S_type == BC_DIRICHLET || S_type == BC_NEUMANN) {
                u(i,jg,k) = u(i,jg+1,k);
            } else if (S_type == BC_PERIODIC) {
                u(i,jg,k) = u(i,y_ce + (jg - y_cs + 1),k);
            }
        }
    });
    Kokkos::fence();
    
    // North Boundary
    Kokkos::parallel_for("Boundary_Conditions_North",
    yBoundary_policy,
    KOKKOS_LAMBDA(const int i, const int k) {
        // Set first ghost cell
        if (N_type == BC_DIRICHLET) {
            u(i,y_ce+1,k) = static_cast<real_t>(2.0)*N_val - u(i,y_ce,k);
        } 
        else if (N_type == BC_NEUMANN) {
            u(i,y_ce+1,k) = u(i,y_ce,k);
        } 
        else if (N_type == BC_PERIODIC) {
            u(i,y_ce+1,k) = u(i,y_cs,k);
        }
        
        // Fill remaining ghost cells
        for (int jg = y_ce+2; jg <= y_be; jg++) {
            if(N_type == BC_DIRICHLET || N_type == BC_NEUMANN) {
                u(i,jg,k) = u(i,jg-1,k);
            } else if (N_type == BC_PERIODIC) {
                u(i,jg,k) = u(i,y_cs + (jg - y_ce - 1),k);
            }
        }
    });
    Kokkos::fence();
    
    #ifndef USE_2D
    // Back Boundary
    Kokkos::parallel_for("Boundary_Conditions_Back",
    zBoundary_policy,
    KOKKOS_LAMBDA(const int i, const int j) {
        // Set first ghost cell
        if (B_type == BC_DIRICHLET) {
            u(i,j,z_cs-1) = static_cast<real_t>(2.0)*B_val - u(i,j,z_cs);
        } 
        else if (B_type == BC_NEUMANN) {
            u(i,j,z_cs-1) = u(i,j,z_cs);
        } 
        else if (B_type == BC_PERIODIC) {
            u(i,j,z_cs-1) = u(i,j,z_ce);
        }
        
        // Fill remaining ghost cells
        for (int kg = z_cs-2; kg >= z_bs; kg--) {
            if(B_type == BC_DIRICHLET || B_type == BC_NEUMANN) {
                u(i,j,kg) = u(i,j,kg+1);
            } else if (B_type == BC_PERIODIC) {
                u(i,j,kg) = u(i,j,z_ce + (kg - z_cs + 1));
            }
        }
    });
    Kokkos::fence();
    
    // Front Boundary
    Kokkos::parallel_for("Boundary_Conditions_Front",
    zBoundary_policy,
    KOKKOS_LAMBDA(const int i, const int j) {
        // Set first ghost cell
        if (F_type == BC_DIRICHLET) {
            u(i,j,z_ce+1) = static_cast<real_t>(2.0)*F_val - u(i,j,z_ce);
        } 
        else if (F_type == BC_NEUMANN) {
            u(i,j,z_ce+1) = u(i,j,z_ce);
        } 
        else if (F_type == BC_PERIODIC) {
            u(i,j,z_ce+1) = u(i,j,z_cs);
        }
        
        // Fill remaining ghost cells
        for (int kg = z_ce+2; kg <= z_be; kg++) {
            if(F_type == BC_DIRICHLET || F_type == BC_NEUMANN) {
                u(i,j,kg) = u(i,j,kg-1);
            } else if (F_type == BC_PERIODIC) {
                u(i,j,kg) = u(i,j,z_cs + (kg - z_ce - 1));
            }
        }
    });
    Kokkos::fence();
    #endif // USE_2D
}