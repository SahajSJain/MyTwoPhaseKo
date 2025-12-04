// file: src/CALCULATORS/normals.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"
// If we have a vector field psi = (psi_x, psi_y, psi_z) 
// Normalize it to get unit normal vector field 
void CalculateNormals(  Field &norm_x, 
                        Field &norm_y,
                        Field &norm_z) 
{
    const GridInfo* grid = norm_x.grid;
    // Get grid dimensions using proper bounds
    auto interior_policy = grid->interior_policy;
    // Store device views for capture
    auto norm_x_u = norm_x.u;
    auto norm_y_u = norm_y.u;
    auto norm_z_u = norm_z.u;
    auto small_number = REAL_EPSILON;
    Kokkos::parallel_for("Normalize_Gradient", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        #ifdef USE_2D
            real_t mag = Kokkos::sqrt( norm_x_u(i,j,k)*norm_x_u(i,j,k) + 
                               norm_y_u(i,j,k)*norm_y_u(i,j,k) + small_number );
        #else
            real_t mag = Kokkos::sqrt( norm_x_u(i,j,k)*norm_x_u(i,j,k) + 
                               norm_y_u(i,j,k)*norm_y_u(i,j,k) + 
                               norm_z_u(i,j,k)*norm_z_u(i,j,k) + small_number );
        #endif
        norm_x_u(i,j,k) /= mag;
        norm_y_u(i,j,k) /= mag;
        #ifndef USE_2D
        norm_z_u(i,j,k) /= mag;
        #endif
    });
    Kokkos::fence();
}

