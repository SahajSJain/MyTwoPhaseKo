// file: src/CALCULATORS/absolute_value.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

// max magnitude of a scalar field
void CalculateAbsoluteValue(Field &phi, Field &abs_phi)
{
    const GridInfo* grid = phi.grid;
    // Get grid dimensions using proper bounds
    auto interior_policy = grid->interior_policy;
    // Device view for safe capture
    auto u = phi.u;
    auto abs_u = abs_phi.u;

    Kokkos::parallel_for("Absolute_Value", 
    interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        abs_u(i,j,k) = Kokkos::abs(u(i,j,k));
    });
    Kokkos::fence();
}

// max magnitude of a vector field
void CalculateAbsoluteValue( Field &phi_x, 
                             Field &phi_y, 
                             Field &phi_z, 
                             Field &abs_phi)
{
    const GridInfo* grid = phi_x.grid;
    // Get grid dimensions using proper bounds
    // Device view for safe capture
    auto u = phi_x.u;
    auto v = phi_y.u;
    auto w = phi_z.u;
    auto abs_u = abs_phi.u;
    auto interior_policy = grid->interior_policy;
    Kokkos::parallel_for("Absolute_Value",interior_policy, 
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        #ifndef USE_2D
        abs_u(i,j,k) = Kokkos::sqrt(u(i,j,k)*u(i,j,k) + 
                                     v(i,j,k)*v(i,j,k) + 
                                     w(i,j,k)*w(i,j,k));
        #else
        abs_u(i,j,k) = Kokkos::sqrt(u(i,j,k)*u(i,j,k) + 
                                     v(i,j,k)*v(i,j,k));
        #endif
    });
    Kokkos::fence();
}