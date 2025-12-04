// file: src/CALCULATORS/max_magnitude.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

// max magnitude of a scalar field
void CalculateMaxMagnitude(Field &phi, real_t &max_magnitude)
{
    const GridInfo* grid = phi.grid;
    // Get grid dimensions using proper bounds
    auto interior_policy = grid->interior_policy;
    // Device view for safe capture
    auto u = phi.u;

    // Create a Kokkos reduction to find the maximum magnitude
    real_t local_max = 0.0;

    Kokkos::parallel_reduce("Max_Magnitude",
    interior_policy,
    KOKKOS_LAMBDA(const int i, const int j, const int k, real_t& thread_max) {
        real_t val = Kokkos::abs(u(i,j,k));
        if (val > thread_max) {
            thread_max = val;
        }
    }, 
    Kokkos::Max<real_t>(local_max));

    Kokkos::fence();

    // Update the output maximum magnitude
    max_magnitude = local_max;
}