// file: src/CALCULATORS/phase2sdf.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

void CalculatePhaseToSdf(
    Field& phi,      // Phase field
    Field& epsilon,  // Interface thickness
    Field& psi              // Signed distance function (output)
) {
    auto grid = psi.grid;
    auto phi_d = phi.u;
    auto eps_d = epsilon.u;
    auto psi_d = psi.u;
    
    auto FullPolicy = grid->full_policy; 
    
    const real_t small = REAL_EPSILON;
    
    Kokkos::parallel_for("PhaseToSdf",
        FullPolicy,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            real_t phi_val = phi_d(i, j, k);
            real_t eps_val = eps_d(i, j, k);
            
            // Clamp phi to avoid numerical issues
            phi_val = Kokkos::max(small, Kokkos::min(1.0 - small, phi_val));
            // psi = epsilon * log((phi+small)/  (1 - phi + small))            
            psi_d(i, j, k) = eps_val * 
                        Kokkos::log((phi_val + small) / (1.0 - phi_val + small));
        });
    
    Kokkos::fence();
}