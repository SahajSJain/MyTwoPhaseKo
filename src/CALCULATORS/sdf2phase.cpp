// file: src/CALCULATORS/sdf2phase.cpp
#include "../INCLUDE/include.hpp"
#include "calculators.hpp"

void CalculateSDF2Phase( Field& psi,      // Signed distance function
                        Field& epsilon,  // Interface thickness
                        Field& phi              // Phase field (output)) 
)
{
    auto grid = phi.grid;
    auto psi_d = psi.u;
    auto eps_d = epsilon.u;
    auto phi_d = phi.u;
    
    auto FullPolicy = grid->full_policy;
    
    Kokkos::parallel_for("SdfToPhase",
        FullPolicy,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            real_t psi_val = psi_d(i, j, k);
            real_t eps_val = eps_d(i, j, k);
            
            // phi = 0.5*(1 + tanh(psi/(2*epsilon)))
            phi_d(i, j, k) = 0.5 * (1.0 + Kokkos::tanh(psi_val / (2.0 * eps_val)));
        });
    
    Kokkos::fence();
}