// file: src/INCLUDE/bctype.hpp
#ifndef BCTYPE_HPP
#define BCTYPE_HPP

#include "structs.hpp"

// Boundary condition type constants
#define BC_DIRICHLET 0
#define BC_NEUMANN   1
#define BC_PERIODIC  2

struct bctype {
    // Boundary condition types (0: Dirichlet, 1: Neumann, 2: Periodic)
    int E;  // East (x+)
    int W;  // West (x-)
    int N;  // North (y+)
    int S;  // South (y-)
    int F;  // Front (z+)
    int B;  // Back (z-)
    
    // Boundary condition values (used for Dirichlet and Neumann)
    real_t E_val;  // East boundary value
    real_t W_val;  // West boundary value
    real_t N_val;  // North boundary value
    real_t S_val;  // South boundary value
    real_t F_val;  // Front boundary value
    real_t B_val;  // Back boundary value
    
    // Default constructor - sets all to Dirichlet with zero value
    bctype() : 
        E(BC_DIRICHLET), W(BC_DIRICHLET), 
        N(BC_DIRICHLET), S(BC_DIRICHLET), 
        F(BC_DIRICHLET), B(BC_DIRICHLET),
        E_val(0.0), W_val(0.0), 
        N_val(0.0), S_val(0.0), 
        F_val(0.0), B_val(0.0) {}
    
    // Constructor with type specification (all same type)
    bctype(int type, real_t value = 0.0) : 
        E(type), W(type), N(type), S(type), F(type), B(type),
        E_val(value), W_val(value), N_val(value), 
        S_val(value), F_val(value), B_val(value) {}
    
    // Check if any boundary is periodic
    bool hasPeriodicBC() const {
        return (E == BC_PERIODIC || W == BC_PERIODIC || 
                N == BC_PERIODIC || S == BC_PERIODIC || 
                F == BC_PERIODIC || B == BC_PERIODIC);
    }
    
    // Check if a specific pair is periodic (must be consistent)
    bool isPeriodicX() const { return (E == BC_PERIODIC && W == BC_PERIODIC); }
    bool isPeriodicY() const { return (N == BC_PERIODIC && S == BC_PERIODIC); }
    bool isPeriodicZ() const { return (F == BC_PERIODIC && B == BC_PERIODIC); }
    
    // Validate boundary conditions
    bool validate() const {
        // Check that boundary types are valid
        if (E < 0 || E > 2 || W < 0 || W > 2 || 
            N < 0 || N > 2 || S < 0 || S > 2 || 
            F < 0 || F > 2 || B < 0 || B > 2) {
            return false;
        }
        
        // Check that periodic BCs are paired
        if ((E == BC_PERIODIC) != (W == BC_PERIODIC)) {
            std::cerr << "ERROR: Periodic BC must be set on both E and W boundaries\n";
            return false;
        }
        if ((N == BC_PERIODIC) != (S == BC_PERIODIC)) {
            std::cerr << "ERROR: Periodic BC must be set on both N and S boundaries\n";
            return false;
        }
        if ((F == BC_PERIODIC) != (B == BC_PERIODIC)) {
            std::cerr << "ERROR: Periodic BC must be set on both F and B boundaries\n";
            return false;
        }
        
        return true;
    }
    
    // Print boundary condition info
    void print() const {
        auto bcTypeString = [](int type) {
            switch(type) {
                case BC_DIRICHLET: return "Dirichlet";
                case BC_NEUMANN:   return "Neumann";
                case BC_PERIODIC:  return "Periodic";
                default:           return "Unknown";
            }
        };
        
        std::cout << "Boundary Conditions:\n";
        std::cout << "  East  (x+): " << bcTypeString(E) << ", value = " << E_val << "\n";
        std::cout << "  West  (x-): " << bcTypeString(W) << ", value = " << W_val << "\n";
        std::cout << "  North (y+): " << bcTypeString(N) << ", value = " << N_val << "\n";
        std::cout << "  South (y-): " << bcTypeString(S) << ", value = " << S_val << "\n";
        std::cout << "  Front (z+): " << bcTypeString(F) << ", value = " << F_val << "\n";
        std::cout << "  Back  (z-): " << bcTypeString(B) << ", value = " << B_val << "\n";
    }
};

#endif // BCTYPE_HPP