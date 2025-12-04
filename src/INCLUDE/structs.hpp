// file: src/INCLUDE/structs.hpp
#ifndef TYPES_H
#define TYPES_H

#include <Kokkos_Core.hpp>
#include <stdbool.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

// Define precision based on compiler flag
#ifdef USE_DOUBLE
    typedef double real_t;
    #define REAL_MAX 1.0e10
    #define REAL_MIN -1.0e10
    #define REAL_EPSILON 1e-16
#else
    typedef float real_t;
    #define REAL_MAX 1.0e5
    #define REAL_MIN -1.0e5
    #define REAL_EPSILON 1e-8
#endif

// Math functions that adapt to precision
#ifdef USE_DOUBLE
    #define SQRT(x) sqrt(x)
    #define SIN(x) sin(x)
    #define COS(x) cos(x)
    #define EXP(x) exp(x)
    #define LOG(x) log(x)
    #define POW(x,y) pow(x,y)
    #define FABS(x) fabs(x)
#else
    #define SQRT(x) sqrtf(x)
    #define SIN(x) sinf(x)
    #define COS(x) cosf(x)
    #define EXP(x) expf(x)
    #define LOG(x) logf(x)
    #define POW(x,y) powf(x,y)
    #define FABS(x) fabsf(x)
#endif

// Constants
#ifdef USE_DOUBLE
    #define PI 3.14159265358979323846
    #define ZERO 0.0
    #define ONE 1.0
    #define TWO 2.0
#else
    #define PI 3.14159265358979323846f
    #define ZERO 0.0f
    #define ONE 1.0f
    #define TWO 2.0f
#endif

// Constants
#define MAX_THREADS 256 // Can be used for Kokkos team size

// Initialization types
#define INIT_ZERO 0
#define INIT_MANUFACTURED 1
#define INIT_RANDOM 2

// RHS types
#define RHS_ZERO 0
#define RHS_MANUFACTURED 1
#define RHS_CUSTOM 2

// Dirichlet BC specifier
#define BCD_USER_SPECIFIED 1
#define BCD_MANUFACTURED 2

// Method types
#define METHOD_JACOBI 1
#define METHOD_SRJ 2
#define METHOD_BICGSTAB 3
#define METHOD_MULTIGRID 4

// Simulation type string for file naming
#define SIMTYPE "kokkos"

// Grid Types
// Grid type defines
#define GRID_UNIFORM 0
#define GRID_NONUNIFORM 1

#endif // TYPES_H