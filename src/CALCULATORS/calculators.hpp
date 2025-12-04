// file: src/CALCULATORS/calculators.hpp
#ifndef CALCULATORS_HPP
#define CALCULATORS_HPP

// Header-only declarations for calculator routines implemented in
// the source files under src/CALCULATORS. This header provides only
// function declarations (no .cpp includes) so it is a proper .hpp file.

// Include the project's master include (types, GridInfo, Field, Face, etc.)
#include "../INCLUDE/include.hpp"

// Function prototypes - match implementations in this directory

// face_diff.cpp
void CalculateFaceDiffs(Field& phi, Face& phi_f);

// face_interpolate.cpp
void CalculateFaceVals(Field& phi, Face& phi_f);
void CalculateFaceVals(Field& phi_x,
                       Field& phi_y,
                       Field& phi_z,
                       Face& phi_f);

// gradient.cpp
void CalculateGradients(Field& phi,
                        Field& phi_x,
                        Field& phi_y,
                        Field& phi_z);

// divergence.cpp
void CalculateDivergence(Field& phi_x,
                         Field& phi_y,
                         Field& phi_z,
                         Field& Div);
void CalculateDivergence(Face& phi_f, Field& Div);

// boundary_conditions.cpp
// Note: signature matches implementation (passes bctype by value)
void CalculateBoundaryValues(Field &phi, bctype BC );

// convection.cpp
void CalculateConvection(Face& phi_f,
                         Face& U_f,
                         Field& Conv);

// diffusion.cpp
void CalculateDiffusion(Face& phi_diff_f,
                        Face& K_f,
                        Field& Laplacian);

// normals.cpp
void CalculateNormals(Field& norm_x,
                      Field& norm_y,
                      Field& norm_z);

// absolute_value.cpp
void CalculateAbsoluteValue(Field &phi, Field &abs_phi);
void CalculateAbsoluteValue(Field &phi_x,
                            Field &phi_y,
                            Field &phi_z,
                            Field &abs_phi);

// max_magnitude.cpp
void CalculateMaxMagnitude(Field &phi, real_t &max_magnitude);

// sharpening.cpp
void CalculateSharpening(Face& psi_f,
                         Face& norm_f,
                         Face& epsilon_f,
                         Field& Sharp);

// SDF <-> Phase conversions
void CalculateSDF2Phase(Field& psi,
                        Field& epsilon,
                        Field& phi);

void CalculatePhaseToSdf(Field& phi,
                         Field& epsilon,
                         Field& psi);

#endif // CALCULATORS_HPP

