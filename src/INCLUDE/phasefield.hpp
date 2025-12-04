// file: src/INCLUDE/phasefield.hpp
#ifndef PHASEFIELD_HPP
#define PHASEFIELD_HPP

#include "include.hpp"
#include "inputreader.hpp"
#include <string>

class PhaseField {
public:
    // Grid (owned by PhaseField)
    std::unique_ptr<GridInfo> grid;
    
    // Input parameters
    InputParams params;
    
    // Boundary conditions
    bctype BC;
    
    // Time stepping parameters
    real_t dt;          // Time step
    real_t time;        // Current simulation time
    int step;           // Current time step number
    
    // ACDI parameters
    Field epsilon;  // Interface thickness (space-varying based on grid)
    real_t epsilon_fac;     // Factor for interface thickness
    real_t Gamma;           // Mobility parameter (Gamma = gamma_fac for now)
    real_t gamma_fac;       // Factor for Gamma calculation
    
    // Initial condition parameters (from input)
    real_t x0, y0, z0;      // Center coordinates
    real_t r0;              // Radius
    int init_mode;          // 0: psi positive inside, 1: psi negative inside
    
    // Primary fields
    Field phi;      // Phase field [0,1]
    Field phi_old;  // Old phase field (for diagnostics)
    Field psi;      // Signed distance function
    
    // Velocity fields
    Field u;        // x-velocity
    Field v;        // y-velocity
    Field w;        // z-velocity
    Field vel_mag;  // Velocity magnitude (for computing max velocity)
    
    // Normal vector fields (also used for psi gradients temporarily)
    Field normx;    // Normal x-component
    Field normy;    // Normal y-component
    Field normz;    // Normal z-component
    
    // Face values
    Face phi_f;         // Phase field at faces (interpolated values)
    Face phi_diff_f;    // Phase field gradients at faces (for diffusion)
    Face psi_f;         // SDF at faces
    Face epsilon_f;     // Epsilon at faces
    Face norm_f;        // Normal vectors at faces
    Face vel_f;         // Velocity field at faces    
    // Terms (not fluxes!)
    Field Conv;     // Convection term: div(u*phi)
    Field Diff;     // Diffusion term: div(epsilon*grad(phi))
    Field Sharp;    // Sharpening term: div((1/4)*[1-tanh²(psi/2eps)]*normal)
    Field RHS;      // Right hand side: -Conv + Gamma*Diff - Gamma*Sharp
    
    // RAII Constructor
    PhaseField(const InputParams& params_in, const gridconfig& grid_config);
    
    // Destructor
    ~PhaseField() = default;
    
    // Main initialization function
    void initialize();
    
    // Memory management
    void allocateFields();      // Allocate all fields
    
    // Data transfer
    void uploadToDevice();      // Upload all fields to device
    void downloadFromDevice();  // Download only output fields from device
    
    // Set initial condition for psi (circle/sphere)
    void setInitialConditionPsi(Field& psi_field);
    
    // Parameter computation
    void computeEpsilon(Field& epsilon_field);
    
    // Set velocity field and compute face velocities
    void setVelocity(real_t t, Field& u_field, Field& v_field, 
                     Field& w_field, Face& vel_face);
    
    // Core solver steps
    void interpolateToFaces(Field& phi_field, Field& psi_field,
                           Field& epsilon_field, 
                           Field& normx_field, Field& normy_field, 
                           Field& normz_field,
                           Face& phi_face, Face& psi_face, Face& epsilon_face, Face& norm_face);
    
    void computeNormals(Field& psi_field,
                       Field& normx_out, Field& normy_out, Field& normz_out);
    
    void computeRHS(Field& phi_field, Face& phi_face, Face& psi_face,
                   Face& epsilon_face, Face& norm_face, Face& vel_face,
                   Field& Conv_out, Field& Diff_out, Field& Sharp_out,
                   Field& RHS_out);
    
    void updatePhi(Field& phi_field, Field& RHS_field, real_t dt_local);
    
    // Main time stepping function
    void advanceTimeStep();
    
    // Apply boundary conditions
    void applyBoundaryConditions(Field& field);
    
    // Utilities
    void checkBounds(Field& phi_field);
    real_t computeMass(Field& phi_field);
    real_t getMaxVelocity(Field& u_field, Field& v_field, 
                         Field& w_field, Field& vel_mag_field);
    
    // Output functions
    void printSummary();
    void printDiagnostics();
    void writeTecplot();  // Writes phase.XXXXXXX.dat format
    
private:
    void writeTecplot2D(const std::string& filename);
    void writeTecplot3D(const std::string& filename);

};

// Constructor implementation
inline PhaseField::PhaseField(const InputParams& params_in, const gridconfig& grid_config) 
    : grid(std::make_unique<GridInfo>(grid_config)),
      params(params_in),
      BC(params.toBCType()),
      dt(params.dt),
      time(0.0),
      step(params.nstart),
      epsilon(grid.get(), "epsilon"),
      epsilon_fac(params.epsilon_fac),
      Gamma(params.gamma_fac),
      gamma_fac(params.gamma_fac),
      x0(params.x0),
      y0(params.y0),
      z0(params.z0),
      r0(params.radius),
      init_mode(params.mode),
      phi(grid.get(), "phi"),
      phi_old(grid.get(), "phi_old"),
      psi(grid.get(), "psi"),
      u(grid.get(), "u"),
      v(grid.get(), "v"), 
      w(grid.get(), "w"),
      vel_mag(grid.get(), "vel_mag"),
      normx(grid.get(), "normx"),
      normy(grid.get(), "normy"),
      normz(grid.get(), "normz"),
      phi_f(grid.get(), "phi_f"),
      phi_diff_f(grid.get(), "phi_diff_f"),  // Add this
      psi_f(grid.get(), "psi_f"),
      epsilon_f(grid.get(), "epsilon_f"),
      norm_f(grid.get(), "norm_f"),
      vel_f(grid.get(), "vel_f"),
      Conv(grid.get(), "Conv"),
      Diff(grid.get(), "Diff"),
      Sharp(grid.get(), "Sharp"),
      RHS(grid.get(), "RHS")
{
    initialize();
}


#endif // PHASEFIELD_HPP