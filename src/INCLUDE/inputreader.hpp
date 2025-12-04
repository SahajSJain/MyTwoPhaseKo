// file: src/INCLUDE/inputreader.hpp
#ifndef INPUTREADER_HPP
#define INPUTREADER_HPP

#include "include.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

struct InputParams
{
    // Grid configuration
    int Nx, Ny, Nz, Ngl;
    int xmode, ymode, zmode;
    real_t xstart, xend;
    real_t ystart, yend;
    real_t zstart, zend;

    // Solver configuration
    real_t dt;
    int nstart, nend;
    int nprint, ndump;

    // Field initialization
    real_t x0, y0, z0;
    real_t radius;
    int mode; // 0: psi positive inside, 1: psi negative inside

    // Phase field parameters
    real_t epsilon_fac;
    real_t gamma_fac;

    // Boundary conditions
    int bc_east, bc_west, bc_north, bc_south, bc_front, bc_back;
    real_t valbc_east, valbc_west, valbc_north, valbc_south, valbc_front, valbc_back;

    // Convert to gridconfig
    gridconfig toGridConfig() const
    {
        gridconfig config;
        config.Ngl = Ngl; // Use the value from input file

        // X direction
        config.Nx = Nx;
        config.x_start = xstart;
        config.x_end = xend;
        config.x_option = xmode;

        // Y direction
        config.Ny = Ny;
        config.y_start = ystart;
        config.y_end = yend;
        config.y_option = ymode;

        // Z direction
        config.Nz = Nz;
        config.z_start = zstart;
        config.z_end = zend;
        config.z_option = zmode;

        return config;
    }

    // Convert to bctype
    bctype toBCType() const
    {
        bctype bc;
        bc.E = bc_east;
        bc.W = bc_west;
        bc.N = bc_north;
        bc.S = bc_south;
        bc.F = bc_front;
        bc.B = bc_back;

        bc.E_val = valbc_east;
        bc.W_val = valbc_west;
        bc.N_val = valbc_north;
        bc.S_val = valbc_south;
        bc.F_val = valbc_front;
        bc.B_val = valbc_back;

        return bc;
    }
};

class InputReader
{
public:
    static InputParams readInputFile(const std::string &filename = "input.dat")
    {
        InputParams params;
        std::ifstream file(filename);

        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open input file '" << filename << "'\n";
            exit(1);
        }

        std::string line;

        // Skip header lines
        for (int i = 0; i < 4; i++)
            std::getline(file, line);

        // Grid configuration
        std::getline(file, line);                                  // Nx Ny Nz Ngl
        file >> params.Nx >> params.Ny >> params.Nz >> params.Ngl; // Read Ngl
        std::getline(file, line);                                  // consume rest of line

        std::getline(file, line); // xmode xstart xend
        file >> params.xmode >> params.xstart >> params.xend;
        std::getline(file, line);

        std::getline(file, line); // ymode ystart yend
        file >> params.ymode >> params.ystart >> params.yend;
        std::getline(file, line);

        std::getline(file, line); // zmode zstart zend
        file >> params.zmode >> params.zstart >> params.zend;
        std::getline(file, line);

        // Skip separator
        std::getline(file, line);

        // Solver configuration
        std::getline(file, line); // dt nstart nend nprint ndump
        file >> params.dt >> params.nstart >> params.nend >> params.nprint >> params.ndump;
        std::getline(file, line);

        // Skip separator
        std::getline(file, line);

        // Field initialization
        std::getline(file, line); // x0 y0 z0 radius mode
        file >> params.x0 >> params.y0 >> params.z0 >> params.radius >> params.mode;
        std::getline(file, line);

        // Skip separator
        std::getline(file, line);

        // Phase field parameters
        std::getline(file, line); // epsilon_fac gamma_fac
        file >> params.epsilon_fac >> params.gamma_fac;
        std::getline(file, line);

        // Skip separator
        std::getline(file, line);

        // Boundary conditions
        std::getline(file, line); // bc_east valbc_east
        file >> params.bc_east >> params.valbc_east;
        std::getline(file, line);

        std::getline(file, line); // bc_west valbc_west
        file >> params.bc_west >> params.valbc_west;
        std::getline(file, line);

        std::getline(file, line); // bc_north valbc_north
        file >> params.bc_north >> params.valbc_north;
        std::getline(file, line);

        std::getline(file, line); // bc_south valbc_south
        file >> params.bc_south >> params.valbc_south;

        std::getline(file, line); // bc_front valbc_front
        file >> params.bc_front >> params.valbc_front;

        std::getline(file, line); // bc_back valbc_back
        file >> params.bc_back >> params.valbc_back;

        file.close();

        return params;
    }

    static void printParams(const InputParams &params)
    {
        std::cout << "\n=== Input Parameters ===\n";
        std::cout << "Grid: " << params.Nx << " x " << params.Ny << " x " << params.Nz
                  << " with " << params.Ngl << " ghost layers\n"; // Show Ngl
        std::cout << "Domain: [" << params.xstart << ", " << params.xend << "] x ["
                  << params.ystart << ", " << params.yend << "] x ["
                  << params.zstart << ", " << params.zend << "]\n";
        std::cout << "Time step: " << params.dt << "\n";
        std::cout << "Time steps: " << params.nstart << " to " << params.nend << "\n";
        std::cout << "Output frequency: print=" << params.nprint << ", dump=" << params.ndump << "\n";
        std::cout << "Initial condition: center=(" << params.x0 << ", " << params.y0 << ", " << params.z0
                  << "), radius=" << params.radius << ", mode=" << params.mode << "\n";
        std::cout << "Phase field: epsilon_fac=" << params.epsilon_fac
                  << ", gamma_fac=" << params.gamma_fac << "\n";
        std::cout << "=======================\n\n";
    }
};

#endif // INPUTREADER_HPP