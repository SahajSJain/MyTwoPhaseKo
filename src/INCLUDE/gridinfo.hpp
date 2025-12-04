// file: src/INCLUDE/gridinfo.hpp
#ifndef GRIDINFO_HPP
#define GRIDINFO_HPP

#include "oneDgridinfo.hpp"
#include <memory>
#include <string>

// Grid configuration structure
struct gridconfig
{
    int Ngl; // Number of ghost layers

    // X direction
    int Nx;
    real_t x_start;
    real_t x_end;
    int x_option; // GRID_UNIFORM or GRID_NONUNIFORM

    // Y direction
    int Ny;
    real_t y_start;
    real_t y_end;
    int y_option;

    // Z direction
    int Nz;
    real_t z_start;
    real_t z_end;
    int z_option;
    // Constructor with defaults
    gridconfig() : Ngl(1),
                   Nx(10), x_start(0.0), x_end(1.0), x_option(GRID_UNIFORM),
                   Ny(10), y_start(0.0), y_end(1.0), y_option(GRID_UNIFORM),
                   Nz(10), z_start(0.0), z_end(1.0), z_option(GRID_UNIFORM) {}
};

class GridInfo
{
public:
    // Grid configuration
    gridconfig config;
    using Policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    using Policy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    // 1D grid information for each direction
    std::unique_ptr<oneDgridinfo> xgrid;
    std::unique_ptr<oneDgridinfo> ygrid;
    std::unique_ptr<oneDgridinfo> zgrid;

    // 3D arrays for volumes and areas (cell-centered)
    // Device views (using OffsetView)
    Kokkos::Experimental::OffsetView<real_t ***> Vols;   // Cell volumes
    Kokkos::Experimental::OffsetView<real_t ***> Area_E; // East face areas
    Kokkos::Experimental::OffsetView<real_t ***> Area_W; // West face areas
    Kokkos::Experimental::OffsetView<real_t ***> Area_N; // North face areas
    Kokkos::Experimental::OffsetView<real_t ***> Area_S; // South face areas
    Kokkos::Experimental::OffsetView<real_t ***> Area_F; // Front face areas
    Kokkos::Experimental::OffsetView<real_t ***> Area_B;   // Back face areas 
    Kokkos::Experimental::OffsetView<real_t ***> map;    // Cell maps

    // Host mirrors
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Vols_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Area_E_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Area_W_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Area_N_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Area_S_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Area_F_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror Area_B_h;
    typename Kokkos::Experimental::OffsetView<real_t ***>::HostMirror map_h;    
    Policy3D interior_policy;
    Policy3D full_policy; 
    Policy2D xBoundary_policy; 
    Policy2D yBoundary_policy;
    Policy2D zBoundary_policy;

    // Constructor
    GridInfo(const gridconfig &config_in);

    // Destructor
    ~GridInfo() = default;

    // Print summary
    void printSummary() const;

private:
    void allocateArrays();
    void computeVolumesAndAreas();
    void initializeCellMap();  
    void uploadToDevice();
};

// Implementation

inline GridInfo::GridInfo(const gridconfig &config_in) : config(config_in)
{
// Adjust for 2D case
#ifdef USE_2D
    config.z_start = -0.5;
    config.z_end = 0.5;
    config.Nz = 1;
    config.z_option = GRID_UNIFORM;
    int z_ngl = 0; // No ghost cells in z for 2D
#else
    int z_ngl = config.Ngl;
#endif

    // Create 1D grids
    std::string xfile = config.x_option == GRID_NONUNIFORM ? "xgrid.dat" : "";
    std::string yfile = config.y_option == GRID_NONUNIFORM ? "ygrid.dat" : "";
    std::string zfile = config.z_option == GRID_NONUNIFORM ? "zgrid.dat" : "";

    xgrid = std::make_unique<oneDgridinfo>(config.Nx, config.Ngl, config.x_start, config.x_end,
                                           "x", config.x_option, xfile);
    ygrid = std::make_unique<oneDgridinfo>(config.Ny, config.Ngl, config.y_start, config.y_end,
                                           "y", config.y_option, yfile);
    zgrid = std::make_unique<oneDgridinfo>(config.Nz, z_ngl, config.z_start, config.z_end,
                                           "z", config.z_option, zfile);
    auto x_ce = xgrid->ce;
    auto y_ce = ygrid->ce;
    auto z_ce = zgrid->ce;
    auto x_cs = xgrid->cs;
    auto y_cs = ygrid->cs;
    auto z_cs = zgrid->cs;
    auto x_be = xgrid->be;
    auto y_be = ygrid->be;
    auto z_be = zgrid->be;
    auto x_bs = xgrid->bs;
    auto y_bs = ygrid->bs;
    auto z_bs = zgrid->bs;

    interior_policy = Policy3D({{x_cs, y_cs, z_cs}}, {{x_ce + 1, y_ce + 1, z_ce + 1}});
    full_policy = Policy3D({{x_bs, y_bs, z_bs}}, {{x_be + 1, y_be + 1, z_be + 1}});
    xBoundary_policy = Policy2D({{y_bs, z_bs}}, {{y_be + 1, z_be + 1}});
    yBoundary_policy = Policy2D({{x_bs, z_bs}}, {{x_be + 1, z_be + 1}});
    zBoundary_policy = Policy2D({{x_bs, y_bs}}, {{x_be + 1, y_be + 1}});

    // Allocate 3D arrays
    allocateArrays();

    // Compute volumes and areas
    computeVolumesAndAreas();

    // Initialize cell map
    initializeCellMap();

    // Upload to device
    uploadToDevice();
}

inline void GridInfo::allocateArrays() {
    // Create begins array for OffsetView
    Kokkos::Array<int64_t, 3> begins = {xgrid->bs, ygrid->bs, zgrid->bs};
    Kokkos::Array<int64_t, 3> ends = {xgrid->be + 1, ygrid->be + 1, zgrid->be + 1};
    
    // Allocate device views with offset indexing using std::pair ranges
    Vols = Kokkos::Experimental::OffsetView<real_t***>(
        "Cell_Volumes",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_E = Kokkos::Experimental::OffsetView<real_t***>(
        "Area_East",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_W = Kokkos::Experimental::OffsetView<real_t***>(
        "Area_West",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_N = Kokkos::Experimental::OffsetView<real_t***>(
        "Area_North",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_S = Kokkos::Experimental::OffsetView<real_t***>(
        "Area_South",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_F = Kokkos::Experimental::OffsetView<real_t***>(
        "Area_Front",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_B = Kokkos::Experimental::OffsetView<real_t***>(
        "Area_Back",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    map = Kokkos::Experimental::OffsetView<real_t***>(
        "Cell_Map",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    // Create host mirrors
    Vols_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Cell_Volumes_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_E_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Area_East_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_W_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Area_West_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_N_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Area_North_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_S_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Area_South_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_F_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Area_Front_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    Area_B_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Area_Back_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
    
    map_h = Kokkos::Experimental::OffsetView<real_t***, Kokkos::HostSpace>(
        "Cell_Map_host",
        std::pair<int64_t, int64_t>{begins[0], ends[0]},
        std::pair<int64_t, int64_t>{begins[1], ends[1]},
        std::pair<int64_t, int64_t>{begins[2], ends[2]});
}

inline void GridInfo::computeVolumesAndAreas()
{
    // Get grid data
    auto &dc_x = xgrid->dc_h;
    auto &dc_y = ygrid->dc_h;
    auto &dc_z = zgrid->dc_h;

    // Compute for all cells including ghosts
    for (int i = xgrid->bs; i <= xgrid->be; i++)
    {
        for (int j = ygrid->bs; j <= ygrid->be; j++)
        {
            for (int k = zgrid->bs; k <= zgrid->be; k++)
            {
                // Map to array indices (0-based)
// Direct indexing with OffsetView
// Volume = dx * dy * dz
            Vols_h(i, j, k) = dc_x(i) * dc_y(j) * dc_z(k);

            // Face areas
            Area_E_h(i, j, k) = dc_y(j) * dc_z(k); // East face (x+)
            Area_W_h(i, j, k) = dc_y(j) * dc_z(k); // West face (x-)
            Area_N_h(i, j, k) = dc_x(i) * dc_z(k); // North face (y+)
            Area_S_h(i, j, k) = dc_x(i) * dc_z(k); // South face (y-)
            Area_F_h(i, j, k) = dc_x(i) * dc_y(j); // Front face (z+)
            Area_B_h(i, j, k) = dc_x(i) * dc_y(j); // Back face (z-)
            }
        }
    }
}

inline void GridInfo::initializeCellMap()
{
    // Initialize all cells to -1
    Kokkos::deep_copy(map_h, -1.0);

    // Set halo points (boundary cells) to 0
    // X boundaries
    for (int j = ygrid->bs; j <= ygrid->be; j++)
    {
        for (int k = zgrid->bs; k <= zgrid->be; k++)
        {
            // Left boundary
            for (int i = xgrid->bs; i < xgrid->cs; i++)
            {
                map_h(i, j, k) = 0.0;
            }
            // Right boundary
            for (int i = xgrid->ce + 1; i <= xgrid->be; i++)
            {
                map_h(i, j, k) = 0.0;
            }
        }
    }

    // Y boundaries
    for (int i = xgrid->bs; i <= xgrid->be; i++)
    {
        for (int k = zgrid->bs; k <= zgrid->be; k++)
        {
            // Bottom boundary
            for (int j = ygrid->bs; j < ygrid->cs; j++)
            {
                map_h(i, j, k) = 0.0;
            }
            // Top boundary
            for (int j = ygrid->ce + 1; j <= ygrid->be; j++)
            {
                map_h(i, j, k) = 0.0;
            }
        }
    }

// Z boundaries (if 3D)
#ifndef USE_2D
    for (int i = xgrid->bs; i <= xgrid->be; i++)
    {
        for (int j = ygrid->bs; j <= ygrid->be; j++)
        {
            // Back boundary
            for (int k = zgrid->bs; k < zgrid->cs; k++)
            {
                map_h(i, j, k) = 0.0;
            }
            // Front boundary
            for (int k = zgrid->ce + 1; k <= zgrid->be; k++)
            {
                map_h(i, j, k) = 0.0;
            }
        }
    }
#endif

    // Now propagate the map values
    bool changed = true;
    int current_level = 0;

    while (changed)
    {
        changed = false;

        // Loop through interior cells only
        for (int i = xgrid->cs; i <= xgrid->ce; i++)
        {
            for (int j = ygrid->cs; j <= ygrid->ce; j++)
            {
                for (int k = zgrid->cs; k <= zgrid->ce; k++)
                {

                    // Skip if already set
                    if (map_h(i, j, k) >= 0)
                        continue;

                    // Check all 6 neighbors
                    bool has_neighbor_at_level = false;

                    // Check x neighbors
                    if (map_h(i - 1, j, k) == current_level ||
                        map_h(i + 1, j, k) == current_level)
                    {
                        has_neighbor_at_level = true;
                    }

                    // Check y neighbors
                    if (map_h(i, j - 1, k) == current_level ||
                        map_h(i, j + 1, k) == current_level)
                    {
                        has_neighbor_at_level = true;
                    }

                    // Check z neighbors
                    if (map_h(i, j, k - 1) == current_level ||
                        map_h(i, j, k + 1) == current_level)
                    {
                        has_neighbor_at_level = true;
                    }

                    // Set to next level if has neighbor at current level
                    if (has_neighbor_at_level)
                    {
                        map_h(i, j, k) = current_level + 1;
                        changed = true;
                    }
                }
            }
        }

        // Move to next level
        if (changed)
        {
            current_level++;
            // Cap at 2 as per requirement
            if (current_level >= config.Ngl)
            {
                current_level = config.Ngl;
            }
        }
    }
}

inline void GridInfo::uploadToDevice()
{
    Kokkos::deep_copy(Vols, Vols_h);
    Kokkos::deep_copy(Area_E, Area_E_h);
    Kokkos::deep_copy(Area_W, Area_W_h);
    Kokkos::deep_copy(Area_N, Area_N_h);
    Kokkos::deep_copy(Area_S, Area_S_h);
    Kokkos::deep_copy(Area_F, Area_F_h);
    Kokkos::deep_copy(Area_B, Area_B_h);
    Kokkos::deep_copy(map, map_h); // Add this line
}

inline void GridInfo::printSummary() const
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "3D Grid Information Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nGrid Configuration:" << std::endl;
    std::cout << "  Ghost layers (Ngl): " << config.Ngl << std::endl;
#ifdef USE_2D
    std::cout << "  Mode: 2D (z-direction forced to single cell)" << std::endl;
#else
    std::cout << "  Mode: 3D" << std::endl;
#endif

    std::cout << "\nDomain:" << std::endl;
    std::cout << "  X: [" << xgrid->start << ", " << xgrid->end << "] with Nx = " << config.Nx << std::endl;
    std::cout << "  Y: [" << ygrid->start << ", " << ygrid->end << "] with Ny = " << config.Ny << std::endl;
    std::cout << "  Z: [" << zgrid->start << ", " << zgrid->end << "] with Nz = " << config.Nz << std::endl;

    std::cout << "\nTotal cells (including ghosts):" << std::endl;
    std::cout << "  X: " << (xgrid->N + 2 * xgrid->Ngl) << " cells" << std::endl;
    std::cout << "  Y: " << (ygrid->N + 2 * ygrid->Ngl) << " cells" << std::endl;
    std::cout << "  Z: " << (zgrid->N + 2 * zgrid->Ngl) << " cells" << std::endl;
    int total_cells = (xgrid->N + 2 * xgrid->Ngl) * (ygrid->N + 2 * ygrid->Ngl) * (zgrid->N + 2 * zgrid->Ngl);
    std::cout << "  Total: " << total_cells << " cells" << std::endl;

    std::cout << "\nMinimum grid spacings:" << std::endl;
    std::cout << "  dx_min: " << xgrid->dmin << std::endl;
    std::cout << "  dy_min: " << ygrid->dmin << std::endl;
    std::cout << "  dz_min: " << zgrid->dmin << std::endl;

    // Sample volume and areas at center
    int ic = (xgrid->cs + xgrid->ce) / 2;  // Use actual grid center
    int jc = (ygrid->cs + ygrid->ce) / 2;
    int kc = (zgrid->cs + zgrid->ce) / 2;
    std::cout << "\nSample values at center cell [" << ic << "," << jc << "," << kc << "]:" << std::endl;
    std::cout << "  Volume: " << Vols_h(ic, jc, kc) << std::endl;
    std::cout << "  Area_E: " << Area_E_h(ic, jc, kc) << std::endl;
    std::cout << "  Area_N: " << Area_N_h(ic, jc, kc) << std::endl;
    std::cout << "  Area_F: " << Area_F_h(ic, jc, kc) << std::endl;

    std::cout << "\nDetailed 1D grid information:" << std::endl;
    std::cout << "\n--- X Grid ---" << std::endl;
    xgrid->printSummary();
    std::cout << "\n--- Y Grid ---" << std::endl;
    ygrid->printSummary();
    std::cout << "\n--- Z Grid ---" << std::endl;
    zgrid->printSummary();

    std::cout << "========================================\n"
              << std::endl;
}

#endif // GRIDINFO_HPP