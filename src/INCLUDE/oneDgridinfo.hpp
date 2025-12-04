// file: src/INCLUDE/oneDgridinfo.hpp
#ifndef ONEDGRIDINFO_HPP
#define ONEDGRIDINFO_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <iomanip>
#include "structs.hpp"

// Grid type defines
#define GRID_UNIFORM 0
#define GRID_NONUNIFORM 1

class oneDgridinfo {
public:
    // Scalars
    const int N;
    const int Ngl;
    real_t start;
    real_t end;
    real_t dmin;
    const int init_option;
    const std::string filename;
    const std::string varname;
    
    // Index limits
    const int cs = 1;          // cell start
    const int ce;              // cell end (= N)
    const int fs = 1;          // face start
    const int fe;              // face end (= N+1)
    const int bs; 
    const int be;              // boundary start and end (including ghosts)
    const int bfs;              // face start including ghosts
    const int bfe;              // face end including ghosts (= N+Ngl+1
    // Regular Views (underlying storage)
    Kokkos::View<real_t*> c_view;
    Kokkos::View<real_t*> dc_view;
    Kokkos::View<real_t*> dc_inv_view;
    Kokkos::View<real_t*> f_view;
    Kokkos::View<real_t*> df_view;
    Kokkos::View<real_t*> df_inv_view;
    Kokkos::View<real_t*> inp_view;
    
    // Host mirrors
    typename Kokkos::View<real_t*>::HostMirror c_view_h;
    typename Kokkos::View<real_t*>::HostMirror dc_view_h;
    typename Kokkos::View<real_t*>::HostMirror dc_inv_view_h;
    typename Kokkos::View<real_t*>::HostMirror f_view_h;
    typename Kokkos::View<real_t*>::HostMirror df_view_h;
    typename Kokkos::View<real_t*>::HostMirror df_inv_view_h;
    typename Kokkos::View<real_t*>::HostMirror inp_view_h;
    
    // Offset Views - Cell arrays
    Kokkos::Experimental::OffsetView<real_t*> c;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> c_h;
    Kokkos::Experimental::OffsetView<real_t*> dc;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> dc_h;
    Kokkos::Experimental::OffsetView<real_t*> dc_inv;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> dc_inv_h;
    
    // Offset Views - Face arrays  
    Kokkos::Experimental::OffsetView<real_t*> f;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> f_h;
    Kokkos::Experimental::OffsetView<real_t*> df;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> df_h;
    Kokkos::Experimental::OffsetView<real_t*> df_inv;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> df_inv_h;
    
    // Interpolation metrics
    Kokkos::Experimental::OffsetView<real_t*> inp;
    Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace> inp_h;
    
    // Constructor
    oneDgridinfo(int N_in, int Ngl_in, real_t start_in, real_t end_in, 
                 const std::string& varname_in,
                 int init_option_in = GRID_UNIFORM, 
                 const std::string& filename_in = "");
    
    // Destructor
    ~oneDgridinfo() = default;
    
    // Print summary
    void printSummary() const;
    
private:
    void allocateArrays();
    void initializeUniformGrid();
    void initializeNonUniformGrid();
    void computeCellCenters();
    void computeDifferences();
    void computeInterpolationMetrics();
    void uploadToDevice();
};

// Implementation

inline oneDgridinfo::oneDgridinfo(int N_in, int Ngl_in, real_t start_in, real_t end_in, 
                                  const std::string& varname_in,
                                  int init_option_in, 
                                  const std::string& filename_in) 
    : N(N_in), Ngl(Ngl_in), start(start_in), end(end_in),
      init_option(init_option_in), filename(filename_in), varname(varname_in),
      ce(N), fe(N+1), bs(1-Ngl), be(N+Ngl), bfs(1-Ngl), bfe(N+Ngl+1) {
    
    // Allocate all arrays
    allocateArrays();
    
    // Initialize face centers
    if (init_option == GRID_UNIFORM) {
        initializeUniformGrid();
    } else {
        initializeNonUniformGrid();
    }
    
    // Compute cell centers
    computeCellCenters();
    
    // Compute differences
    computeDifferences();
    
    // Compute interpolation metrics
    computeInterpolationMetrics();
    
    // Upload everything to device
    uploadToDevice();
}

inline void oneDgridinfo::allocateArrays() {
    // Cell arrays - size N+2*Ngl
    c = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_cell_centers",
        std::pair<int64_t, int64_t>{bs, be + 1});
    
    c_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_cell_centers_host",
        std::pair<int64_t, int64_t>{bs, be + 1});
    
    dc = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_cell_diff",
        std::pair<int64_t, int64_t>{bs, be + 1});
    
    dc_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_cell_diff_host",
        std::pair<int64_t, int64_t>{bs, be + 1});
    
    dc_inv = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_cell_diff_inv",
        std::pair<int64_t, int64_t>{bs, be + 1});
    
    dc_inv_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_cell_diff_inv_host",
        std::pair<int64_t, int64_t>{bs, be + 1});
    
    // Face arrays - size N+2*Ngl+1
    f = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_face_centers",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    f_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_face_centers_host",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    df = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_face_diff",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    df_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_face_diff_host",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    df_inv = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_face_diff_inv",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    df_inv_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_face_diff_inv_host",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    // Interpolation metrics
    inp = Kokkos::Experimental::OffsetView<real_t*>(
        varname + "_interp_factor",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    inp_h = Kokkos::Experimental::OffsetView<real_t*, Kokkos::HostSpace>(
        varname + "_interp_factor_host",
        std::pair<int64_t, int64_t>{bfs, bfe + 1});
    
    // Initialize all arrays to zero
    Kokkos::deep_copy(c_h, 0.0);
    Kokkos::deep_copy(c, 0.0);
    Kokkos::deep_copy(dc_h, 0.0);
    Kokkos::deep_copy(dc, 0.0);
    Kokkos::deep_copy(dc_inv_h, 0.0);
    Kokkos::deep_copy(dc_inv, 0.0);
    Kokkos::deep_copy(f_h, 0.0);
    Kokkos::deep_copy(f, 0.0);
    Kokkos::deep_copy(df_h, 0.0);
    Kokkos::deep_copy(df, 0.0);
    Kokkos::deep_copy(df_inv_h, 0.0);
    Kokkos::deep_copy(df_inv, 0.0);
    Kokkos::deep_copy(inp_h, 0.0);
    Kokkos::deep_copy(inp, 0.0);
}

inline void oneDgridinfo::initializeUniformGrid() {
    real_t dx = (end - start) / N;
    
    // Initialize interior faces (1 to N+1)
    for (int i = 1; i <= N+1; i++) {
        f_h(i) = start + (i-1) * dx;
    }
    
    // Set boundary faces
    for (int i = 1; i <= Ngl; i++) {
        f_h(1-i) = f_h(1) - i * dx;
        f_h(N+1+i) = f_h(N+1) + i * dx;
    }
}

inline void oneDgridinfo::initializeNonUniformGrid() {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int idx;
    real_t value;
    real_t min_val = std::numeric_limits<real_t>::max();
    real_t max_val = std::numeric_limits<real_t>::lowest();
    
    // Read interior faces
    while (file >> idx >> value) {
        if (idx >= 1 && idx <= N+1) {
            f_h(idx) = value;
            min_val = std::min(min_val, value);
            max_val = std::max(max_val, value);
        }
    }
    file.close();
    
    // Override start and end with values from file
    start = min_val;
    end = max_val;
    
    // Set boundary faces using extrapolation
    real_t dx_left = f_h(2) - f_h(1);
    real_t dx_right = f_h(N+1) - f_h(N);
    
    for (int i = 1; i <= Ngl; i++) {
        f_h(1-i) = f_h(1) - i * dx_left;
        f_h(N+1+i) = f_h(N+1) + i * dx_right;
    }
}

inline void oneDgridinfo::computeCellCenters() {
    // Interior cells: c(i) = 0.5*(f(i) + f(i+1))
    for (int i = 1; i <= N; i++) {
        c_h(i) = 0.5 * (f_h(i) + f_h(i+1));
    }
    
    // Boundary cells - equidistant
    real_t dx_left = f_h(2) - f_h(1);  // Use face spacing
    real_t dx_right = f_h(N+1) - f_h(N);

    for (int i = 1; i <= Ngl; i++) {
        c_h(1-i) = c_h(1) - i * dx_left;
        c_h(N+i) = c_h(N) + i * dx_right;
    }
}

inline void oneDgridinfo::computeDifferences() {
    // Cell differences: dc(i) = f(i+1) - f(i)
    for (int i = 1-Ngl; i <= N+Ngl; i++) {
        dc_h(i) = f_h(i+1) - f_h(i);
        dc_inv_h(i) = 1.0 / dc_h(i);
    }
    
    // Face differences: df(i) = c(i) - c(i-1)
    for (int i = 2-Ngl; i <= N+Ngl; i++) {
        df_h(i) = c_h(i) - c_h(i-1);
    }
    
    // Boundary face differences
    df_h(1-Ngl) = df_h(2-Ngl);
    df_h(N+Ngl+1) = df_h(N+Ngl);
    
    // Compute inverses
    for (int i = 1-Ngl; i <= N+Ngl+1; i++) {
        df_inv_h(i) = 1.0 / df_h(i);
    }
    
    // Find dmin
    dmin = std::numeric_limits<real_t>::max();
    for (int i = 1; i <= N+1; i++) {
        dmin = std::min(dmin, df_h(i));
    }
}

inline void oneDgridinfo::computeInterpolationMetrics() {
    // Interpolation factor: inp(i) = (f(i) - c(i-1)) / (c(i) - c(i-1))
    // That means, for central difference, 
    //   east face i+1/2 uses cell i and i+1 
    //   then phi(i+1/2) = (f(i+1) - c(i)) / (c(i+1) - c(i)) * phi(i+1) + ...
    //                     (c(i+1) - f(i+1)) / (c(i+1) - c(i)) * phi(i) 
    //   i.e. phi(i+1/2) = inp(i+1) * phi(i+1) + (1 - inp(i+1)) * phi(i)
    // similaryly for west face i-1/2 uses cell i-1 and i 
    //   then phi(i-1/2) = inp(i) * phi(i) + (1 - inp(i)) * phi(i-1) 
    for (int i = 1-Ngl; i <= N+Ngl+1; i++) {
        if (i >= 2-Ngl && i <= N+Ngl) {
            inp_h(i) = (f_h(i) - c_h(i-1)) / (c_h(i) - c_h(i-1));
        } else {
            inp_h(i) = 0.5;  // Default for boundaries
        }
    }
}

inline void oneDgridinfo::uploadToDevice() {
    // Direct copy from host to device offset views
    Kokkos::deep_copy(c, c_h);
    Kokkos::deep_copy(dc, dc_h);
    Kokkos::deep_copy(dc_inv, dc_inv_h);
    Kokkos::deep_copy(f, f_h);
    Kokkos::deep_copy(df, df_h);
    Kokkos::deep_copy(df_inv, df_inv_h);
    Kokkos::deep_copy(inp, inp_h);
}

inline void oneDgridinfo::printSummary() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "1D Grid Information Summary: " << varname << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Grid Type: " << (init_option == GRID_UNIFORM ? "UNIFORM" : "NON-UNIFORM") << std::endl;
    if (init_option == GRID_NONUNIFORM) {
        std::cout << "Grid File: " << filename << std::endl;
    }
    std::cout << "Domain: [" << start << ", " << end << "]" << std::endl;
    std::cout << "Domain Length: " << (end - start) << std::endl;
    
    std::cout << "\nGrid Points:" << std::endl;
    std::cout << "  Interior cells (N): " << N << std::endl;
    std::cout << "  Ghost layers (Ngl): " << Ngl << std::endl;
    std::cout << "  Cell range: [" << (1-Ngl) << ", " << (N+Ngl) << "]" << std::endl;
    std::cout << "  Face range: [" << (1-Ngl) << ", " << (N+Ngl+1) << "]" << std::endl;
    
    std::cout << "\nArray Extents:" << std::endl;
    std::cout << "  Cell arrays:" << std::endl;
    std::cout << "    c: begin=" << c_h.begin(0) << ", end=" << c_h.end(0) 
              << ", extent=" << c_h.extent(0) 
              << " [indices: " << c_h.begin(0) << " to " << (c_h.end(0)-1) << "]" << std::endl;
    std::cout << "    dc: begin=" << dc_h.begin(0) << ", end=" << dc_h.end(0) 
              << ", extent=" << dc_h.extent(0)
              << " [indices: " << dc_h.begin(0) << " to " << (dc_h.end(0)-1) << "]" << std::endl;
    std::cout << "  Face arrays:" << std::endl;
    std::cout << "    f: begin=" << f_h.begin(0) << ", end=" << f_h.end(0) 
              << ", extent=" << f_h.extent(0)
              << " [indices: " << f_h.begin(0) << " to " << (f_h.end(0)-1) << "]" << std::endl;
    std::cout << "    df: begin=" << df_h.begin(0) << ", end=" << df_h.end(0) 
              << ", extent=" << df_h.extent(0)
              << " [indices: " << df_h.begin(0) << " to " << (df_h.end(0)-1) << "]" << std::endl;
    std::cout << "  Interpolation array:" << std::endl;
    std::cout << "    inp: begin=" << inp_h.begin(0) << ", end=" << inp_h.end(0) 
              << ", extent=" << inp_h.extent(0)
              << " [indices: " << inp_h.begin(0) << " to " << (inp_h.end(0)-1) << "]" << std::endl;
    
    std::cout << "\nGrid Spacing:" << std::endl;
    std::cout << "  Minimum face spacing (dmin): " << dmin << std::endl;
    if (init_option == GRID_UNIFORM) {
        std::cout << "  Uniform spacing (dx): " << (end - start) / N << std::endl;
    } else {
        // Find max spacing
        real_t dmax = 0.0;
        for (int i = 1; i <= N+1; i++) {
            dmax = std::max(dmax, df_h(i));
        }
        std::cout << "  Maximum face spacing: " << dmax << std::endl;
        std::cout << "  Spacing ratio (max/min): " << dmax/dmin << std::endl;
    }
    
    std::cout << "\nSample Grid Points:" << std::endl;
    std::cout << "  First interior face f[1]: " << f_h(1) << std::endl;
    std::cout << "  Last interior face f[" << (N+1) << "]: " << f_h(N+1) << std::endl;
    std::cout << "  First interior cell c[1]: " << c_h(1) << std::endl;
    std::cout << "  Last interior cell c[" << N << "]: " << c_h(N) << std::endl;
    
    // Print ghost cells
    std::cout << "\nGhost Cells:" << std::endl;
    for (int i = 1; i <= Ngl; i++) {
        std::cout << "  Left ghost: c[" << (1-i) << "] = " << c_h(1-i) 
                  << ", f[" << (1-i) << "] = " << f_h(1-i) << std::endl;
    }
    for (int i = 1; i <= Ngl; i++) {
        std::cout << "  Right ghost: c[" << (N+i) << "] = " << c_h(N+i) 
                  << ", f[" << (N+i+1) << "] = " << f_h(N+i+1) << std::endl;
    }
    
    std::cout << "\nSample Differences:" << std::endl;
    std::cout << "  dc[1] = " << dc_h(1) << ", df[1] = " << df_h(1) << std::endl;
    std::cout << "  dc[" << N << "] = " << dc_h(N) << ", df[" << (N+1) << "] = " << df_h(N+1) << std::endl;
    
    std::cout << "\nInterpolation Factors:" << std::endl;
    std::cout << "  inp[1] = " << inp_h(1) << " (for interpolating to f[1])" << std::endl;
    std::cout << "  inp[" << N << "] = " << inp_h(N) << " (for interpolating to f[" << N << "])" << std::endl;
    
    std::cout << "========================================\n" << std::endl;
}

#endif // ONEDGRIDINFO_HPP