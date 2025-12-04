// file: src/INCLUDE/include.hpp
#ifndef INCLUDE_HPP
#define INCLUDE_HPP

// Include all project headers in dependency order
#include "structs.hpp"
#include "oneDgridinfo.hpp"
#include "gridinfo.hpp"
#include "field.hpp"
#include "face.hpp"
#include "bctype.hpp"
#include "inputreader.hpp"
#include "phasefield.hpp"

// Include Kokkos
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

// Include common standard libraries
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <limits>

#endif 
// INCLUDE_HPP