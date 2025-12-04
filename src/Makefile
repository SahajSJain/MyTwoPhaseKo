# Makefile for Phase Field Kokkos project

# Compiler - Intel icpx
CXX = g++

# Include directories
INCLUDES = -I. -I/usr/include

# Library directories
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu

# Libraries - order matters!
LIBS = -lkokkoscore -lkokkoscontainers -ldl -lpthread

# Compiler flags for Intel icpx
CXXFLAGS = -O3 -std=c++20 -fopenmp
CXXFLAGS += -Wall -Wextra
CXXFLAGS +=  -g 
CXXFLAGS += -DUSE_DOUBLE

# Linker flags
LDFLAGS = $(LIB_DIRS) $(LIBS) 

# Object directories
OBJDIR_2D = .obj/2d
OBJDIR_3D = .obj/3d

# Source files
SRCS = main.cpp \
       INCLUDE/phasefield.cpp \
       INCLUDE/face.cpp \
       INCLUDE/field.cpp \
       CALCULATORS/face_interpolate.cpp \
       CALCULATORS/face_diff.cpp \
       CALCULATORS/gradient.cpp \
       CALCULATORS/divergence.cpp \
       CALCULATORS/normals.cpp \
       CALCULATORS/convection.cpp \
       CALCULATORS/diffusion.cpp \
       CALCULATORS/sharpening.cpp \
       CALCULATORS/boundary_conditions.cpp \
       CALCULATORS/sdf2phase.cpp \
       CALCULATORS/phase2sdf.cpp \
       CALCULATORS/absolute_value.cpp \
       CALCULATORS/max_magnitude.cpp

# Object files with directory structure preserved
OBJS2D = $(patsubst %.cpp,$(OBJDIR_2D)/%.o,$(SRCS))
OBJS3D = $(patsubst %.cpp,$(OBJDIR_3D)/%.o,$(SRCS))

# Executable names
TARGET_2D = phasefield2d
TARGET_3D = phasefield3d

# Default target - build both
all: $(TARGET_2D) $(TARGET_3D)

# Create object directories
$(OBJDIR_2D) $(OBJDIR_3D):
	@mkdir -p $@
	@mkdir -p $@/INCLUDE
	@mkdir -p $@/CALCULATORS

# Build 2D version
$(TARGET_2D): $(OBJS2D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built 2D version: $(TARGET_2D)"

# Build 3D version
$(TARGET_3D): $(OBJS3D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built 3D version: $(TARGET_3D)"

# Pattern rule for 2D object files
$(OBJDIR_2D)/%.o: %.cpp | $(OBJDIR_2D)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -DUSE_2D $(INCLUDES) -c $< -o $@

# Pattern rule for 3D object files
$(OBJDIR_3D)/%.o: %.cpp | $(OBJDIR_3D)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -rf .obj
	rm -f $(TARGET_2D) $(TARGET_3D)
	rm -f *.optrpt
	rm -rf TECOUT *.tmp

# Clean and rebuild
rebuild: clean all

# Debug build with additional Intel debugging options
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: CXXFLAGS += -check=uninit -fp-stack-check
debug: CXXFLAGS += -ftrapuv  # Initialize uninitialized variables
debug: clean all

# Single precision build
single: CXXFLAGS := $(filter-out -DUSE_DOUBLE,$(CXXFLAGS))
single: clean all

# Intel-specific optimized build
intel-opt: CXXFLAGS = -O3 -xHost -qopenmp -ipo -no-prec-div -fp-model fast=2
intel-opt: CXXFLAGS += -std=c++20 -traceback -DUSE_DOUBLE
intel-opt: clean all

# Show configuration
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Kokkos path: $(KOKKOS_PATH)"
	@echo "Includes: $(INCLUDES)"
	@echo "Libraries: $(LDFLAGS)"
	@echo "Source files: $(SRCS)"
	@echo "Targets: $(TARGET_2D) $(TARGET_3D)"
	@echo "2D Objects: $(OBJS2D)"
	@echo "3D Objects: $(OBJS3D)"

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Build both 2D and 3D versions"
	@echo "  make 2d       - Build only 2D version"
	@echo "  make 3d       - Build only 3D version"
	@echo "  make run2d    - Build and run 2D version"
	@echo "  make run3d    - Build and run 3D version"
	@echo "  make run2d-4  - Build and run 2D with 4 OpenMP threads"
	@echo "  make run3d-4  - Build and run 3D with 4 OpenMP threads"
	@echo "  make clean    - Remove build files and output"
	@echo "  make rebuild  - Clean and rebuild both versions"
	@echo "  make debug    - Build with debug flags"
	@echo "  make single   - Build with single precision"
	@echo "  make intel-opt - Build with aggressive Intel optimizations"
	@echo "  make info     - Show build configuration"
	@echo "  make help     - Show this help message"

# Convenience targets
2d: $(TARGET_2D)
3d: $(TARGET_3D)

run2d: $(TARGET_2D)
	./$(TARGET_2D)

run3d: $(TARGET_3D)
	./$(TARGET_3D)

run2d-4: $(TARGET_2D)
	OMP_NUM_THREADS=4 ./$(TARGET_2D)

run3d-4: $(TARGET_3D)
	OMP_NUM_THREADS=4 ./$(TARGET_3D)

# Dependencies
$(OBJDIR_2D)/main.o $(OBJDIR_3D)/main.o: main.cpp INCLUDE/phasefield.hpp INCLUDE/inputreader.hpp
$(OBJDIR_2D)/INCLUDE/phasefield.o $(OBJDIR_3D)/INCLUDE/phasefield.o: INCLUDE/phasefield.cpp INCLUDE/phasefield.hpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/INCLUDE/face.o $(OBJDIR_3D)/INCLUDE/face.o: INCLUDE/face.cpp INCLUDE/face.hpp
$(OBJDIR_2D)/INCLUDE/field.o $(OBJDIR_3D)/INCLUDE/field.o: INCLUDE/field.cpp INCLUDE/field.hpp
$(OBJDIR_2D)/CALCULATORS/face_interpolate.o $(OBJDIR_3D)/CALCULATORS/face_interpolate.o: CALCULATORS/face_interpolate.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/face_diff.o $(OBJDIR_3D)/CALCULATORS/face_diff.o: CALCULATORS/face_diff.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/gradient.o $(OBJDIR_3D)/CALCULATORS/gradient.o: CALCULATORS/gradient.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/divergence.o $(OBJDIR_3D)/CALCULATORS/divergence.o: CALCULATORS/divergence.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/convection.o $(OBJDIR_3D)/CALCULATORS/convection.o: CALCULATORS/convection.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/normals.o $(OBJDIR_3D)/CALCULATORS/normals.o: CALCULATORS/normals.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/diffusion.o $(OBJDIR_3D)/CALCULATORS/diffusion.o: CALCULATORS/diffusion.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/sharpening.o $(OBJDIR_3D)/CALCULATORS/sharpening.o: CALCULATORS/sharpening.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/boundary_conditions.o $(OBJDIR_3D)/CALCULATORS/boundary_conditions.o: CALCULATORS/boundary_conditions.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/sdf2phase.o $(OBJDIR_3D)/CALCULATORS/sdf2phase.o: CALCULATORS/sdf2phase.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/phase2sdf.o $(OBJDIR_3D)/CALCULATORS/phase2sdf.o: CALCULATORS/phase2sdf.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/absolute_value.o $(OBJDIR_3D)/CALCULATORS/absolute_value.o: CALCULATORS/absolute_value.cpp CALCULATORS/calculators.hpp
$(OBJDIR_2D)/CALCULATORS/max_magnitude.o $(OBJDIR_3D)/CALCULATORS/max_magnitude.o: CALCULATORS/max_magnitude.cpp CALCULATORS/calculators.hpp

.PHONY: all 2d 3d run2d run3d run2d-4 run3d-4 clean rebuild debug single intel-opt info help