// file: src/INCLUDE/field.hpp
#ifndef FIELD_HPP
#define FIELD_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include "gridinfo.hpp"
#include "structs.hpp"
#include <iostream>
#include <cstdlib>
#include <string>

class Field
{
public:
    // Grid information pointer
    const GridInfo *grid;

    // Field name
    const std::string name;

    // Dimensions
    const int Nx, Ny, Nz;
    const int Ngl;
    const int Ntotal;

    // OffsetViews ONLY
    Kokkos::Experimental::OffsetView<real_t ***> u;                      // Device
    Kokkos::Experimental::OffsetView<real_t ***, Kokkos::HostSpace> u_h; // Host

    // Constructor - RAII with GridInfo
    Field(const GridInfo *grid_ptr, const std::string &field_name);

    // Deep copy constructor
    Field(const Field &other);

    // Assignment operator
    Field &operator=(const Field &other);

    // Destructor
    ~Field() = default;

    // Device operations
    void upload();
    void download();

    // Deep copy operation
    void deepCopy(const Field &source);

    // Additional methods
    void fill(real_t value);
    void swap(Field &other);
    bool isOnDevice() const { return true; } // Always true for Kokkos

    // Print info
    void printInfo() const;

    // this = a + b
    void add(const Field &a, const Field &b);

    // this = a * b
    void multiply(const Field &a, const Field &b);

    // Operator overloading
    Field &operator+=(const Field &other);
    Field &operator*=(const Field &other);
    Field &operator-=(const Field &other);

    // Binary operators (create new Field)
    Field operator+(const Field &other) const;
    Field operator*(const Field &other) const;
    Field operator-(const Field &other) const;

    // Scalar operations
    Field &operator*=(real_t scalar);
    Field &operator/=(real_t scalar);
    Field &operator+=(real_t scalar);
    Field &operator-=(real_t scalar);
    Field operator*(real_t scalar) const;
    Field operator/(real_t scalar) const;
    Field operator-(real_t scalar) const;
    Field operator+(real_t scalar) const;

    // inverse
    Field inverse(real_t scalar = 1.0) const;

private:
    void allocateArrays();
};

// Friend functions for scalar operations
Field operator*(real_t scalar, const Field &field);
Field operator/(real_t scalar, const Field &field);

#endif // FIELD_HPP
