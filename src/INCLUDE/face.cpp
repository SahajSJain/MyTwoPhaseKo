// file: src/INCLUDE/face.cpp
#include "face.hpp"
#include "field.hpp"

#include <iostream>

Face::Face(const GridInfo* grid_ptr, const std::string& base_name) 
    : grid(grid_ptr), 
      basename(base_name),
      E(grid, basename + "_E"),
      W(grid, basename + "_W"),
      N(grid, basename + "_N"),
      S(grid, basename + "_S")
      #ifndef USE_2D
      ,F(grid, basename + "_F"),
      B(grid, basename + "_B")
      #endif
{
    if (!grid) {
        std::cerr << "ERROR: Face constructor received null GridInfo pointer\n";
        exit(EXIT_FAILURE);
    }
}

Face::Face(const Face& other)
    : grid(other.grid),
      basename(other.basename + "_copy"),
      E(other.E),
      W(other.W),
      N(other.N),
      S(other.S)
      #ifndef USE_2D
      ,F(other.F),
      B(other.B)
      #endif
{
}

Face& Face::operator=(const Face& other) {
    if (this != &other) {
        // Grid pointer should remain the same, just copy the data
        basename = other.basename;
        E = other.E;
        W = other.W;
        N = other.N;
        S = other.S;
        #ifndef USE_2D
        F = other.F;
        B = other.B;
        #endif
    }
    return *this;
}

void Face::upload() {
    E.upload();
    W.upload();
    N.upload();
    S.upload();
    #ifndef USE_2D
    F.upload();
    B.upload();
    #endif
}

void Face::download() {
    E.download();
    W.download();
    N.download();
    S.download();
    #ifndef USE_2D
    F.download();
    B.download();
    #endif
}

void Face::fill(real_t value) {
    E.fill(value);
    W.fill(value);
    N.fill(value);
    S.fill(value);
    #ifndef USE_2D
    F.fill(value);
    B.fill(value);
    #endif
}

Face& Face::operator+=(const Face& other) {
    E += other.E;
    W += other.W;
    N += other.N;
    S += other.S;
    #ifndef USE_2D
    F += other.F;
    B += other.B;
    #endif
    return *this;
}

Face& Face::operator*=(const Face& other) {
    E *= other.E;
    W *= other.W;
    N *= other.N;
    S *= other.S;
    #ifndef USE_2D
    F *= other.F;
    B *= other.B;
    #endif
    return *this;
}

Face& Face::operator-=(const Face& other) {
    E -= other.E;
    W -= other.W;
    N -= other.N;
    S -= other.S;
    #ifndef USE_2D
    F -= other.F;
    B -= other.B;
    #endif
    return *this;
}

Face& Face::operator*=(real_t scalar) {
    E *= scalar;
    W *= scalar;
    N *= scalar;
    S *= scalar;
    #ifndef USE_2D
    F *= scalar;
    B *= scalar;
    #endif
    return *this;
}

Face& Face::operator/=(real_t scalar) {
    E /= scalar;
    W /= scalar;
    N /= scalar;
    S /= scalar;
    #ifndef USE_2D
    F /= scalar;
    B /= scalar;
    #endif
    return *this;
}

Face& Face::operator+=(real_t scalar) {
    E += scalar;
    W += scalar;
    N += scalar;
    S += scalar;
    #ifndef USE_2D
    F += scalar;
    B += scalar;
    #endif
    return *this;
}

Face& Face::operator-=(real_t scalar) {
    E -= scalar;
    W -= scalar;
    N -= scalar;
    S -= scalar;
    #ifndef USE_2D
    F -= scalar;
    B -= scalar;
    #endif
    return *this;
}

Face Face::operator+(const Face& other) const {
    Face result(*this);  // Copy constructor
    result += other;
    return result;
}

Face Face::operator*(const Face& other) const {
    Face result(*this);
    result *= other;
    return result;
}

Face Face::operator-(const Face& other) const {
    Face result(*this);
    result -= other;
    return result;
}

Face Face::operator+(real_t scalar) const {
    Face result(*this);
    result += scalar;
    return result;
}

Face Face::operator-(real_t scalar) const {
    Face result(*this);
    result -= scalar;
    return result;
}

Face Face::operator*(real_t scalar) const {
    Face result(*this);
    result *= scalar;
    return result;
}

Face Face::operator/(real_t scalar) const {
    Face result(*this);
    result /= scalar;
    return result;
}

void Face::printInfo() const {
    std::cout << "\nFace collection: " << basename << std::endl;
    E.printInfo();
    W.printInfo();
    N.printInfo();
    S.printInfo();
    #ifndef USE_2D
    F.printInfo();
    B.printInfo();
    #endif
}