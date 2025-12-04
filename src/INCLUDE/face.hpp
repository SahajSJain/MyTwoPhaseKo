// file: src/INCLUDE/face.hpp
#ifndef FACE_HPP
#define FACE_HPP

#include "field.hpp"
#include "gridinfo.hpp"
#include <string>

class Face {
public:
    // Grid information pointer
    const GridInfo* grid;
    
    // Base name for all face fields
    std::string basename;
    
    // Face fields - direct objects, not pointers
    Field E;  // East face (x+)
    Field W;  // West face (x-)
    Field N;  // North face (y+)
    Field S;  // South face (y-)
    #ifndef USE_2D
    Field F;  // Front face (z+)
    Field B;  // Back face (z-)
    #endif
    
    // Constructor
    Face(const GridInfo* grid_ptr, const std::string& base_name);
    
    // Copy constructor
    Face(const Face& other);
    
    // Assignment operator
    Face& operator=(const Face& other);
    
    // Upload all faces to device
    void upload();
    
    // Download all faces from device
    void download();
    
    // Fill all faces with a value
    void fill(real_t value);
    
    // In-place operators
    Face& operator+=(const Face& other);
    Face& operator*=(const Face& other);
    Face& operator-=(const Face& other);
    Face& operator*=(real_t scalar);
    Face& operator/=(real_t scalar);
    Face& operator+=(real_t scalar);
    Face& operator-=(real_t scalar);
    
    // Binary operators (create new Face)
    Face operator+(const Face& other) const;
    Face operator*(const Face& other) const;
    Face operator-(const Face& other) const;
    Face operator+(real_t scalar) const;
    Face operator-(real_t scalar) const;
    Face operator*(real_t scalar) const;
    Face operator/(real_t scalar) const;
    
    // Print info
    void printInfo() const;
};

// Friend function for scalar * Face
inline Face operator*(real_t scalar, const Face& face) {
    return face * scalar;
}

#endif // FACE_HPP