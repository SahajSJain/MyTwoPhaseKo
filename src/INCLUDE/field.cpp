// file: src/INCLUDE/field.cpp
#include "field.hpp"
// Constructor
Field::Field(const GridInfo *grid_ptr, const std::string &field_name)
    : grid(grid_ptr),
      name(field_name),
      Nx(grid->xgrid->N),
      Ny(grid->ygrid->N),
      Nz(grid->zgrid->N),
      Ngl(grid->config.Ngl),
      Ntotal((Nx + 2 * Ngl) * (Ny + 2 * Ngl) * (Nz + 2 * Ngl))
{
    if (!grid)
    {
        std::cerr << "ERROR: Field constructor received null GridInfo pointer\n";
        exit(EXIT_FAILURE);
    }
    allocateArrays();
}

void Field::allocateArrays()
{
    auto x_bs = grid->xgrid->bs;
    auto x_be = grid->xgrid->be;
    auto y_bs = grid->ygrid->bs;
    auto y_be = grid->ygrid->be;
    auto z_bs = grid->zgrid->bs;
    auto z_be = grid->zgrid->be;
    // Create offset views directly with the proper bounds
    // Goes from (1-Ngl, 1-Ngl, 1-Ngl) to (N+Ngl, N+Ngl, N+Ngl)
    u = Kokkos::Experimental::OffsetView<real_t ***>(
        name,
        std::pair<int64_t, int64_t>{x_bs, x_be + 1},
        std::pair<int64_t, int64_t>{y_bs, y_be + 1},
        std::pair<int64_t, int64_t>{z_bs, z_be + 1});

    u_h = Kokkos::Experimental::OffsetView<real_t ***, Kokkos::HostSpace>(
        name + "_host",
        std::pair<int64_t, int64_t>{x_bs, x_be + 1},
        std::pair<int64_t, int64_t>{y_bs, y_be + 1},
        std::pair<int64_t, int64_t>{z_bs, z_be + 1});

    Kokkos::deep_copy(u_h, 0.0);
    Kokkos::deep_copy(u, 0.0);
}

// Copy constructor
Field::Field(const Field &other)
    : grid(other.grid),
      name(other.name + "_copy"),
      Nx(other.Nx),
      Ny(other.Ny),
      Nz(other.Nz),
      Ngl(other.Ngl),
      Ntotal(other.Ntotal)
{
    allocateArrays();
    deepCopy(other);
}

// Assignment operator
Field &Field::operator=(const Field &other)
{
    if (this != &other)
    {
        if (Ntotal != other.Ntotal)
        {
            std::cerr << "ERROR: Cannot assign fields of different sizes\n";
            exit(EXIT_FAILURE);
        }
        deepCopy(other);
    }
    return *this;
}

void Field::deepCopy(const Field &source)
{
    if (Ntotal != source.Ntotal)
    {
        std::cerr << "ERROR: Cannot deep copy fields of different sizes\n";
        exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(u_h, source.u_h);
    Kokkos::deep_copy(u, source.u);
}

void Field::upload()
{
    Kokkos::deep_copy(u, u_h);
}

void Field::download()
{
    Kokkos::deep_copy(u_h, u);
}

void Field::fill(real_t value)
{
    for (int i = 1 - Ngl; i <= Nx + Ngl; i++)
    {
        for (int j = 1 - Ngl; j <= Ny + Ngl; j++)
        {
            for (int k = 1 - Ngl; k <= Nz + Ngl; k++)
            {
                u_h(i, j, k) = value;
            }
        }
    }
}

void Field::swap(Field &other)
{
    if (Ntotal != other.Ntotal)
    {
        std::cerr << "ERROR: Cannot swap fields of different sizes\n";
        exit(EXIT_FAILURE);
    }
    std::swap(u, other.u);
    std::swap(u_h, other.u_h);
}

void Field::printInfo() const
{
    std::cout << "\nField: " << name << std::endl;
    std::cout << "  Dimensions: " << Nx << " x " << Ny << " x " << Nz << std::endl;
    std::cout << "  Ghost layers: " << Ngl << std::endl;
    std::cout << "  Total size: " << Ntotal << std::endl;
    std::cout << "  Index ranges: [" << (1 - Ngl) << ":" << (Nx + Ngl)
              << "] x [" << (1 - Ngl) << ":" << (Ny + Ngl)
              << "] x [" << (1 - Ngl) << ":" << (Nz + Ngl) << "]" << std::endl;
}

void Field::add(const Field &a, const Field &b)
{
    if (Ntotal != a.Ntotal || Ntotal != b.Ntotal)
    {
        std::cerr << "ERROR: Cannot add fields of different sizes\n";
        exit(EXIT_FAILURE);
    }

    auto u_dst = this->u;
    auto u_a = a.u;
    auto u_b = b.u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_add", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) = u_a(i, j, k) + u_b(i, j, k); 
        });

    Kokkos::fence();
}

void Field::multiply(const Field &a, const Field &b)
{
    if (Ntotal != a.Ntotal || Ntotal != b.Ntotal)
    {
        std::cerr << "ERROR: Cannot multiply fields of different sizes\n";
        exit(EXIT_FAILURE);
    }

    auto FullPolicy = grid->full_policy;

    auto u_dst = this->u;
    auto u_a = a.u;
    auto u_b = b.u;

    Kokkos::parallel_for("Field_multiply", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) = u_a(i, j, k) * u_b(i, j, k); 
        });
    Kokkos::fence();
}

Field &Field::operator+=(const Field &other)
{
    if (Ntotal != other.Ntotal)
    {
        std::cerr << "ERROR: Cannot add fields of different sizes\n";
        exit(EXIT_FAILURE);
    }

    auto FullPolicy = grid->full_policy;

    auto u_dst = this->u;
    auto u_other = other.u;
    
    Kokkos::parallel_for("Field_add_inplace", 
        FullPolicy,
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) += u_other(i, j, k); 
        });
    Kokkos::fence();
    return *this;
}

Field &Field::operator*=(const Field &other)
{
    if (Ntotal != other.Ntotal)
    {
        std::cerr << "ERROR: Cannot multiply fields of different sizes\n";
        exit(EXIT_FAILURE);
    }

    auto u_dst = this->u;
    auto u_src = other.u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_multiply_assign", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) *= u_src(i, j, k); 
        });
    Kokkos::fence();
    return *this;
}

Field &Field::operator-=(const Field &other)
{
    if (Ntotal != other.Ntotal)
    {
        std::cerr << "ERROR: Cannot subtract fields of different sizes\n";
        exit(EXIT_FAILURE);
    }

    auto u_dst = this->u;
    auto u_src = other.u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_subtract_assign", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) -= u_src(i, j, k); 
        });
    Kokkos::fence();
    return *this;
}

Field Field::operator+(const Field &other) const
{
    Field result(this->grid, this->name + "_plus_" + other.name);
    result.add(*this, other);
    return result;
}

Field Field::operator*(const Field &other) const
{
    Field result(this->grid, this->name + "_times_" + other.name);
    result.multiply(*this, other);
    return result;
}

Field Field::operator-(const Field &other) const
{
    Field result(this->grid, this->name + "_minus_" + other.name);

    auto u_dst = result.u;
    auto u_a = this->u;
    auto u_b = other.u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_subtract", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) = u_a(i, j, k) - u_b(i, j, k); 
        });
    Kokkos::fence();
    return result;
}

Field &Field::operator*=(real_t scalar)
{
    auto u_dst = this->u;
    auto FullPolicy = grid->full_policy;    
    Kokkos::parallel_for("Field_scalar_multiply", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) *= scalar; 
        });
    Kokkos::fence();
    return *this;
}

Field &Field::operator/=(real_t scalar)
{
    auto u_dst = this->u;
    auto FullPolicy = grid->full_policy;    
    Kokkos::parallel_for("Field_scalar_divide", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) /= scalar; 
        });
    Kokkos::fence();
    return *this;
}

Field &Field::operator+=(real_t scalar)
{
    auto u_dst = this->u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_scalar_add_assign", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) += scalar; 
        });
    Kokkos::fence();
    return *this;
}

Field &Field::operator-=(real_t scalar)
{
    auto u_dst = this->u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_scalar_subtract_assign", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) -= scalar; 
        });
    Kokkos::fence();
    return *this;
}

Field Field::operator*(real_t scalar) const
{
    Field result(*this); // Copy constructor
    result *= scalar;
    return result;
}

Field Field::operator/(real_t scalar) const
{
    Field result(*this); // Copy constructor
    result /= scalar;
    return result;
}

Field Field::operator-(real_t scalar) const
{
    Field result(this->grid, this->name + "_minus_scalar");

    auto u_dst = result.u;
    auto u_src = this->u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_scalar_subtract", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) = u_src(i, j, k) - scalar; 
        });
    Kokkos::fence();
    return result;
}

Field Field::operator+(real_t scalar) const
{
    Field result(this->grid, this->name + "_plus_scalar");

    auto u_dst = result.u;
    auto u_src = this->u;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_scalar_add", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) { 
            u_dst(i, j, k) = u_src(i, j, k) + scalar; 
        });
    Kokkos::fence();
    return result;
}

Field Field::inverse(real_t scalar) const
{
    Field result(this->grid, this->name + "_inverse");

    auto u_dst = result.u;
    auto u_src = this->u;
    const real_t epsilon = REAL_EPSILON;
    auto FullPolicy = grid->full_policy;
    Kokkos::parallel_for("Field_inverse", 
        FullPolicy, 
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            real_t denominator = u_src(i, j, k);
            // Avoid division by zero
            if (Kokkos::abs(denominator) < epsilon) {
                denominator = (denominator >= 0) ? epsilon : -epsilon;
            }
            u_dst(i, j, k) = scalar / denominator; 
        });
    Kokkos::fence();
    return result;
}

// Friend functions for scalar operations
Field operator*(real_t scalar, const Field &field)
{
    return field * scalar;
}

Field operator/(real_t scalar, const Field &field)
{
    return field.inverse(scalar);
}
