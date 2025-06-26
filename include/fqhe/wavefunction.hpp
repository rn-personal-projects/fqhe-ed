#pragma once

#include <complex>
#include <memory>
#include <Eigen/Dense>
#include "geometry.hpp"

namespace fqhe {

/**
 * Abstract class representing a wavefunction.
 */
class Wavefunction {
public:
    using Complex = std::complex<double>;
    using ComplexVector = Eigen::VectorXcd;
    using ComplexMatrix = Eigen::MatrixXcd;
    using RealVector = Eigen::VectorXd;
    using RealMatrix = Eigen::MatrixXd;

    /**
     * @brief Constructs a wavefunction object with empty data.
     */
    Wavefunction() :  computed_(false) {};
    /**
     * @brief Virtual destructor for inheritors storing additional data.
     */
    virtual ~Wavefunction() = default;

    /**
     * @brief Get the geometry type of this wavefunction
     * @return Geometry type
     */
    virtual Geometry geometry() const = 0;

    /**
     * @brief Computes the wavefunction matrix.
     */
    virtual void compute() = 0;

    /**
     * @brief Checks if wavefunction matrix precomputed
     * @return True if wavefunction has been precomputed
     */
    virtual bool is_computed() const = 0;

    /**
     * @brief Compute overlap between this wavefunction and another
     * @param other Other wavefunction
     * @return Complex overlap value
     */
    virtual Complex overlap(const Wavefunction& other) const = 0;

    /**
     * @brief Compute density profile at given positions
     * @param positions Vector of complex positions
     * @return Density values at each position
     */
    virtual RealVector density_profile(const ComplexVector& positions) const = 0;


    /**
     * @brief Evaluate the wavefunction at a complex position
     * @param z Complex position
     * @return Complex value
     */
    virtual inline Complex evaluate(Complex z) const = 0;


protected:
    ComplexMatrix psi;
    bool computed_;
};

}