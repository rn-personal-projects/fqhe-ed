#pragma once

#include "wavefunction.hpp"
#include <Eigen/Dense>
#include <optional>

namespace fqhe {

/**
* Single LLL orbital in symmetric gauge on the disk,
* implements the abstract wavefunction class.
*/
class LLLOrbital : public Wavefunction {
public:
    /**
     * @brief Constructor with specified angular momentum and grid size.
     * @param m Angular momentum quantum number
     * @param grid_size Grid size (radial and angular)
     * Implicit: r_max = 2 sqrt(m) + 5
     */
    explicit LLLOrbital(std::size_t m, std::size_t grid_size);

    /**
     * @brief Constructor with specified r_max
     * @param m Angular momentum quantum number
     * @param grid_size Grid size (radial and angular)
     * @param r_max Maximum radius
     */
    LLLOrbital(std::size_t m, std::size_t grid_size, double r_max);

    /**
     * @brief Get the geometry type (always DISK)
     * @return DISK geometry
     */
    Geometry geometry() const override;

    /**
     * @brief Computes the wavefunction matrix.
     */
    void compute() override;

    /**
     * @brief Checks if wavefunction matrix precomputed
     * @return True if wavefunction has been precomputed
     */
    bool is_computed() const override;

    /**
     * @brief Compute overlap with another wavefunction
     * @param other Other LLLOrbital
     * @return Complex overlap value if other has been computed.
     */
    std::optional<Complex> overlap(const LLLOrbital &other) const;

    /**
     * @brief Compute density profile at given positions
     * @param positions Vector of complex positions
     * @return Density values at each position
     */
    RealVector density_profile(const ComplexVector &positions) const override;

    /**
     * @brief Evaluate the orbital at a complex position
     * @param z Complex position
     * @return Complex value
     */
    Complex evaluate(Complex z) const override;

    /**
     * @brief Evaluate in polar coordinates
     * @param r Radius
     * @param theta Angle
     * @return Complex value
     */
    Complex evaluate(double r, double theta) const;

    /**
     * @brief Get the angular momentum quantum number
     * @return Angular momentum m
     */
    std::size_t angular_momentum() const noexcept;

    /**
     * @brief Compute overlap with another wavefunction
     * @param other Other wavefunction
     * @return Complex overlap value if other has been computed.
     */
    Complex overlap(const Wavefunction &other) const override;

    /**
     * @brief Getter of grid size.
     * @return Grid size.
     */
    std::size_t grid_size() const { return grid_size_; }

    /**
     * @brief Getter of max radius
     * @return Max radius
     */
    std::size_t r_max() const { return r_max_; }

    /**
     * @brief Getter of radius differential.
     * @return Radius differential.
     */
    double d_r() const { return d_theta_; }

    /**
     * @brief Getter of angular differential.
     * @return Angular differential.
     */
    double d_theta() const { return d_r_; }

private:
    std::size_t m_;
    std::size_t grid_size_;
    double r_max_;
    double d_r_;
    double d_theta_;
    double norm_const;
};

}