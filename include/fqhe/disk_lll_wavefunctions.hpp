#pragma once

#include "lll_orbital.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <optional>

namespace fqhe {

/**
 *  Wavefunction manager for Disk LLLs.
 */
class DiskLLLWavefunctions {
public:
    using Complex = std::complex<double>;
    using ComplexVector = Eigen::VectorXcd;
    using ComplexMatrix = Eigen::MatrixXcd;

    /**
     * @brief Constructor with no computation, max radius unspecified
     * @param max_m Maximum angular momentum quantum number
     * Implicit: r_max = sqrt(2 max_m) + 5.
     */
    explicit DiskLLLWavefunctions(std::size_t max_m);

    /**
    * @brief Constructor with no computation, max radius specified
    * @param max_m Maximum angular momentum quantum number
    * @param r_max Maximum radius stored in orbital wavefunction grid.
    */
    DiskLLLWavefunctions(std::size_t max_m, double r_max);

    /**
    * @brief Constructor with no computation, max radius and grid size specified.
    * @param max_m Maximum angular momentum quantum number.
    * @param r_max Maximum radius stored in orbital wavefunction grid.
    * @param grid_size Grid size of orbitals (radial and angular)
    */
    DiskLLLWavefunctions(std::size_t max_m, double r_max, std::size_t grid_size);

    /**
     * @brief Getter of maximum angular momentum quantum number.
     * @return Maximum m value.
     */
    std::size_t max_angular_momentum() const noexcept;

    /**
     * @brief Getter of number of orbitals.
     * @return Number of orbitals (max_m + 1).
     */
    std::size_t num_orbitals() const noexcept;

    /**
     * @brief Get a specific orbital by angular momentum, returning nullopt if failure.
     * @param m Angular momentum quantum number
     * @return Shared pointer to the LLL orbital, or nullopt if failure
     */
    [[nodiscard]] std::optional<std::shared_ptr<LLLOrbital>> get_orbital(std::size_t m) const;



    /**
     * @brief Get all orbitals.
     * @return Vector of pointers to orbitals
     */
    [[nodiscard]] const std::vector<std::shared_ptr<LLLOrbital>>& get_all_orbitals() const;

    /**
     * @brief Updates/adds grid_size: requires orbitals to not be computed.
     * @param grid_size Size of the wavefunction grid.
     */
    void enable_compute(std::size_t grid_size);

    /**
     * @brief Compute LLL orbitals: requires previous specification of grid size.
    */
    void compute();

    /**
     * @brief Checks whether computation is possible.
     * @return True if the orbitals can been computed (grid size specified).
    */
    bool can_compute() const;

    /**
     * @brief Checks whether orbitals have been computed.
     * @return True if the orbital wavefunctions have been computed.
    */
    bool is_computed() const;

    /**
     * @brief Evaluates specified wavefunction at location
     * @param z Location of wavefunction evaluation.
     * @return Complex wavefunction amplitude at location
     */
     Complex evaluate(std::size_t m, Complex z) const;

     /**
     * @brief Get the geometry type (always DISK in present iteration)
     * @return DISK geometry
     */
    Geometry geometry() const;

private:
    std::size_t max_m_;
    std::size_t grid_size_;
    std::vector<std::shared_ptr<LLLOrbital>> orbitals_;
    bool ready_to_compute_;
    bool computed_;
    double r_max_;
};

}