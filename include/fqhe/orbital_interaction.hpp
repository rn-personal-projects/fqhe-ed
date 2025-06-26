#pragma once

#include "disk_lll_wavefunctions.hpp"
#include <memory>
#include <unordered_map>
#include <tuple>

namespace fqhe {


struct TupleHash {
    template<typename T>
    std::size_t operator()(const T& t) const {
        return std::hash<std::size_t>{}(std::get<0>(t)) ^
               (std::hash<std::size_t>{}(std::get<1>(t)) << 1) ^
               (std::hash<std::size_t>{}(std::get<2>(t)) << 2) ^
               (std::hash<std::size_t>{}(std::get<3>(t)) << 3);
    }
};

/**
 * Computes and stores matrix interaction elements.
 */
class OrbitalInteraction {
public:
    using Complex = std::complex<double>;
    /**
     * @brief Constructor.
     * @param wfs A shared pointer to the manager for LLL orbitals.
     * @param alpha The exponent of the power-law interaction potential.
     * @param lambda The strength of the interaction potential.
     */
    OrbitalInteraction(std::shared_ptr<DiskLLLWavefunctions> wfs, double alpha, double lambda);

    /**
     * @brief On-demand access to a matrix element V_mnpq.
     * If the element is not in the cache, it is computed, stored, and then returned.
     * @return The value of the matrix element.
     */
    Complex operator()(std::size_t m, std::size_t n, std::size_t p, std::size_t q) const;

    /**
     * @brief Computes all wavefunctions in the wavefunction manager object, with given grid size.
     * This must be performed before matrix elements can be computed.
     * Null-op if computation has already been performed, or if grid_size is not passed.
     * @param grid_size Grid size for computation.
     */
    void compute_all_wfs(std::optional<std::size_t> grid_size = std::nullopt);

    /**
     * @brief Computes all matrix elements up to a specified angular momentum.
     * This will populate the internal cache with all relevant elements.
     */
    void compute_all_elements();

    /**
     * @brief Checks if a specific element has been computed and is in the cache.
     * @return True if the element is in the cache, false otherwise.
     */
    inline bool is_cached(std::size_t m, std::size_t n, std::size_t p, std::size_t q) const;

    /**
     * @brief Clears the entire cache of computed matrix elements.
     */
    void clear_cache();

    /**
     * @brief Gets the underlying wavefunction manager object.
     */
    std::shared_ptr<DiskLLLWavefunctions> wavefunctions() const;

    /**
     * @brief Gets the interaction potential parameters.
     * @return Tuple of the form (lambda, alpha), see constructor.
     */
    std::tuple<double, double> parameters() const;

private:
    /**
     * @brief The core function to compute a single matrix element.
     * This performs the actual integration using the stored wavefunctions.
     * @return The computed value of the matrix element.
     */
    Complex compute_element(std::size_t m, std::size_t n, std::size_t p, std::size_t q) const;

    std::shared_ptr<DiskLLLWavefunctions> wfs_;
    double alpha_;
    double lambda_;
    std::size_t m_max_;
    bool wfs_computed_;
    bool elements_computed_;

    mutable std::unordered_map<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>, Complex, TupleHash> cache_;

    inline Complex evaluate_interaction(Complex z1, Complex z2) const;

    inline Complex evaluate_interaction(double r1, double r2, double theta1, double theta2) const;
};

} // namespace fqhe 