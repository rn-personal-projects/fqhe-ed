#pragma once

#include "basis_state.hpp"
#include "orbital_interaction.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace fqhe {
using Complex = std::complex<double>;
/**
 * Fock FQHE Hamiltonian with basis of symmetric gauge Slater states.
 */
class FockHamiltonian {
public:
    /**
     * @brief Constructor
     * @param basis Vector of basis states
     * @param V Orbital interaction calculator
     */
    FockHamiltonian(std::shared_ptr<std::vector<BasisState>> basis,
                    std::shared_ptr<OrbitalInteraction> V);

    /**
     * @brief Getter of Hamiltonian matrix
     * @return Const reference to Hamiltonian matrix.
     */
    [[nodiscard]] const Eigen::SparseMatrix<Complex>& matrix() const;

    /**
     * @brief Diagonalize the Hamiltonian to find the lowest energy eigenvalues
     * @param n_eigenvalues Number of eigenvalues to compute
     * @param max_iterations Max iterations for eigensolver
     */
    void diagonalize(int n_eigenvalues = 1, int max_iterations = 1000);

    /**
     * @brief Getter of computed eigenvalues
     * @return Eigenvalues vector
     */
    [[nodiscard]] const Eigen::VectorXcd& eigenvalues() const;

    /**
     * @brief Getter of computed eigenvectors
     * @return Eigenvectors matrix
     */
    [[nodiscard]] const Eigen::MatrixXcd& eigenvectors() const;

    /**
     * @brief Gets basis states used by this Hamiltonian
     * @return Vector of basis states
     */
    [[nodiscard]] std::shared_ptr<const std::vector<BasisState>> basis() const;

    /**
     * @brief Get the dimension of the Hilbert space
     * @return Dimension of the basis
     */
    [[nodiscard]] std::size_t dimension() const;

    /**
     * @brief Describes how many eigenvalues have been found
     * @return True if diagonalization is complete, false otherwise
     */
    [[nodiscard]] bool is_diagonalized() const;

    /**
     * @brief Compute expectation value of an operator
     * @param state_index Index of the eigenstate
     * @param operator_matrix Operator matrix
     * @return Expectation value
     */
    [[nodiscard]] double expectation_value(int state_index, const Eigen::SparseMatrix<Complex>& operator_matrix) const;

    /**
     * @brief Compute density matrix for a given eigenstate
     * @param state_index Index of the eigenstate
     * @return Density matrix
     */
    [[nodiscard]] Eigen::MatrixXcd density_matrix(int state_index) const;

private:
    std::shared_ptr<std::vector<BasisState>> basis_;           ///< Basis states
    std::shared_ptr<OrbitalInteraction> V_;                    ///< Orbital interaction calculator
    Eigen::SparseMatrix<Complex> H_;           ///< Hamiltonian matrix
    Eigen::VectorXcd evals_;                   ///< Eigenvalues
    Eigen::MatrixXcd evecs_;                   ///< Eigenvectors
    bool diagonalized_;                ///< True if diagonalization is complete, false otherwise

    /**
     * @brief Build the Hamiltonian matrix
     */
    void build_matrix();
};

} // namespace fqhe 