#include "fqhe/fock_hamiltonian.hpp"
#include <Spectra/HermEigsSolver.h>
#include <Spectra/MatOp/SparseHermMatProd.h>

namespace fqhe {
using Complex = std::complex<double>;

FockHamiltonian::FockHamiltonian(std::shared_ptr<std::vector<BasisState>> basis,
                                 std::shared_ptr<OrbitalInteraction> V)
    : basis_(std::move(basis)), V_(std::move(V)), H_(), evals_(), evecs_(), diagonalized_(false) {
    build_matrix();
}

const Eigen::SparseMatrix<Complex>& FockHamiltonian::matrix() const {
    return H_;
}

void FockHamiltonian::diagonalize(int n_eigenvalues, int max_iterations) {
    if (H_.rows() == 0 || H_.cols() == 0) {
        build_matrix();
    }

    Spectra::SparseHermMatProd<Complex> op(H_);
    Spectra::HermEigsSolver<Spectra::SparseHermMatProd<Complex>> eigs(op, n_eigenvalues, 2 * n_eigenvalues);

    eigs.init();
    eigs.compute(Spectra::SortRule::SmallestAlge, max_iterations, 1e-10);

    if (eigs.info() == Spectra::CompInfo::Successful) {
        evals_ = eigs.eigenvalues();
        evecs_ = eigs.eigenvectors();
        diagonalized_ = true;
    } else {
        throw std::runtime_error("Spectra eigenvalue computation failed.");
    }
}


const Eigen::VectorXcd& FockHamiltonian::eigenvalues() const {
    return evals_;
}

const Eigen::MatrixXcd& FockHamiltonian::eigenvectors() const {
    return evecs_;
}

std::shared_ptr<const std::vector<BasisState>> FockHamiltonian::basis() const {
    return basis_;
}

size_t FockHamiltonian::dimension() const {
    return basis_->size();
}

bool FockHamiltonian::is_diagonalized() const {
    return diagonalized_;
}

double FockHamiltonian::expectation_value(int state_index, const Eigen::SparseMatrix<Complex>& operator_matrix) const {
    Eigen::VectorXcd psi = evecs_.col(state_index);
    Complex res = psi.adjoint().dot(operator_matrix * psi);
    return std::real(res);
}

Eigen::MatrixXcd FockHamiltonian::density_matrix(int state_index) const {
    (void)state_index;
    Eigen::VectorXcd psi = evecs_.col(state_index);
    return psi * psi.adjoint();
}
void FockHamiltonian::build_matrix() {
    const auto dim = basis_->size();
    H_.resize(dim, dim);
    std::vector<Eigen::Triplet<Complex>> tripletList;

    int m = -1, n = -1, p = -1, q = -1;

    for (int i = 0; i < dim; ++i) {
        for (int j = i; j < dim; ++j) {
            if ((*basis_)[i].differs_by_two((*basis_)[j], m, n, p, q)) {
                Complex val = (*V_)(m, n, p, q);
                if (val != Complex(0.0, 0.0)) {
                    tripletList.emplace_back(i, j, val);
                    if (i != j) {
                        tripletList.emplace_back(j, i, std::conj(val));
                    }
                }
            }
        }
    }

    H_.setFromTriplets(tripletList.begin(), tripletList.end());
}



}