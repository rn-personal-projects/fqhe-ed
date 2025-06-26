#include <iostream>
#include <memory>
#include "fqhe/fqhe.hpp"

int main() {
    
    try {
        const std::size_t grid_size = 15;
        const int N = 3; // Number of particles
        const int M = 6; // Number of orbitals
        const int Lz = 6; // Total angular momentum

        auto basis_states = std::make_shared<std::vector<fqhe::BasisState>>(
            fqhe::BasisState::generate_fixed_Lz(N, M, Lz)
        );

        if (basis_states->size() <= 1) {
            return 1;
        }
        
        for (size_t i = 0; i < basis_states->size(); ++i) {
            std::cout << "Basis state " << i << ":" << std::bitset<64>((*basis_states)[i].raw()) << "\n";
        }

        auto wfs = std::make_shared<fqhe::DiskLLLWavefunctions>(M, grid_size);
        wfs->enable_compute(grid_size); 
        wfs->compute(); 
        auto interaction_v = std::make_shared<fqhe::OrbitalInteraction>(wfs, 1.0, 1.0);
        interaction_v->compute_all_elements();
        for (size_t i = 0; i < basis_states->size(); ++i) {
            for (size_t j = i + 1; j < basis_states->size(); ++j) {
                int m, n, p, q;
                if ((*basis_states)[i].differs_by_two((*basis_states)[j], m, n, p, q)) {
                    auto V_element = (*interaction_v)(m, n, p, q);
                }
            }
        }

        fqhe::FockHamiltonian hamiltonian(basis_states, interaction_v);
        for(fqhe::BasisState state : (*basis_states)) {
            std::cout << std::bitset<64>(state.raw()) << "\n";
        }
        std::cout << "Hamiltonian: \n" << hamiltonian.matrix() << "\n";

        hamiltonian.diagonalize(1);
        auto eigenvalues = hamiltonian.eigenvalues();
        std::cout << "GS Energy:" << eigenvalues(0) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

