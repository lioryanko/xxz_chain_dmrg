import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib
import cProfile

pr = cProfile.Profile()

sz = np.array([[1, 0], [0, -1]])
sp = np.array([[0, 1], [0, 0]])
sm = sp.T


class MatrixCompressor(object):
    def __init__(self, max_dimension):
        self.max_dimension = max_dimension
        self.U = None
        self.V = None

    def _update_compression_matrices(self, U, V):
        # Compress only if the max dimension has been exceeded.
        if self.max_dimension < len(U):
            self.U = U[:self.max_dimension]
            self.V = V[:, :self.max_dimension]

    def update(self, rho_reduced):
        pass

    def compress_operator(self, operator):
        # If the compression matrices have not been updated yet, do nothing (this is for optimization purposes).
        if not self.U:
            return operator

        return self.U @ operator @ self.V

    def get_max_dimension(self):
        return self.max_dimension


class SVDCompressor(MatrixCompressor):
    def update(self, rho_reduced):
        U, _, V = np.linalg.svd(rho_reduced)
        self._update_compression_matrices(U, V)


class EigenCompressor(MatrixCompressor):
    def update(self, rho_reduced):
        U = scipy.linalg.eig(rho_reduced)[1]
        self._update_compression_matrices(np.conjugate(U.T), U)


class XXZChain(object):
    def __init__(self, compressor, h, Jz, J):
        self.compressor = compressor
        self.h = h
        self.Jz = Jz
        self.J = J
        self.block_H = - h * sz
        self.last_site_sz = sz
        self.last_site_sp = sp
        self.last_site_sm = sm
        self.block_M = sz

    def build_super_block_hamiltonian(self):
        block_dimension = len(self.block_H)
        d = len(sz)

        # Add the single block terms.
        super_block_H = np.kron(self.block_H, np.eye(d * block_dimension * d))
        super_block_H += np.kron(np.eye(block_dimension * d), np.kron(self.block_H, np.eye(d)))

        # Add the single site terms.
        super_block_H += np.kron(np.kron(np.eye(block_dimension), -self.h * sz), np.eye(d * block_dimension))
        super_block_H += np.kron(np.eye(block_dimension * d * block_dimension), -self.h * sz)

        # Add the site-site interaction terms.
        site_sz = np.kron(np.eye(block_dimension), sz)
        site_sp = np.kron(np.eye(block_dimension), sp)
        site_sm = np.kron(np.eye(block_dimension), sm)
        super_block_H += self.Jz * np.kron(site_sz, site_sz)
        super_block_H += self.J / 2 * np.kron(site_sp, site_sp)
        super_block_H += self.J / 2 * np.kron(site_sm, site_sm)

        # Add the block-site interaction terms.
        block_site_interaction = self.Jz * np.kron(self.last_site_sz, sz)
        block_site_interaction += self.J / 2 * np.kron(self.last_site_sp, sp)
        block_site_interaction += self.J / 2 * np.kron(self.last_site_sm, sm)

        super_block_H += np.kron(block_site_interaction, np.eye(len(block_site_interaction)))
        super_block_H += np.kron(np.eye(len(block_site_interaction)), block_site_interaction)

        return super_block_H

    def get_super_block_ground_state(self):
        super_block_H = self.build_super_block_hamiltonian()
        eigvals, eigvects = scipy.sparse.linalg.eigsh(super_block_H, k=1, which='SA')
        return eigvals[0].real, eigvects[:, 0]

    def get_super_block_magnetization_expectation_value(self, state):
        block_dimension = len(self.block_H)
        d = len(sz)

        super_block_M = np.kron(self.block_M, np.eye(d * block_dimension * d))
        super_block_M += np.kron(np.eye(block_dimension * d), np.kron(self.block_M, np.eye(d)))
        super_block_M += np.kron(np.kron(np.eye(block_dimension), sz), np.eye(d * block_dimension))
        super_block_M += np.kron(np.eye(block_dimension * d * block_dimension), sz)

        row_state = np.conjugate(state.T)
        return (row_state @ super_block_M @ state) / (row_state @ state)

    def update_compressor(self):
        gs = self.get_super_block_ground_state()[1]
        rho = np.einsum('i,j->ij', gs, np.conjugate(gs))
        rho_reduced_dimension = len(self.block_H) * len(sz)
        rho_reduced = rho.reshape(rho_reduced_dimension, rho_reduced_dimension, rho_reduced_dimension,
                                  rho_reduced_dimension).trace(axis1=2, axis2=3)
        self.compressor.update(rho_reduced)

    def expand(self, iterations):
        for i in range(iterations):
            block_dimension = len(self.block_H)
            next_site_sz = np.kron(np.eye(block_dimension), sz)
            next_site_sp = np.kron(np.eye(block_dimension), sp)
            next_site_sm = np.kron(np.eye(block_dimension), sm)

            block_site_interaction = self.Jz * np.kron(self.last_site_sz, sz)
            block_site_interaction += self.J / 2 * (np.kron(self.last_site_sp, sp) + np.kron(self.last_site_sm, sm))
            block_H = np.kron(self.block_H, np.eye(len(sz))) - self.h * next_site_sz + block_site_interaction
            block_M = np.kron(self.block_M, np.eye(len(sz))) + next_site_sz

            self.update_compressor()

            # Now the block has another site, and the operators operating on it are (possibly) compressed.
            self.block_H = self.compressor.compress_operator(block_H)
            self.block_M = self.compressor.compress_operator(block_M)
            self.last_site_sz = self.compressor.compress_operator(next_site_sz)
            self.last_site_sp = self.compressor.compress_operator(next_site_sp)
            self.last_site_sm = self.compressor.compress_operator(next_site_sm)


def plot_as_function_of_Jz_and_h(N, J, resolution, compressor):
    Jz = np.arange(-5, 5 + resolution, resolution, dtype='f')
    h = np.arange(-10, 10 + resolution, resolution, dtype='f')

    h, Jz = np.meshgrid(h, Jz)
    gs_energy = np.zeros_like(Jz)
    gs_magnetization = np.zeros_like(gs_energy)

    for i in range(len(Jz)):
        for j in range(len(Jz[i])):
            #pr.enable()
            chain = XXZChain(compressor, h=h[i][j], Jz=Jz[i][j], J=J)

            iterations = int(N / 2)
            chain.expand(iterations)

            gs_energy[i][j], gs = chain.get_super_block_ground_state()
            gs_magnetization[i][j] = chain.get_super_block_magnetization_expectation_value(gs)
            print("Jz={}, h={}, gs energy: {}, gs magnetization: {}".format(Jz[i][j], h[i][j], gs_energy[i][j], gs_magnetization[i][j]))
            #pr.disable()
            #pr.print_stats()

    general_title_info = "J={}, N={}, max_dimension={}, compressor={}".format(J, N, compressor.get_max_dimension(), compressor)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, gs_energy, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('E_gs [a.u.]')
    ax.set_title('Ground state energy as function of J_z and h, {}'.format(general_title_info))
    ax.view_init(30, 45)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, gs_magnetization, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('<M>_gs [a.u.]')
    ax.set_title('Ground state magnetization as function of J_z and h, {}'.format(general_title_info))
    ax.view_init(30, 45)


def main():
    N = 8  # Chain length
    J = 2
    resolution = 0.5
    high_compressor_max_dimension = 10
    low_compressor_max_dimension = 1000

    low_eigen_compressor = EigenCompressor(low_compressor_max_dimension)
    plot_as_function_of_Jz_and_h(N, J, resolution, low_eigen_compressor)

    high_eigen_compressor = EigenCompressor(high_compressor_max_dimension)
    plot_as_function_of_Jz_and_h(N, J, resolution, high_eigen_compressor)

    """
    low_svd_compressor = SVDCompressor(low_compressor_max_dimension)
    plot_as_function_of_Jz_and_h(N, J, resolution, low_svd_compressor)

    high_svd_compressor = SVDCompressor(high_compressor_max_dimension)
    plot_as_function_of_Jz_and_h(N, J, resolution, high_svd_compressor)
    """

    plt.show()


if __name__ == '__main__':
    main()