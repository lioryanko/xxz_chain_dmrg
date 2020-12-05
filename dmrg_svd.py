import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib
import cProfile

pr = cProfile.Profile()

sz = np.array([[1, 0], [0, -1]])
sp = np.array([[0, 1], [0, 0]])
sm = sp.T


class XXZChain(object):
    def __init__(self, max_dimension, h, Jz, J):
        self.max_dimension = max_dimension
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

    def get_compression_matrices(self):
        _, gs = self.get_super_block_ground_state()
        rho = np.einsum('i,j->ij', gs, np.conjugate(gs))
        rho_reduced_dimension = len(self.block_H) * len(sz)
        rho_reduced = rho.reshape(rho_reduced_dimension, rho_reduced_dimension, rho_reduced_dimension,
                                  rho_reduced_dimension).trace(axis1=2, axis2=3)
        U, s, V = np.linalg.svd(rho_reduced)
        d = min(self.max_dimension, len(s))
        U_c = U[:d]
        V_c = V[:, :d]
        return U_c, V_c

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

            U, V = self.get_compression_matrices()

            # Now the block has another site, and the operators operating on it are (possibly) compressed.
            self.block_H = U @ block_H @ V
            self.block_M = U @ block_M @ V
            self.last_site_sz = U @ next_site_sz @ V
            self.last_site_sp = U @ next_site_sp @ V
            self.last_site_sm = U @ next_site_sm @ V


def plot_as_function_of_Jz_and_h(N, J, resolution):
    Jz = np.arange(-5, 5 + resolution, resolution, dtype='f')
    h = np.arange(-10, 10 + resolution, resolution, dtype='f')

    h, Jz = np.meshgrid(h, Jz)
    exact_gs_energy = np.zeros_like(Jz)
    approximated_gs_energy = np.zeros_like(exact_gs_energy)
    exact_gs_magnetization = np.zeros_like(exact_gs_energy)
    approximated_gs_magnetization = np.zeros_like(exact_gs_energy)

    for i in range(len(Jz)):
        for j in range(len(Jz[i])):
            #pr.enable()
            exact_chain = XXZChain(max_dimension=1000, h=h[i][j], Jz=Jz[i][j], J=J)
            approximated_chain = XXZChain(max_dimension=10, h=h[i][j], Jz=Jz[i][j], J=J)

            iterations = int(N / 2)
            exact_chain.expand(iterations)
            approximated_chain.expand(iterations)

            exact_gs_energy[i][j], exact_gs = exact_chain.get_super_block_ground_state()
            approximated_gs_energy[i][j], approximated_gs = approximated_chain.get_super_block_ground_state()
            exact_gs_magnetization[i][j] = exact_chain.get_super_block_magnetization_expectation_value(exact_gs)
            approximated_gs_magnetization[i][j] = approximated_chain.get_super_block_magnetization_expectation_value(approximated_gs)
            print("Jz={}, h={}, exact gs energy: {}, approximated gs energy: {}, exact gs magnetization: {}, approximated gs magnetization: {}".format(Jz[i][j], h[i][j], exact_gs_energy[i][j], approximated_gs_energy[i][j], exact_gs_magnetization[i][j], approximated_gs_magnetization[i][j]))
            #pr.disable()
            #pr.print_stats()



    general_title_info = "J={}, N={}".format(J, N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, exact_gs_energy, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('E_gs [a.u.]')
    ax.set_title('Exact ground state energy as function of J_z and h, {}'.format(general_title_info))
    ax.view_init(30, 45)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, approximated_gs_energy, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('E_gs [a.u.]')
    ax.set_title('Approximated ground state energy as function of J_z and h, {}'.format(general_title_info))
    ax.view_init(30, 45)

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, exact_gs_magnetization, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('<M>_gs [a.u.]')
    ax.set_title('Exact ground state magnetization as function of J_z and h, {}'.format(general_title_info))
    ax.view_init(30, 45)

    fig4 = plt.figure()
    ax = fig4.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, approximated_gs_magnetization, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('<M>_gs [a.u.]')
    ax.set_title('Approximated ground state magnetization as function of J_z and h, {}'.format(general_title_info))
    ax.view_init(30, 45)

    plt.show()


def main():
    N = 8  # Chain length
    J = 2
    resolution = 0.5

    plot_as_function_of_Jz_and_h(N, J, resolution)


if __name__ == '__main__':
    main()