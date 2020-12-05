import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sp = np.array([[0, 1], [0, 0]])
sz = np.array([[1, 0], [0, -1]])

sz_sz = np.kron(sz, sz)
sp_sp = np.kron(sp, sp)
sm_sm = np.kron(sp.T, sp.T)


def left_block_operator_to_right_block(operator):
    U = np.flip(np.eye(len(operator)), 0)
    return U @ operator @ U


class XXZChain(object):
    def __init__(self, max_dimension, h, Jz, J):
        self.max_dimension = max_dimension
        self.h = h
        self.Jz = Jz
        self.J = J
        self.left_block_H = - h * sz
        self.last_site_sz = sz
        self.last_site_sp = sp
        self.last_site_sm = sp.T
        self.left_block_M = sz

    def build_super_block_hamiltonian(self):
        block_dimension = len(self.left_block_H)
        d = len(sz)

        # Add the single block terms.
        right_block_H = left_block_operator_to_right_block(self.left_block_H)
        super_block_H = np.kron(self.left_block_H, np.eye(d**2 * block_dimension))
        super_block_H += np.kron(np.eye(block_dimension * d**2), right_block_H)

        # Add the single site terms.
        super_block_H += np.kron(np.kron(np.eye(block_dimension), -self.h * sz), np.eye(d * block_dimension))
        super_block_H += np.kron(np.kron(np.eye(block_dimension * d), -self.h * sz), np.eye(block_dimension))

        # Add the site-site interaction terms.
        site_site_interaction = self.J / 2 * (sp_sp + sm_sm) + self.Jz * sz_sz
        super_block_H += np.kron(np.kron(np.eye(block_dimension), site_site_interaction), np.eye(block_dimension))

        # Add the block-site interaction terms.
        left_block_edge_interaction = -self.Jz * np.kron(self.last_site_sz, sz)
        left_block_edge_interaction += -self.J / 2 * np.kron(self.last_site_sp, sp)
        left_block_edge_interaction += -self.J / 2 * np.kron(self.last_site_sm, sp.T)

        U = np.flip(np.eye(block_dimension * d), 0)
        right_block_edge_interaction = U @ left_block_edge_interaction @ U

        super_block_H += np.kron(left_block_edge_interaction, np.eye(len(left_block_edge_interaction)))
        super_block_H += np.kron(np.eye(len(right_block_edge_interaction)), right_block_edge_interaction)

        super_block_H += np.kron(left_block_edge_interaction, np.eye(d * block_dimension))

        return super_block_H

    def get_super_block_ground_state(self):
        super_block_H = self.build_super_block_hamiltonian()
        eigvals, eigvects = np.linalg.eigh(super_block_H)
        return eigvals[0], eigvects[:, 0]

    def get_super_block_ground_state_magnetization(self):
        block_dimension = len(self.left_block_H)
        d = len(sz)

        right_block_M = left_block_operator_to_right_block(self.left_block_M)
        super_block_M = np.kron(self.left_block_M, np.eye(d ** 2 * block_dimension))
        super_block_M += np.kron(np.eye(block_dimension * d ** 2), right_block_M)
        super_block_M += np.kron(np.kron(np.eye(block_dimension), sz), np.eye(d * block_dimension))
        super_block_M += np.kron(np.kron(np.eye(block_dimension * d), sz), np.eye(block_dimension))

        _, gs = self.get_super_block_ground_state()
        return (gs.T @ super_block_M @ gs) / (gs.T @ gs)

    def get_compression_matrices(self):
        _, ground_state = self.get_super_block_ground_state()
        rho = np.einsum('i,j->ij', ground_state, np.conjugate(ground_state))
        rho_reduced_dimension = len(self.left_block_H) * len(sz)
        rho_reduced = rho.reshape(rho_reduced_dimension, rho_reduced_dimension, rho_reduced_dimension,
                                  rho_reduced_dimension).trace(axis1=2, axis2=3)
        U, s, V = np.linalg.svd(rho_reduced)
        d = min(self.max_dimension, len(s))
        U_c = U[:d]
        V_c = V[:,:d]
        return U_c, V_c

    def expand(self, iterations):
        for i in range(iterations):
            block_dimension = len(self.left_block_H)
            next_site_sz = np.kron(np.eye(block_dimension), sz)
            next_site_sp = np.kron(np.eye(block_dimension), sp)
            next_site_sm = np.kron(np.eye(block_dimension), sp.T)

            block_site_interaction = self.Jz * np.kron(self.last_site_sz, sz)
            block_site_interaction += self.J / 2 * (np.kron(self.last_site_sp, sp) + np.kron(self.last_site_sm, sp.T))

            U, V = self.get_compression_matrices()

            self.left_block_H = np.kron(self.left_block_H, np.eye(len(sz)))
            self.left_block_H += -self.h * next_site_sz + block_site_interaction
            self.left_block_M = np.kron(self.left_block_M, np.eye(len(sz)))
            self.left_block_M += next_site_sz

            # Now the block has another site, and the operators operating on it are (possibly) compressed.
            self.left_block_H = U @ self.left_block_H @ V
            self.left_block_M = U @ self.left_block_M @ V
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
            exact_chain = XXZChain(max_dimension=1000, h=h[i][j], Jz=Jz[i][j], J=J)
            approximated_chain = XXZChain(max_dimension=10, h=h[i][j], Jz=Jz[i][j], J=J)

            iterations = int(N / 2)
            exact_chain.expand(iterations)
            approximated_chain.expand(iterations)

            exact_gs_energy[i][j], _ = exact_chain.get_super_block_ground_state()
            approximated_gs_energy[i][j], _ = approximated_chain.get_super_block_ground_state()
            exact_gs_magnetization[i][j] = exact_chain.get_super_block_ground_state_magnetization()
            approximated_gs_magnetization[i][j] = approximated_chain.get_super_block_ground_state_magnetization()
            print("Jz={}, h={}, exact gs energy: {}, approximated gs energy: {}, exact gs magnetization: {}, approximated gs magnetization: {}".format(Jz[i][j], h[i][j], exact_gs_energy[i][j], approximated_gs_energy[i][j], exact_gs_magnetization[i][j], approximated_gs_magnetization[i][j]))


    general_title_info = "J={}, N={}, even_site_perturbation={}".format(J, N, even_site_perturbation)

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