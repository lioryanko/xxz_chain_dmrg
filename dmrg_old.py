import numpy as np
import matplotlib.pyplot as plt
import matplotlib


sp = np.array([[0, 1], [0, 0]])
sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])


def single_to_mutli_particle_operator(i, N, operator):
    """
    Takes a single particle operator, that is an operator in a Hilbert space of a single particle,
    and transforms it to an operator on the i-th (count starts from 0) particle in an N particle Hilbert space
    """
    I2 = np.eye(2)

    result = 1
    for j in range(N):
        if j==i:
            left_factor = operator
        else:
            left_factor = I2
        result = np.kron(left_factor, result)

    return result

def single_to_mutli_particle_operator_list(N, operator):
    """
    Creates a list of the results of single_to_mutli_particle_operator from i=0 to i=N-1
    """
    operator_i = []
    for i in range(N):
        operator_i.append(single_to_mutli_particle_operator(i, N, operator))

    return operator_i


def build_heisenberg_hamiltonian(N, J, Jz, h, dh_i):
    """
    Builds a matrix representation of an Heisenberg hamiltonian for a one dimensional spin lattice
    """
    sp_i = single_to_mutli_particle_operator_list(N, sp)
    sm_i = [sp.T for sp in sp_i]
    sz_i = single_to_mutli_particle_operator_list(N, sz)

    sp_sm_term = 0
    sz_sz_term = 0

    for i in range(N-1):
        sp_sm_term += np.matmul(sp_i[i], sm_i[i+1])
        sz_sz_term += np.matmul(sz_i[i], sz_i[i+1])

    sp_sm_term = float(J)/2*sp_sm_term
    sm_sp_term = sp_sm_term.T
    sz_sz_term = Jz*sz_sz_term
    h_i = np.ones(N) * h + dh_i
    sz_term = sum([h_i[i] * sz_i[i] for i in range(len(sz_i))])

    return (sp_sm_term + sm_sp_term) + sz_sz_term - sz_term


def plot_as_function_of_Jz_and_h(N, J, left_site_perturbation, resolution):
    sz_average = sum(single_to_mutli_particle_operator_list(N, sz))/N
    sz_i = single_to_mutli_particle_operator_list(N, sz)
    sx_i = single_to_mutli_particle_operator_list(N, sx)

    sz_sz_average = 0
    sx_sx_average = 0
    for i in range(N - 1):
        sz_sz_average += np.matmul(sz_i[i], sz_i[i + 1]) * 1.0 / (N - 1)
        sx_sx_average += np.matmul(sx_i[i], sx_i[i + 1]) * 1.0 / (N - 1)

    # Use a small per-site perturbation in order to break symmetry and get a pure orientation ground state
    dh_i = np.zeros(N, dtype='f')
    dh_i[0] = left_site_perturbation

    Jz = np.arange(-5, 5 + resolution, resolution, dtype='f')
    h = np.arange(-10, 10 + resolution, resolution, dtype='f')

    h, Jz = np.meshgrid(h, Jz)
    gs_energy = np.zeros_like(Jz)
    gs_sz_average = np.zeros_like(Jz)
    gs_sz_sz_average = np.zeros_like(Jz)
    gs_sx_sx_average = np.zeros_like(Jz)
    ultra_gs_energy = np.zeros_like(gs_energy)
    ultra_ultra_gs_energy = np.zeros_like(gs_energy)

    for i in range(len(Jz)):
        for j in range(len(Jz[i])):
            H = build_heisenberg_hamiltonian(N, J, Jz[i][j], h[i][j], dh_i)
            # print(H)
            ev, es = np.linalg.eig(H)
            gs_energy[i][j] = min(ev)
            sorted_ev = list(ev)
            sorted_ev.sort()
            ultra_gs_energy[i][j] = sorted_ev[1]
            ultra_ultra_gs_energy[i][j] = sorted_ev[2]
            # print(min(ev))
            # print(list(ev).index(min(ev)))
            # print(es[:, list(ev).index(min(ev))])
            sorted_ev = list(ev)
            sorted_ev.sort()
            gs = es[:, list(ev).index(sorted_ev[0])]
            gs_sz_average[i][j] = np.matmul(gs, np.matmul(sz_average, gs))
            gs_sz_sz_average[i][j] = np.matmul(gs, np.matmul(sz_sz_average, gs))
            gs_sx_sx_average[i][j] = np.matmul(gs, np.matmul(sx_sx_average, gs))
            print('parameters: J={}, J_z={}, h={}'.format(J, Jz[i][j], h[i][j]))
            sorted_gs = list(gs)
            sorted_gs.sort()

            #print('gs energy: {}'.format(sorted_ev[0]))
            print('first max gs component weight: {}'.format(sorted_gs[-1]))
            print('first max gs component: {}'.format(format(list(gs).index(sorted_gs[-1]), "#010b")))
            """
            print('second max gs component weight: {}'.format(sorted_gs[-2]))
            print('second max gs component: {}'.format(format(list(gs).index(sorted_gs[-2]), "#010b")))
            """

            ultra_gs = es[:, list(ev).index(sorted_ev[1])]
            sorted_ultra_gs = list(ultra_gs)
            sorted_ultra_gs.sort()
            """
            print('ultra_gs energy: {}'.format(sorted_ev[1]))
            print('first max ultra_gs component weight: {}'.format(sorted_ultra_gs[-1]))
            print('first max ultra_gs component: {}'.format(
                format(list(ultra_gs).index(sorted_ultra_gs[-1]), "#010b")))
            print('second max ultra_gs component weight: {}'.format(sorted_ultra_gs[-2]))
            print('second max ultra_gs component: {}'.format(
                format(list(ultra_gs).index(sorted_ultra_gs[-2]), "#010b")))
            """

        # print(ev)
        # print(es)
        # print(min(ev))
        # print(list(ev).index(min(ev)))
        # print(es[:, list(ev).index(min(ev))])
        gs = es[:, list(ev).index(min(ev))]
        # print(gs.dot(gs))
        # print(max(list(gs)))
        # print(format(list(gs).index(max(list(gs))), "#010b"))
        sorted_gs = list(gs)
        sorted_gs.sort()
        # print(max(sorted_gs[:-1]))
        # print(format(list(gs).index(max(sorted_gs[:-1])), "#010b"))
        # print(max(sorted_gs[:-2]))
        # print(format(list(gs).index(max(sorted_gs[:-2])), "#010b"))
        # print(max(sorted_gs[:-3]))
        # print(format(list(gs).index(max(sorted_gs[:-3])), "#010b"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, gs_energy, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('E_gs [a.u.]')
    ax.set_title('Ground state energy as function of J_z and h, J={}, left_site_perturbation={}'.format(J, left_site_perturbation))
    ax.view_init(30, 45)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, ultra_gs_energy-gs_energy, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('E_gs [a.u.]')
    ax.set_title('Ultra ground state ground state energy gap as function of J_z and h, J={}, left_site_perturbation={}'.format(J, left_site_perturbation))
    ax.view_init(30, 45)

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, ultra_ultra_gs_energy - ultra_gs_energy, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('E_gs [a.u.]')
    ax.set_title('3rd lowest eigenstate 2nd lowest eigenstate energy gap as function of J_z and h, J={}, left_site_perturbation={}'.format(J, left_site_perturbation))
    ax.view_init(30, 45)

    fig4 = plt.figure()
    ax = fig4.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, gs_sz_average, cmap=matplotlib.cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('<sz> [a.u.]')
    ax.set_title('Ground state average site sz expectation value as function of J_z and h, J={}, left_site_perturbation={}'.format(J, left_site_perturbation))
    ax.view_init(30, 45)

    fig5 = plt.figure()
    ax = fig5.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, gs_sz_sz_average, cmap=matplotlib.cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('<sz_sz> [a.u.]')
    ax.set_title('Ground state average sz_i*sz_(i+1) expectation value as function of J_z and h, J={}, left_site_perturbation={}'.format(J, left_site_perturbation))
    ax.view_init(30, 45)

    fig6 = plt.figure()
    ax = fig6.add_subplot(111, projection='3d')
    ax.plot_surface(Jz, h, gs_sx_sx_average, cmap=matplotlib.cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_xlabel('J_z [a.u.]')
    ax.set_ylabel('h [a.u.]')
    ax.set_zlabel('<sx_sx> [a.u.]')
    ax.set_title('Ground state average sx_i*sx_(i+1) expectation value as function of J_z and h, J={}, left_site_perturbation={}'.format(J, left_site_perturbation))
    ax.view_init(30, 45)

    plt.show()


class Operator(object):
    def __init__(self, single_site_matrix_representation):
        self.matrix_representation = single_site_matrix_representation

    def get_matrix_representation(self):
        return self.matrix_representation

    def set_matrix_representation(self, new_matrix_representation):
        self.matrix_representation = new_matrix_representation


class Chain(object):
    d = 2 # The spin multiplicity of each site.

    def __init__(self, J, Jz, h):

        self.J = float(J)
        self.Jz = float(Jz)
        self.h = float(h)

        # The matrix of the projection of the current (perhaps reduced) block basis |a_l> on the basis |a_l-1, s_l>,
        # where |s_l> is a basis of the state space of the newest site added to the block. Initialized as None because
        # there is no "old block basis" |a_l-1> yet (because the block size l is 1).
        self.basis_projection_matrix = None

        # The Hamiltonian terms exclusive for the block. That is, the super-block Hamiltonian has interactions term
        # which are not corresponding to a single block, but rather to a block-site or site-site interaction, that are
        # not accounted for by this Hamiltonian. Initialized as 0 because there are no terms in the Hamiltonian yet.
        self.H = np.zeros(shape=(2,2))

        # The block operators which act on the edge site of the block (the last site INSIDE the block). Initialized as
        # single site operators because initially the block is a single site.
        self.sz_edge = np.array([[1, 0], [0, -1]])
        self.sp_edge = np.array([[0, 1], [0, 0]])


    def initialize_operator(self, site_operator):
        """
        Takes a site operator and produces a block operator.
        """

        # The dimension of the block state is the number of columns in the basis projection matrix
        block_state_dimension = len(self.basis_projection_matrix[0])

        # The dimension of the old block state is the number of rows (old block+site basis before decimation) divided by
        # the number of site states.
        old_block_state_dimension = len(self.basis_projection_matrix) / self.d

        # Any operator operating on the block state can be represented by a square matrix with the fitting size.
        new_operator = np.zeros((block_state_dimension, block_state_dimension))

        for alo in range(block_state_dimension):
            for alot in range(block_state_dimension):
                for al in range(old_block_state_dimension):
                    for slo in range(self.d):
                        sum_term = 0
                        for slot in range(self.d):
                            sum_term += site_operator[slo][slot] * self.basis_projection_matrix[slot * self.d + al][alot]
                        new_operator[alo][alot] += self.basis_projection_matrix[slo * self.d + al][
                                        alo].conjugate() * sum_term

        return new_operator


    def initialize_product_operator(self, block_operator, site_operator):
        """
        Takes a block operator and a site operator (operating on a site to be added to the block), and outputs an operator
        which is the product ("the" product because the operators commute) of both operators and operates on the new block.
        """

        # The dimension of the block state is the number of columns in the basis projection matrix
        block_state_dimension = len(self.basis_projection_matrix[0])

        # The dimension of the old block state is the number of rows (old block+site basis before decimation) divided by
        # the number of site states.
        old_block_state_dimension = len(self.basis_projection_matrix) / self.d

        # Any operator operating on the block state can be represented by a square matrix with the fitting size.
        new_operator = np.zeros((block_state_dimension, block_state_dimension))

        for alo in range(block_state_dimension):
            for alot in range(block_state_dimension):
                for al in range(old_block_state_dimension):
                    for alt in range(old_block_state_dimension):
                        outer_sum_term = 0
                        for slo in range(self.d):
                            inner_sum_term = 0
                            for slot in range(self.d):
                                inner_sum_term += site_operator[slo][slot] * \
                                                  self.basis_projection_matrix[slot * self.d + alt][alot]
                            outer_sum_term += self.basis_projection_matrix[slo * self.d + al][
                                                  alo].conjugate() * inner_sum_term
                        new_operator[alo][alot] += outer_sum_term * block_operator[al][alot]

        return new_operator

    def update_operator(self, operator):
        """
        Updates an old block operator to the new block basis.
        """

        # The dimension of the block state is the number of columns in the basis projection matrix
        block_state_dimension = len(self.basis_projection_matrix[0])

        # The dimension of the old block state is the number of rows (old block+site basis before decimation) divided by
        # the number of site states.
        old_block_state_dimension = len(self.basis_projection_matrix) / self.d

        # Any operator operating on the block state can be represented by a square matrix with the fitting size.
        new_operator = np.zeros((block_state_dimension, block_state_dimension))

        # 'a' denotes basis state of the block, 's' (for "sigma") denotes a basis state a site, 'l' denotes that the
        # state dimension is l, 'lo' (for "l plus one") denotes that the state dimension is l+1, 't' denotes tag. So,
        # alt for example corresponds to |a_l'>
        for alo in range(block_state_dimension):
            for alot in range(block_state_dimension):
                for slo in range(self.d):
                    for al in range(old_block_state_dimension):
                        sum_term = 0
                        for alt in range(len(self.basis_projection_matrix)):
                            sum_term += operator[al][alt] * self.basis_projection_matrix[slo * self.d + alt][alot]
                        new_operator[alo][alot] += self.basis_projection_matrix[slo * self.d + al][alo].conjugate() * sum_term

        return new_operator


    def build_super_block_H(self):
        H_AssB = np.zeros((len(self.H) * self.d) ** 2)
        for al in range(len(self.H)):
                for alt in range(len(self.H)):
                    for slo in range(self.d)


    def expand(self):
        super_block_H = self.build_super_block_H()



        self.H = self.update_operator(self.H)

        # Add the new terms to H.
        sp_sp = self.initialize_product_operator(self.sp_edge, sp)
        sm_sm = sp_sp.getH()
        sz_sz = self.initialize_product_operator(self.sz_edge, sz)
        sz_lo = self.initialize_operator(self, sz)
        sp_lo = self.initialize_operator(self, sp)

        self.H += self.J/2 * (sp_sp + sm_sm) + self.Jz * sz_sz - self.h * sz_lo

        # Now the new site has become the edge site of the block.
        self.sz_edge = sz_lo
        self.sp_edge = sp_lo




def expand_chain(A, operators_to_initiate, operators_to_update, operators_to_multiply):
    """
    A is the basis of the current left block of the chain (Z_2 symmetry is assumed). operators_to_initiate is a list of
    of operators on the added site to transform into block operators. operators_to_update is a list of the previously
    initiated operators that require updating to the new block basis. operators_to_multiply is a list of pairs operators,
    such that the first is an operator updated to the former block basis and the second is an operation on the new site
    added to the block, and the result is a block operator that is the multiplication of both.
    """
    pass

def bootstrap_chain()

def dmrg_chain_ground_state_calculation(N, D):
    A = np.zeros(2, dtype=float)

    #for i in range(N/2):
    #    A = expand_chain(A, operators)


def main():
    plot_as_function_of_Jz_and_h(10, 1, 0.1, 0.25)


if __name__ == '__main__':
    main()