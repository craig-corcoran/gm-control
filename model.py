import copy
import numpy as np
import itertools
import theano.tensor as T
import theano.sparse
import theano
import scipy.optimize
from scipy.sparse import *

class Model:
    def __init__(self, n_state_nodes, n_action_nodes, n_next_state_nodes, n_reward_nodes = 1):
        # X = <S,A,S',R>
        n = 0
        self.state_nodes      = range(n, n + n_state_nodes)
        n += n_state_nodes
        self.action_nodes     = range(n, n + n_action_nodes)
        n += n_action_nodes
        self.next_state_nodes = range(n, n + n_next_state_nodes)
        n += n_next_state_nodes
        self.reward_nodes     = range(n, n + n_reward_nodes)

        self.edges = []
        self.edges += list(itertools.product(self.state_nodes, self.action_nodes))
        self.edges += list(itertools.product(self.state_nodes, self.next_state_nodes))
        self.edges += list(itertools.product(self.action_nodes, self.next_state_nodes))
        self.edges += list(itertools.product(self.state_nodes, self.reward_nodes))
        self.edges += list(itertools.product(self.action_nodes, self.reward_nodes))
        self.edges = np.array(self.edges)

        self.n_nodes = n_state_nodes + n_action_nodes + n_next_state_nodes + n_reward_nodes
        self.n_edges = len(self.edges)
        self.chi     = [0,1]

        self.n_params = self.n_nodes*len(self.chi) + self.n_edges*len(self.chi)*len(self.chi)

        self.n_nz_phi = self.n_nodes + self.n_edges
        self.sparse_identity = scipy.sparse.coo_matrix( (np.ones(self.n_params), (np.arange(self.n_params), np.arange(self.n_params))),
                                                shape = (self.n_params, self.n_params),
                                                dtype = np.int8)
        self.sparse_identity = scipy.sparse.csr_matrix(self.sparse_identity)
        
        # Theta = <node-wise params, edge-wise params>
        # theta_s,j = |chi| * s + j
        # theta_edge,j,k = |chi|*|V| + (|chi|^2)*edge + |chi|*j + k
        self.theta = np.zeros((self.n_params,1), dtype = np.float)
        #self.theta = np.random.standard_normal((self.n_params, 1))
    
        print 'building index arrays for each node'
        rows = []
        cols = []
        vals = []
        self.num_indxs = np.zeros(self.n_nodes)
        for i in xrange(self.n_nodes):
            new_vals = range(2*i, 2*i+2)
            for indx,e in enumerate(self.edges):
                if i in e:
                    num = 2*self.n_nodes + indx*4
                    new_vals += range(num, num +4) 
            
            rows += [i]*len(new_vals)
            cols += range(len(new_vals))
            vals += new_vals
            self.num_indxs[i] = len(vals) # keep track of number of nz to remove zero end padding

        
        # a silly way to build a matrix whose size you don't know
        mat = scipy.sparse.coo_matrix((vals, (rows, cols)), dtype = np.int32)
        self.I = np.array(mat.todense())
        self.row_range = np.arange(self.I.shape[1])
 
        # define theano stuff
        t_phi = theano.sparse.csr_matrix('phi', dtype='int8')
        t_phi_p = theano.sparse.csr_matrix('phi_p', dtype='int8')
        t_theta = T.dmatrix('theta')

        
        t_T = theano.sparse.csc_matrix('T', dtype='int8')

        t_phi_i = theano.sparse.structured_dot(t_T, t_phi)
        t_phi_i_p = theano.sparse.structured_dot(t_T, t_phi_p)
        t_theta_i = theano.sparse.structured_dot(t_T, t_theta )

        t_a = theano.sparse.structured_dot(t_phi_i.T, t_theta_i)
    
        # var_likelihood_func returns likelihood for a single variable on a single data point
        t_var_likelihood = (t_a - T.log(T.exp(t_a) + T.exp(theano.sparse.structured_dot(t_phi_i_p.T, t_theta_i))))[0,0]
        self.t_var_likelihood_func = theano.function([t_phi, t_phi_p, t_theta, t_T], t_var_likelihood)


        # loop over variables to get the per-data-point loss:
        n_ind = T.iscalar('n_ind')
        n_params = T.iscalar('n_params')
        phi_p_indxs = T.ivector('phi_p_indxs')
        cols = T.ivector('cols') # of tranformation matrix
        rows = T.ivector('rows')
        t_sparse_identity = theano.sparse.csc_matrix('sp_ident', dtype='int8')
        
        # first we need a function to transform the outer loops iterables into 
        # the tensors used by var_likelihood_func.
        c1,_ = theano.scan(fn = self.add_index,
                            outputs_info = T.zeros((n_params,1), dtype='int8'), #theano.sparse.csr_from_dense(
                            sequences = phi_p_indxs,
                            non_sequences = [0, t_sparse_identity, (n_params,1)])
        phi_p = c1[-1]
        
        c2,_ = theano.scan(fn = self.add_index,
                            outputs_info = theano.sparse.csr_from_dense(T.zeros((n_ind, n_params), dtype='int8')),
                            sequences = [rows[:n_ind], cols[:n_ind]],
                            non_sequences = [t_sparse_identity, (n_ind, n_params)])
        trans_mat = c2[-1]
        
        outer_var_likelihood = self.t_var_likelihood_func(t_phi, phi_p, t_theta, trans_mat)
        self.outer_var_likelihood_func = theano.function([n_ind, phi_p_indxs, cols, rows, n_params, t_phi, t_theta, zero], outer_var_likelihood)



        # now the data loop:
        t_num_indxs = T.ivector('num_indexes')
        t_I = T.imatrix('I')
        t_PHI_indxs_p = T.imatrix('PHI_indxs_p')
        t_var_losses,_ = theano.scan(fn = self.out_var_likelihood_func,
                                outputs_info=None,
                                sequences=[t_num_indxs, t_PHI_indxs_p, t_I],
                                non_sequences=[rows, n_params, t_phi, t_theta, zero])

        data_loss = t_var_losses.sum()
        self.data_loss_func = theano.function([t_num_indxs, t_PHI_indxs_p, t_I,
                                    rows, n_params, t_phi, t_theta], data_loss)

        data_loss_grad = T.grad(data_loss, [t_theta])
        self.data_loss_grad_func = theano.function([t_num_indxs, t_PHI_indxs_p, t_I,
                                    rows, n_params, t_phi, t_theta], data_loss_grad)

    def add_index(self,vec,i,j, identity, size):
        n_cols = size[1]
        vec[i,0:n_cols] = vec[i,0:n_cols] + identity[j,0:n_cols]
        return vec

    def get_scan_vars(self, n_ind, inds, phi_p_indxs, phi, theta):

        inds = inds[:n_ind] 

        phi_p = scipy.sparse.coo_matrix(( np.ones(self.n_nz_phi),
                                        ([0]*self.n_nz_phi, phi_p_indxs) ),
                                        shape=(self.n_params, 1),
                                        dtype='int8')
        phi_p = scipy.sparse.csr_matrix(phi_p)

        t = scipy.sparse.coo_matrix(( np.ones(n_ind), 
                            (range(n_ind), inds) ), 
                            shape = (n_ind, self.n_params),
                            dtype='int8')
        t = scipy.sparse.csc_matrix(t)

        return self.t_var_likelihood_func(phi, phi_p, theta, t)


    def get_phi_indexes(self, d):
        index_list = np.zeros(self.n_nodes + self.n_edges, dtype = np.int32)
            
        # 2 * indx + val
        index_list[:self.n_nodes] = d + np.array(range(0,self.n_nodes*2,2), dtype = np.int32)
        
        index_list[self.n_nodes:] = np.array(range(0,self.n_edges*4,4), dtype = np.int32) + \
                    np.sum((2,1) * d[self.edges], axis = 1)

        return index_list
        
                
    def get_phi(self, d):

        index_list = self.get_phi_indexes(d)
        phi = coo_matrix(([1]*len(index_list), ([0] * len(index_list), index_list)), 
                                        shape = (1,self.n_params), dtype = np.int8)
        phi = csc_matrix(phi)
        return phi

    def get_PHI_indxs(self, d):
        ''' returns a n_nodes by n_nz_row array PHI_indxs where each row 
        represents the nonzero indices in phi(d_prime_i) for each variable i'''
         
        PHI_indxs = np.zeros(self.n_nodes, self.n_nz_phi, dtype=np.int32)
        
        PHI_indxs[0,:] = self.get_phi_indexes(d)
        d_prime = copy.deepcopy(d)            
        for i in xrange(self.n_nodes):
            d_prime[i] = 1 if d[i] == 0 else 0
            PHI_indxs[i,:] = self.get_phi_indexes(d_prime)
            d_prime[i] = d[i]
            
        return PHI_indxs
            
    



    def pseudo_likelihood_loss(self, theta, data):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes - returns negative sum of log psl over the 
        data'''

        if theta.ndim == 1:
            theta = theta[:,None]

        l = 0.
        for d in data:
    
            PHI_indxs = self.get_PHI_indxs(d)
            n_nz = phi_p_indxs.shape[1]
            phi = scipy.sparse.coo_matrix(( np.ones(n_nz),
                                        ( phi_p_indxs[0,:]), [0]*len(n_nz) ),
                                        shape=(self.n_params, 1),
                                        dtype='int8')
            phi = scipy.sparse.csr_matrix(phi)
            PHI_indxs_p = PHI_indxs[1:,:]

            l += self.data_loss_func(self.num_indxs, PHI_indxs_p, self.I, 
                                    self.row_range, phi, theta)
    
        return (-1. * l) / len(data)

    def pseudo_likelihood_grad(self, theta, data):
        
        if theta.ndim == 1:
            theta = theta[:,None]

        g = np.zeros_like(theta)
        for d in data:

            PHI_indxs = self.get_PHI_indxs(d)
            n_nz = phi_p_indxs.shape[1]
            phi = scipy.sparse.coo_matrix(( np.ones(n_nz),
                                        ( phi_p_indxs[0,:]), [0]*len(n_nz) ),
                                        shape=(self.n_params, 1),
                                        dtype='int8')
            phi = scipy.sparse.csr_matrix(phi)
            PHI_indxs_p = PHI_indxs[1:,:]

            g += self.data_loss_grad_func(self.num_indxs, PHI_indxs_p, self.I, 
                                    self.row_range, phi, theta)[0]
 
        return (-1. * g.flatten()) / len(data)

    def get_data(self, mdp, n_samples, n_test_samples, state_rep = 'factored'):
    
        all_data = mdp.sample_grid_world(n_samples + n_test_samples, state_rep )
        np.random.shuffle(all_data)

        data = all_data[:n_samples]
        test_data = all_data[n_samples:]

        return data, test_data
                



def train_model_cg(minibatch = 10, n_samples = 30, n_test_samples = 10, cg_max_iter = 3):
    import grid_world
    
    #m = Model(81, 2, 81)
    m = Model(18, 2, 18)

    print 'Generating Samples Trajectory from Gridworld...'
    mdp = grid_world.MDP()
    data, test_data = m.get_data(mdp, n_samples, n_test_samples,  
                                        state_rep = 'factored')
    n_iters = n_samples / minibatch

    print 'initial loss: ', m.pseudo_likelihood_loss(m.theta, test_data)

    for i in xrange(n_iters):

        print 'iter: ', i+1, ' of ', n_iters

        mb = data[i*minibatch: (i+1)*minibatch]

        n_theta, val, fc, gc, w = scipy.optimize.fmin_cg(
                                m.pseudo_likelihood_loss,
                                m.theta,
                                fprime = m.pseudo_likelihood_grad, 
                                args = (mb,), 
                                full_output = True,
                                epsilon=1.e-12,
                                gtol = 1e-180,

                                maxiter = cg_max_iter)

        print 'theta: ', n_theta

        print 'theta sum: ', np.sum(n_theta)
        
        delta = np.linalg.norm(n_theta - m.theta)
        
        print 'function calls', fc
        print 'gradient calls', gc 
        print 'delta theta: ', delta 

        m.theta = n_theta
        print 'current training min: ', val
        print 'new test loss: ', m.pseudo_likelihood_loss(m.theta, test_data)

        if delta < 1e-20:
            break

    return m


def train_model_sgd( minibatch = 250, n_samples = 2500, n_test_samples =500, alpha = 1e-1):
    
    import grid_world
    
    #m = Model(81, 2, 81)
    m = Model(18, 2, 18)

    print 'Generating Samples Trajectory from Gridworld...'
    mdp = grid_world.MDP()
    data, test_data = m.get_data(mdp, n_samples, n_test_samples,  
                                        state_rep = 'factored')
    n_iters = n_samples / minibatch

    print 'initial loss: ', m.pseudo_likelihood_loss(m.theta, test_data)

    for i in xrange(n_iters):
        
        alpha *= 0.9
        
        mb = data[i*minibatch: (i+1)*minibatch]

        m.theta -= alpha * m.pseudo_likelihood_grad(m.theta, mb)[:,None]
        
        print 'new loss: ', m.pseudo_likelihood_loss(m.theta, test_data)
    
    return m


if __name__ == "__main__":
    train_model_cg()
