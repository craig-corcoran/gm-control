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
        self.n_state_nodes = n_state_nodes
        self.n_action_nodes = n_action_nodes

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
        
        # Theta = <node-wise params, edge-wise params>
        # theta_s,j = |chi| * s + j
        # theta_edge,j,k = |chi|*|V| + (|chi|^2)*edge + |chi|*j + k
        self.theta = np.zeros((self.n_params,1), dtype = np.float)
    
        print 'building transformation matrices for each node'
        self.T = [None]*self.n_nodes
        for i in xrange(self.n_nodes):
            cols = range(2*i, 2*i+2)
            for indx,e in enumerate(self.edges):
                if i in e:
                    num = 2*self.n_nodes + indx*4
                    cols = cols + range(num, num +4) 

            mat = scipy.sparse.coo_matrix(([1]*len(cols), (range(len(cols)),cols)), shape = (len(cols), self.n_params), dtype = np.int8)
            self.T[i] = scipy.sparse.csr_matrix(mat)
        
        # define theano stuff
        t_phi = theano.sparse.csr_matrix('phi', dtype='int8')
        t_phi_p = theano.sparse.csr_matrix('phi_p', dtype='int8')

        t_theta = T.dmatrix('theta')
        t_T = theano.sparse.csc_matrix('T', dtype='int8')

        t_theta_i = theano.sparse.structured_dot(t_T, t_theta)
        t_phi_i = theano.sparse.structured_dot(t_T, t_phi)
        t_phi_i_p = theano.sparse.structured_dot(t_T, t_phi_p)
        t_a = theano.sparse.structured_dot(t_phi_i.T, t_theta_i)

        t_var_likelihood = (t_a - T.log(T.exp(t_a) + T.exp(theano.sparse.structured_dot(t_phi_i_p.T, t_theta_i))))[0,0]
        self.t_var_likelihood_func = theano.function([t_phi, t_phi_p, t_theta, t_T], t_var_likelihood)

        t_var_likelihood_grad = T.grad(t_var_likelihood, [t_theta])
        self.t_gradient_func = theano.function([t_phi, t_phi_p, t_theta, t_T], t_var_likelihood_grad)

        t_sparsity = abs(t_theta)
        self.t_regularization_func = theano.function([t_theta], t_sparsity)


    def get_phi_indexes(self, d):
        index_list = np.zeros(self.n_nodes + self.n_edges, dtype = np.int32)
            
        # 2 * indx + val
        index_list[:self.n_nodes] = d + np.array(range(0,self.n_nodes*2,2), dtype = np.int32)
        
        index_list[self.n_nodes:] = np.array(range(0,self.n_edges*4,4), dtype = np.int32) + \
                    np.sum((2,1) * d[self.edges], axis = 1)

        return index_list
        
                
    def get_phi(self, d):

        index_list = self.get_phi_indexes(d)
        phi = coo_matrix(([1]*len(index_list), (index_list, [0] * len(index_list))), shape = (self.n_params, 1), dtype = np.int8)
        phi = csc_matrix(phi)
        return phi

    def get_PHI(self, d):
        
        n_nz_col = self.n_nodes + self.n_edges    # number of nonzeros per column in PHI
        n_nz = n_nz_col * self.n_nodes # total number of nonzeros in PHI

        rows = np.zeros( n_nz, dtype = np.int32)
        cols = np.zeros( n_nz, dtype = np.int32)
        vals = np.ones( n_nz, dtype = np.int8)


        rows[:n_nz_col] = self.get_phi_indexes(d)
        cols[:n_nz_col] = 0
        d_prime = copy.deepcopy(d)            
        for i in xrange(self.n_nodes):
            d_prime[i] = 1 if d[i] == 0 else 0
            rows[i*n_nz_col:(i+1)*n_nz_col] = self.get_phi_indexes(d_prime)
            cols[i*n_nz_col:(i+1)*n_nz_col] = i+1
            d_prime[i] = d[i]
            
        Phi = scipy.sparse.coo_matrix((vals,(rows,cols)), shape=(self.n_params,self.n_nodes+1))
        return scipy.sparse.csc_matrix(Phi)
            

    def pseudo_likelihood_loss(self, theta, data):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes - returns negative sum of log psl over the 
        data'''

        if theta.ndim == 1:
            theta = theta[:,None]

        l = 0.
        for d in data:

            d_prime = copy.deepcopy(d)
            phi = self.get_phi(d)
            for i in xrange(self.n_nodes):
                d_prime[i] = 1 if d[i] == 0 else 0
                phi_p = self.get_phi(d_prime)
                t = self.T[i]
                d_prime[i] = d[i]

                l += self.t_var_likelihood_func(phi, phi_p, theta, t)
            
        
        return (-1. * l) / len(data)

    def pseudo_likelihood_grad(self, theta, data):
        
        if theta.ndim == 1:
            theta = theta[:,None]
        
        g = np.zeros_like(theta)
        for d in data:
            d_prime = copy.deepcopy(d)
            phi = self.get_phi(d)
            for i in xrange(self.n_nodes):
                d_prime[i] = 1 if d[i] == 0 else 0
                phi_p = self.get_phi(d_prime)
                t = self.T[i]
                d_prime[i] = d[i]
                g += self.t_gradient_func(phi, phi_p, theta, t)[0]
        
        return (-1. * g.flatten()) / len(data)

    def get_data(self, mdp, n_samples, n_test_samples, state_rep = 'factored',
                                            distribution = 'uniform'):
    
        all_data = mdp.sample_grid_world(n_samples + n_test_samples, state_rep,
                                        distribution)
        np.random.shuffle(all_data)

        data = all_data[:n_samples]
        test_data = all_data[n_samples:]

        return data, test_data
                



def train_model_cg(model_size = (18, 2, 18), minibatch = 50, n_samples = 250, 
                    n_test_samples = 50, cg_max_iter = 3, dist = 'uniform', use_opt_policy = True):
    import grid_world
    
    m = Model(*model_size)
    
    print 'Generating Samples Trajectory from Gridworld...'
    mdp = grid_world.MDP()

    if use_opt_policy:
        print 'using optimal policy for training'
        mdp.policy = grid_world.OptimalPolicy(mdp.env)
    else:
        print 'using random policy for training'

    data, test_data = m.get_data(mdp, n_samples, n_test_samples,  
                                        state_rep = 'factored', distribution = dist)
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


if __name__ == "__main__":
    train_model_cg()
