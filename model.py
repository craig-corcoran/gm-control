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
        
        # Theta = <node-wise params, edge-wise params>
        # theta_s,j = |chi| * s + j
        # theta_edge,j,k = |chi|*|V| + (|chi|^2)*edge + |chi|*j + k
        #self.theta = csc_matrix( (self.n_params,1), dtype=np.float)
        #self.theta = np.zeros((self.n_params,1), dtype = np.float)
        self.theta = np.random.standard_normal((self.n_params, 1))

        t_Phi = theano.sparse.csr_matrix('Phi', dtype='int8')
        t_theta = T.dmatrix('theta')
        t_n_nodes = T.iscalar('n_nodes')
        t_g = theano.sparse.structured_dot(t_Phi.T, t_theta)

        t_likelihood =  t_n_nodes * t_g[0,0] - T.sum( T.log( T.exp(T.repeat(t_g[0,0], t_n_nodes)) + T.exp(t_g[1:,0]))) 
 
        self.t_likelihood_func = theano.function([t_Phi, t_theta, t_n_nodes], t_likelihood)
        
        t_likelihood_gradient = T.grad(t_likelihood, [t_theta])
        self.t_gradient_func = theano.function([t_Phi, t_theta, t_n_nodes], t_likelihood_gradient)


    def get_phi_indexes(self, d):
        index_list = np.zeros(self.n_nodes + self.n_edges, dtype = np.int32)
            
        # 2 * indx + val
        index_list[:self.n_nodes] = d + np.array(range(0,self.n_nodes*2,2), dtype = np.int32)
        
        index_list[self.n_nodes:] = np.array(range(0,self.n_edges*4,4), dtype = np.int32) + \
                    np.sum((2,1) * d[self.edges], axis = 1)

        return index_list
        
                
    def get_phi(self, d):
        ''' Convert data array into phi(X)'''
        # Here we assume len(chi) = 2
        phi = csc_matrix((self.n_params,1), dtype=np.int8)
        for indx,val in enumerate(d):
            phi[2 * indx + val, 0] = 1 
        for indx,e in enumerate(self.edges):
            phi[2 * self.n_nodes + indx*4 + 2*d[e[0]] + d[e[1]], 0] = 1
        return phi

    def get_PHI(self, d):
        
        n_nz_col = self.n_nodes + self.n_edges    # number of nonzeros per column in PHI
        n_nz = n_nz_col * self.n_nodes # total number of nonzeros in PHI
        #Phi = csc_matrix( (self.n_params,self.n_nodes+1), dtype=np.int8)
        #Phi[:,0] = self.get_phi(d)
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
        return scipy.sparse.csr_matrix(Phi)
            

    def pseudo_likelihood(self, theta, data):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes - returns negative sum of log psl over the 
        data'''

        if theta.ndim == 1:
            theta = theta[:,None]

        l = 0.
        for d in data:
            Phi = self.get_PHI(d)
            l += self.t_likelihood_func(Phi, theta, self.n_nodes)
        
        return (-1. * l) / len(data)

    def pseudo_likelihood_grad(self, theta, data):
        
        if theta.ndim == 1:
            theta = theta[:,None]

        g = np.zeros_like(theta)
        for d in data:
            Phi = self.get_PHI(d)
            g += self.t_gradient_func(Phi, theta, self.n_nodes)[0]
        
        return (-1. * g.flatten()) / len(data)

    def get_data(self, mdp, n_samples, n_test_samples, state_rep = 'factored'):
    
        all_data = mdp.sample_grid_world(n_samples + n_test_samples, state_rep )
        np.random.shuffle(all_data)

        data = all_data[:n_samples]
        test_data = all_data[n_samples:]

        return data, test_data
                



def train_model_cg(minibatch = 250, n_samples = 2500, n_test_samples = 500, cg_max_iter = 3):
    import grid_world
    
    #m = Model(81, 2, 81)
    m = Model(18, 2, 18)

    print 'Generating Samples Trajectory from Gridworld...'
    mdp = grid_world.MDP()
    data, test_data = m.get_data(mdp, n_samples, n_test_samples,  
                                        state_rep = 'factored')
    n_iters = n_samples / minibatch

    print 'initial loss: ', m.pseudo_likelihood(m.theta, test_data)

    for i in xrange(n_iters):

        print 'iter: ', i+1, ' of ', n_iters

        mb = data[i*minibatch: (i+1)*minibatch]

        n_theta, val, fc, gc, w = scipy.optimize.fmin_cg(
                                m.pseudo_likelihood,
                                m.theta,
                                fprime = m.pseudo_likelihood_grad, 
                                args = (mb,), 
                                full_output = True,
                                gtol = 1e-180,
                                maxiter = cg_max_iter)
        
        delta = np.linalg.norm(n_theta - m.theta)
        
        print 'function calls', fc
        print 'gradient calls', gc 
        print 'delta theta: ', delta 

        m.theta = n_theta
        print 'current training min: ', val
        print 'new test loss: ', m.pseudo_likelihood(m.theta, test_data)

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

    print 'initial loss: ', m.pseudo_likelihood(m.theta, test_data)

    for i in xrange(n_iters):
        
        alpha *= 0.9
        
        mb = data[i*minibatch: (i+1)*minibatch]

        m.theta -= alpha * m.pseudo_likelihood_grad(m.theta, mb)[:,None]
        
        print 'new loss: ', m.pseudo_likelihood(m.theta, test_data)
    
    return m


if __name__ == "__main__":
    train_model_cg()
