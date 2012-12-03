import copy
import numpy as np
import itertools
# import theano.tensor
# import theano.function
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
        self.theta = np.zeros(self.n_params)

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

    def new_pseudo_likelihood(self, data, theta):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes'''

        l = 0.
        for d in data:
            
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
                rows[i*n_nz_col:(i+1)*n_nz_col] = self.get_phi_indexes(d)
                cols[i*n_nz_col:(i+1)*n_nz_col] = i+1
                d_prime[i] = d[i]

            Phi = scipy.sparse.coo_matrix((vals,(rows,cols)), shape=(self.n_params,self.n_nodes+1))
            
            # g = <|V| + 1>
            g = Phi.T.dot(theta)

            l += self.n_nodes * g[0] - np.sum(np.log(np.exp(np.repeat(g[0], self.n_nodes)) + np.exp(g[1:])))
        return l / data.shape[0]
                
    def pseudo_likelihood(self, data, theta):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes'''

        l = 0.
        for d in data:
            # Ensure d is a list of length |V| whose values are in Chi
            assert(len(d) == self.n_nodes)
            for i in d:
                assert(i in self.chi)

            #Phi = csc_matrix( (self.n_params,self.n_nodes+1), dtype=np.int8)
            #Phi[:,0] = self.get_phi(d)
            Phi = hstack( [self.get_phi(d)], format='csc')
            d_prime = copy.deepcopy(d)
            for i in xrange(self.n_nodes):
                d_prime[i] = 1 if d[i] == 0 else 0
                Phi = hstack( [Phi, self.get_phi(d_prime)], format='csc', dtype=np.int8)
                #Phi[:,i+1] = self.get_phi(d_prime)
                d_prime[i] = d[i]
            
            # g = <|V| + 1>
            g = Phi.T.dot(theta).todense()

            l += self.n_nodes * g[0] - np.sum(np.log(np.exp(np.repeat(g[0], self.n_nodes)) + np.exp(g[1:])))
        return l / data.shape[0]


    # def grad_pseudo_likelihood(self, theta, data):
    #     grad = []
    #     for key in theta.keys():
    #         if len(key) == 2:   # Node theta
    #             grad.append(self.grad_pseudo_likelihood_nodewise(key,theta,data))
    #         elif len(key) == 4: # Edge theta
    #             grad.append(self.grad_pseudo_likelihood_edgewise(key,theta,data))
    #         else:
    #             assert(False)
    #     return grad

    # def grad_pseudo_likelihood_nodewise(self, key, theta, data):
    #     ''' Gradient W.R.T. node s taking value j '''
    #     assert(len(key) == 2)
    #     s = key[0]
    #     j = key[1]

    #     grad = 0.
    #     for d in data:
    #         if d[s] == j:
    #             grad += 1

    #         alpha = theta[(s,j)]
    #         for t in self.neighbors[s]:
    #             alpha += theta[(s,t,j,d[t])] if s<t else theta[(t,s,d[t],j)]
    #         alpha = np.exp(alpha)

    #         alpha_denom = 0.
    #         for J in self.chi:
    #             b = theta[(s,J)]
    #             for t in self.neighbors[s]:
    #                 b += theta[(s,t,J,d[t])] if s<t else theta[(t,s,d[t],J)]
    #             alpha_denom += np.exp(b)

    #         grad -= alpha / alpha_denom
            
    #     return grad

    # def grad_pseudo_likelihood_edgewise(self, key, theta, data):
    #     ''' Gradient W.R.T. egde s,t taking value j,k '''
    #     assert(len(key) == 4)
    #     s = key[0]
    #     t = key[1]
    #     j = key[2]
    #     k = key[3]
        
    #     grad = 0.
    #     for d in data:
    #         if d[s] == j and d[t] == k:
    #             grad += 2
                
    #             # First set of alphas
    #             alpha = theta[(t,k)]
    #             for T in self.neighbors[t]:
    #                 alpha += theta[(t,T,k,d[T])] if t<T else theta[(T,t,d[T],k)]
    #             alpha = np.exp(alpha)

    #             alpha_denom = 0.
    #             for J in self.chi:
    #                 b = theta[(t,J)]
    #                 for T in self.neighbors[t]:
    #                     b += theta[(t,T,J,d[T])] if t<T else theta[(T,t,d[T],J)]
    #                 alpha_denom += np.exp(b)

    #             grad -= alpha / alpha_denom

    #             # Second set of alphas
    #             alpha = theta[(s,j)]
    #             for T in self.neighbors[s]:
    #                 alpha += theta[(s,T,j,d[T])] if s<T else theta[(T,s,d[T],j)]
    #             alpha = np.exp(alpha)

    #             alpha_denom = 0.
    #             for J in self.chi:
    #                 b = theta[(s,J)]
    #                 for T in self.neighbors[s]:
    #                     b += theta[(s,T,J,d[T])] if s<T else theta[(T,s,d[T],J)]
    #                 alpha_denom += np.exp(b)

    #             grad -= alpha / alpha_denom
            
    #     return grad

    # def learn(self, data):
    #     ''' Attempts to find the self.theta maximizing the pseudo likelihood'''
    #     x0 = 
    #     scipy.optimize.fmin(pseudo_likelihood,


def main():
    import grid_world
    import time

    a = Model(18, 2, 18)

    print 'Generating Samples Trajectory from Gridworld...'
    start = time.time()
    mdp = grid_world.MDP()
    data = mdp.sample_grid_world(5)
    elapsed = (time.time() - start)
    print elapsed, 'seconds'

    # data = []
    # for i in range(5):
    #     data.append(np.random.randint(2, size=a.n_nodes))

    print 'Computing Pseudo-Likelihood...'
    start = time.time()
    val = a.new_pseudo_likelihood(data, a.theta)
    elapsed = (time.time() - start)
    print elapsed, 'seconds'

    print val
    print np.exp(val)

    # alpha = -.2
    # for i in xrange(10):
    #     print a.pseudo_likelihood(data, a.theta)

    #     grad = a.grad_pseudo_likelihood(a.theta, data)
    #     indx = 0
    #     for key in a.theta.keys():
    #         a.theta[key] += alpha * grad[indx]
    #         indx += 1
            
    


if __name__ == "__main__":
    main()
