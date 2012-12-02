import numpy
import theano.tensor
import theano.function

class Model:
    def __init__(self, n_state_nodes, n_action_nodes, n_next_state_nodes, n_reward_nodes = 1, chi = [0,1]):
        self.n_nodes   = n_state_nodes + n_action_nodes + n_next_state_nodes + n_reward_nodes
        self.chi       = chi
        self.neighbors = {}
        self.theta     = {}
        
        # X = <S,A,S',R>
        n = 0
        self.state_nodes      = range(n, n + n_state_nodes)
        n += n_state_nodes
        self.action_nodes     = range(n, n + n_action_nodes)
        n += n_action_nodes
        self.next_state_nodes = range(n, n + n_next_state_nodes)
        n += n_next_state_nodes
        self.reward_nodes     = range(n, n + n_reward_nodes)

        # Initialize the adjacency dictionary
        for i in self.state_nodes:
            self.neighbors[i] = list(set(range(self.n_nodes)) - set(self.state_nodes))
        for i in self.action_nodes:
            self.neighbors[i] = list(set(range(self.n_nodes)) - set(self.action_nodes))
        for i in self.next_state_nodes:
            self.neighbors[i] = list(set(range(self.n_nodes)) - set(self.reward_nodes) - set(self.next_state_nodes))
        for i in self.reward_nodes:
            self.neighbors[i] = list(set(range(self.n_nodes)) - set(self.reward_nodes) - set(self.next_state_nodes))

        # Initialize Node-wise parameters
        for s in range(self.n_nodes):
            for j in self.chi:
                self.theta[(s,j)] = 0

        # Initialize Edge-wise parameters
        for s in range(self.n_nodes):
            for t in self.neighbors[s]:
                for j in self.chi:
                    for k in self.chi:
                        self.theta[(min(s,t),max(s,t),j,k)] = 0
                
                
    def pseudo_likelihood(self, data, theta):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes'''
        
        l = 0.
        for s in xrange(self.n_nodes):
            for d in data:
                # Ensure d is a list of length |V| whose values are in Chi
                assert(len(d) == self.n_nodes)
                for i in d:
                    assert(i in self.chi)
                    
                # theta(X_i) + Sum_neighbors theta(X_i,X_t)
                a = theta[(s,d[s])]
                for t in self.neighbors[s]:
                    a += theta[(s,t,d[s],d[t])] if s<t else theta[(t,s,d[t],d[s])]

                # log(b) = log( Sum_chi exp{ theta(X_j) + Sum_neighbors theta(X_j,X_t) } )
                b = 0
                for j in self.chi:
                        
                    if j == d[s]:
                        c = a
                    else:
                        c = theta[(s,j)]
                        for t in self.neighbors[s]:
                            c += theta[(s,t,d[s],d[t])] if s<t else theta[(t,s,d[t],d[s])]

                    b += numpy.exp(c)
                l -= numpy.log(b)
        return l / data.shape[0]


    def grad_pseudo_likelihood(self, theta, data):
        grad = []
        for key in theta.keys():
            if len(key) == 2:   # Node theta
                grad.append(self.grad_pseudo_likelihood_nodewise(key,theta,data))
            elif len(key) == 4: # Edge theta
                grad.append(self.grad_pseudo_likelihood_edgewise(key,theta,data))
            else:
                assert(False)
        return grad

    def grad_pseudo_likelihood_nodewise(self, key, theta, data):
        ''' Gradient W.R.T. node s taking value j '''
        assert(len(key) == 2)
        s = key[0]
        j = key[1]

        grad = 0.
        for d in data:
            if d[s] == j:
                grad += 1

            alpha = theta[(s,j)]
            for t in self.neighbors[s]:
                alpha += theta[(s,t,j,d[t])] if s<t else theta[(t,s,d[t],j)]
            alpha = numpy.exp(alpha)

            alpha_denom = 0.
            for J in self.chi:
                b = theta[(s,J)]
                for t in self.neighbors[s]:
                    b += theta[(s,t,J,d[t])] if s<t else theta[(t,s,d[t],J)]
                alpha_denom += numpy.exp(b)

            grad -= alpha / alpha_denom
            
        return grad

    def grad_pseudo_likelihood_edgewise(self, key, theta, data):
        ''' Gradient W.R.T. egde s,t taking value j,k '''
        assert(len(key) == 4)
        s = key[0]
        t = key[1]
        j = key[2]
        k = key[3]
        
        grad = 0.
        for d in data:
            if d[s] == j and d[t] == k:
                grad += 2
                
                # First set of alphas
                alpha = theta[(t,k)]
                for T in self.neighbors[t]:
                    alpha += theta[(t,T,k,d[T])] if t<T else theta[(T,t,d[T],k)]
                alpha = numpy.exp(alpha)

                alpha_denom = 0.
                for J in self.chi:
                    b = theta[(t,J)]
                    for T in self.neighbors[t]:
                        b += theta[(t,T,J,d[T])] if t<T else theta[(T,t,d[T],J)]
                    alpha_denom += numpy.exp(b)

                grad -= alpha / alpha_denom

                # Second set of alphas
                alpha = theta[(s,j)]
                for T in self.neighbors[s]:
                    alpha += theta[(s,T,j,d[T])] if s<T else theta[(T,s,d[T],j)]
                alpha = numpy.exp(alpha)

                alpha_denom = 0.
                for J in self.chi:
                    b = theta[(s,J)]
                    for T in self.neighbors[s]:
                        b += theta[(s,T,J,d[T])] if s<T else theta[(T,s,d[T],J)]
                    alpha_denom += numpy.exp(b)

                grad -= alpha / alpha_denom
            
        return grad


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
    #     data.append(numpy.random.randint(2, size=a.n_nodes))

    # print 'Computing Pseudo-Likelihood...'
    # start = time.time()
    # val = a.pseudo_likelihood(data, a.theta)
    # elapsed = (time.time() - start)
    # print elapsed, 'seconds'

    # print val
    # print numpy.exp(val)

    alpha = -.2
    for i in xrange(10):
        print a.pseudo_likelihood(data, a.theta)

        grad = a.grad_pseudo_likelihood(a.theta, data)
        indx = 0
        for key in a.theta.keys():
            a.theta[key] += alpha * grad[indx]
            indx += 1
            
    


if __name__ == "__main__":
    main()
