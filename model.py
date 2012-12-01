import numpy

class Model:
    def __init__(self, n_state_nodes, n_action_nodes, n_next_state_nodes, n_reward_nodes = 1, chi = [0,1]):
        self.n_nodes   = n_state_nodes + n_action_nodes + n_next_state_nodes + n_reward_nodes
        self.chi       = chi
        self.neighbors = {}
        self.theta     = {}
        
        # P = <S,A,S',R>
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
                self.theta[(s,j)] = numpy.random.random()

        # Initialize Edge-wise parameters
        for s in range(self.n_nodes):
            for t in self.neighbors[s]:
                for j in self.chi:
                    for k in self.chi:
                        self.theta[(s,t,j,k)] = numpy.random.random()
                
                
    def pseudo_likelihood(self, data, theta):
        ''' data is a list of lists where each inner list is a assigment of
        values to all of the nodes'''
        
        l = 0
        for s in range(self.n_nodes):
            for d in data:
                # Ensure d is a list of length |S| whose values are in Chi
                assert(len(d) == self.n_nodes)
                for i in d:
                    assert(i in self.chi)
                    
                # theta(X_i) + Sum_neighbors theta(X_i,X_t)
                l += theta[(s,d[s])]
                for t in self.neighbors[s]:
                    l += theta[(s,t,d[s],d[t])]

                # log(b) = log( Sum_chi exp{ theta(X_j) + Sum_neighbors theta(X_j,X_t) } )
                b = 0
                for j in self.chi:
                    c = theta[(s,j)]
                    for t in self.neighbors[s]:
                        c += theta[(s,t,j,d[t])]
                    b += numpy.exp(c)
                l -= numpy.log(b)
        return l
            
                
def main():
    a = Model(3,2,3)

    data = []
    for i in range(5):
        data.append(numpy.random.randint(2, size=a.n_nodes))

    print a.pseudo_likelihood(data, a.theta)


if __name__ == "__main__":
    main()
