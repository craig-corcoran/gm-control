import cPickle as pickle
import numpy as np
import pulp
import util
import grid_world
import model

def get_map(m, state):
    ''' state here is the partial vector of values assigned to s in the full 
    configuration x = [s, a, s', r] '''

    LP = pulp.LpProblem("Grid MDP", pulp.LpMaximize)
    mu = pulp.LpVariableDict('state-action value')
    theta = m.theta 

    # add node variables, constraints
    for i in xrange(m.n_nodes):
        ind0=2*i
        ind1=2*i+1
        
        mu[ind0] = pulp.LpVariable('mu%i%i'%(i, 0))
        mu[ind1] = pulp.LpVariable('mu%i%i'%(i, 1))

        # add non-negative constraints
        LP += mu[ind0] >= 0
        LP += mu[ind1] >= 0

        # add node-wise normalization constraint
        LP += mu[ind0] + mu[ind1] == 1
    
    offset = 2*m.n_nodes
    for i,e in enumerate(m.edges):
        inds = offset + 4*i + np.arange(4)
        
        # non-negative
        for num,ind in enumerate(inds): 
            mu[ind] = pulp.LpVariable('mu%i%i-%i'%(e[0],e[1],num))
            LP += mu[ind] >=0
        
        # marginalization constraints
        # marginalize out the second node e[1]
        LP += (mu[inds[0]] + mu[inds[1]]) == mu[2*e[0]]
        LP += (mu[inds[2]] + mu[inds[3]]) == mu[2*e[0]+1]

        # marginalize out the second node e[2]
        LP += (mu[inds[0]] + mu[inds[2]]) == mu[2*e[1]]
        LP += (mu[inds[1]] + mu[inds[3]]) == mu[2*e[1]+1]

    # add constraints for observed states
    for i,s in enumerate(state):
        LP += mu[i*2] == (1-s)
        LP += mu[i*2+1] == s
    
    # adding the objective to the LP
    LP += sum(map(lambda key, th: mu[key]*th,
                mu.keys(), theta))

    # solving the LP
    LP.solve(pulp.GLPK(msg = 0)) # then solve    
    map_config = np.zeros(m.n_nodes)

    # read off the map configuration from mu
    for i in xrange(m.n_nodes):
        
        if pulp.value(mu[2*i+1]) == 1:
            map_config[i] = 1
        else:
            assert pulp.value(mu[2*i+1]) == 0
        
        assert not(pulp.value(mu[2*i]) == pulp.value(mu[2*i+1]))

    return map_config

def parse_config(map_config, m, env, state_rep = 'factored'):
    # separate components
    n = [0, m.n_state_nodes, m.n_state_nodes + m.n_action_nodes,
        m.n_state_nodes*2 + m.n_action_nodes]
    s = map_config[n[0]:n[1]]
    a = map_config[n[1]:n[2]]
    s_p = map_config[n[2]:n[3]]
    r = map_config[-1]
    
    # convert from encoding to array
    if state_rep == 'factored':
        pos = s.nonzero()[0]
        assert len(pos) == 2
        pos[1] = pos[1] % (m.n_state_nodes/2)

        pos_p = s_p.nonzero()[0]
        print pos_p
        try:
            assert len(pos_p) == 2
            pos_p[1] = pos_p[1] % (m.n_state_nodes/2)
        except AssertionError:
            print 'there are more or less than 2 nz elements in the s_p config'
            print pos_p
            if len(pos_p) == 1:
                pos_p = np.array([pos_p[0],-1]) # insert bogus y pos
            elif len(pos_p) == 0:
                pos_p = np.array([-1,-1]) # insert bogus y pos

    else:
        print 'representations other than factored not implemented'
        assert False

    act = env.code_to_action[tuple(a)]
    
    return pos, act, pos_p, r

class MapPolicy:
    
    def __init__(self, m, env):
        self.m = m
        self.env = env

    def choose_actions(self, actions=None):
        '''ignores list of actions and does map to choose next action'''
        map_config = get_map(self.m, self.env.state)
        action_code = map_config[self.m.n_state_nodes:self.m.n_state_nodes+2]
        return self.env.code_to_action(action_code)

def main(model_size = (18,2,18), dist = 'uniform', n_iters = 10):
    
    file_str = 'model.%i.%i.%s.pickle.gz'%(model_size[0],model_size[1],dist)

    try:
        with util.openz(file_str) as f:
            print 'found previous file, using: ', f
            m = pickle.load(f)
    except IOError:
        print 'no serialized model found, training a new one'
        m = model.train_model_cg(model_size, dist = dist)
        with util.openz(file_str, 'wb') as f:
            pickle.dump(m, f)

    mdp = grid_world.MDP() 
    map_policy = MapPolicy(m, mdp.env)
    mdp.policy = map_policy


    for i in xrange(n_iters):
        state = mdp.env.state
        print 'current state: ', state

        # convert state to binary vector
        phi_s = np.zeros(model_size[0])
        phi_s[state[0]] = 1 
        phi_s[mdp.env.n_rows + state[1]] = 1

        pos, act, pos_p, r = parse_config(get_map(m, phi_s), m, mdp.env)

        mdp.env.take_action(act)

    print 'map configuration: ', pos, act, pos_p, r



if __name__ == '__main__':
    main() 
