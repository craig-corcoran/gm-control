import copy
import numpy
import scipy.sparse
import grid_world
import model

def test_learned_model(m, mdp, n_samples = 1000):
    
    data, _ = m.get_data(mdp, n_samples, 0)
    #numpy.round(numpy.random.random(data.shape))
    rand_configs = copy.deepcopy(data)
    map(numpy.random.shuffle, rand_configs) 
        
    score_d = 0.
    score_r = 0.
    for i in xrange(data.shape[0]):
        
        rows_d = m.get_phi_indexes(data[i])
        rows_r = m.get_phi_indexes(rand_configs[i])
        
        vals = numpy.ones_like(rows_d)
        cols = numpy.zeros_like(rows_d)
        phi_d = scipy.sparse.coo_matrix((vals, (rows_d, cols)), shape = (m.n_params, 1))
        phi_r = scipy.sparse.coo_matrix((vals, (rows_r, cols)), shape = (m.n_params, 1))

        phi_d = scipy.sparse.csr_matrix(phi_d)
        phi_r = scipy.sparse.csr_matrix(phi_r)

        score_d += numpy.exp(phi_d.T.dot(m.theta))
        score_r += numpy.exp(phi_r.T.dot(m.theta))

    score_d = score_d / n_samples
    score_r = score_r / n_samples

    return score_d, score_r

def probability_experiment():
    
    m = model.train_model_cg()
    mdp = grid_world.MDP()

    score_d, score_r = test_learned_model(m, mdp)

    print 'unnormalized data probability: ', score_d
    print 'unnormalized random probability: ', score_r

if __name__ == "__main__":
    probability_experiment()


