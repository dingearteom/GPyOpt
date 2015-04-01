import numpy as np

from ..util.general import best_value, reshape, spawn
from ..core.optimization import mp_batch_optimization, random_batch_optimization, predictive_batch_optimization

try:
    from ..plotting.plots_bo import plot_acquisition, plot_convergence, plot_convergence_gradients
except:
    pass

class BO(object):
    def __init__(self, acquisition_func, bounds=None, model_optimize_interval=None, model_optimize_restarts=None, model_data_init=None, normalize=None, verbosity=None):
        self.input_dim = len(self.bounds)        
        self.acquisition_func = acquisition_func
        self.model_optimize_interval = model_optimize_interval
        self.model_optimize_restarts = model_optimize_restarts
        if  model_data_init ==None: 
            self.model_data_init = 2*self.input_dim      # default number or samples for initial random exploration
        else: 
            self.model_data_init = model_data_init  
        self.normalize = normalize
        self.verbosity = verbosity
    
 
    def _init_model(self):
        pass
        
    def run_optimization(self, max_iter=15, n_inbatch=1, acqu_optimize_method='random', batch_method='predictive', acqu_optimize_restarts=20, stop_criteria = 1e-16, n_procs=1, true_gradients = True, verbose=True):
        """ 
        Starts Bayesian Optimization for a number H of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of iterations  
	    :param n_inbatch: number of samples to collect in each batch (one by default)
        :param acqu_optimize_method: method to optimize the acquisition function 
	        -'brute': uses a uniform lattice with 'acqu_optimize_restarts' points per dimension. A local CG gradient is run the best point.
	        -'random': takes the best of 'acqu_optimize_restarts' local random optimizers.
            -'fast_brute':
            -'fast_random':
        :param nb: number of samples to collect in each batch (one by default)
	    :param batch_method: method to collect samples in batches
            -'predictive': uses the predicted mean in the selected sample to update the acquisition function.
            -'mp': used a penalization of the acquisition function to based on exclusion zones.
            -'random': collects the element of the batch randomly
        :param acqu_optimize_restarts: numbers of random restarts in the optimization of the acquisition function, default = 10.
    	:param stop_criteria: minimum distance between two consecutive x's to keep running the model
    	:param n_procs: The number of processes used for evaluating the given function *f*
        :param true_gradients: If the true gradients (can be slow) of the acquisition ar an approximation is used (True, default).

        ..Note : X and Y can be None. In this case Nrandom*model_dimension data are uniformly generated to initialize the model.
    
        """
        # load the parameters of the function into the object.
        self.num_acquisitions = 0
        self.n_inbatch=n_inbatch
        self.batch_method = batch_method
        self.stop_criteria = stop_criteria 
        self.acqu_optimize_method = acqu_optimize_method
        self.acqu_optimize_restarts = acqu_optimize_restarts
        self.batch_method = batch_method
        self.acquisition_func.model = self.model
        self.n_procs = n_procs

        # decide wether we use the true gradients to optimize the acquitision function
        if true_gradients !=True:
            self.true_gradients = False  
            self.acquisition_func.d_acquisition_function = None
        else: 
            self.true_gradients = true_gradients

        # optimize model and acquisition function by first time
        self._update_model()
        prediction = self.model.predict(self.X)       
        self.m_in_min = prediction[0]
        prediction[1][prediction[1]<0] = 0
        self.s_in_min = np.sqrt(prediction[1])

        k=0
        distance_lastX = self.stop_criteria + 1
        while k<max_iter and distance_lastX > self.stop_criteria:
                
            # ------- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))
            
            # ------- Evaluate *f* in X and augment Y
            if self.n_procs==1:
                self.Y = np.vstack((self.Y,self.f(np.array(self.suggested_sample))))
            else:
                try:
                    # Parallel evaluation of *f* is several cores are available
                    from multiprocessing import Process, Pipe
                    from itertools import izip          
                    divided_samples = [self.suggested_sample[i::self.n_procs] for i in xrange(self.n_procs)]
                    pipe=[Pipe() for i in xrange(self.n_procs)]
                    proc=[Process(target=spawn(self.f),args=(c,x)) for x,(p,c) in izip(divided_samples,pipe)]
                    [p.start() for p in proc]
                    [p.join() for p in proc]
                    rs = [p.recv() for (p,c) in pipe]
                    self.Y = np.vstack([self.Y]+rs)
                except:
                    if not hasattr(self, 'parallel_error'):
                        print 'Error in parallel computation. Fall back to single process!'
                        self.parallel_error = True 
                    self.Y = np.vstack((self.Y,self.f(np.array(self.suggested_sample))))
                
            # -------- Update internal elements (needed for plotting)
            self.num_acquisitions += 1
            pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))       
            self.m_in_min = np.vstack((self.m_in_min,pred_min[0]))
            self.s_in_min = np.vstack((self.s_in_min,np.sqrt(abs(pred_min[1]))))
                
            # -------- Update model
            try:
                self._update_model()                
            except np.linalg.linalg.LinAlgError:
                break

            # ------- Update stop conditions
            k +=1
            distance_lastX = np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))     

        # ------- Stop messages            
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = min(self.Y)
        if verbose: print '*Optimization completed:'
        if k==max_iter:
            if verbose: print '   -Maximum number of iterations reached.'
            return 1
        else: 
            if verbose: print '   -Method converged.'
            return 0

    
    def change_to_sparseGP(self, num_inducing):
        """
        Changes standard GP estimation to sparse GP estimation
	       
	    :param num__inducing: number of inducing points for sparse-GP modeling
	     """
        if self.sparse == True:
            raise 'Sparse GP is already in use'
        else:
            self.num_inducing = num_inducing
            self.sparse = True
            self._init_model(self.X,self.Y)

    def change_to_standardGP(self):
        """
        Changes sparse GP estimation to standard GP estimation

        """
        if self.sparse == False:
            raise 'Sparse GP is already in use'
        else:
            self.sparse = False
            self._init_model(self.X,self.Y)
    
        
    def _optimize_acquisition(self):
        """
        Optimizes the acquisition function. It combines initial grid search with local optimization starting on the minimum of the grid

        """
        # ------ Elements of the acquisition function
        acqu_name = self.acqu_name
        acquisition = self.acquisition_func.acquisition_function
        d_acquisition = self.acquisition_func.d_acquisition_function
        acquisition_par = self.acquisition_par
        model = self.model
        
        # ------  Parameters to optimize the acquisition
        acqu_optimize_restarts = self.acqu_optimize_restarts
        acqu_optimize_method = self.acqu_optimize_method
        n_inbatch = self.n_inbatch
        bounds = self.bounds

        # ------ Selection of the batch method (if any, predictive used when n_inbathc=1)
        if self.batch_method == 'predictive':
            X_batch = predictive_batch_optimization(acqu_name, acquisition_par, acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch)            
        elif self.batch_method == 'mp':
            X_batch = mp_batch_optimization(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch)
        elif self.batch_method == 'random':
            X_batch = random_batch_optimization(acquisition, d_acquisition, bounds, acqu_optimize_restarts,acqu_optimize_method, model, n_inbatch)        
        return X_batch


    def _update_model(self):
        """        
        Updates X and Y in the model and re-optimizes the parameters of the new model

        """  
        # ------- Normalize acquisition function (if needed)
        if self.normalize:      
            self.model.set_XY(self.X,(self.Y-self.Y.mean())/self.Y.std())
        else:
            self.model.set_XY(self.X,self.Y)
        
        # ------- Optimize model when required
        if (self.num_acquisitions%self.model_optimize_interval)==0:
            self.model.optimization_runs = [] # clear previous optimization runs so they don't get used.
            self.model.optimize_restarts(num_restarts=self.model_optimize_restarts, verbose=self.verbosity)            
        
        # ------- Optimize acquisition function
        self.suggested_sample = self._optimize_acquisition()


    def plot_acquisition(self,filename=None):
        """        
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """  
        return plot_acquisition(self.bounds,self.input_dim,self.model,self.model.X,self.model.Y,self.acquisition_func.acquisition_function,self.suggested_sample,filename)

    def plot_convergence(self,filename=None):
        """
        Makes three plots to evaluate the convergence of the model
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
            plot 3: Iterations vs. the variance of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        if self.acqu_name == 'GMF':
            return plot_convergence_gradients(self.X)  ##check imputs  
        else:
            return plot_convergence(self.X,self.Y_best,self.s_in_min)



















