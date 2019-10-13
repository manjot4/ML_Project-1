''' This script does hyperparameter tuning (lambda and gamma) using cross-validation '''

''' Importing Libraries '''
import numpy as np
from implementations import reg_logistic_regression, compute_loss_reg_logistic_regression


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
    ''' return the loss for logistic regression for the specific kth-fold '''
    ''' splitting the dataset into k-folds for a particular k-value, 
     kth fold - testing set, (k-1 folds) - training set '''
    k_test = k_indices[k]
    k_train = k_indices[(np.arange(k_indices.shape[0]) != k)]
    k_train = k_train.reshape(-1)
    x_train = tx[k_train]
    x_test = tx[k_test]
    y_train = y[k_train]
    y_test = y[k_test]
	# Logistic Regression, getting weights
    _, weight = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    # calculating loss for train and test data     
    train_loss = compute_loss_reg_logistic_regression(y_train, x_train, lambda_, weight)
    test_loss = compute_loss_reg_logistic_regression(y_test, x_test, lambda_, weight)
    
    return train_loss, test_loss


def gamma_lambda_selection_cv(y, tx, k_fold, lambdas, initial_w, max_iters, gammas):
    """ Implementing cross_validation to hypertune gamma and lambda 

    Parameters
    ----------
    y: ndarray
    	Labels
    tx: ndarray
    	Input Matrix
    lambdas: Regularisation Parameter
    	set of lambdas from which optimal lambda_ is chosen	
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma: The step size
        set of gammas from which optimal gamma is chosen
    Returns
    -------
    tuple
        optimal_lambda_ and optimal_gamma
    """

    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # list for storing minimum loss values for a particular lambda and gamma
    # and then choosing optimal lambda and gamma value with minimum loss
    min_loss = []
    good_lambda = []
    for gamma in range(len(gammas)):
    	log_reg_tr = [] 
    	log_reg_te = []
    	for lambda_ in range(len(lambdas)):
    		train_loss = []
    		test_loss = []
    		for k in range(k_fold):
    			train, test = cross_validation(y, tx, k_indices, k, lambdas[lambda_], initial_w, max_iters, gammas[gamma])
    			train_loss.append(train)
    			test_loss.append(test)
    		log_reg_tr.append(np.mean(train_loss))
        	log_reg_te.append(np.mean(test_loss))	
        optimal_lambda_ = lambdas[np.argmin(log_reg_te)]
        good_lambda.append(optimal_lambda_)
        min_loss.append(log_reg_te[np.argmin(log_reg_te)])
    optimal_lambda_ = lambdas[np.argmin(min_loss)] 
    optimal_gamma = gammas[np.argmin(min_loss)]

    return optimal_lambda_, optimal_gamma

# cross_validation_visualization(lambdas, log_reg_tr, log_reg_te) - plots from ex4. 


# # choosing best lambda or gamma - only one
# def optimal_lambda_selection(y, tx, k_fold, lambdas, initial_w, max_iters, gamma):
#     seed = 1
#     # split data in k fold
#     k_indices = build_k_indices(y, k_fold, seed)
#     # define lists to store the loss of training data and test data
#     log_reg_tr = [] 
#     log_reg_te = []
#     for j in range(len(lambdas)):
#         train_loss = []
#         test_loss = []
#         for j in range(k_fold):
#             tr, te = cross_validation(y, tx, k_indices, j, lambdas[j], initial_w, max_iters, gamma)
#             train_loss.append(tr)
#             test_loss.append(te)
#         log_reg_tr.append(np.mean(train_loss))
#         log_reg_te.append(np.mean(test_loss))
#     optimal_lambda_ = lambdas[np.argmin(log_reg_te)]     
#     return optimal_lambda_