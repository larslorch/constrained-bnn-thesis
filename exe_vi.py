import os
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from plot import *
from utils import *
from bbb import bayes_by_backprop_variational_inference
from bnn import make_BNN
from npv import nonparametric_variational_inference


'''
Runs variational inference optimization procedure and returns results
'''
def run_experiment(experiment):

    # always, this is done to hide import code in editor
    if True:

        '''BNN '''
        architecture = experiment['nn']['architecture']
        nonlinearity = experiment['nn']['nonlinearity']
        prior_ds = experiment['nn']['prior_ds']

        '''Data '''
        noise_ds = experiment['data']['noise_ds']
        X = experiment['data']['X'] 
        Y = experiment['data']['Y']

        X_v_id = experiment['data']['X_v_id']
        Y_v_id = experiment['data']['Y_v_id']

        X_v_ood = experiment['data']['X_v_ood']
        Y_v_ood = experiment['data']['Y_v_ood']


        '''VI settings'''
        rv_samples = experiment['vi']['rv_samples']
        batch_size = experiment['vi']['batch_size']
        num_batches = int(torch.ceil(torch.tensor(X.shape[0] / batch_size))) if batch_size else 1
        lr = experiment['vi']['lr']

        # regular
        iterations_regular = experiment['vi']['regular']['iterations']
        restarts_regular = experiment['vi']['regular']['restarts']
        reporting_every_regular_ = experiment['vi']['regular']['reporting_every_']
        cores_regular = experiment['vi']['regular']['cores_used']

        # constrained
        iterations_constr = experiment['vi']['constrained']['iterations']
        restarts_constr = experiment['vi']['constrained']['restarts']
        reporting_every_constr_ = experiment['vi']['constrained']['reporting_every_']
        cores_constr = experiment['vi']['constrained']['cores_used']
        gamma = experiment['vi']['constrained']['gamma']
        tau = experiment['vi']['constrained']['tau_tuple']
        violation_samples = experiment['vi']['constrained']['violation_samples']
        constrained_region_sampler = experiment['vi']['constrained']['constrained_region_sampler']
        constr = experiment['constraints']['constr']

        S = experiment['vi']['posterior_predictive_analysis']['posterior_samples']
        

        '''Experiment settings'''
        regular_BbB = experiment['experiment']['run_regular_vi']
        constrained_BbB = experiment['experiment']['run_constrained_vi']
        multithread = experiment['experiment']['multithread_computation']
        compute_held_out_loglik_id = experiment['experiment']['compute_held_out_loglik_id']
        compute_held_out_loglik_ood = experiment['experiment']['compute_held_out_loglik_ood']

        compute_RMSE_id = experiment['experiment']['compute_RMSE_id']
        compute_RMSE_ood = experiment['experiment']['compute_RMSE_ood']

    '''Make directory for results'''
    current_directory = make_unique_dir(experiment)
    
    '''Define BNN'''
    num_weights, forward, log_prob = \
        make_BNN(layer_sizes=architecture,
                 prior_ds=prior_ds,
                 noise_ds=noise_ds,
                 nonlinearity=nonlinearity,
                 num_batches=num_batches)

    '''Defines log posterior with minibatching'''
    if batch_size == 0:
        # full dataset
        def log_posterior(weights, iter):
            return log_prob(weights, X, Y)
    else:
        # minibatching
        def batch_indices(iter):
            # same seed/batches indices for one iteration over X
            seed, effective_iter = divmod(iter, num_batches)
            torch.manual_seed(seed)
            batches = torch.split(torch.randperm(X.shape[0]), batch_size)
            return batches[effective_iter]

        def log_posterior(weights, iter):
            batch = batch_indices(iter)
            return log_prob(weights, X[batch], Y[batch])


    '''Inference functions'''
    both_runs = []

    '''Computes held-out log likelihood of x,y given distribution implied by param'''
    def held_out_loglikelihood(x, y, param):
        mean, log_std = param[:, 0], param[:, 1]
        ws = mean + torch.randn(S, mean.shape[0]) * log_std.exp() # torch.log(1.0 + log_std.exp())
        samples = forward(ws, x)
        mean = samples.mean(0).squeeze()
        std = samples.std(0).squeeze()
        return ds.Normal(mean, std).log_prob(y).sum()

    '''Compute RMSE of validation dataset given optimizated params'''
    def compute_rmse(x, y, param):
        mean, log_std = param[:, 0], param[:, 1]
        ws = mean + torch.randn(S, mean.shape[0]) * log_std.exp()
        samples = forward(ws, x)
        pred = samples.mean(0) # prediction is mean
        rmse = (pred - y).pow(2).mean(0).pow(0.5)
        return rmse
    
    '''Computes expected violation via constraint function, of distribution implied by param'''
    def violation(param):
        mean, log_std = param[:, 0], param[:, 1]
        weights = mean + torch.randn(S, mean.shape[0]) * log_std.exp()
        x = constrained_region_sampler(violation_samples)
        y = forward(weights, x)
        tau_c, tau_g = tau
        c = torch.zeros(y.shape)
        for region in constr:
            d = torch.ones(y.shape)
            for constraint in region:
                d *= psi(constraint(x, y), tau_c, tau_g)
            c += d
        l = gamma * c.sum() / y.numel()
        # l = gamma * c.max() # returns max violation along y.shape (might be better than average across all)
        return l

    '''Runs Bayes by Backprop for one random restart'''
    def run_bbb(r, constrained):
                
        variational_objective, evidence_lower_bound = \
            bayes_by_backprop_variational_inference(
                log_posterior, violation, num_samples=rv_samples, constrained=constrained, num_batches=num_batches)

        # initialization
        print(50 * '-')
        init_mean = experiment['vi']['bbb_param']['initialize_q']['mean'] * \
            torch.randn(num_weights, 1)
        init_log_std = experiment['vi']['bbb_param']['initialize_q']['std'] * \
            torch.ones(num_weights, 1)
        params = Variable(
            torch.cat([init_mean, init_log_std], dim=1),
            requires_grad=True)

        # specific settings
        if constrained:
            iterations = iterations_constr
            reporting_every_ = reporting_every_constr_
        else:
            iterations = iterations_regular
            reporting_every_ = reporting_every_regular_

        # ADAM optimizer
        optimizer = optim.Adam([params], lr=lr)
        # optimizer = optim.LBFGS([params], lr=1)
        # optimizer = optim.SGD([params], lr=0.01, momentum=0.9)


        # evaluation
        training_evaluation = dict(
            objective=[], 
            elbo=[], 
            violation=[], 
            held_out_ll_indist=[], 
            held_out_ll_outofdist=[],
            rmse_id=[],
            rmse_ood=[])

        for t in range(iterations):

            # optimization
            optimizer.zero_grad()
            loss = variational_objective(params, t)
            loss.backward()
            optimizer.step()


            # compute evaluation every 'reporting_every_' steps
            if not t % reporting_every_:
                elbo = evidence_lower_bound(params, t)
                viol = violation(params).detach()
                training_evaluation['objective'].append(loss.detach())
                training_evaluation['elbo'].append(elbo.detach())
                training_evaluation['violation'].append(viol)

                if compute_held_out_loglik_id:
                    training_evaluation['held_out_ll_indist'].append(
                        held_out_loglikelihood(X_v_id, Y_v_id, params.detach()))

                if compute_held_out_loglik_ood:
                    training_evaluation['held_out_ll_outofdist'].append(
                        held_out_loglikelihood(X_v_ood, Y_v_ood, params.detach()))
                
                if compute_RMSE_id:
                    rmse_id_cache = compute_rmse(
                        X_v_id, Y_v_id, params.detach())
                    training_evaluation['rmse_id'].append(rmse_id_cache)
                
                if compute_RMSE_ood:
                    training_evaluation['rmse_ood'].append(
                        compute_rmse(X_v_ood, Y_v_ood, params.detach()))

                # command line printing
                str = 'Step {:7}  ---  Objective: {:15}  ELBO: {:15}  Violation: {:10}'.format(
                    t, round(loss.item(), 4), round(elbo.item(), 4), round(viol.item(), 4))

                if compute_RMSE_id:
                    str += '   ID-RMSE {:10}'.format(round(rmse_id_cache.item(), 4))

                print(str)

        return params.detach(), loss.detach(), training_evaluation

    '''Runs nonparametric VI for one random restart'''
    def run_npv(r, constrained):
        
        # initialization: params has shape (mixtures, weights + 1)
        print(50 * '-')
        mixtures = experiment['vi']['npv_param']['mixtures']
        mean_opt_step = steps = experiment['vi'][
            'npv_param']['optim']['steps_per_mean']
        std_opt_step = steps = experiment['vi'][
            'npv_param']['optim']['steps_std']

        params = torch.zeros(mixtures, num_weights + 1)
        
        for m in range(mixtures):
            params[m, :] = experiment['vi']['bbb_param']['initialize_q']['mean'] * \
                torch.randn(num_weights + 1)
            params[m, 0] = experiment['vi']['bbb_param']['initialize_q']['std'] 

        params = Variable(params, requires_grad=True)
        
        elbo_approx_1, elbo_approx_2 = \
            nonparametric_variational_inference(
                log_posterior, 
                violation, 
                num_samples=rv_samples, 
                constrained=constrained, 
                num_batches=num_batches)


        # specific settings
        if constrained:
            iterations = iterations_constr
            reporting_every_ = reporting_every_constr_
        else:
            iterations = iterations_regular
            reporting_every_ = reporting_every_regular_
    

        training_evaluation = dict(
            objective=[],
            elbo=[],
            violation=[],
            held_out_ll_indist=[],
            held_out_ll_outofdist=[],
            rmse_id=[],
            rmse_ood=[])

        for t in range(iterations):
            
            # TODO
            
            # optimize means individually
            for n in range(mixtures):
                
                # UNPACK params + CHANGE
                mean_optimizer = optim.Adam([params], lr=lr)

                for _ in range(mean_opt_step):
                    mean_optimizer.zero_grad()

                    # compute l1 loss and step on mean[n]

                    pass

            # optimize stds

            # UNPACK params + CHANGE
            std_optimizer = optim.Adam([params], lr=lr)

            for _ in range(std_opt_step):
                std_optimizer.zero_grad()

                # compute l2 loss and step
                pass

        

            print('Not implemented.')
            exit(1)

            # optimization
            optimizer.zero_grad()
            loss = variational_objective(params, t)
            loss.backward()
            optimizer.step()

            # compute evaluation every 'reporting_every_' steps
            if not t % reporting_every_:
                elbo = elbo_approx_1(params, t)
                viol = violation(params).detach()
                training_evaluation['objective'].append(loss.detach())
                training_evaluation['elbo'].append(elbo.detach())
                training_evaluation['violation'].append(viol)

                if compute_held_out_loglik_id:
                    training_evaluation['held_out_ll_indist'].append(
                        held_out_loglikelihood(X_v_id, Y_v_id, params.detach()))

                if compute_held_out_loglik_ood:
                    training_evaluation['held_out_ll_outofdist'].append(
                        held_out_loglikelihood(X_v_ood, Y_v_ood, params.detach()))

                if compute_RMSE_id:
                    rmse_id_cache = compute_rmse(
                        X_v_id, Y_v_id, params.detach())
                    training_evaluation['rmse_id'].append(rmse_id_cache)

                if compute_RMSE_ood:
                    training_evaluation['rmse_ood'].append(
                        compute_rmse(X_v_ood, Y_v_ood, params.detach()))

                # command line printing
                str = 'Step {:7}  ---  Objective: {:15}  ELBO (first-order): {:15}  Violation: {:10}'.format(
                    t, round(loss.item(), 4), round(elbo.item(), 4), round(viol.item(), 4))

                if compute_RMSE_id:
                    str += '   ID-RMSE {:10}'.format(
                        round(rmse_id_cache.item(), 4))

                print(str)

        return params.detach(), loss.detach(), training_evaluation



    '''Runs multiple restarts'''
    def run_all(constrained):

        # choose alg
        algs = {
            'bbb' : run_bbb,
            'npv': run_npv,
        }
        code = experiment['vi']['alg']
        run_alg = algs[code]

        # specific settings
        if constrained:
            restarts = restarts_constr
            cores = cores_constr
        else:
            restarts = restarts_regular
            cores = cores_regular


        if multithread:
            
            # TODO multithreading via pytorch


            core_use = min(cores, mp.cpu_count())
            p = Pool(core_use)
            print('Cores used: {} Cores available: {}'.format(
                core_use, mp.cpu_count()))

            params, best_objectives, training_evaluations = [], [], []
            for param, obj, eval in p.map(lambda r: run_alg(r, constrained), range(restarts)):
                params.append(param)
                best_objectives.append(obj)
                training_evaluations.append(eval)


        else:
            print('Not multithreading.')
            
            params, best_objectives, training_evaluations = [], [], []
            for param, obj, eval in map(lambda r: run_alg(r, constrained), range(restarts)):
                params.append(param)
                best_objectives.append(obj)
                training_evaluations.append(eval)

        
        best = torch.min(torch.tensor(best_objectives), 0)[1]

        print('Best objective: {}'.format(best_objectives[best]))

        # store results
        str = 'constrained_BbB' if constrained else 'regular_BbB'
        joblib.dump((best, params, training_evaluations),
                    current_directory + '/optimization_data_results' + '/' + str + '_data.pkl')

        # print(params)
        return best, params, training_evaluations, str

    # run both experiments
    both_runs = []

    if regular_BbB:
        both_runs.append(run_all(False))

    if constrained_BbB:
        both_runs.append(run_all(True))

    return both_runs, forward, current_directory


'''
Computes mass of posterior predictive in constrained region
(i.e. independent of constraint function parameters)

This is an interpretable evaluation metric rather than the violation part of the objective.
'''

def compute_posterior_predictive_violation(params, prediction, experiment):

    S = experiment['vi']['posterior_predictive_analysis']['posterior_samples']
    T = experiment['vi']['posterior_predictive_analysis']['constrained_region_samples_for_pp_violation']

    constrained_region_sampler = experiment['vi']['constrained']['constrained_region_sampler']
    integral_constrained_region = experiment['data']['integral_constrained_region']

    constr = experiment['constraints']['constr']

    violations = []

    # for each random restart
    for j, param in enumerate(params):


        mean, log_std = param[:, 0], param[:, 1]

        '''Collect samples from optimized variational distribution'''
        ws = mean + torch.randn(S, param.shape[0]) *  log_std.exp() # torch.log(1.0 + log_std.exp()) #

        '''Integral of posterior predictive over total constrained region, evaluation metric'''

        # 1 - find random x samples form constrained region (via passed in sampling function)
        #     and approximation of area of constrained x region

        all_mc_points = constrained_region_sampler(T).unsqueeze(1)

        # 2 - approximate integral using monte carlo 

        integral = 0
        all_x_mc_points = []
        all_y_mc_points = []
        mc_points_color = []

        for x_ in all_mc_points:
            

            # 2.1 - sample ys from p(y' | x', X, Y) using MC samples of W
            ys = prediction(ws, x_)

            # 2.2 - approximate mass in constrained region by ys that satisfy constraint
            ys_violated = 0

            for y_ in ys:
                
                polytopes_violated = []
                for region in constr:

                    polytopes_violated.append(all(
                        [c_x(x_, y_) <= 0 for c_x in region]))

                if any(polytopes_violated):
                    ys_violated += 1

                    all_x_mc_points.append(x_)
                    all_y_mc_points.append(y_)
                    mc_points_color.append('red' if any(
                        polytopes_violated) else 'green')

            mass_violated = ys_violated / ys.shape[0]
            integral += ((1 / T) * mass_violated) * \
                integral_constrained_region

        violations.append(integral)

    return violations


'''
Make unique directory for results
'''
def make_unique_dir(experiment):
    directory = 'experiment_results/' + experiment['title'] + '_v'
    j = 0
    while os.path.exists(directory + str(j)):
        j += 1
    current_directory = directory + str(j)
    os.makedirs(current_directory)

    joblib.dump(experiment, current_directory +
            '/experiment_settings_dict.pkl')
    
    q_param_directory = current_directory + '/optimization_data_results'
    os.makedirs(q_param_directory)

    return current_directory
