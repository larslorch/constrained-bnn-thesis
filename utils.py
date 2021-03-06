import inspect
import pprint
import torch
import os
import joblib

# stable computation using tanh

def sig(z, tau_c):
    return 0.5 * (torch.tanh(- tau_c * z) + 1) 

def psi(z, tau_c, tau_g):
    return 0.25 * (torch.tanh(- tau_c * z) + 1) * (torch.tanh(- tau_g * z) + 1)


'''
Make unique directory for results
'''
def make_unique_dir(experiment, method='vi'):
    directory = 'experiment_results/' + experiment['title'] + '_v'
    j = 0
    while os.path.exists(directory + str(j)):
        j += 1
    current_directory = directory + str(j)
    os.makedirs(current_directory)

    joblib.dump(experiment, current_directory +
            '/experiment_settings_dict.pkl')
    
    if method == 'vi':
        q_param_directory = current_directory + '/' + method
        os.makedirs(q_param_directory)

    if method == 'hmc':
        hmc_directory = current_directory + '/' + method
        os.makedirs(hmc_directory)

    return current_directory



'''
Converts most important info of experiment to string for logfile
'''

def experiment_to_string(experiment):

    # fields of experiment dict to be printed
    regular_BbB = [
        ('Iterations', experiment['vi']['regular']['iterations']),
        ('BbB r.v. samples for grad.',
         experiment['vi']['rv_samples']),
    ]

    constrained_BbB = [
        ('Iterations', experiment['vi']['constrained']['iterations']),
        ('BbB r.v. samples for gradient',
         experiment['vi']['rv_samples']),
        ('Gamma', experiment['vi']['constrained']['gamma']),
        ('Tau', experiment['vi']['constrained']['tau_tuple']),
    ]

    nn = [
        ('Architecture', experiment['nn']['architecture']),
        ('Nonlinearity', experiment['nn']['nonlinearity']),
    ]

    data = [
        ('Input data X', experiment['data']['X']),
        ('Output data Y', experiment['data']['Y']),
        ('ID test set', experiment['data']['X_v_id']),
        ('OOD test set', experiment['data']['X_v_ood']),
    ]

    constr = [
        ('Constraints for regions', experiment['constraints']['constr']),
    ]

    # do not change below
    s = '-' * 70 + '\n--- EXPERIMENT SETUP ---'
    for title, l in [('Regular BbB', regular_BbB),
                     ('Constrained BbB', constrained_BbB),
                     ('Constraints in (x space, y-space)', constr),
                     ('NN', nn),
                     ('Data', data),
                     ]:

        s += '\n' + '-' * 70 + "\n\n- {}:\n".format(title)
        for descr, elt in l:
            s += '\n' + descr + ':\n'
            if callable(elt):
                # function
                try:
                    str = inspect.getsource(elt)
                except:
                    str = 'custom torch.autograd.funtion' # this happens for relu
                s += str + '\n'
            else:
                # constraints
                if type(elt) is list and elt != []:
                    if type(elt[0]) is tuple:
                        for l1, l2 in elt:
                            s += 'X constraints: {}\nY constraints:{}\n\n'.format(
                                l1, l2)
                # else
                else:
                    s += '{}\n'.format(elt)

    s += '\n\n' + '-' * 70 + '\n' + '-' * 70 + '\n\n --- Full dictionary ---\n\n' + \
        pprint.pformat(experiment, indent=1)
    return s

