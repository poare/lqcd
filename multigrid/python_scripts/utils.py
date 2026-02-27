
import numpy as np

def read_eval_txt(file):
    """Reads eigenvalue input from a file. Format for each line is `idx eval ritzEstimate`. """
    evals = []
    ritzValues = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(' ')
            idx = int(tokens[0])

            # parse eigenvalue
            re, im = tokens[1].replace(')', '').replace('(', '').split(',')
            eval = float(re) + 1j*float(im)

            # parse ritz estimate (TODO)
            ritz = -1
            if len(tokens) > 2:
                ritz = float(tokens[2])

            # print output
            if ritz != -1:
                print(f'Eval {idx} = {eval}, ritz estimate = {ritz}')
                ritzValues.append(ritz)
            else:
                print(f'Eval {idx} = {eval}')
            evals.append(eval)
    return np.array(evals), np.array(ritzValues)

def read_chulwoo_file(file, all_evals = False):
    """Reads eigenvalue input from a file. Format for each line is `eval_re eval_im`. """
    evals = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            if all_evals:
                print(tokens)
                idx = int(tokens[0])
                if len(evals) == idx:
                    evals.append([])
                evals[idx].append(float(tokens[1]) + 1j*float(tokens[2]))
            else:
                evals.append(float(tokens[0]) + 1j*float(tokens[1]))
    if not all_evals:
        evals = np.array(evals)
    return evals