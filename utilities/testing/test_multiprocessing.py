# Just trying to get a hang of multiprocessing
import os
import time
import numpy as np
import multiprocessing
from multiprocessing import Pool, Process
import threading

def f(x):
    np.random.seed(20)
    print(f'Process id: {os.getpid()}')
    time.sleep(1.0)
    # return x*x
    return np.random.rand()

N = 10
start = time.time()
if __name__ == '__main__':
    n_procs = 2
    with Pool(n_procs) as p:
        x = p.map(f, [n for n in range(N)])
        #print(p.map(f, [n for n in range(N)]))
    print(f'Time with 2 processes: {time.time() - start}')
    
    start = time.time()
    n_procs = 10
    with Pool(n_procs) as p:
        x = p.map(f, [n for n in range(N)])
        #print(p.map(f, [n for n in range(N)]))
    print(f'Time with 10 processes: {time.time() - start}')

    print(x)



# if __name__ == '__main__':
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()