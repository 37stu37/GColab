'''
test hydrogen, tqdm and numba
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


#%%
# test fucntionality tqdm
def tqdm_test(x):
    y = (x*1000)
    for i in trange(x, desc="outer loop"):
        for j in range(y):
            time.sleep(0.001)


#%%
if __name__ == '__main__':
   tqdm_test(2)

#%%
# test speed numba

import random
from numba import jit

@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

def monte_carlo_pi_no_numba(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

#%%
%time monte_carlo_pi_no_numba(10000)
#%%
%time monte_carlo_pi(10000)
%time monte_carlo_pi(10000)
#%%
from numba import jit, njit, types, vectorize
#%%
def sane_function(input_list):
    output_list = []
    for item in input_list:
        if item % 2 == 0:
            output_list.append(2)
        else:
            output_list.append(1)
    return output_list

test_list = list(range(100000))

#%%
%time sane_function(test_list)[0:5]
#%%
njitted_sane_function = njit()(sane_function)
%time njitted_sane_function(test_list)[0:5]
#%%
# speed up ...
import numpy as np
test_list = np.arange(100000)
%time njitted_sane_function(test_list)[0:5]
#%%
@vectorize(nopython=True)
def non_list_function(item):
    if item % 2 == 0:
        return 2
    else:
        return 1

# This allows us to write a function to operate on a single element, but then call it on a list!
#%%
%time non_list_function(test_list)
%time non_list_function(test_list)
#%%
# more complex function
@njit(nogil=True)
def friction_fn(v, vt):
    if v > vt:
        return - v * 3
    else:
        return - vt * 3 * np.sign(v)

@njit(nogil=True)
def simulate_spring_mass_funky_damper(x0, T=10, dt=0.0001, vt=1.0):
    times = np.arange(0, T, dt)
    positions = np.zeros_like(times)

    v = 0
    a = 0
    x = x0
    positions[0] = x0/x0

    for ii in range(len(times)):
        if ii == 0:
            continue
        t = times[ii]
        a = friction_fn(v, vt) - 100*x
        v = v + a*dt
        x = x + v*dt
        positions[ii] = x/x0
    return times, positions

# compile
_ = simulate_spring_mass_funky_damper(0.1)

# plot
plt.plot(*simulate_spring_mass_funky_damper(0.1))
plt.plot(*simulate_spring_mass_funky_damper(1))
plt.plot(*simulate_spring_mass_funky_damper(10))
plt.legend(['0.1', '1', '10'])
