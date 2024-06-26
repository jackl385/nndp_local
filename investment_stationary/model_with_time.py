from typing import Callable
import jax 
import jax.numpy as jnp
from jax._src.basearray import Array
from jax.tree_util import Partial
from jax._src.prng import PRNGKeyArray
from jax import random

# Economic parameters used in functions
T = 10 # number of periods: t=0, ...,T where death occurs at T
alpha = 1.0/3.0 # capital share
delta = 0.1 # depreciation
r = 0.04 # interest rate
beta = 1/(1+r) # discount factor

rho_z = 1 # persistence of TFP
sigma_z = 0 # TFP shock variance

z_bound = [-0.6, 0.6] # bound for TFP
k_bound = [0, 15] # bound for capital

@jax.jit
def u(state:Array, action:Array) -> Array:
    '''
    Reward function
    '''
    t, z, k = state[...,0], state[...,1], state[...,2]
    k_next = action[...,0]
    return ((1/(1+r))**t)*(jnp.exp(z)*k**alpha - (k_next-(1-delta)*k))


@jax.jit
def m(key:jax.random.PRNGKey, state:Array, action:Array) -> Array:
    '''
    State evolution equation
    '''
    t, z = state[...,0], jnp.atleast_1d(state[...,1])
    k_next = action[...,0]
    t_next = t + 1
    z_next = rho_z * z + sigma_z * jax.random.normal(key, shape = (len(z),))
    return jnp.column_stack([t_next, z_next, k_next])


@jax.jit
def v_T(state:Array) -> Array:
    '''
    value function in death period t=T
    '''
    t = state[:,[0]]
    z = state[:,[1]]
    k = state[:,[2]]
    
    return ((1/(1+r))**T)*(jnp.exp(z) * k**alpha + (1 - delta)*k)


# @Partial(jax.jit,static_argnames='N')
# def F(key:PRNGKeyArray, N:int) -> Array:
#     '''
#     Alternative way to sample from N initial states. 
#     The original F() function does not work that well for regions where k is small. 
#     Fro small_k_frac fraction * N points, this function uses an exponential distribution to sample from a region where k is small.   
#     '''    
#     #sample for z
#     key, subkey = jax.random.split(key)
#     z = jax.random.uniform(subkey, shape = (N,1), minval=z_bound[0], maxval=z_bound[1])
        
#     # sample for k
#     small_k_frac = 3/4 #fraction of points to sample in the small region
#     exp_beta = 0.1 #parameter for exponential distribution
    
#     key, subkey = jax.random.split(key)
#     k_small = jax.random.exponential(subkey, shape = (int(N * small_k_frac),1))
#     k_small = (1 / exp_beta) * (k_small ** (1 / exp_beta))
   
#     key, subkey = jax.random.split(key)
#     k_large = jax.random.uniform(subkey, shape = (int(N*(1-small_k_frac)),1), minval=2.5, maxval=k_bound[1])
    
#     k = jnp.vstack([k_small, k_large])
    
#     # append and return state variables
#     state = jnp.column_stack([z, k])
#     return state


@Partial(jax.jit, static_argnames='N')
def F(key:jax.random.PRNGKey, N:int) -> Array:
    '''
    Sample N initial states
    '''
    key, *subkey = jax.random.split(key, 4)

    t = jax.random.randint(subkey[0], shape = (N,), minval=0, maxval=T)
    z = jax.random.uniform(subkey[1], shape = (N,), minval=z_bound[0], maxval=z_bound[1])
    k = jax.random.uniform(subkey[2], shape = (N,), minval=k_bound[0], maxval=k_bound[1])
    
    return jnp.column_stack([t, z, k])


@jax.jit
def nn_to_action(state: jnp.ndarray, 
                 params: dict,
                 nn: Callable[[dict, jnp.ndarray], jnp.ndarray]
                 ) -> jnp.ndarray:
    '''
    Defines how a Haiku Neural Network, nn, with parameters, params, is mapped
    into an action.
    
    Parameters:
    -----------
    state: current state = N_simul x n_states
    nn: Haiku Neural Network with signature nn(params, state)
    params: dictionary of parameters used by nn. 

    Returns:
    -----------
    action: action to take = N_simul x n_actions.
    '''

    t, z, k = state[:, 0], state[:, 1], state[:, 2]
    state_norm = jnp.column_stack([t, z, k])
    
    action = nn(params, state_norm)
    return action