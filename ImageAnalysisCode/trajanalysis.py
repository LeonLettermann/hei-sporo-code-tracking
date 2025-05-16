# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import relu
from functools import partial

def Rx(theta):
  return jnp.array([[ 1, 0           , 0           ],
                   [ 0, jnp.cos(theta),-jnp.sin(theta)],
                   [ 0, jnp.sin(theta), jnp.cos(theta)]])
def Ry(theta):
  return jnp.array([[ jnp.cos(theta), 0, jnp.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-jnp.sin(theta), 0, jnp.cos(theta)]])
def Rz(theta):
  return jnp.array([[ jnp.cos(theta), -jnp.sin(theta), 0 ],
                   [ jnp.sin(theta), jnp.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
def rot_matrix(a1,a2,a3):
    return Rx(a1)@Ry(a2)@Rz(a3)

def helix(t, r0, a, R1, R2, rot, p, r, inverted):
    return r0 + jnp.einsum('ij,mj->mi', rot, p*t[:,jnp.newaxis]*a[jnp.newaxis]+r*(jnp.cos((1-2*inverted)*t)[:,jnp.newaxis]*R1[jnp.newaxis]-jnp.sin((1-2*inverted)*t)[:,jnp.newaxis]*R2[jnp.newaxis]))

@jax.jit
def helixloss(params,ts,a_guess, R1_guess, R2_guess, testtraj, inverted=False):
    helix_guess = helix(ts,params[2:5],a_guess,R1_guess,R2_guess,rot_matrix(*params[5:8]),params[0],params[1], inverted)
    return jnp.nansum(jnp.nanmin(jnp.linalg.norm(helix_guess[jnp.newaxis]-testtraj[:,jnp.newaxis], axis=-1), axis=1)**2)

@jax.jit
def helixloss_ts(params,ts,a_guess, R1_guess, R2_guess, testtraj, valid, inverted=False):
    helix_guess = helix(ts,params[2:5],a_guess,R1_guess,R2_guess,rot_matrix(*params[5:8]),params[0],params[1], inverted)
    return jnp.sum(jnp.linalg.norm(valid[:,jnp.newaxis]*jnp.nan_to_num(helix_guess-testtraj), axis=-1)**2) + 20*jnp.nansum(relu(-jnp.diff(ts)))

def trajPCA(traj):
    cov = np.cov(traj, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    sorting = np.argsort(eigvals)
    return eigvals[sorting], eigvecs[:,sorting]

def jaxtrajPCA(traj):
    eigvals, eigvecs = jnp.linalg.eigh(jnp.cov(jnp.nan_to_num(traj-jnp.nanmean(traj, axis=0)), rowvar=False))
    sorting = jnp.argsort(eigvals)
    return eigvals[sorting], eigvecs[:,sorting]

normalize = lambda x: x/jnp.linalg.norm(x, axis=-1)[...,jnp.newaxis]

span_pre, span_fit = 10, 5
span_diff = span_pre-span_fit

def min_valid_ind(x):
    return jnp.argmax(jnp.logical_not(jnp.isnan(x)))

def max_valid_ind(x):
    return len(x)-jnp.argmax(jnp.logical_not(jnp.flip(jnp.isnan(x))))-1

def estimate_a_helix(eigvals, eigvecs, testtraj, framedur):
    a_guess = eigvecs[:,2]
    a_guess = jnp.sign(jnp.dot(a_guess, testtraj[max_valid_ind(testtraj[:,0])]-testtraj[min_valid_ind(testtraj[:,0])]))*a_guess
    p_guess = 2.
    ts = np.linspace(-10,10,100)
    return a_guess, p_guess, ts

def estimate_a_circle(eigvals, eigvecs, testtraj, framedur):
    a_guess = eigvecs[:,0]
    p_guess = 0.
    ts = np.linspace(-5,5,100)
    return a_guess, p_guess, ts

def estimate_a_undet(eigvals, eigvecs, testtraj, framedur):
    a_guess = testtraj[max_valid_ind(testtraj[:,0])]-testtraj[min_valid_ind(testtraj[:,0])]
    p_guess = 1.
    a_guess = normalize(a_guess)
    ts = np.linspace(-7,7,100)
    return a_guess, p_guess, ts

@jax.jit
def est_helix_params(testtraj, framedur, inverted=False):
    """Estimate the parameters of a helix from a trajectory using PCA."""
    eigvals, eigvecs = jaxtrajPCA(testtraj)

    selector = 0 + (eigvals[0]<0.05) + 2*(eigvals[2]<10)*(eigvals[0]>0.05)

    a_guess, p_guess, ts = jax.lax.switch(selector, [estimate_a_helix,estimate_a_circle, estimate_a_undet], eigvals, eigvecs, testtraj, framedur)
    
    r0_guess = jnp.nanmean(testtraj,axis=0)
    t0_ind = jnp.nanargmin(jnp.linalg.norm(testtraj-r0_guess, axis=1))
    r_guess = 2*jnp.nanmin(jnp.linalg.norm(testtraj-r0_guess, axis=1))
    R1_guess = testtraj[t0_ind]-r0_guess
    R1_guess = normalize(R1_guess-jnp.dot(R1_guess, a_guess)*a_guess)
    R2_guess = normalize(jnp.cross(a_guess, R1_guess))
    guesshelix = helix(ts, r0_guess, a_guess, R1_guess, R2_guess, np.eye(3), p_guess, r_guess, inverted=inverted)
    ts_guess = jnp.linspace(ts[jnp.nanargmin(jnp.linalg.norm(guesshelix[:34]-testtraj[min_valid_ind(testtraj[:,0])][np.newaxis], axis=-1))],
                        ts[66+jnp.nanargmin(jnp.linalg.norm(guesshelix[-33:]-testtraj[max_valid_ind(testtraj[:,0])][np.newaxis], axis=-1))], len(testtraj))

    return r0_guess, a_guess, R1_guess, R2_guess, p_guess, r_guess, ts, ts_guess, guesshelix

@jax.jit
def _fit_traj_once(testtraj, traintraj, valid, framedur, inverted, epochs):
    """Fit a helix to a trajectory using gradient descent.
    This is one fit, either right or left handed.
    """
    r0_guess, a_guess, R1_guess, R2_guess, p_guess, r_guess, ts, ts_guess, guesshelix = est_helix_params(testtraj, framedur, inverted)
    params0 = jnp.array([p_guess, r_guess]+list(r0_guess)+3*[0])
    val_and_grad = jax.value_and_grad(helixloss_ts, argnums=(0,1))
    lr = 1e-3
    params=jnp.array(params0)
    ts_opt = ts_guess[span_diff:len(ts_guess)-span_diff]

    rotreduce = np.ones(len(params))

    args = 0, params, ts_opt, a_guess, R1_guess, R2_guess, traintraj, valid, inverted
    def body(i,args):
        lossval, grads = val_and_grad(*args[1:])
        grad, grad_ts = grads
        params = args[1] - lr*grad*rotreduce
        ts_opt = args[2] - lr*grad_ts
        return (lossval, params, ts_opt, *args[3:])
    
    lossval, params, ts_opt, a_guess, R1_guess, R2_guess, traintraj, valid, inverted = jax.lax.fori_loop(0,epochs, body, args)

    reshelix = helix(ts,params[2:5],a_guess,R1_guess,R2_guess, rot_matrix(*params[5:8]),params[0],params[1], inverted)
    reshelix_ts = helix(ts_opt,params[2:5],a_guess,R1_guess,R2_guess, rot_matrix(*params[5:8]),params[0],params[1], inverted)
    return lossval, params, ts_opt, guesshelix, reshelix, reshelix_ts, inverted

def _fit_traj_twice(testtraj, traintraj, valid, framedur, epochs):
    """Fit a helix to a trajectory using gradient descent.
    This is two fits, one right and one left handed.
    """
    res = _fit_traj_once(jnp.copy(testtraj), jnp.copy(traintraj), jnp.copy(valid), framedur, False, epochs)
    res_inv = _fit_traj_once(jnp.copy(testtraj), jnp.copy(traintraj), jnp.copy(valid), framedur, True, epochs)
    return jax.lax.cond(res[0]<res_inv[0], lambda: res, lambda: res_inv)

def _fit_traj_once_dummy(testtraj, traintraj, valid, framedur, epochs):
    return np.nan, np.nan*np.ones(8), np.nan*jnp.ones(len(traintraj)), np.nan*jnp.ones((100,3)), np.nan*jnp.ones((100,3)), np.nan*jnp.ones_like(traintraj), False

def fit_traj(sporo, center, biocents, epochs, framedur):
    """Fit a helix to a trajectory using gradient descent."""
    testtraj = jax.lax.dynamic_slice(jnp.pad(jax.lax.dynamic_index_in_dim(biocents, sporo, keepdims=False), ((span_pre, span_pre),(0,0)), constant_values=jnp.nan),(center,0),(2*span_pre+1,3))
    traintraj = testtraj[span_diff:len(testtraj)-span_diff]#[valid]
    valid = jnp.logical_not(jnp.isnan(traintraj[:,0]))
    selector = jnp.logical_or(jnp.isnan(testtraj[span_pre,0]), (jnp.sum(valid)<5))
    res = jax.lax.cond(selector, _fit_traj_once_dummy, _fit_traj_twice, testtraj, traintraj, valid, framedur, epochs)
    return res
        
def fit_all_traj(biocents, epochs, framedur):
    """Fit a helix to a trajectory using gradient descent."""
    sporo, center = np.arange(biocents.shape[0]),np.arange(biocents.shape[1])#8, 50
    res = jax.vmap(jax.vmap(fit_traj, in_axes=(None,0,None,None,None)), in_axes=(0,None,None,None,None))(sporo, center, biocents, epochs, framedur)
    return res



## This is experimental code to estimate geometric parameters of the sporozoite shape

def circle(t, r0, R1, R2, rot, r):
    return r0 + jnp.einsum('ij,mj->mi', rot, r*(jnp.cos(t)[:,jnp.newaxis]*R1[jnp.newaxis]-jnp.sin(t)[:,jnp.newaxis]*R2[jnp.newaxis]))

@jax.jit
def circleloss(params,R1_guess, R2_guess, points, valid):
    ts = np.linspace(-2,2,71)
    circle_guess = circle(ts,params[1:4],R1_guess,R2_guess,rot_matrix(*params[4:7]),params[0])
    return jnp.nanmean(valid*jnp.nanmin(jnp.linalg.norm(circle_guess[jnp.newaxis]-jnp.nan_to_num(points)[:,jnp.newaxis], axis=-1), axis=1)**2) + 1/6*jnp.linalg.norm(circle_guess[35]-jnp.nanmean(points,axis=0))**2

@jax.jit
def est_shape_circle_params(points):
    eigvals, eigvecs = jaxtrajPCA(points)

    a_guess = eigvecs[:,0]
    ts = jnp.linspace(-2,2,100)
    r0_guess = jnp.nanmean(points,axis=0)
    r_guess = 5
    #t0_ind = jnp.nanargmin(jnp.linalg.norm(points-r0_guess, axis=1))
    #R1_guess = points[t0_ind]-r0_guess
    #R1_guess = normalize(R1_guess-jnp.dot(R1_guess, a_guess)*a_guess)
    R1_guess = eigvecs[:,1]
    r0_guess = r0_guess# - r_guess*R1_guess
    R2_guess = normalize(jnp.cross(a_guess, R1_guess))
    guesscircle = circle(ts, r0_guess, R1_guess, R2_guess, np.eye(3), r_guess)
    #ts_guess = ts[jnp.argmin(jnp.linalg.norm(guesscircle[:,jnp.newaxis]-points[jnp.newaxis], axis=-1), axis=0)]
    return r0_guess, R1_guess, R2_guess, r_guess, ts, guesscircle

@jax.jit
def fit_circle(points, epochs, key):
    r0_guess, R1_guess, R2_guess, r_guess, ts, guesscircle = est_shape_circle_params(points)
    params0 = jnp.array([r_guess]+list(r0_guess)+list(2*np.pi*jax.random.uniform(key, (3,))))
    val_and_grad = jax.value_and_grad(circleloss, argnums=0)
    lr = 4e-1
    params=jnp.array(params0)

    rotreduce = np.ones(len(params))
    rotreduce[4:7]=0.1
    valid = jnp.logical_not(jnp.isnan(points[:,0]))

    args = 0, params, R1_guess, R2_guess, points, valid
    def body(i,args):
        lossval, grads = val_and_grad(*args[1:])
        params = args[1] - (lr*(epochs-i)/epochs+lr*i/1000/epochs)*grads*rotreduce
        return (lossval, params, *args[2:])
    
    lossval, params, R1_guess, R2_guess, points, valid = jax.lax.fori_loop(0,epochs, body, args)

    rescircle = circle(ts,params[1:4],R1_guess,R2_guess, rot_matrix(*params[4:7]),params[0])
    return lossval, params, guesscircle, rescircle

def fit_all_shapes(pointdata, epochs, key, multishot=5):
    keys = jax.random.split(key, (multishot*pointdata.shape[0]*pointdata.shape[1])).reshape((multishot, pointdata.shape[0], pointdata.shape[1],2))
    res = jax.vmap(jax.vmap(jax.vmap(fit_circle, in_axes=(0,None,0)), in_axes=(0,None,0)), in_axes=(None,None,0))(pointdata, epochs, keys)
    return res