
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Model, Sequential
import tensorflow_probability as tfp

import sqlite3
conn = sqlite3.connect('work007db.sqlite', 
    detect_types = sqlite3.PARSE_COLNAMES|sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()

import sys
sys.path.append('../../lib')

from work007module import *

def env_construct():
    env = {
        "t": 0,
        "f": 10,
        "amp": 0.1,
        "y": 1 * np.ones((1,1)).astype(np.float32), # (1,1) 
        "u_in": 1,
        "dt": 0.001,
    }
    return env

def env_init(env):
    env["t"] = 0
    env["y"] = 1 * np.ones((1,1)).astype(np.float32) # (1,1)

def env_observe(env):
    return env["y"]

def env_step(env, u):
    #u: (1,1) 
    u_out = u[0,0]
    if u[0,0] > 2:
        u_out = 2
    if u[0,0] < 0:
        u_out = 0

    dv = env["amp"] * env["f"] * 2 * np.pi * \
        np.sin(env["f"] * 2 * np.pi * env["t"])
    y = env["y"] + env["dt"] * (env["u_in"] - u_out + dv) # (1,1) 
    env["y"] = y # (1,1)
    env["t"] += env["dt"]

    is_violated = False
    r = -np.abs(y[0,0] - 1.0)
    if y[0,0] > 2:
        is_violated = True
        r += -1000. * np.abs(y[0,0] - 2.0)

    if y[0,0] < 0:
        is_violated = True 
        r += -1000. * np.abs(y[0,0])

    if u[0,0] > 2:
        is_violated = False
        r += -1000. * np.abs(u[0,0] - 2)

    if u[0,0] < 0:
        is_violated = False
        r += -1000. * np.abs(u[0,0])

    return r, is_violated


def agent_construct(Nu, Nz, Nhidden, lr_actor, lr_critic):

    agent = Agent(Nz, Nu, Nhidden)

    optimizer_actor  = tf.keras.optimizers.RMSprop(lr = lr_actor)
    optimizer_critic = tf.keras.optimizers.RMSprop(lr = lr_actor)

    return agent, optimizer_actor, optimizer_critic

def agent_init(Nz):
    z = np.zeros((1, Nz)).astype(np.float32) # (1, Nz)
    return z

def agent_step(y, z, agent, randomstate):
# y: (1, Ny), z: (1, Nz)

    _y = tf.constant(y) # (1, Ny)
    _z = tf.constant(z) # (1, Ny)
    _z_next = agent.step(_y, _z) # (1, Nz)
    _mu = agent.action(_z_next) # (1, Nu)
    mu = _mu.numpy() # (1, Nu)
    u = randomstate.randn(1, Nu) * np.sqrt(s2) + mu # (1, Nu)
    u = u.astype(np.float32) # (1, Nu)
    z_next = _z_next.numpy() # (1, Nz)

    return z_next, u

def nstep_train(Y, Z, U, R, agent, gamma, s2):
# Y: (N+1, Ny)
# Z: (N+1, Nz)
# U: (N, Nu)
# R: (N, 1)

    N = Y.shape[0] - 1

    _Y = tf.constant(Y) # (N+1, Ny)
    _Z = tf.constant(Z) # (N+1, Ny)

    with tf.GradientTape() as gtape:
        _V = agent.value(agent.step(_Y, _Z)) # (N+1, 1)
        _gamma_V = gamma ** np.arange(N+1).reshape(-1, 1) * _V # (N+1, 1)
        gamma_R = gamma ** np.arange(N).reshape(-1,1) * R # (N, 1)

        A = []
        for i in range(N):
            _a = (1-gamma) * np.sum(gamma_R[i:, 0]) + _gamma_V[-1,0] - \
                _gamma_V[i,0] # (,)
            A.append(_a)
        _A = tf.stack(A) # (N,)
        _loss = tf.reduce_mean(_A ** 2) # (,)
        A_numpy = _A.numpy() # (N,)
    _grad_critic = gtape.gradient(_loss, 
        agent.Z2Z.trainable_variables + agent.Z2V.trainable_variables)

    with tf.GradientTape() as gtape:
        _Mu = agent.action(agent.step(_Y[:-1,:], _Z[:-1,:])) # (N+1, Nu)
        dist = tfp.distributions.Normal(loc = _Mu, 
            scale = np.sqrt(s2).astype(np.float32))
        _LL_PI = dist.log_prob(U) # (N, Nu)
        _loss = -tf.reduce_sum(_LL_PI * A_numpy) # (,)
    _grad_actor = gtape.gradient(_loss, 
        agent.Z2Z.trainable_variables + agent.Z2Mu.trainable_variables)

    return _grad_actor, _grad_critic


if __name__ == "__main__":

# initlize env and agent
    env = env_construct()

    Ny = 1
    Nu = 1

    Nz        = 2**3
    Nhidden   = 2**4
    lr_actor  = 1e-3
    lr_critic = 1e-3

    agent, optimizer_actor, optimizer_critic = \
        agent_construct(Nu, Nz, Nhidden, lr_actor, lr_critic)

# training
    s2      = (1e-1) ** 2
    Nbatch  = 20
    gamma   = 1 - 1/Nbatch
    Nstep   = 2**12
    randomstate = np.random.RandomState(seed = 1)
    Nepisode = 2**10

# DB
    cur.execute('''Select count(id) From Train''')
    cnt, = cur.fetchone()
    train_id = cnt + 1
    cur.execute('''\
        Insert Into Train (id, lr_actor, lr_critic, gamma, Nhidden) 
        values (?, ?, ?, ?, ?)''',
        (train_id, lr_actor, lr_critic, gamma, Nhidden))
# DB

    episode = 0
    while True:
# @Env
        env_init(env) 
        y = env_observe(env) # (1, Ny)

# @Agent
        z = agent_init(Nz) # (1, Nz)

        counter_batch = 0
        Y = [y,]
        Z = [z,]
        U = []
        R = []
        for k1 in range(Nstep):
#
            z_next, u = agent_step(y, z, agent, randomstate)
# z_next: (1, Nz), u: (1, Nu)

# @ Env
            r, is_violated = env_step(env, u) # (,)
            y_next = env_observe(env) # (1, Ny)
# DB
            cur.execute('''\
                Insert Into History (train_id, epi, t, y, u) 
                Values (?, ?, ?, ?, ?)''',
                (train_id, episode, env["t"], float(y), float(u)))
            if k1 % 2**7 == 0 or k1 == Nstep - 1 or is_violated:
                conn.commit()
# DB
            sys.stdout.write(
                '\r episode %04d step %04d V %10.2f Qout %10.2f' 
                % (episode, k1, y[0,0], u[0,0]))
            if is_violated:
                print('some condtions are violated')

# train
            counter_batch += 1
            Y.append(y_next)
            Z.append(z_next)
            U.append(u)
            R.append(r)
            if counter_batch < Nbatch and not is_violated:
                pass
            else:
                Y_n = np.concatenate(Y, axis=0) # (Nbatch+1, Ny)
                Z_n = np.concatenate(Z, axis=0) # (Nbatch+1, Nz)
                U_n = np.concatenate(U, axis=0) # (Nbatch, Nu)
                R_n = np.stack(R, axis=0).reshape(-1,1) # (Nbatch, 1)

                _grad_actor, _grad_critic = nstep_train(Y_n, Z_n, U_n, 
                    R_n, agent, gamma, s2)

                optimizer_actor.apply_gradients(zip(_grad_actor, 
                    agent.Z2Z.trainable_variables + 
                    agent.Z2Mu.trainable_variables))
                optimizer_critic.apply_gradients(zip(_grad_critic, 
                    agent.Z2Z.trainable_variables + 
                    agent.Z2V.trainable_variables))

                counter_batch = 0
                Y.clear()
                Z.clear()
                U.clear()
                R.clear()

                Y.append(y_next)
                Z.append(z_next)

# update (s, u)
            if is_violated:
                break # end of this episode
            else:
                y = y_next
                z = z_next
        episode += 1

