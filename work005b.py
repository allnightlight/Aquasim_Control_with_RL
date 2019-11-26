
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Model, Sequential
import tensorflow_probability as tfp

Ns = 1
Ny = 1
Nu = 1
Nhidden = 2**3
s2 = (0.1)**2
Nstep = 2**3
gamma = 0.0
lr_optimizer = 1e-2

assert Nu == Ns
assert Ny == Ns

def env_init():
	s = np.zeros((1, Ns)).astype(np.float32) # (1, Ns)
	return s # (1, Ns)
	
def env_reward(s):
	sv = np.ones((1,Ns)).astype(np.float32) # (1, Ns)
	r = -np.sum(np.abs(s - sv), axis=1, keepdims = True).astype(np.float32)  
	# (1, 1)
	return r # (1, 1)

def env_observe(s):
	y = s
	return y # (1, Ny)

def env_step(s, u):
	alpha_env = 0.1
	s_next = (1-alpha_env) * s + alpha_env * u
	return s_next # (1, Ns)

def agent_init(y):
	u = np.zeros((1, Nu)).astype(np.float32) # (1, Nu)

	net_pi = Sequential()
	net_pi.add(Dense(Nu, activation='linear', 
		kernel_initializer=tf.zeros_initializer()))

	net_v = Sequential()
	net_v.add(Dense(Nhidden, activation='linear'))
	net_v.add(LeakyReLU())
	net_v.add(Dense(1, activation='linear'))

	optimizer_actor = tf.keras.optimizers.RMSprop(lr=lr_optimizer)
	optimizer_critic = tf.keras.optimizers.RMSprop(lr=lr_optimizer)

	return u, net_pi, net_v, optimizer_actor, optimizer_critic

def agent_step(y, net_pi):
# y: (1, Ny)

	_y = tf.constant(y) # (1, Ny)
	_mu = net_pi(_y) # (1, Nu)
	mu = _mu.numpy() # (1, Nu)
	u = np.random.randn(1, Nu) * np.sqrt(s2) + mu # (1, Nu)
	u = u.astype(np.float32) # (1, Nu)

	return u

def actor_step(y, u, y_next, u_next, r, net_pi, net_v):
	# y, y_next: (1, Ny), u, u_next: (1, Nu), r: (1, 1)

	_y = tf.constant(y) # (1, Ny)
	_y_next = tf.constant(y_next) # (1, Ny)

# actor
# a(s,u) = (1-gamma) * r(s,u) + gamma * _v(s+) - v(s)
	_v = net_v(_y) # (1, 1) as V(s)
	_v_next = net_v(_y_next) # (1, 1) as V(s+)
	_a = (1-gamma) * r + gamma * _v_next - _v # (1,1)
	a = _a.numpy() # (1,1)

	with tf.GradientTape() as gtape:
		_mu = net_pi(_y) # (1, Nu)
		_dist = tfp.distributions.Normal(loc = _mu, 
			scale = np.sqrt(s2).astype(np.float32))
		_loss_actor = -a * _dist.log_prob(u)
	_grad_actor = gtape.gradient(_loss_actor, net_pi.trainable_variables)

	return _grad_actor

def critic_step(y, u, y_next, u_next, r, net_pi, net_v):
	# y, y_next: (1, Ny), u, u_next: (1, Nu), r: (1, 1)

	_y = tf.constant(y) # (1, Ny)
	_y_next = tf.constant(y_next) # (1, Ny)

# critic
# loss = (1-gamma) * r(s,u) + gamma * v(s+) - v(s,u)
	with tf.GradientTape() as gtape:
		_v = net_v(_y) # (1, 1) as V(s)
		_v_next = net_v(_y_next) # (1, 1) as V(s+)
		_delta = (1-gamma) * r + gamma * _v_next - _v # (1,1)
		_loss = tf.reduce_mean((_delta**2)/2)
	_grad_critic = gtape.gradient(_loss, net_v.trainable_variables)

	return _grad_critic

if __name__ == "__main__":

# @ Env
	s = env_init() # (1, Ns)
	y = env_observe(s) # (1, Ny)

# @ Agent
	u, net_pi, net_v, optimizer_actor, optimizer_critic = agent_init(y) 
# u  (1, Nu)

	for k1 in range(Nstep):
		print(s,u)
# @ Env
		s_next = env_step(s, u) # (1, Ns)
		r = env_reward(s_next) # (1, 1)
		y_next = env_observe(s_next) # (1, Ny)

# @ Agent
		u_next = agent_step(y_next, net_pi) # (1, Nu)

# update actor
		if k1 > 2**10:
			_grad_actor = actor_step(y, u, y_next, u_next, r, net_pi, net_v)
			optimizer_actor.apply_gradients(
				zip(_grad_actor, net_pi.trainable_variables))

# update critic
		_grad_critic = critic_step(y, u, y_next, u_next, r, net_pi, net_v)
		optimizer_critic.apply_gradients(
			zip(_grad_critic, net_v.trainable_variables))

# update (s, u)
		s = s_next
		u = u_next
		y = y_next
