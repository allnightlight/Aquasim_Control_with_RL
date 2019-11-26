
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
		kernel_initializer = tf.zeros_initializer()))

	net_q = Sequential()
	net_q.add(Dense(Nhidden, activation='linear'))
	net_q.add(LeakyReLU())
	net_q.add(Dense(1, activation='linear'))

	optimizer_actor = tf.keras.optimizers.RMSprop(lr=lr_optimizer)
	optimizer_critic = tf.keras.optimizers.RMSprop(lr=lr_optimizer)

	return u, net_pi, net_q, optimizer_actor, optimizer_critic

def agent_step(y, net_pi):
# y: (1, Ny)

	_y = tf.constant(y) # (1, Ny)
	_mu = net_pi(_y) # (1, Nu)
	mu = _mu.numpy() # (1, Nu)
	u = np.random.randn(1, Nu) * np.sqrt(s2) + mu # (1, Nu)
	u = u.astype(np.float32) # (1, Nu)

	return u

def actor_step(y, u, y_next, u_next, r, net_pi, net_q):
	# y, y_next: (1, Ny), u, u_next: (1, Nu), r: (1, 1)

	_y = tf.constant(y) # (1, Ny)
	_u = tf.constant(u) # (1, Nu)
	_y_next = tf.constant(y_next) # (1, Ny)
	_u_next = tf.constant(u_next) # (1, Nu)

# actor
	_q = net_q(tf.concat((_y, _u), axis=1)) # (1, 1) as Q(s,u)

	Nsample = 2**5

	_mu = net_pi(_y) # (1, Nu)
	mu = _mu.numpy()  # (1, Nu)
	u_sample = np.random.randn(Nsample, Nu) * np.sqrt(s2) + mu
	u_sample = u_sample.astype(np.float32) # (*, Nu)
	_u_sample = tf.constant(u_sample) # (*, Nu)
	_q_sample = net_q(tf.concat((_y * tf.ones((Nsample, 1)), _u_sample), 
		axis=1)) # (*, 1)
	_v = tf.reduce_mean(_q_sample, axis=0, keepdims=True) 
	# (1, 1), as V(s)

	_a = _q - _v # (1,1)
	a = _a.numpy() # (1,1), as A(s,u)

	with tf.GradientTape() as gtape:
		_mu = net_pi(_y) # (1, Nu)
		_dist = tfp.distributions.Normal(loc = _mu, 
			scale = np.sqrt(s2).astype(np.float32))
		_loss_actor = -a * _dist.log_prob(u)
	_grad_actor = gtape.gradient(_loss_actor, net_pi.trainable_variables)

	return _grad_actor

def critic_step(y, u, y_next, u_next, r, net_pi, net_q):
	# y, y_next: (1, Ny), u, u_next: (1, Nu), r: (1, 1)

	_y = tf.constant(y) # (1, Ny)
	_u = tf.constant(u) # (1, Nu)
	_y_next = tf.constant(y_next) # (1, Ny)
	_u_next = tf.constant(u_next) # (1, Nu)

# critic
	with tf.GradientTape() as gtape:
		_q = net_q(tf.concat((_y, _u), axis=1)) # (1, 1) as Q(s,u)
		_q_next = net_q(tf.concat((_y_next, _u_next), axis=1)) 
		# (1, 1) as Q(s+,u+)
		_loss = tf.reduce_mean(((1-gamma)*r + gamma * _q_next - _q)**2/2)
	_grad_critic = gtape.gradient(_loss, net_q.trainable_variables)

	return _grad_critic

if __name__ == "__main__":

# @ Env
	s = env_init() # (1, Ns)
	y = env_observe(s) # (1, Ny)

# @ Agent
	u, net_pi, net_q, optimizer_actor, optimizer_critic = agent_init(y) 
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
		_grad_actor = actor_step(y, u, y_next, u_next, r, net_pi, net_q)
		optimizer_actor.apply_gradients(
			zip(_grad_actor, net_pi.trainable_variables))

# update critic
		_grad_critic = critic_step(y, u, y_next, u_next, r, net_pi, net_q)
		optimizer_critic.apply_gradients(
			zip(_grad_critic, net_q.trainable_variables))

# update (s, u)
		s = s_next
		u = u_next
		y = y_next
