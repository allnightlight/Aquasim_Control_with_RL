
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Model, Sequential
import tensorflow_probability as tfp

import unittest


class Agent(Model):
    def __init__(self, Nz, Nu, Nhidden):
        super(Agent, self).__init__()
        self.Nz = Nz

        self.Z2Z = Sequential()
        self.Z2Z.add(Dense(Nz, activation = "tanh"))

        self.Z2V = Sequential()
        self.Z2V.add(Dense(Nhidden, activation = "linear"))
        self.Z2V.add(LeakyReLU())
        self.Z2V.add(Dense(1, activation='linear',
            kernel_initializer = tf.zeros_initializer()))

        self.Z2Mu = Sequential()
        self.Z2Mu.add(Dense(Nhidden, activation = "linear"))
        self.Z2Mu.add(LeakyReLU())
        self.Z2Mu.add(Dense(Nu, activation='linear',
            kernel_initializer = tf.zeros_initializer()))

    def step(self, _Y, _Z):
        # _Y: (*, Ny)
        # _Z: (*, Nz)
        _Z_next = self.Z2Z(tf.concat((_Y, _Z), axis=1)) # (*, Nz)
        return _Z_next # (*, Nz)

    def value(self, _Z):
        # _Z: (*, Nz)
        _V = self.Z2V(_Z) # (*, 1)
        return _V

    def action(self, _Z):
        # _Z: (*, Nz)
        _Mu = self.Z2Mu(_Z) # (*, Nu)
        return _Mu


class TestCase(unittest.TestCase):
    def test_001(self):
        Nz          = 2**1
        Nu          = 2**3
        Nhidden     = 2**3
        Ny          = 2
        Nbatch      = 2**5

        agent = Agent(Nz, Nu, Nhidden)

        _Y = tf.random.normal((Nbatch, Ny)) # (*, Ny)
        _Z = tf.random.normal((Nbatch, Nz)) # (*, Ny)

        _Z_next = agent.step(_Y, _Z)
        _V = agent.value(_Z_next)
        _Mu = agent.value(_Z_next)


if __name__ == "__main__":
    unittest.main()
    pass
