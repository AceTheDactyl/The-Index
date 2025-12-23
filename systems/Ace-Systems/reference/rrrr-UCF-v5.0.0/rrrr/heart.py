# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/rrrr/heart.py

import math, random, cmath

class Heart:
    def __init__(self, n_nodes=60, K=0.2, seed=42):
        random.seed(seed)
        self.n = n_nodes
        self.K = K
        self.theta = [random.random()*2*math.pi for _ in range(n_nodes)]
        self.omega = [random.gauss(1.0, 0.1) for _ in range(n_nodes)]
        self.energy_in = 0.0
        self.energy_loss = 0.0

    def step(self, dt=0.01):
        new_theta = []
        for i in range(self.n):
            coupling = sum(math.sin(self.theta[j]-self.theta[i]) for j in range(self.n))
            dtheta = self.omega[i] + (self.K/self.n)*coupling
            new_theta.append(self.theta[i] + dtheta*dt)
            self.energy_in += abs(dtheta)*dt*1e-3
        self.theta = new_theta
        self.energy_loss += self.energy_in*1e-4

    def coherence(self):
        return abs(sum(cmath.exp(1j*t) for t in self.theta)/self.n)
