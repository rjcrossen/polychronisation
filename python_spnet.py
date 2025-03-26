import numpy as np

class IzhikevichNetwork:
    def __init__(self, Ne=800, Ni=200, M=100):
        self.Ne = Ne
        self.Ni = Ni
        self.N = Ne + Ni
        self.M = M  # synapses per neuron
        self.D = 20  # max delay
        
        # Neuron parameters
        self.a = np.concatenate([0.02 * np.ones(Ne), 0.1 * np.ones(Ni)])
        self.d = np.concatenate([8 * np.ones(Ne), 2 * np.ones(Ni)])
        
        # Neuron state variables
        self.v = -65 * np.ones(self.N)
        self.u = self.v * 0.2
        
        # Create post-synaptic target matrix
        self.post = np.zeros((self.N, M), dtype=int)
        self.s = np.zeros((self.N, M))  # synaptic weights
        self.sd = np.zeros((self.N, M))  # synaptic derivatives
        
        # Initialize delays dictionary
        self.delays = {}
        for i in range(self.N):
            self.delays[i] = {d: [] for d in range(1, self.D+1)}
        
        # Create random connections
        for i in range(Ne):
            # Random postsynaptic targets (no self-connections)
            p = np.random.permutation(self.N)
            p = p[p != i]  # remove self-connection
            self.post[i] = p[:M]
            
            # Random delays for excitatory neurons
            for j in range(M):
                delay = np.random.randint(1, self.D+1)
                self.delays[i][delay].append(j)
            
            # Set excitatory weights
            self.s[i] = 6
        
        # Create inhibitory connections
        for i in range(Ne, self.N):
            p = np.random.permutation(Ne)
            self.post[i] = p[:M]
            self.delays[i][1] = list(range(M))  # all inhibitory delays are 1ms
            self.s[i] = -5
        
        # Create reverse lookups for STDP
        self.pre = [[] for _ in range(self.N)]
        self.aux = [[] for _ in range(self.N)]
        
        for i in range(Ne):
            for d in range(1, self.D+1):
                for syn in self.delays[i][d]:
                    target = self.post[i, syn]
                    # Store both pre-synaptic neuron index and synapse index
                    self.pre[target].append((i, syn))
                    self.aux[target].append(self.D-1-d)
        
        # Convert to numpy arrays for faster access
        self.pre = [np.array(p) for p in self.pre]
        self.aux = [np.array(a) for a in self.aux]
        
        # STDP traces
        self.STDP = np.zeros((self.N, 1001 + self.D))
        
        # Spike timing storage
        self.firings = [(-self.D, 0)]  # [(time, neuron_idx), ...]
    
    def step(self, t, sec):
        # Initialize current
        I = np.zeros(self.N)
        I[np.random.randint(0, self.N)] = 20  # random thalamic input
        
        # Find neurons that have fired
        fired = np.where(self.v >= 30)[0]
        
        # Reset fired neurons
        self.v[fired] = -65
        self.u[fired] += self.d[fired]
        
        # Update STDP trace for fired neurons
        self.STDP[fired, t + self.D] = 0.1
        
        # Process fired neurons
        for k in fired:
            # Update synaptic derivatives based on STDP traces
            if k < self.Ne:  # only for excitatory neurons
                for pre_idx, syn_idx in self.pre[k]:
                    self.sd[pre_idx, syn_idx] += self.STDP[k, t + self.aux[k][0]]
        
        # Store firings
        if len(fired) > 0:
            self.firings.extend([(t, i) for i in fired])
        
        # Process earlier spikes: loop through recent firing history
        k = len(self.firings) - 1
        while k > 0 and self.firings[k][0] > t - self.D:
            spike_time, i = self.firings[k]
            if t - spike_time < self.D:  # if spike is still relevant
                d = t - spike_time
                if d >= 0:
                    for syn_idx in self.delays[i][d+1]:
                        target = self.post[i, syn_idx]
                        I[target] += self.s[i, syn_idx]
                        if i < self.Ne:  # only for excitatory neurons
                            self.sd[i, syn_idx] -= 1.2 * self.STDP[target, t + self.D]
            k -= 1
        
        # Update membrane potential and recovery variable
        self.v += 0.5 * ((0.04 * self.v + 5) * self.v + 140 - self.u + I)
        self.v += 0.5 * ((0.04 * self.v + 5) * self.v + 140 - self.u + I)
        self.u += self.a * (0.2 * self.v - self.u)
        
        # Decay STDP traces
        self.STDP[:, t + self.D + 1] = 0.95 * self.STDP[:, t + self.D]
        
        # Every second
        if t == 999:
            # Shift STDP traces
            self.STDP[:, :self.D] = self.STDP[:, 1001:1001+self.D]  # Fixed indexing here
            
            # Update synaptic weights for excitatory neurons
            self.s[:self.Ne] = np.clip(0.01 + self.s[:self.Ne] + self.sd[:self.Ne], 0, 10)
            
            # Decay synaptic derivatives
            self.sd *= 0.9
            
            # Reset firings, keeping only recent ones
            recent = [(t-1000, n) for t, n in self.firings if t > 1001-self.D]
            self.firings = [(-self.D, 0)] + recent
        
        return fired

    def run(self, duration_seconds=5):
        all_spikes = []
        for sec in range(duration_seconds):
            for t in range(1000):  # 1000ms = 1sec
                fired = self.step(t, sec)
                if len(fired) > 0:
                    all_spikes.extend([(t + sec*1000, n) for n in fired])
        return all_spikes

if __name__ == "__main__":
    import time
    
    network = IzhikevichNetwork()
    
    start_time = time.time()
    spikes = network.run(5)  # run for 5 seconds
    end_time = time.time()
    
    print(f"Number of spikes: {len(spikes)}")
    print(f"Simulation time: {end_time - start_time:.2f} seconds")
    
        # Optional: Plot results using matplotlib
    try:
        import matplotlib.pyplot as plt
        
        spike_times = [s[0] for s in spikes]
        spike_neurons = [s[1] for s in spikes]
        
        plt.figure(figsize=(12, 8))
        plt.scatter(spike_times, spike_neurons, marker='.', s=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.title('Spike Raster Plot')
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")
        
    print(network.firings)