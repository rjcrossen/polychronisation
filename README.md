This is a repository featuring scripts used for a biologically plausible spiking neural network for my thesis.

analysis\_pipeline/network_sim_ab.m, analysis\_pipeline/polychron.m and analysis\_pipeline/polygroup.m are scripts written by Izhikevich (2006) for his paper "Polychronisation: Computation with spikes".

analysis\_pipeline/network_sim_ab.m is modified to administer two repeated stimuli which alternate each second.

analysis\_pipeline/polychron_parallelised implements CPU multiprocessing to speed up the group-finding algorithm. It is recommended to use this version as the original algorithm is computationally expensive.

analysis\_pipeline/output_classifier_jitter.m and analysis\_pipeline/apply_stimulus_jitter.m are used to apply a stimulus to a network repeatedly and attempt to predict the input stimulus label based on readout firing rates.

analysis\_pipeline/group_activation.m simulates the network receiving input stimuli and tracks polychronous group activation.
