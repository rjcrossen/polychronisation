function network_sim_ab(varargin)
p = inputParser;

addParameter(p, 'Overlap', 0);
addParameter(p, 'Neurons', 1000);
addParameter(p, 'Synapses', 100);
addParameter(p, 'MaxDelay', 20);
addParameter(p, 'EIRatio', 4); % default 4:1 ratio of E:I
addParameter(p, 'MaxSynapseStrength', 10);
addParameter(p, 'Seed', 1);
addParameter(p, 'SaveFolder', "");
addParameter(p, 'FileName', "");
addParameter(p, 'SimTime', 3600); % sim time in seconds

parse(p, varargin{:});

% percentage overlap of input neurons
overlap = p.Results.Overlap;

% simulation time
sim_time = p.Results.SimTime;

% number of synapses per neuron
M = p.Results.Synapses;

% max delay
D = p.Results.MaxDelay;

% EI Ratio
EI = p.Results.EIRatio;

% number of neurons
N = p.Results.Neurons;

Ni = round(N / (EI + 1));

% number of E neurons
Ne = N - Ni;

% number of I neurons
Ni = N - Ne;

% max synaptic strength
sm = p.Results.MaxSynapseStrength;

% Creates a column vector of 0.02 of size Ne
a=[0.02*ones(Ne,1);    0.1*ones(Ni,1)];
d=[   8*ones(Ne,1);    2*ones(Ni,1)];

% path to save workspace
if ismember('SaveFolder', p.UsingDefaults)
    disp('SaveFolder not provided, will save to current directory');
    save_folder = "";
else
    save_folder = p.Results.SaveFolder;
    if ~endsWith(save_folder, "/")
        save_folder = save_folder + "/";
    end
    if ~isfolder(save_folder)
        mkdir(save_folder)
        disp("Directory has been created: " + save_folder);
    end
end

if ismember('FileName', p.UsingDefaults)
    disp('FileName not provided, will save with default name');
    file_name = num2str(Ne) + "e_" + num2str(Ni) + "i_" + num2str(overlap) + "overlap_" + num2str(EI) + "ratio";
    save_path = save_folder + file_name;
else
    file_name = p.Results.FileName;
    save_path = save_folder + file_name;
end

% Set random seed
rand_seed = p.Results.Seed;
%disp("Using Seed: " + num2str(rand_seed));
rng(rand_seed);

% Creates two matrices for postsynaptic connections and appends them to
% form an N*M matrix
post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)]);

% Take special care not to have multiple connections between neurons
delays = cell(N,D);
for i=1:Ne
    %Create permutation from 1 to 1000 with no repeating elements, and set
    %the ith row of the postsynaptic connections matrix to the first M
    %elements of permutation
    p=randperm(N);
    post(i,:)=p(1:M);
    for j=1:M
        delays{i, ceil(D*rand)}(end+1) = j;  % Assign random exc delays
    end;
end;

%Loop through all inhibitory neurons
for i=Ne+1:N
    p=randperm(Ne);
    post(i,:)=p(1:M);
    delays{i,1}=1:M;                    % all inh delays are 1 ms.
end;

s=[6*ones(Ne,M);-5*ones(Ni,M)];         % synaptic weights
sd=zeros(N,M);                          % their derivatives

% Make links at postsynaptic targets to the presynaptic weights
pre = cell(N,1);
aux = cell(N,1);

%For every excitatory neuron
for i=1:Ne
    %For every delay of every excitatory neuron
    for j=1:D
        %For every neuron of a given delay for each excitatory neuron
        for k=1:length(delays{i,j})
            pre{post(i, delays{i, j}(k))}(end+1) = N*(delays{i, j}(k)-1)+i;
            aux{post(i, delays{i, j}(k))}(end+1) = N*(D-1-j)+i; % takes into account delay
        end;
    end;
end;


STDP = zeros(N,1001+D);
v = -65*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
firings=[-D 0];                         % spike timings

% Generate spike trains for stimuli A and B

% I have to reset the random seed because random functions above used
% variables which could change (e.g. Ne, Ni, N) and affect the seed
rng(rand_seed);

% Stimulus: 200ms @ 100Hz (arbitrary)
Hz = 100;
stim_A = rand(1,200) < Hz/1000;
stim_A = [stim_A, zeros(1, 1000 - length(stim_A))];

stim_B = rand(1,200) < Hz/1000;
stim_B = [stim_B, zeros(1, 1000 - length(stim_B))];

% Select the neurons for stim A and stim B (Excitatory only)
% Stimulus will be delivered to 50 neurons with varying levels of overlap
n_input = 50;
n_overlap = n_input * (overlap/100);
b_start_idx = n_input - n_overlap + 1;
b_end_idx = b_start_idx + n_input - 1;

% When I change EI balance, Ne changes which affects the permutation we
% generate. To get around this I just take the first X neurons as input
% neurons

%p = randperm(Ne);
neurons_A = 1:n_input;
neurons_B = b_start_idx:b_end_idx;

% Determine the delay that each spike in the stimulus will arrive at the
% input neuron with
input_delays = randi(20, 100);
disp("First 5 values: " + num2str(input_delays(1:5)));
delays_A = input_delays(1:50);
delays_B = input_delays(51:100);

for sec=1:sim_time
    % On odd seconds we will present stimulus A and vice versa
    if mod(sec, 2) == 0
        stim_train = stim_B;
        stim_targets = neurons_B;
        stim_delays = delays_B;
    else
        stim_train = stim_A;
        stim_targets = neurons_A;
        stim_delays = delays_A;
    end

    for t=1:1000                         % simulation of 1 sec
        ms = t;
        I=zeros(N,1);
        indices = t - stim_delays;
        valid_mask = (indices >= 1);
        delayed_spikes = zeros(size(indices));
        delayed_spikes(valid_mask) = stim_train(indices(valid_mask));
        I(stim_targets) = I(stim_targets) + delayed_spikes' * 20;
        I(ceil(N*rand))=20;                 % random thalamic input

        fired = find(v>=30);                % indices of fired neurons
        v(fired)=-65;                       % reset fired neurons back to resting membrane potential
        u(fired)=u(fired)+d(fired);
        STDP(fired,t+D)=0.1;

        for k=1:length(fired)               % loop through every fired neuron
            sd(pre{fired(k)})=sd(pre{fired(k)})+STDP(N*t+aux{fired(k)});      %adjust derivate of synaptic weights
        end;
        firings=[firings;t*ones(length(fired),1),fired];
        k=size(firings,1);
        while firings(k,1)>t-D
            del=delays{firings(k,2),t-firings(k,1)+1};
            ind = post(firings(k,2),del);
            I(ind)=I(ind)+s(firings(k,2), del)';
            sd(firings(k,2),del)=sd(firings(k,2),del)-1.2*STDP(ind,t+D)';
            k=k-1;
        end;

        v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
        v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
        u=u+a.*(0.2*v-u);                   % step is 0.5 ms
        STDP(:,t+D+1)=0.95*STDP(:,t+D);     % tau = 20 ms
    end;
    %plot(firings(:,1),firings(:,2),'.');
    %axis([0 1000 0 N]); drawnow;
    STDP(:,1:D+1)=STDP(:,1001:1001+D);
    ind = find(firings(:,1) > 1001-D);
    firings=[-D 0;firings(ind,1)-1000,firings(ind,2)];
    % Taking the max of 0 and (minimum of 10 or synaptic weight + synaptic
    % derivative
    s(1:Ne,:)=max(0,min(sm,0.01+s(1:Ne,:)+sd(1:Ne,:)));
    sd=0.9*sd;

end;
%drawnow;

save(save_path, 'pre', 'aux', 's', 'delays', 'post', 'stim_A', 'stim_B', ...
    'neurons_A', 'neurons_B', 'delays_A', 'delays_B', 'a', 'd', ...
    'N', 'Ne', 'Ni', 'D', 'sm')
%disp(workspace_dir + " saved")
disp("done")
end
