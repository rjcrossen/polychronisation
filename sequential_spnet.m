% Runs neural networks with spike-timing dependent plasticity and
% conduction delays sequentially and saves information to a file to gain
% information about polychronous groups.
% Adapted from Eugene M. Izhikevich's spnet.m script (2004).
% Adapted/written by Robert J. Crossen (November, 2024).

M=100;                  % number of synapses per neuron
D=20;                   % maximal conduction delay 
Ne=800;                 % excitatory neurons                   
Ni=200;                 % inhibitory neurons                         
N=Ne+Ni;                % total number 
sm=10;                  % maximal synaptic strength

% parameters of Izhikevich model (Nx1 matrix)
a=[0.02*ones(Ne,1); 0.1*ones(Ni,1)];
d=[8*ones(Ne,1); 2*ones(Ni,1)];

post=zeros(N, M);       % postsynaptic connections (NxM)

delays=cell(N,D);       % synaptic conduction delay
[delays{Ne+1:N, 1}]=deal(1:M); % all inhibitory delays are 1ms

s=[6*ones(Ne,M);-5*ones(Ni,M)]; % synaptic weights (NxM)
sd=zeros(N,M);          % their derivatives (NxM)

% randomly set the targets and delays for each neuron's M synapses
for i=1:Ne
    post(i,:)=randperm(N, M);
    for j=1:M
        delays{i, ceil(D*rand)}(end+1) = j; % randomly assigning excitatory delays
    end
end

% inhibitory neurons are not connected to other inhibitory neurons
for i=Ne+1:N
    post(i,:)=randperm(Ne,M);
end

pre=cell(N, 1); % list of each neuron's presynaptic connections
aux=cell(N, 1); % this matrix will help us to query the STDP matrix later by
% containing the correct offsets for each neuron

for i=1:Ne
    for j=1:D
        for k=1:length(delays{i,j})
            % fill in each neuron's presynaptic connections with convoluted
            % query
            % multiply by N to leverage column-major ordering of matrices
            % in MATLAB
            pre{post(i, delays{i,j}(k))}(end+1) = N*(delays{i, j}(k)-1)+i;

            % look a number of columns ahead corresponding to the
            % conductance delay of the synapse
            aux{post(i, delays{i, j}(k))}(end+1) = N*(D-1-j)+i; 
        end
    end
end

STDP = zeros(N,1001+D);                 % contains D extra columns for correct calculation of STDP
v = -65*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
firings=[-D 0];                         % track the timing of spikes

% Network loop
for sec=1:60 % Simulating one minute
    for t=1:1000 % Simulating one second
        I = zeros(N,1); % Input vector
        I(ceil(N*rand)) = 20; % Giving random 20mv thalamic input to a neuron
        fired = find(v>=30);
        v(fired) = -65;
        u(fired) = u(fired) + d(fired);
        STDP(fired, t+D) = 0.1;

        for k=1:length(fired)
            % Set the synaptic derivative of all presynaptic neurons based
            % on the timing of the firing of this neuron
            sd(pre{fired(k)}) = sd(pre{fired(k)})+STDP(N*t+aux{fired(k)});
        end

        firings=[firings;t*ones(length(fired),1),fired];
        k=size(firings,1);
    
        while firings(k,1)>t-D % if neuron has fired within D ms
          del=delays{firings(k,2),t-firings(k,1)+1};
          ind=post(firings(k,2),del);
    
          % increment inputs of postsynaptic neurons by the synaptic weight
          I(ind)=I(ind)+s(firings(k,2), del)';
    
          % penalise synaptic derivative for presynaptic neurons that fired
          % late
          sd(firings(k,2),del)=sd(firings(k,2),del)-1.2*STDP(ind,t+D)';
          k=k-1;
        end;

        v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical 
        v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time 
        u=u+a.*(0.2*v-u);                   % step is 0.5 ms
        STDP(:,t+D+1)=0.95*STDP(:,t+D);     % tau = 20 ms
    
    end;

  plot(firings(:,1),firings(:,2),'.');
  axis([0 1000 0 N]); drawnow;
  STDP(:,1:D+1)=STDP(:,1001:1001+D);
  ind = find(firings(:,1) > 1001-D);
  firings=[-D 0;firings(ind,1)-1000,firings(ind,2)];
  % Taking the max of 0 and (min of 10 or synaptic weight + synaptic
  % derivative) to maintain synaptic weight between 0 and 10
  s(1:Ne,:)=max(0,min(sm,0.01+s(1:Ne,:)+sd(1:Ne,:)));
  sd=0.9*sd;
end;
