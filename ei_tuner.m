% This script takes a network structure and attempts to reach a target
% network balance for excitatory neurons by adjuting inhibitory weights
% only. It will leave the PNG structure intact.

rng(1);

network = load("network.mat");

% Load in the network structure
post = network.post;
delays = network.delays;
N = network.N;
Ne = network.Ne;
D = network.D;
d = network.d;
a = network.a;
sm = network.sm;
pre = network.pre;
M = network.M;

% Create inhibitory pre cell
pre_inhib = cell(Ne, 1);
for i = Ne+1:N
    for j = 1:length(delays{i,1})
        pre_inhib{post(i, delays{i,1}(j))}(end+1) = N*(delays{i, 1}(j)-1)+i; 
    end
end

s = network.s;

v = -65*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
firings=[-D 0];                         % spike timings

sim_time = 60*30;

% Target E/I balance
target_balance = 3.5;

% Tolerance
beta_tolerance = 1e-3;

% Learning rate
balance_rate = 1e-2;

% Vector to store global beta history
beta_history = zeros(sim_time, 1);


for sec=1:sim_time
    local_betas = zeros(N, 1);
    for t=1:1000                        % simulation of 1 sec
        target_neuron = randi(N);
        I=zeros(N,1);
        I(target_neuron)=20;                 % random thalamic input
        fired = find(v>=30);                % indices of fired neurons
        v(fired)=-65;                       % reset fired neurons back to resting membrane potential
        u(fired)=u(fired)+d(fired);

        firings=[firings;t*ones(length(fired),1),fired];
        k=size(firings,1);
        while firings(k,1)>t-D
            del=delays{firings(k,2),t-firings(k,1)+1};
            ind = post(firings(k,2),del);
            I(ind)=I(ind)+s(firings(k,2), del)';
            local_betas(ind) = local_betas(ind) + s(firings(k,2), del)';
            k=k-1;
        end

        v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
        v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
        u=u+a.*(0.2*v-u);                   % step is 0.5 ms
    end

    plot(firings(:,1),firings(:,2),'.');
    axis([0 1000 0 N]); drawnow;

    ind = find(firings(:,1) > 1001-D);
    firings=[-D 0;firings(ind,1)-1000,firings(ind,2)];

    % Average balance per timestep
    local_betas = local_betas / 1000;

    % Step toward target balance
    balance_delta = target_balance - local_betas(1:Ne);
    step_size = balance_delta .* balance_rate;

    for i = 1:Ne
        s(pre_inhib{i}) = s(pre_inhib{i}) + step_size(i);
    end
    s(Ne+1:end, :) = clip(s(Ne+1:end, :), -Inf, 0);

    % Record balance history
    global_beta = mean(local_betas(1:Ne));
    beta_history(sec) = global_beta;

    if sec > 60
        if isbetween(mean(beta_history(sec-60:sec)), target_balance-beta_tolerance, target_balance+beta_tolerance) 
            disp("Target achieved. Terminating")
            break
        end
    end
end

network_weights = s;

save("network_ei_" + target_balance + ".mat", "network_weights");
