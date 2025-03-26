% This script takes PNGs from a network and checks how they are activated
% in response to different stimuli as a form of representation analysis.
% To start, import a workspace file from polychron.m and a workspace file
% from network_sim.m to get the stimuli. It is assumed that these are both
% in the same directory, with the PNG scan workspace suffixed by "_pg_scan". 

function group_activation(varargin)

p = inputParser;

addParameter(p, 'WorkspaceName', "");
addParameter(p, 'WorkspaceFolder', "");
addParameter(p, 'SaveFolder', "");
addParameter(p, 'PathLength', 4);
addParameter(p, 'Seed', 1);

parse(p, varargin{:});

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


% workspaces to load in
ws_name = p.Results.WorkspaceName;
ws_folder = p.Results.WorkspaceFolder;
if ~endsWith(ws_folder, "/")
    ws_folder = ws_folder + "/";
end

network = load(ws_folder + ws_name);
pg_scan = load(ws_folder + ws_name + "_pg_scan");

groups = pg_scan.groups;
D = network.D;
a = network.a;
d = network.d;
s = network.s;
stim_A = network.stim_A;
stim_B = network.stim_B;
neurons_A = network.neurons_A;
neurons_B = network.neurons_B;
delays_A = network.delays_A;
delays_B = network.delays_B;
post = network.post;
N = network.N;
delays = network.delays;

% save path
save_path = save_folder + ws_name + "_pg_activation";

rand_seed = p.Results.Seed;

rng(rand_seed);

% can choose to select only a subset, but right now we are selecting all
% groups
num_groups = length(groups);
%rand_groups = randi(group_count, num_groups, 1);
selected_groups = groups(1:num_groups);
disp(num2str(length(selected_groups)))

error_range = 1; % How much jitter we tolerate in neuron activation time (ms) in PNGs
min_activation = 0.75; % This is how much of the group needs to activate.

% Take the last neuron to fire in each PNG and use it as a
% key to start searching for that PNG
output_neurons = [];

for i = 1:num_groups
    output_neuron = selected_groups{1,i}.firings(end);
    output_neurons(end+1) = output_neuron;
end

% Now we simulate the network (with no STDP) and search for these groups

v = -65*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
firings=[-D 0];                         % spike timings
group_activations = 0;
% Row encodes group index, column encodes stimulus (1 = A, 2 = B)
group_activation_matrix = zeros(num_groups, 2);
pg_activation_times = cell(num_groups, 2);
tic;
for sec=1:60*60               % Simulate 1 hour
    disp(num2str(sec));
    % On odd seconds we will present stimulus A and vice versa
    if mod(sec, 2) == 0
        stim_train = stim_B;
        stim_targets = neurons_B;
        stim_delays = delays_B;
        column = 2;
    else
        stim_train = stim_A;
        stim_targets = neurons_A;
        stim_delays = delays_A;
        column = 1;
    end

    for t=1:1000                         % simulation of 1 sec
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

        mask = ismember(output_neurons, fired);
        % Search for PNGs that have activated
        if any(mask)
            activated_pngs = find(mask);

            % Now that we have the activated PNGs we need to retroactively
            % search previous firings to see if they've been activated
            for j=1:length(activated_pngs)
                png_index = activated_pngs(j);
                group = selected_groups{1,png_index};
                num_firings = length(group.firings);
                max_misses = floor(num_firings * (1 - min_activation));
                missed_firings = 0;
                final_firing = group.firings(num_firings, 1);

                if final_firing > t % This condition could be refined a bit
                    %disp("PNG could not have activated fully since stimulus presentation. Trying next group.");
                    continue
                end

                k = num_firings-1;
                while k > 0
                    neuron_index = group.firings(k, 2);
                    fire_time = t - (group.firings(num_firings, 1) - group.firings(k, 1));
                    lower_bound = fire_time - error_range;
                    upper_bound = fire_time + error_range;

                    % Now I need to check the firings record to see if the
                    % correct neuron has fired at these times
                    p = length(firings);

                    %Implement binary search to find the upper bound index
                    %of firings
                    left_pointer = 1;
                    right_pointer = p;
                    upper_index = p+1; % Default to p+1

                    while left_pointer <= right_pointer
                        mid = floor((left_pointer + 3*right_pointer) / 4); % Bias towards end for efficiency

                        if firings(mid,1) <= upper_bound
                            upper_index = mid; % Store the best index so far
                            left_pointer = mid + 1; % Search in the right half to find the last valid one
                        else
                            right_pointer = mid - 1; % Search in the left half
                        end
                    end

                    if upper_index <= p
                        while upper_index > 0 && firings(upper_index,1) >= lower_bound
                            if (firings(upper_index,2) == neuron_index)
                                %disp("Neuron found");
                                break;
                            end
                            upper_index = upper_index - 1;
                        end
                        % If we exited the loop and the firing is out of bounds, count it as a miss
                        if upper_index == 0 || firings(upper_index,1) < lower_bound
                            missed_firings = missed_firings + 1;
                            if missed_firings > max_misses
                                break;
                            end
                        end
                    else
                        %disp("Neuron not found. Skipping this group")
                        break;
                    end
                    k = k - 1;
                end
                if k == 0
                    %disp("Group activation found :)");
                    group_activations = group_activations + 1;
                    group_activation_matrix(png_index, column) = group_activation_matrix(png_index, column) + 1;
                    pg_activation_times{png_index, column}(end+1) = t;
                end
            end
        end

        firings=[firings;t*ones(length(fired),1),fired];
        k=size(firings,1);
        while firings(k,1)>t-D
            del=delays{firings(k,2),t-firings(k,1)+1};
            ind = post(firings(k,2),del);
            I(ind)=I(ind)+s(firings(k,2), del)';
            k=k-1;
        end;

        v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
        v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
        u=u+a.*(0.2*v-u);                   % step is 0.5 ms
    end;
    %plot(firings(:,1),firings(:,2),'.');
    %axis([0 1000 0 N]); drawnow;
    ind = find(firings(:,1) > 1001-D);
    firings=[-D 0;firings(ind,1)-1000,firings(ind,2)];
end;
elapsed_time = toc;
%drawnow;

disp(num2str(group_activations));
disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
save(save_path)
end