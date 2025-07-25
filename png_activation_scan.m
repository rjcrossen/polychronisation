% Scans a spike train to search for PNG activations
% This script requires a recorded spike train and the PNG scan for the
% corresponding network. It's currently written so that the spike train for
% each trial is a cell entry but it could easily be modified

%% PREPARING PNGS
pg_full = load("pg_scan.mat").groups;

% Minor optimisation: stop the scan each trial at the point where no PNGs
% can possibly activate anymore
% Implemented by finding the shortest PNG so we know where to stop
shortest_group = 150;

for i = 1:length(pg_full)
    curr_pg = pg_full{i};
    % Figuring out the shortest possible span of time that the shortest
    % group could be classified as having fired in to save loops
    curr_exc_firings = curr_pg.firings(curr_pg.firings(:, 2) <= 800, :);
    threshold = ceil(size(curr_exc_firings, 1) / 2);
    span = curr_exc_firings(threshold, 1) - curr_exc_firings(1, 1);
    if span < shortest_group
        shortest_group = span;
    end

    % Take only excitatory firings
    curr_pg.firings = curr_exc_firings;

    % Align time to start at 0
    curr_pg.firings(:, 1) = curr_pg.firings(:, 1) - min(curr_pg.firings(:, 1));

    pg_full{i} = curr_pg;
end

% Load spike train
% Here, we load a cell called "spike_trains" where each cell entry is a
% trial of a task
spike_train = load("spike_train.mat").spike_trains;
num_trials = length(spike_trains);

% Jitter allowance
tolerance = 1;

% Refractory period for PNGs
refractory_period = 3;

% Prepare output data structures
pg_activation_scans = struct();
activations = cell(length(spike_train), 1);
pg_activation_counts = zeros(part_length, length(pg_full), 'uint8');
total_activations = 0;

for sec = 1:num_trials
    firings = spike_train{sec};
    firings = firings(firings(:, 2) <= 800, :);

    max_t = max(firings(:, 1))-shortest_group;

    % Optimisation: create a dictionary of the indices where each t starts
    % using binary search.
    t_index = cell(max_t, 1);
    t_index{1} = 1;
    low = 1;
    for t = 2:length(t_index)
        high = size(firings, 1);
        j = size(firings, 1) + 1;  % Default to one-past-end if not found
        while low <= high
            mid = floor((low + high) / 2);
            if firings(mid, 1) >= t
                j = mid;
                high = mid - 1;  % Look to the left
            else
                low = mid + 1;   % Look to the right
            end
        end
        t_index{t} = j;
        low = j;
    end

    refractory = zeros(length(pg_full), 1);

    % Check for PNG activation at every possible starting point
    % We start from 2 to avoid issues with the jitter (so t-jitter > 0).
    % In practice, a PNG is unlikely to activate at t=1 because the network
    % has not received sufficient input.

    for t = 2:max_t
        refractory(refractory < t-refractory_period) = 0;

        % Loop through every group and check if 50% of excitatory neurons
        % activated
        for group_num = 1:length(pg_full)
            if refractory(group_num) ~= 0
                continue;
            end
            group = pg_full{group_num};
            group_firings = group.firings;

            % only check excitatory neurons
            group_firings = group_firings(group_firings(:, 2) <= 800, :);

            % if we hit this many the group must be "activated"
            hit_threshold = ceil(length(group_firings) / 2);
            hit_counter = 0;

            % if we miss this many the group cannot be "activated"
            miss_threshold = length(group_firings) - (ceil(length(group_firings) / 2));
            miss_counter = 0;

            % start firing times at current time
            group_firings(:, 1) = group_firings(:, 1) + double(t);

            % work out the first firing in the group (there is no wiggle
            % room for this one)
            neuron_ind = group_firings(1, 2);

            j = t_index{t};
            
            % Continue from j while firings(j,1) <= t
            while j <= length(firings) && firings(j, 1) == t
                if firings(j, 2) == neuron_ind
                    hit_counter = hit_counter + 1;
                    break;
                end
                j = j + 1;
            end

            if j > length(firings) || firings(j,1) ~= t
                miss_counter = miss_counter + 1;
                if miss_counter >= miss_threshold
                    continue
                end
            end
            if hit_counter >= hit_threshold
                refractory(group_num) = t;
                activations{sec}(end+1, :) = [t group_num];
                total_activations = total_activations + 1;
                pg_activation_counts(sec, group_num) = pg_activation_counts(sec, group_num) + 1;
                continue
            end

            % now start looking from the second firing with wiggle room
            for k = 2:length(group_firings)
                t_fire = group_firings(k, 1);
                neuron_ind = group_firings(k, 2);
                t_min = t_fire-tolerance;
                t_max = t_fire+tolerance;
                
                if t_min <= max_t
                    j = t_index{t_min};
                else
                    break;
                end
                % Continue from j up to the point firings(j,1) > t_max
                while j <= length(firings) && firings(j, 1) <= t_max
                    if firings(j, 2) == neuron_ind && firings(j,1) >= t_min
                        hit_counter = hit_counter + 1;
                        break;
                    end
                    j = j + 1;
                end
                if j > length(firings) || firings(j,1) > t_max
                    miss_counter = miss_counter + 1;
                    if miss_counter >= miss_threshold
                        break
                    end
                end
                if hit_counter >= hit_threshold
                    refractory(group_num) = t;
                    activations{sec}(end+1, :) = [t group_num];
                    total_activations = total_activations + 1;
                    pg_activation_counts(sec, group_num) = pg_activation_counts(sec, group_num) + 1;
                    break
                end
            end
        end
    end
end

pg_activation_scans.activations = activations;
pg_activation_scans.total_activations = total_activations;
pg_activation_scans.pg_activation_counts = pg_activation_counts;
save("pg_activation_scan", "pg_activation_scans");