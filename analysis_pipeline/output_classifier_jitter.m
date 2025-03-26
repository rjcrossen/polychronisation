function output_classifier_jitter(varargin)
p = inputParser;

addParameter(p, 'Seed', 1);
addParameter(p, 'WorkspaceName', "");
addParameter(p, 'WorkspaceFolder', "");
addParameter(p, 'SaveFolder', "");

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

% workspace to load in
ws_name = p.Results.WorkspaceName;
ws_folder = p.Results.WorkspaceFolder;
if ~endsWith(ws_folder, "/")
    ws_folder = ws_folder + "/";
end

% save path
save_path = save_folder + ws_name + "_logr";

% set random seed
rng(p.Results.Seed);

data = load(ws_folder + ws_name);

N = data.N;
Ne = data.Ne;
D = data.D;
delays = data.delays;
post = data.post;
d = data.d;
a = data.a;
s = data.s;
neurons_A = data.neurons_A;
neurons_B = data.neurons_B;
stim_A = data.stim_A;
stim_B = data.stim_B;
delays_A = data.delays_A;
delays_B = data.delays_B;

%Choose a random readout layer of 50 neurons

n_readout = 50;
% NOTE: I AM HARDCODING THE READOUT LAYER IN FOR THE EI_PNG_ACTIVATION_ANALYSIS EXPERIMENT
p = randperm(800); % minimum number of excitatory neurons in my experiment
%p = randperm(Ne); 
readout = p(1:n_readout);

% Applying stimuli to generate label-firing rate pairs
trial_count = 500;
num_stim = 2;
readout_time = 1000;
firing_rate_matrix = zeros(trial_count*num_stim, length(readout));
labels = [ones(trial_count, 1); 2*ones(trial_count, 1)];
T_sec = readout_time / 1000; % readout time in seconds 

for i=1:trial_count
    disp(num2str(i))
    % Simulate stimulus A
    raster = apply_stimulus_jitter(N, D, post, delays, s, d, a, neurons_A, stim_A, delays_A);
    raster = raster(ismember(raster(:, 2), readout), :);
    spike_counts = zeros(length(readout), 1);  % Preallocate spike count array
    for j = 1:length(readout)
        neuron_id = readout(j);
        spike_counts(j) = sum(raster(:,2) == neuron_id);
    end

    % Compute firing rates (spikes per second)
    firing_rates = spike_counts / T_sec;

    firing_rate_matrix(i, :) = firing_rates;
    
    % Simulate stimulus B
    raster = apply_stimulus_jitter(N, D, post, delays, s, d, a, neurons_B, stim_B, delays_B);
    raster = raster(ismember(raster(:, 2), readout), :);
    spike_counts = zeros(length(readout), 1);  % Preallocate spike count array
    for j = 1:length(readout)
        neuron_id = readout(j);
        spike_counts(j) = sum(raster(:,2) == neuron_id);
    end

    % Compute firing rates (spikes per second)
    firing_rates = spike_counts / T_sec;

    firing_rate_matrix(i+trial_count, :) = firing_rates;
end


% Training logistic regression
k = 10; % number of folds
X = firing_rate_matrix;
y = labels;
y = (y == 2); % converting labels from 1 and 2 to 0 and 1

cv = cvpartition(size(X, 1), 'KFold', k);  % k-fold partition

accuracies = zeros(k, 1);


for i = 1:k
    % Get train and test indices for the current fold
    X_train = X(training(cv, i), :);
    y_train = y(training(cv, i));
    X_test = X(test(cv, i), :);
    y_test = y(test(cv, i));

    % Train logistic regression model
    model = fitclinear(X_train, y_train, 'Learner', 'logistic'); 
    
    % Predict on test set
    y_pred = predict(model, X_test); 
    
    % Compute accuracy
    accuracies(i) = mean(y_pred == y_test);
    
    disp("Fold " + i + " Accuracy: " + num2str(accuracies(i) * 100) + "%");
end

% Compute overall mean accuracy and standard deviation
mean_accuracy = mean(accuracies);
std_accuracy = std(accuracies);

disp("Mean Test Accuracy: " + num2str(mean_accuracy * 100) + "%");
disp("Standard Deviation: " + num2str(std_accuracy * 100) + "%");

save(save_path);

end
