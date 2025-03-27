% Original PNG-finding algorithm created by Eugene M. Izhikevich (2005) 
% and modified based on suggestions of Petra Vertes (2008). 
% Script adapted by Robert J. Crossen (2025).

% Main idea: for each mother neuron, consider various combinations of 
% pre-synatic (anchor) neurons and see whether any activity of a silent
% network could emerge if these anchors are fired. 

function parallel_polychron(varargin)

% Calculate how many cores the script should use.
% There is some instability when using all available cores, so default to
% available cores - 2
cores = feature('numCores');
if cores - 2 >= 1
    cores = cores - 2;
end

p = inputParser;

addParameter(p, 'WorkspaceName', "");
addParameter(p, 'WorkspaceFolder', "");
addParameter(p, 'SaveFolder', "");
addParameter(p, 'PathLength', 4);
addParameter(p, 'Cores', cores);

parse(p, varargin{:});

% available cores
available_cores = p.Results.Cores;

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
save_path = save_folder + ws_name + "_pg_scan";

data = load(ws_folder + ws_name);

a = data.a;
d = data.d;
N = data.N;
D = data.D;
s = data.s;
sm = data.sm;
Ne = data.Ne;
N = data.N;
delays = data.delays;
post = data.post;
pre = data.pre;

%uncomment the command below to test the algorithm for shuffled (randomized e->e) synapses
%e2e = find(s>=0 & post<Ne); s(e2e) = s(e2e(randperm(length(e2e))));
            
groups=cell(Ne, 1);          % the list of all polychronous groups
anchor_width=3;     % the number of anchor neurons, from which a group starts
min_group_path=p.Results.PathLength;   % discard all groups having shorter paths from the anchor neurons 
T=150;              % the max length of a group to be considered;
                    % longer groups will be cut at t=T

% Make necessary initializations to speed-up simulations.

% This matrix provides the delay values for each synapse.
for i=1:N
    for j=1:D
        delay(i, delays{i,j})=j;
    end;
end;

%This cell element tells what the presynaptic delay is; 
for i=1:N
    dpre{i}=delay( pre{i} );
end;

%This cell element tells where to put PSPs in the matrix I (N by 1000)
for i=1:N
    pp{i}=post(i,:)+N*(delay(i,:)-1);
end;

%This cell element tells who (indexes) the pre-synaptic neurons are; 
for i=1:N
    ppre{i}=mod( pre{i}, N);
end;

sm_threshold = 0.95*sm;     % discard all weak exc->exc synapses
s(find(post<Ne & s>0 & s<=sm_threshold))=0;

parpool('Processes', available_cores);

parfor i=1:Ne
    local_groups = {}; % Store results for this worker

    anchors=1:anchor_width;                     % initial choice of anchor neurons
    strong_pre=find(s(pre{i})>sm_threshold);    % candidates for anchor neurons
    if length(strong_pre) >= anchor_width       % must be enough candidates
     while 1        % will get out of the loop via the 'break' command below
         
        gr=polygroup( ppre{i}(strong_pre(anchors)), D-dpre{i}(strong_pre(anchors)), ...
            a, d, N, D, pp, s, ppre, dpre, pre, T ); 
        
        %Calculate the longest path from the first to the last spike
        fired_path=sparse(N,1);        % the path length of the firing (from the anchor neurons)       
        for j=1:length(gr.gr(:,2))
            fired_path( gr.gr(j,4) ) = max( fired_path( gr.gr(j,4) ), 1+fired_path( gr.gr(j,2) ));
        end;
        
        longest_path = max(fired_path);
        
        if longest_path>=min_group_path 
            
            gr.longest_path = longest_path(1,1); % the path is a cell
            
            % How many times were the spikes from the anchor neurons useful?
            % (sometimes an anchor neuron does not participate in any
            % firing, because the mother neuron does its job; such groups
            % should be excluded. They are found when the mother neuron is
            % an anchor neuron for some other neuron).
            useful = zeros(1,anchor_width);
            anch = ppre{i}(strong_pre(anchors));
            for j=1:anchor_width
                useful(j) = length( find(gr.gr(:,2) == anch(j) ) );
            end;
       
            
            
            if all(useful>=2)
                
                local_groups{end+1}=gr;           % add found group to the list
                %disp([num2str(round(100*i/Ne)) '%: size=' num2str(size(gr.firings,1)) ', path_length=' num2str(gr.longest_path)])   % display of the current status
            
                %plot(gr.firings(:,1),gr.firings(:,2),'o');
                %hold on;
                %for j=1:size(gr.gr,1)
                %    plot(gr.gr(j,[1 3 5]),gr.gr(j,[2 4 4]),'.-');
                %end;
                %axis([0 T 0 N]);
                %hold off
                %drawnow;
            end;
        end

        % Now, get a different combination of the anchor neurons
        k=anchor_width;
        while k>0 & anchors(k)==length(strong_pre)-(anchor_width-k)
            k=k-1;
        end;
        
        if k==0, break, end;    % exhausted all possibilities
        
        anchors(k)=anchors(k)+1;
        for j=k+1:anchor_width
            anchors(j)=anchors(j-1)+1;
        end;
        
        pause(0); % to avoid feezing when no groups are found for long time
        
     end;
    end;
    groups{i} = local_groups;
    disp(num2str(i) + " complete")
end;

delete(gcp('nocreate'))

% Flatten results
groups = [groups{:}];

mean_size = mean(cellfun(@(s) length(s.firings), groups));
mean_length = mean(cellfun(@(s) full(s.longest_path), groups));
group_count = length(groups);

save(save_path);

disp("PG scan complete for workspace " + ws_name);
disp("# Groups: " + num2str(group_count));
disp("Average PG Size: " + num2str(mean_size));
disp("Average PG Length (longest path): " + num2str(mean_length));

end