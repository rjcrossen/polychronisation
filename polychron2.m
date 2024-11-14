tic

fname = 'Networks/500.mat';

global a d N D pp s ppre dpre post pre delay T

load(fname);

groups={};
anchor_width=3;
T=150;
min_group_path=7;

% Make necessary initializations to speed-up simulations.

% This matrix provides the delay values for each synapse.
% N x M
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

%This cell element tells who (indexes 1:N) the pre-synaptic neurons are; 
for i=1:N
    ppre{i}=mod( pre{i}, N);
end;


sm_threshold = 0.95*sm;     % discard all weak exc-exc synapses
% skeptical if we need the s>0 or if we even need this at all given the next lines
s(find(post<Ne & s>0 & s<=sm_threshold))=0;

% find all possible combinations of anchor neurons (saves some computation later)
anchor_combinations = cell(Ne, 1);
for i=1:Ne
    strong_pre = find(s(pre{i})>sm_threshold);
    if(length(strong_pre) >= anchor_width)
        anchor_combinations{i} = nchoosek(strong_pre, anchor_width);
    else
        anchor_combinations{i} = 0;
    end
end

for i=1:Ne
    if(anchor_combinations{i} == 0)
        j=j+1;
        continue
    end
    for j=1:height(anchor_combinations{i})
        anchors = anchor_combinations{i}(j, :);
        group = polygroup(ppre{i}(anchors), D-dpre{i}(anchors));

        %Calculate the longest path from the first to the last spike
        fired_path=sparse(N,1);        % the path length of the firing (from the anchor neurons)       
        for k=1:length(group.gr(:,2))
            fired_path( group.gr(k,4) ) = max( fired_path( group.gr(k,4) ), 1+fired_path( group.gr(k,2) ));
        end

        longest_path = max(fired_path);
        
                
        if longest_path>=min_group_path 
            
            group.longest_path = longest_path(1,1); % the path is a cell
            
            % How many times were the spikes from the anchor neurons useful?
            % (sometimes an anchor neuron does not participate in any
            % firing, because the mother neuron does its job; such groups
            % should be excluded. They are found when the mother neuron is
            % an anchor neuron for some other neuron).
            useful = zeros(1,anchor_width);
            anch = ppre{i}(anchors);
            for k=1:anchor_width
                useful(k) = length( find(group.gr(:,2) == anch(k) ) );
            end;
       
            
            
            if all(useful>=2)
                groups{end+1}=group;           % add found group to the list
                disp([num2str(round(100*i/Ne)) '%: groups=' num2str(length(groups)) ', size=' num2str(size(group.firings,1)) ', path_length=' num2str(group.longest_path)])   % display of the current status
            
                plot(group.firings(:,1),group.firings(:,2),'o');
                hold on;
                for k=1:size(group.gr,1)
                    plot(group.gr(k,[1 3 5]),group.gr(k,[2 4 4]),'.-');
                end;
                axis([0 T 0 N]);
                hold off
                drawnow;
            end;
        end
        pause(0);
    end   
end

toc