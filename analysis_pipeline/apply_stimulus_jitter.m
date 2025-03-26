function raster = apply_stimulus_jitter(N, D, post, delays, s, d, a, stim_targets, stim_train, stim_delays)
    %rng(1); It seems to make sense to not use a random seed here or else
    %it would not be realistic
    v = -70*ones(N,1);                      % initial values
    u = 0.2.*v;                             % initial values
    firings = [-D 0];                       % initial values
    for t=1:1000                             % simulation of 1000ms (should capture all PG activations)
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
    raster = firings;
end
