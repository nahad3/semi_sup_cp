function [seg_array, Z, Y] = get_segs_and_generate_series(X,cp_indices,buff,rep_times)

%Input : X signal
%cp_indices : change segments
%buffer: buffer that is not included in the segs
%rep times: how often is a segment identifeid by a change point is repeated


%Outputs:

%Z: generated signal
%Y: indices where segments joined
seg_array = {};
k = 1;
sim_batch = {};
sim_btch = 1;
for i = 1:length(cp_indices)-2
    temp = X(cp_indices(i) -  buff : cp_indices(i+1) +  buff , :);
    temp2 = X(cp_indices(i+1) -  buff : cp_indices(i+2) +  buff , :);
    
    
    
    if length(temp) <  400
        continue
    elseif length(temp)> 1000
        similar_cells= {};
        s = 1;
        for j = 1:floor(length(temp)/400)
            similar_cells{s} = repmat(temp((j-1)*400+1:j*400,:),[rep_times,1]);
            s = s+1;
            
        end
        sim_batch{sim_btch} = similar_cells;
        sim_btch = sim_btch + 1;
    else
        seg_array{k} = repmat(temp,[rep_times,1]);
        k = k+1;
    end
end

L  = length(seg_array);


numb_segments = 2000;
r = randi([1,L],numb_segments,1);

Z = [];

Y = [];
Y(end+1) = 1;
for i = 1:numb_segments
    Z = cat(1,Z,seg_array{r(i)});
   Y(end+1) = Y(end) + length(seg_array{r(i)});
end

Y = Y';