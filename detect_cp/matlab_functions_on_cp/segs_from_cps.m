function [X_seg, Y_seg]  = segs_from_cps(X,Y,cp_indices)

%segs_from_cp
%This function gets segments from a signal using change points identified
%on the signal. Used for getting training points

X_seg = {};
Y_seg = {};
k = 1;
for i = 1:length(cp_indices)-1
    
   temp_X = X( cp_indices(i) : cp_indices( i + 1 ), : );
   temp_Y = Y( cp_indices(i) : cp_indices( i + 1 ) , : );
   
   if (length(temp_X)) > 1200
        for t = 1:floor(length(temp_X)/600)
           temp_X_t = temp_X( (t - 1)*600 + 1 : t*600,:); 
           X_seg{k} = temp_X_t;
           Y_seg{k} = median(temp_Y);
           k = k+1;
       end 
   elseif length(temp_X) > 350
         temp_X = X( cp_indices(i) -200 : cp_indices( i + 1 )+200 , : );
        if length(temp_X) >= 600
             amount_to_trunc_end = (length(temp_X) - 600)/2;
            temp_X(1:floor(amount_to_trunc_end), :) = [];
            temp_X(length(temp_X) - floor(amount_to_trunc_end):end,:) = [];
            
        end
        if length(temp_X) < 600
            temp_X(end+1,:) = temp_X(end,:);
        end
       X_seg{k} = temp_X;
       Y_seg{k} = median(temp_Y);
       k = k + 1;
   else 
       'here'
   end
   
    
end
    