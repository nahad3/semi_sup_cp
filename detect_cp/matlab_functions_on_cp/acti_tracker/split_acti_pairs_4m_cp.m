
%%This script filters out cp for good users on the Acti tracker dataset

%This ensures that only clean pairwise information is provided



clear all
close all

save_path = '/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/actitracker/pairs4mactualCP_refined.mat';

data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/actitracker/cp_detected_WSDM_MMD.mat');


data_user_list = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/human_activity/mat_files/Train_concat_users.mat');

user_array = data_user_list.users_train(1001:end);

%removing low variance points
var_1_at_cp = data.var_arr_1(data.c_indices);
var_2_at_cp = data.var_arr_2(data.c_indices);
cp_stats = data.Cp_stat(data.c_indices);
cp_indices = data.c_indices;

cp_indicator = zeros(length(data.X),1);

find_id_var_too_low = var_1_at_cp < 0.001 & var_2_at_cp < 0.001 & cp_stats < 1.5;
cp_indices(find_id_var_too_low) = [];
cp_indicator(cp_indices) = 1;


%users to be removed
remove_user_list = [ 6, 36, 18, 14,23, 12,8 ,31,1,7,24];


temp = user_array ~= remove_user_list(1);
for i = 2 : length(remove_user_list)
    temp= user_array ~= remove_user_list(i) & temp;
end


refined_X = data.X(temp ,:);
data.Y = data.Y';
data.Cp_stat = data.Cp_stat';
data.var_arr_1 = data.var_arr_1';
refined_Y = data.Y(temp == 1,: );
refined_cp = cp_indicator(temp == 1, :);
refined_user = user_array(temp == 1 ,: );
refined_cp_stat = data.Cp_stat(temp == 1,:);

%removing false cps (User switching)
%refined_cp(516637) = [];
refined_cp = find(refined_cp == 1);
refined_cp(refined_cp == 545676) = [];
refined_cp(refined_cp == 552610) = [];
refined_cp(refined_cp == 553533) = [];
refined_cp(refined_cp == 564755) = [];
refined_cp(refined_cp == 563812) = [];
refined_cp(refined_cp == 576480) = [];
refined_cp(refined_cp == 588918 ) = [];
refined_cp(refined_cp == 516627 ) = [];
refined_var = data.var_arr_1(temp == 1,:);
ax1 = subplot(4,1,1)
plot(refined_X );
ax2 = subplot(4,1,2)
plot(refined_cp_stat)
hold on
scatter(refined_cp ,refined_cp_stat(refined_cp ))
ax3  = subplot(4,1,3)
plot(refined_Y)
hold on
plot(refined_var)
ax4 = subplot(4,1,4)
plot(refined_user);
legend('User list')
linkaxes([ax1,ax2,ax3,ax4],'x')

var_1_at_cp = data.var_arr_1(data.c_indices);
var_2_at_cp = data.var_arr_2(data.c_indices);
cp_stats = data.Cp_stat(data.c_indices);
cp_indices = refined_cp;



pair_size = 200;

buff= 100;
X1_dissim = [  ];
X2_dissim = [  ];

X1_sim = [   ] ;
X2_sim = [   ] ;

X1_sim_labels = [];
X1_dissim_labels = [];

X2_sim_labels = [];
X2_dissim_labels = [];

change_indices = cp_indices;

X = refined_X;
label_array = refined_Y;

for i = 2:length(change_indices)-1
   
   %concatentating dissimilar items
   X1 = X( change_indices(i) - pair_size - buff : change_indices(i) -buff , :) ;
   X1_dissim_labels= [X1_dissim_labels ; label_array(change_indices(i) -buff)];
   [A, B ] = size(X1);
   X1_dissim = cat( 1, X1_dissim,  reshape(X1,1,A,B));
   X2 = X( change_indices(i) + buff : change_indices(i) + pair_size +buff, : ) ;
   
   [A, B ] = size(X2);
   X2_dissim = cat( 1, X2_dissim,  reshape(X2,1,A,B)  );
   X2_dissim_labels= [X2_dissim_labels ; label_array(change_indices(i) + buff)];
   %concatenating similar_item
   if i == 100
       'pause'
   end
   X1 = X(change_indices(i) - pair_size -buff : change_indices(i) - buff  , :) ;
   [A, B ] = size(X1);
   X1_sim_labels  = [X1_sim_labels; label_array(change_indices(i)-buff)];
   X1_sim = cat( 1, X1_sim,  reshape(X1,1,A,B));
   X2 = X(change_indices(i) - 2*pair_size - buff : change_indices(i) - pair_size -buff , :) ;
   [A, B ] = size(X2);
   X2_sim = cat( 1, X2_sim,  reshape(X2,1,A,B)  );
   X2_sim_labels = [X2_sim_labels ; label_array(change_indices(i)-buff)];
   
end



[A1,~,~ ] = size(X1_sim);
[A2,~,~] = size(X1_dissim);

Y_dissim = -1*ones(A1,1);
Y_sim = ones(A2,1);

Y = [Y_dissim; Y_sim];
X1_label= [ X1_dissim_labels ; X1_sim_labels ];
X2_label = [X2_dissim_labels ; X2_sim_labels];
X1 = [X1_dissim ; X1_sim];
X2 = [X2_dissim ; X2_sim];

save(save_path,'X1','X2','Y','X1_label','X2_label');