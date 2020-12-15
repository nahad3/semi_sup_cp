
%This script loads data and obtains pairwise comparisions for Ys.
%By pairwise comparision, it provides X1 and X2 and a Y Y indicating if
%the X are similar or not
clear all
close all
%save_path = '/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/human_activity/mat_files/Train_pairs_from_label_scaled_0_1_sim_dissim_from_cp_with_labels_ActivTraker2011_.mat';

%data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/actitracker/cp_detected_WSDM_MMD.mat');

data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/gen_simulations/mat_files/mackay_switch/cp_detected_switch_mackay.mat')'
ax1 = subplot(3,1,1)
plot(data.X);
legend('Sequence value')
ylabel('Sequence values')
set(gca,'FontSize',18)
title('Change detection : Mackay-Glass sequence')
ax2 = subplot(3,1,2)
plot(data.Cp_stat,'Linewidth',2)
hold on
scatter(data.c_indices,data.Cp_stat(data.c_indices),80,'Linewidth',2)
[l3,l2] = legend({'MMD statistic','Detected change points'},'FontSize',18);
l3.FontSize = 18;
ylabel('MMD statistic')
set(gca,'FontSize',18)
ch = findobj(l2, 'type', 'patch'); %// children of legend of type line
set(ch, 'Markersize', 10);
set(ch,'Linewidth',2);

ax3  = subplot(3,1,3)
plot(data.Y,'Linewidth',2)
hold on
%plot(data.var_arr_1)
legend('Labels')
ylabel('Sequence labels')
xlabel('Sequence length')
set(gca,'FontSize',18)
linkaxes([ax1,ax2,ax3'],'x')

pair_size = 200;

X1_sim = [];
X2_sim = [ ];

X1_dissim = [];
X2_dissim = [ ];

Ys_X1_sim = [ ];
Ys_X1_dissim = [ ];

Ys_X2_sim = [];
Ys_X2_dissim = [];
%buffer before and after change points
buff= 10;

xlim([0,16000])
ylim([0,5])
set(gca,'FontSize',18)
for i = 2 : length(cp_indices)-1
    if ((cp_indices(i) - (cp_indices(i-1))) > 3*pair_size ) &&   (((cp_indices(i+1) - cp_indices(i-1))) > 3*pair_size)
        X1_temp= data.X(cp_indices(i) - pair_size -buff :cp_indices(i) - buff,:);
        X1_temp_1= data.X(cp_indices(i) - 2*pair_size-buff :cp_indices(i) - pair_size-buff,:);
        [A, B ] = size(X1_temp);
        X1_dissim = cat( 1, X1_dissim,  reshape(X1_temp,1,A,B));
        X1_dissim = cat(1,X1_dissim,reshape(X1_temp_1,1,A,B));
        Ys_X1_dissim(end+1) = data.Y(cp_indices(i) - buff);
        Ys_X1_dissim(end+1) = data.Y(cp_indices(i) - buff);
        X2_temp = data.X(cp_indices(i) + buff: cp_indices(i) + buff + pair_size,: );
        X2_temp_1 = data.X(cp_indices(i) + buff +pair_size : cp_indices(i) + buff+ 2*pair_size,: );
        [A, B ] = size(X2_temp);
        X2_dissim = cat( 1, X2_dissim,  reshape(X2_temp,1,A,B));
        X2_dissim = cat(1,X2_dissim,reshape(X2_temp_1,1,A,B));
        Ys_X2_dissim(end+1) = data.Y(cp_indices(i) + buff);
        Ys_X2_dissim(end+1) = data.Y(cp_indices(i) + buff);
    end
end 




for i = 2 : length(cp_indices)-1
    if cp_indices(i) - cp_indices(i-1) > 450
        X1_temp = data.X(cp_indices(i-1) + buff   : cp_indices(i-1) + pair_size+buff, :);
        [A, B ] = size(X1_temp);
        X1_sim = cat( 1, X1_sim,  reshape(X1_temp,1,A,B));
        Ys_X1_sim(end+1) = data.Y(cp_indices(i-1)+buff+2);
        X2_temp =  data.X(cp_indices(i)   -  pair_size -buff : cp_indices(i)  - buff, :);
        [A, B ] = size(X2_temp);
        X2_sim = cat( 1, X2_sim,  reshape(X2_temp,1,A,B));
        Ys_X2_sim(end+1) = data.Y(cp_indices(i)-buff);
    end
end 


[A1,~,~ ] = size(X1_sim);
[A2,~,~] = size(X1_dissim);

Y_dissim = -1*ones(A1,1);
Y_sim = ones(A2,1);

Y = [Y_dissim; Y_sim];
X1_label= [ Ys_X1_dissim' ; Ys_X1_sim' ];
X2_label = [Ys_X2_dissim' ; Ys_X2_sim'];
X1 = [X1_dissim ; X1_sim];
X2 = [X2_dissim ; X2_sim];

save(save_path,'X1','X2','Y','X1_label','X2_label');