clear all
data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/hci_cp_untrunc_freehand_raw_rescaled01.mat');


%data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/human_activity/mat_files/Train_scaled_0_1_sim_dissim_from_cp_with_labels_ActivTraker2011_.mat');
close all


X = data.X;
ax1 = subplot(2,1,1)
plot(data.X)



ax2 = subplot(2,1,2)
plot(data.Y)

linkaxes([ax1,ax2],'x')


Y = data.Y;
c_indices = diff(data.Y);
c_indices = find(c_indices ~= 0) ;
[X,Y]= segs_from_cps(X, Y, c_indices);

idx = randperm(length(X));
Y = Y(idx);
X = X(idx);
save('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_HCI.mat','X','Y');