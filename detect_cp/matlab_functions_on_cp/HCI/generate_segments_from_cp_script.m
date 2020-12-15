%This script takes in finer cp detected on HCI dataset and generates
%segments (X) along with their labels



%handguided data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/cp_detected_hci_cp_untrunc_guided_raw_rescaled01.mat');
%nadguided data2 = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/hci_cp_untrunc_guided_raw_rescaled01.mat');

%freehand
rng('default')
s = rng

data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/cp_detected_hci_cp_untrunc_freehand_raw_rescaled01.mat');
data2  = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/hci_cp_untrunc_freehand_raw_rescaled01.mat');

Y = data2.Y(1000:end);
%data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/human_activity/mat_files/Train_scaled_0_1_sim_dissim_from_cp_with_labels_ActivTraker2011_.mat');
close all


X = data.X;
ax1 = subplot(2,1,1)
plot(data.X)

Y_diff = [  diff(Y)];
Y_diff = (Y_diff ~= 0);

ax2 = subplot(2,1,2)
plot(data.Cp_stat)
hold on
scatter(data.c_indices, data.Cp_stat(data.c_indices))
plot(Y)
linkaxes([ax1,ax2],'x')


c_indices = diff(Y);
c_indices = find(c_indices ~= 0) ;


[X,Y]= segs_from_cps(data.X, Y, c_indices);
idx = randperm(length(X));
Y = Y(idx);
X = X(idx);

train_len = 50;

Y_true = Y;
X_true = X;

Y = Y_true(1:50);
X = X_true(1:50);
save('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_train_HCI.mat','X','Y');


Y = Y_true(50:end);
X = X_true(50:end);
save('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_test_HCI.mat','X','Y');
