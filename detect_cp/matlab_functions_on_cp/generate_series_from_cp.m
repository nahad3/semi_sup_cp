 data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/cp_detected_hci_cp_untrunc_guided_raw_rescaled01.mat');
 close all
 ax1 = subplot(2,1,1)
 plot(data.X)
 
 ax2 = subplot(2,1,2)
 plot(data.Cp_stat)
 hold on
%[~, data.Cp] = findpeaks(data.Cp,'MinPeakDistance',200,'MinPeakHeight',1);
 scatter(data.c_indices,data.Cp_stat(data.c_indices));
 
 linkaxes([ax1,ax2],'x')
 
 
 [segs,Z,Y] = get_segs_and_generate_series(data.X,data.c_indices,30, 6);
 
 
 
 X = Z;d

 save('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/real_world_data/HCI_continuous/hci_cp_augmented_scaled.mat','X','Y')