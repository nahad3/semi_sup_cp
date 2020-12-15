data = load('/Users/naumanahad/Documents/GitHub/semisup_with_cp/semi_ts_clustering/data_files/gen_simulations/mat_files/mean_var_change/mean_var_chg_train_labels.mat');



X = data.X;
N = length(X)
window = 50;
q = 0.99;
delta_array = zeros(N,1);
thresh = 100;
%thresh  = 60;
 change_vec_mean = [];
 change_vec_var = [];
 X_p = X( 1 : window);
 pre_mean = 0;
 pre_var = var(X_p);
 pre_var = 0.1;
 second_mean_vec = [];
 first_mean_vec = [];
  
  in_seat_changes = [];
  

  first_mean_vec(end+1) = pre_mean;
  
  second_change_list = [];
  first_change_list = [];
  cp_stat_array = zeros(100000,1);
for i = window+1 :1: 100000 - window
    
    X_f =  X(  i : i + window );
  
    X_p =  X(i - window:i);
    
    pre_mean = mean(X_p);
    pre_var = var(X_p);
    post_mean = mean(X_f);
    post_var = var(X_f);
    cp_stat_array(i) = kl_divergence([pre_mean, pre_var],[post_mean post_var]) + kl_divergence([post_mean post_var], [pre_mean, pre_var]) ;

        
end
    
[pks,loc] = findpeaks(cp_stat_array,'MinPeakHeight',2,'MinPeakDistance',300)
close all
ax1 = subplot(2,1,1)
plot(X(1:100000))
hold on
ax2 = subplot(2,1,2)
plot(cp_stat_array)
hold on
scatter(loc,cp_stat_array(loc));
ylabel('Change statistic')
xlabel('Time(seconds)')
set(gca,'FontSize',14)

linkaxes([ax1,ax2],'x')