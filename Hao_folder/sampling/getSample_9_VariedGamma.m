%% this will generate Nsmaples of parameter sets and also correponding 
%% shear stress from 6 different simple shears

close all; clc,clear;
addpath('./lhs_sampling');
% Latin hypercube sampling from uniform distribution.
%     a     b   af   bf   as    bs  afs  bfs gamma
lb = [0.05 0.1, 0.1, 0.1, 0.05, 0.1 0.05 0.1, 0.001];
ub = [10,   10, 10,  10,  10,   10, 10,  10, 0.5];
Nsamples = 5000;
% samples = lhsu(ones(1,4)*0.1, ones(1,4)*5, 10000);
samples = lhsu(lb, ub, Nsamples);
% save('sample.mat','sample')


%%% now calculate the shear stress for each set of parameters 
samples_para_stress = zeros( [Nsamples, 15] ); 
for i = 1 : size(samples, 1)
    a = samples(i,1);
    b = samples(i,2);
    af = samples(i,3);
    bf = samples(i,4);
    as = samples(i,5);
    bs = samples(i,6);
    afs = samples(i,7);
    bfs = samples(i,8);
    gamma = samples(i,9);
    
    samples_para_stress(i,1:9) = samples(i, 1:9);
    
    [sig_fs_fs, sig_sf_fs, sig_fn_fn, sig_nf_fn, ...
              sig_ns_sn, sig_sn_sn] = shears_6experiments( ...
              a, b, af, bf, as, bs, afs, bfs, gamma);
          
    samples_para_stress(i,10:15)= [sig_fs_fs, sig_sf_fs, sig_fn_fn, ...
                                   sig_nf_fn,sig_ns_sn,sig_sn_sn  ];
    
end

%% save the data for ML
% xlswrite('samples_only_para.xlsx', samples)
xlswrite('samples_para_stress_variedGamma.xlsx', samples_para_stress);



%% plot the distriubtions for inspection
figure
boxplot([samples(:,1), samples(:,2),samples(:,3), samples(:,4),...
         samples(:,5), samples(:,6),samples(:,7), samples(:,8), ...
         samples(:,9)],'Notch','on','Labels',{'a','b','af','bf', 'as','bs','afs','bfs','gamma'});


figure
boxplot([ standaraize_data(samples_para_stress(:,10)), ...
         standaraize_data(samples_para_stress(:,11)),...
    standaraize_data(samples_para_stress(:,12)),... 
    standaraize_data(samples_para_stress(:,13)),...
    standaraize_data(samples_para_stress(:,14)),... 
    standaraize_data(samples_para_stress(:,15))],'Notch','on',...
  'Labels',{'fs_fs','sf_fs','fs_fn','nf_fn', 'ns_sn','sn_sn'});



