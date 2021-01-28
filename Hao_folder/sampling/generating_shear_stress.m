%%% generating the stress data from shear experiments 

%% paramters from Table 1 for the shear in figure 7, the HO paper
a = 0.59; 
b = 8.023;
af = 18.472;
bf = 16.026;
as = 2.481;
bs = 11.120;
afs = 0.216;
bfs = 11.436;

gamma_data = 0:0.01:0.5;

for i = 1 : length(gamma_data)
    gamma = gamma_data(i);
    
    [sig_fs_fs, sig_sf_fs, sig_fn_fn, sig_nf_fn, ...
              sig_ns_sn, sig_sn_sn] = shears_6experiments( ...
              a, b, af, bf, as, bs, afs, bfs, gamma);
     
    Sig_fs_fs(i) = sig_fs_fs;
    Sig_sf_fs(i) = sig_sf_fs;
    Sig_nf_fn(i) = sig_nf_fn;
    Sig_fn_fn(i) = sig_fn_fn;
    Sig_ns_sn(i) = sig_ns_sn;
    Sig_sn_sn(i) = sig_sn_sn;
    
end


figure; hold on;
plot(gamma_data, Sig_fs_fs, 'LineWidth', 2);
plot(gamma_data, Sig_sf_fs, 'LineWidth', 2);
plot(gamma_data, Sig_nf_fn, 'LineWidth', 2);
plot(gamma_data, Sig_fn_fn, 'LineWidth', 2);
plot(gamma_data, Sig_ns_sn, 'LineWidth', 2);
plot(gamma_data, Sig_sn_sn, 'LineWidth', 2);
legend('fs\_fs', 'sf\_fs', 'nf\_fn', 'fn\_fn', 'ns\_sn', 'sn\_sn');

      