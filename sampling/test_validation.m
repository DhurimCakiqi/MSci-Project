%% for generateing finally test case
clear all; close all; clc;
samples = [2,	5,	3,	6,	8,	7,	4,	8];
gamma = 0:0.01:0.5;

for i = 1 : length(gamma)
    a = samples(1);
    b = samples(2);
    af = samples(3);
    bf = samples(4);
    as = samples(5);
    bs = samples(6);
    afs = samples(7);
    bfs = samples(8);
    gamma_t = gamma(i);
    
    [sig_fs_fs, sig_sf_fs, sig_fn_fn, sig_nf_fn, ...
              sig_ns_sn, sig_sn_sn] = shears_6experiments( ...
              a, b, af, bf, as, bs, afs, bfs, gamma_t);
          
    samples_para_stress(i,1:7)= [sig_fs_fs, sig_sf_fs, sig_fn_fn, ...
                                   sig_nf_fn,sig_ns_sn,sig_sn_sn, gamma_t  ];
    
end

xlswrite('test_stress_variedGamma.xlsx', samples_para_stress);


