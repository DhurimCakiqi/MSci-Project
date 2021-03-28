%% for shear experiments using the standard HO model 
function [sig_fs_fs, sig_sf_fs, sig_fn_fn, sig_nf_fn, ...
          sig_ns_sn, sig_sn_sn] =shears_6experiments( ...
          a, b, af, bf, as, bs, afs, bfs, gamma)
      
      %% shear in fs along f0
      F = [1, gamma, 0; ...
           0, 1,     0; ...
           0, 0,     1];
      [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs);
      sig_sf_fs = 2*(phi_1+ phi_4s)*gamma + phi_8fs;
      %% an alternative way to clacualte the shear stress
      sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs);
      sig_sf_fs_1 = sigma_tensor(1,2);
      if ( abs(sig_sf_fs_1- sig_sf_fs) > 1e-6)
          disp('something wrong in sig_sf_fs');
          pause;
      end
      
      %% shear in fs along s0
      F = [1, 0, 0; ...
           gamma, 1,     0; ...
           0, 0,     1];
      [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs);
      sig_fs_fs = 2*(phi_1+ phi_4f)*gamma + phi_8fs;
      %% an alternative way to clacualte the shear stress
      sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs);
      sig_fs_fs_1 = sigma_tensor(1,2);
      if ( abs(sig_fs_fs_1- sig_fs_fs) > 1e-6)
          disp('something wrong in sig_fs_fs');
          pause;
      end
      
      
     %% shear in the sn plane along s0
     F = [1, 0, 0; ...
           0, 1,     gamma; ...
           0, 0,     1];
      [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs);
      sig_ns_sn = 2*(phi_1)*gamma;
      %% an alternative way to clacualte the shear stress
      sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs);
      sig_ns_sn_1 = sigma_tensor(2,3);
      if ( abs(sig_ns_sn_1- sig_ns_sn) > 1e-6)
          disp('something wrong in sig_ns_sn');
          pause;
      end
      
      %% shear in the sn along n0
      F = [1, 0, 0; ...
           0, 1,     0; ...
           0, gamma,     1];
      [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs);
      sig_sn_sn = 2*(phi_1+ phi_4s)*gamma;
      %% an alternative way to clacualte the shear stress
      sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs);
      sig_sn_sn_1 = sigma_tensor(2,3);
      if ( abs(sig_sn_sn_1- sig_sn_sn) > 1e-6)
          disp('something wrong in sig_sn_sn');
          pause;
      end
      
      
      %% shear in the fn along f0
      F = [1, 0, gamma; ...
           0, 1,     0; ...
           0, 0,     1];
      [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs);
      sig_nf_fn = 2*(phi_1)*gamma;
      %% an alternative way to clacualte the shear stress
      sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs);
      sig_nf_fn_1 = sigma_tensor(1,3);
      if ( abs(sig_nf_fn_1- sig_nf_fn) > 1e-6)
          disp('something wrong in sig_nf_fn');
          pause;
      end
      
       %% shear in the fn along n0
      F = [1, 0, 0; ...
           0, 1,     0; ...
           gamma, 0,     1];
      [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs);
      sig_fn_fn = 2*(phi_1 + phi_4f)*gamma;
      %% an alternative way to clacualte the shear stress
      sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs);
      sig_fn_fn_1 = sigma_tensor(1,3);
      if ( abs(sig_fn_fn_1- sig_fn_fn) > 1e-6)
          disp('something wrong in sig_fn_fn');
          pause;
      end
      
      
 return       
       
       
       
       
function [phi_1, phi_4f, phi_4s, phi_8fs]=  dWHO_dI(F,a, b, af, bf, as, bs, afs, bfs)
%% this fuction will try to figure out the derivatives with respect strain invariants
B = F*F';
f0 = [1; 0; 0];
s0 = [0; 1; 0];
n0 = [0; 0; 1];

f = F*f0;
s = F*s0; 
n = F*n0;

I1 = B(1,1) + B(2,2) + B(3,3);
I4f = dot(f, f);
I4s = dot(s,s);
I8fs = dot(f,s);

phi_1 = a/2*exp( b*(I1-3) );
phi_4f = af*(I4f-1)*exp( bf*(I4f-1)^2  );
phi_4s = as*(I4s-1)*exp( bs*(I4s-1)^2  );
phi_8fs = afs*I8fs*exp( bfs*I8fs^2 );

function sigma_tensor = compute_stress(F,a, b, af, bf, as, bs, afs, bfs)
B = F*F';

f0 = [1; 0; 0];
s0 = [0; 1; 0];
n0 = [0; 0; 1];

f = F*f0;
s = F*s0; 
n = F*n0;


I1 = trace(B);
I4f = f'*f; 
I4s = s'*s;
I8fs = f'*s;

fxf = f*f';
sxs = s*s';
fxs = f*s';
sxf = s*f';

if I4f < 1
    I4f = 1;
end
if I4s < 1
    I4s = 1;
end

sigma_tensor = a*exp( b*(I1 - 3) )*B + ...
    2*af*(I4f-1)*exp( bf*(I4f-1)^2 )*fxf+ ...
    2*as*(I4s-1)*exp( bs*(I4s-1)^2 )*sxs+ ...
    afs*I8fs*exp( bfs*I8fs^2 )*( fxs + sxf );




        


