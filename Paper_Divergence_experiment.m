%%%% The Fig 2 in the TCI paper
%% Non-convergent case VS convergence case
%% Convergence case includes 
%% To calculate directly the norm of slhl(wl)-s*h*(w*)=sl(Dlul)-s*(D*u*)
%% w^{l+1}-w^{*}
%% D^{l} = D*+\Xi^{l} ; and \belta^{l} = \belta*+\gamma^l
%% where \Xi^{l} and \gamma^l are random sequence
clc; clear all; close all;
load('ustar.mat');
ustar = u_ast;
load('Levin09blurdata/im05_flit02.mat');
k = rot90(f, 2); % blur kernel
y = conv2(x, k); % sharp image * kernel
varmin = 0.000001; mean = 0;
y = imnoise(y,'gaussian',mean,varmin);
lmax = 30;
for L = 1:1:lmax
options.max_iter = L;
options.is_isotropic = true;
options.beta = 2e3;
[x_hat_L1,cost1] = deconv_hqs_divergence(y, k,options);
u_L = x_hat_L1;
Dxstar = [-1, 1; 0, 0];
Dystar = [-1, 0; 1, 0];
Dx_L = Dxstar + normrnd(0,L/60,2,2);
Dy_L = Dystar + normrnd(0,L/60,2,2);
h_L_wx_L = conv2(u_L,Dx_L);
h_L_wy_L= conv2(u_L,Dy_L);
belta_ast = 2e3; 
belta_L = belta_ast + normrnd(0,L/60,1,1);
lambda_L = 1/ belta_L;
[S_L_wx_L, S_L_wy_L] = thresh_l2(h_L_wx_L, h_L_wy_L, lambda_L);

h_star_wx_star = conv2(ustar,Dxstar);
h_star_wy_star = conv2(ustar,Dystar);
Lambda_star = 1/ belta_ast;
[S_star_wx_star, S_star_wy_star] = thresh_l2(h_star_wx_star, h_star_wy_star, Lambda_star);

Diff_x = S_L_wx_L - S_star_wx_star;
Diff_y = S_L_wy_L - S_star_wy_star;
Diff = [Diff_x Diff_y];
Diff_star = [S_star_wx_star S_star_wy_star];
Diff_norm_div(L) = norm(Diff)./norm(Diff_star); % \|w^l+1-w^*\|./\|w^*\|
end
Dl = 0.8;
lmax = 30;
for L = 1:1:lmax
options.max_iter = L;
options.is_isotropic = true;
options.beta = 2e3;
convergence_base_gradient_filter = Dl;
convergence_base_beta = Dl;
[x_hat_L,cost1] = deconv_hqs_conver(y,k,convergence_base_gradient_filter,convergence_base_beta,options);
u_L = x_hat_L;
Dxstar = [-1, 1; 0, 0];
Dystar = [-1, 0; 1, 0];
Dx_L = Dxstar + convergence_base_gradient_filter.^(L).*ones(2);
Dy_L = Dystar + convergence_base_gradient_filter.^(L).*ones(2);
h_L_wx_L = conv2(u_L,Dx_L);
h_L_wy_L = conv2(u_L,Dy_L);
belta_ast = 2e3; 
belta_L = belta_ast + Dl.^(L);
lambda_L = 1/ belta_L;
[S_L_wx_L, S_L_wy_L] = thresh_l2(h_L_wx_L, h_L_wy_L, lambda_L);

h_star_wx_star = conv2(ustar,Dxstar);
h_star_wy_star = conv2(ustar,Dystar);
Lambda_star = 1/ belta_ast;
[S_star_wx_star, S_star_wy_star] = thresh_l2(h_star_wx_star, h_star_wy_star, Lambda_star);

Diff_x = S_L_wx_L - S_star_wx_star;
Diff_y = S_L_wy_L - S_star_wy_star;
Diff = [Diff_x Diff_y];
Diff_star = [S_star_wx_star S_star_wy_star];
Diff_norm_conver_D(L) = norm(Diff)./norm(Diff_star); % \|w^l+1-w^*\|./\|w^*\|
end
p = 1;
for L = 1:1:lmax
options.max_iter = lmax;
options.is_isotropic = true;
options.beta = 2e3;
options. p = p;
[x_hat_L,cost1] = deconv_hqs_conver_frac(y,k,options);
u_L = x_hat_L;
Dxstar = [-1, 1; 0, 0];
Dystar = [-1, 0; 1, 0];
Dx_L = Dxstar + 1/(L.^p).*ones(2);
Dy_L = Dystar + 1/(L.^p).*ones(2);
h_L_wx_L = conv2(u_L,Dx_L);
h_L_wy_L = conv2(u_L,Dy_L);
belta_ast = 2e3; 
belta_L = belta_ast + 1/(L.^p);
lambda_L = 1/ belta_L;
[S_L_wx_L, S_L_wy_L] = thresh_l2(h_L_wx_L, h_L_wy_L, lambda_L);

h_star_wx_star = conv2(ustar,Dxstar);
h_star_wy_star = conv2(ustar,Dystar);
Lambda_star = 1/ belta_ast;
[S_star_wx_star, S_star_wy_star] = thresh_l2(h_star_wx_star, h_star_wy_star, Lambda_star);

Diff_x = S_L_wx_L - S_star_wx_star;
Diff_y = S_L_wy_L - S_star_wy_star;
Diff_star = [S_star_wx_star S_star_wy_star];
Diff = [Diff_x Diff_y];
Diff_norm_frac(L) = norm(Diff)./norm(Diff_star);
end
figure(1);
plot(1:1:lmax,Diff_norm_div,'k',1:1:lmax,Diff_norm_conver_D,'b',1:1:lmax,Diff_norm_frac,'r','LineWidth',1);
ylabel('Error','FontSize',11); xlabel('Iteration','FontSize',11); 
leg = legend('\sigma_{\xi^l} = l/60 and \sigma_{\beta^l} = l/60','\xi^l = 0.8^l and \gamma^l = 0.8^l','p = 1');
set(leg,'FontName','Times New Roman','FontSize',10.5,'FontWeight','normal')
grid on;