%%% The Fig 4 (a) and (b)
%% To calculate directly the norm of w^l+1-w^*=slhl(wl)-s*h*(w*)=sl(Dlul)-s*(D*u*)
%% D^l = D^*+\Xi^l;  \belta^{l} = \belta*+\gamma^l
%% \Xi^l and \gamma^l (1/L)^p with p =1,2,3,4
clc; clear all; close all;
load('ustar.mat');
ustar = u_ast;
load('Levin09blurdata/im05_flit02.mat');
k = rot90(f, 2); % blur kernel
y = conv2(x, k); % sharp image * kernel
varmin = 0.000001; mean = 0;
y = imnoise(y,'gaussian',mean,varmin);
lmax = 30;
for p = 1:1:4
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
Diff_norm(L) = norm(Diff)./norm(Diff_star);
end
Diff_Total(p,:) = Diff_norm;
end
figure(1);
plot(1:1:lmax,Diff_Total(1,:),'k',1:1:lmax,Diff_Total(2,:),'b',1:1:lmax,Diff_Total(3,:),'g',1:1:lmax,Diff_Total(4,:),'r','LineWidth', 1); 
ylabel('Error','FontSize',11); xlabel('Iteration','FontSize',11); 
leg = legend('p=1','p=2','p=3','p=4');
set(leg,'FontName','Times New Roman','FontSize',10.5,'FontWeight','normal')
grid on;
figure(2);
plot(1:1:lmax,log(Diff_Total(1,:)),'k',1:1:lmax,log(Diff_Total(2,:)),'b',1:1:lmax,log(Diff_Total(3,:)),'g',1:1:lmax,log(Diff_Total(4,:)),'r','LineWidth', 1); 
ylabel('Error','FontSize',11); xlabel('Iteration','FontSize',11); 
leg = legend('p=1','p=2','p=3','p=4');
set(leg,'FontName','Times New Roman','FontSize',10.5,'FontWeight','normal')
grid on;
