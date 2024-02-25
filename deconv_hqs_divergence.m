function [x,cost] = deconv_hqs_divergence(y, k,options)
    % Deconvolution using alternating minimization
    % \Xi^l and \beta^l random sequence with respect to l
    % Inputs:
    %  y: blurred image
    %  k: blur kernel
    %  options: struct holding parameters and configurations
    % Outputs:
    %  x: deblurred latent image
    % 
    % Parse input arguments
    if isfield(options, 'max_iter')
        max_iter = options.max_iter;
    else
        max_iter = 1000;
    end
    if isfield(options, 'is_isotropic')
        is_isotropic = options.is_isotropic;
    else
        is_isotropic = true;
    end
    if isfield(options, 'beta')
        beta = options.beta;
    else
        beta  = 2e5;
    end
    if isfield(options, 'mu')
        mu = options.mu;
    else
        mu = 5e4;
    end

    [mk, nk] = size(k);
    [mb, nb] = size(y);
    % Size of the latent image
    mi = mb - mk + 1;
    ni = nb - nk + 1;
    % Gradient filters
    fx = [-1, 1; 0, 0];
%     +normrnd(Mu, Sigma, 2, 2);
    fy = [-1, 0; 1, 0];
%     +normrnd(Mu, Sigma, 2, 2);
    mg = mi + max(size(fx, 1), size(fy, 1)) - 1;
    ng = ni + max(size(fx, 2), size(fy, 2)) - 1;
    % Optimal size for FFT
    mf = 2^nextpow2(max(mg, mb));
    nf = 2^nextpow2(max(ng, nb));
    % Some terms can be pre-computed for efficiency
    Ffx = fft2(fx, mf, nf);
    Ffy = fft2(fy, mf, nf);
    Fk = fft2(k, mf, nf);
    gx = zeros(mg, ng);
    gy = zeros(mg, ng);
    Fy = fft2(y, mf, nf);

    iter = 0;
    while iter < max_iter
        betal = beta + normrnd(0,(iter+1)/60,1,1) ;
        if is_isotropic
            [zx, zy] = thresh_l2(gx, gy, 1 / (betal));
        else
            zx = thresh_l1(gx, 1 / (betal));
            zy = thresh_l1(gy, 1 / (betal));
        end
        Fzx = fft2(zx, mf, nf);
        Fzy = fft2(zy, mf, nf);
        num = conj(Ffx).*Fzx + conj(Ffy).*Fzy + mu/(betal)*conj(Fk).*Fy;
        den = abs(Ffx).^2 + abs(Ffy).^2 + mu/(betal)*abs(Fk).^2;
        x = real(ifft2(num ./ den, mf, nf));
        x = x(1: mi, 1: ni);
        % here, we add Gaussian array for convergence array modeling
       % gx = conv2(x, fx+(convergence_base_gradient_filter^(iter+1)).*fspecial('gaussian',[2 2], sigma));
        fxl = fx + normrnd(0,(iter+1)/60,2,2);
        gx = conv2(x, fxl);
        
       %  gx = conv2(x, fx+(1/((iter+1).^2)));
%         gx = conv2(x, fx+(1/2^((iter+1))));
       % gy = conv2(x, fy+(convergence_base_gradient_filter^(iter+1)).*fspecial('gaussian',[2 2], sigma));
        fyl = fy + normrnd(0,(iter+1)/60,2,2);
        gy = conv2(x, fyl);
       %   gy = conv2(x, fy+(1/((iter+1).^2)));
        temp = 0.5 * mu * norm(conv2(x, k) - y, 'fro')^2 ...
            + 0.5 * (betal) * (norm(zx - gx, 'fro')^2 + norm(zy - gy, 'fro')^2);
        if is_isotropic
            cost(iter+1) = temp + norm(sqrt(gx(:).^2 + gy(:).^2), 1);
        else
            cost(iter+1) = temp + norm(gx(:), 1) + norm(gy(:), 1);
        end
        fprintf('Iteration %03d, cost %.2f\n', iter, cost(iter+1));
        iter = iter + 1;
%         figure(1);  % compare blurred and deblurred images
%         subplot(121), imshow(y);
%         subplot(122), imshow(x);
%         figure(2);  % visualize image gradients
%         subplot(121), imshow(zx, []);
%         subplot(122), imshow(zy, []);
%         pause(0.0001);  % allow some time for the figure to show up
    end
end

function z = thresh_l1(x, lambda)
    % Soft-thresholding for anisotropic total variation
    z = sign(x).*max(abs(x) - lambda, 0);
end

function [zx, zy] = thresh_l2(gx, gy, lambda)
    % Soft-thresholding for isotropic total variation
    g = sqrt(gx.^2 + gy.^2);
    zx = max(g - lambda, 0) .* gx ./ (g + eps);
    zy = max(g - lambda, 0) .* gy ./ (g + eps);
end
