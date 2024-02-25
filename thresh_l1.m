function z = thresh_l1(x, lambda)
    % Soft-thresholding for anisotropic total variation
    z = sign(x).*max(abs(x) - lambda, 0);
end