function [zx, zy] = thresh_l2(gx, gy, lambda)
    % Soft-thresholding for isotropic total variation
    g = sqrt(gx.^2 + gy.^2);
    zx = max(g - lambda, 0) .* gx ./ (g + eps);
    zy = max(g - lambda, 0) .* gy ./ (g + eps);
end