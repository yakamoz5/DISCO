import torch


def ensure_2d(tensor):
    """Ensure the input tensor is 2-dimensional.

    Converts a 1-D tensor to shape ``(n, 1)`` and raises an error for tensors
    with more than 2 dimensions. Non-tensor inputs are first converted to a
    ``torch.float32`` tensor.

    Args:
        tensor: Input array or tensor of shape ``(n,)``, ``(n, d)``, or
            batched higher-dimension shapes (which raise).

    Returns:
        torch.Tensor: A 2-D tensor of shape ``(n, d)`` or ``(n, 1)``.
    """
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)  # Convert shape (n,) → (n,1)
    elif tensor.ndim > 2:
        raise ValueError(
            f"Expected a 1D or 2D tensor, but got shape {tensor.shape}"
        )
    return tensor


def sdisco_metric(X, Y, Z, h=1.0, eps=1e-12, method="mean"):
    r"""Compute the s-differential correlation metric :math:`\widehat{\rho}_n`.

    The metric measures the conditional correlation :math:`\rho(X, Y \mid Z)`
    using a kernel-based local weighting scheme.  Gradients **only** flow
    through ``X`` (model predictions); ``Y`` (bias attribute) and ``Z``
    (conditioning variable) are detached from the computation graph.

    Args:
        X: Model predictions (differentiable tensor).  Shape ``(n, d_x)``
            or ``(n,)``.
        Y: Bias attribute / Protected variable.  Shape ``(n, d_y)`` or
            ``(n,)``.
        Z: Conditioning variable / Target label.  Shape ``(n, d_z)`` or
            ``(n,)``.
        h: Bandwidth parameter for the Gaussian kernel on ``Z``.
            Default: ``1.0``.
        eps: Small numerical-stability epsilon used in weight normalization
            and variance clamping.  Default: ``1e-12``.
        method: Aggregation method across samples.  One of ``"mean"``,
            ``"max"``, or ``"standard"``.  Default: ``"mean"``.

    Returns:
        torch.Tensor: A scalar tensor containing the aggregated
        :math:`\widehat{\rho}_n` value.
    """
    n = X.shape[0]

    # X requires gradients (model predictions)
    X = ensure_2d(X).float()

    # ---------------------------------------------------------
    # NON-GRADIENT BLOCK: Target (Y) and Bias (Z) operations
    # ---------------------------------------------------------
    with torch.no_grad():
        Y = ensure_2d(Y).float()
        Z = ensure_2d(Z).float()

        # Pairwise distances for Y and Z
        DY = torch.cdist(Y, Y, p=2)
        D2_Z = torch.cdist(Z, Z, p=2) ** 2

        # Row-normalized weights W
        Kz = torch.exp(-D2_Z / (2.0 * h * h))
        W = Kz / (Kz.sum(dim=1, keepdim=True) + eps)  # (n, n)

        # Local and Grand Means for Y
        MY = W @ DY  # (n, n)
        gY = (W * MY).sum(dim=1)  # (n,)

        # --- Compute Variance V(Y,Y|Z) ---
        DYY = DY.pow(2)
        T1_Y = (W * (W @ DYY)).sum(dim=1)
        T2_Y = -2.0 * (W * MY * MY).sum(dim=1)
        T3_Y = gY * gY
        varY = T1_Y + T2_Y + T3_Y  # (n,)

    # ---------------------------------------------------------
    # GRADIENT BLOCK: Prediction (X) operations
    # ---------------------------------------------------------
    # 1. Pairwise Distances for X
    DX = torch.cdist(X, X, p=2)

    # 2. Local Row Means via MatMul
    MX = W @ DX  # W does not require grad, DX does

    # 3. Local Grand Means
    gX = (W * MX).sum(dim=1)  # (n,)

    # --- Compute Covariance V(X,Y|Z), see paper for proof ---
    DXY = DX * DY  # Element-wise product
    M_XY = W @ DXY
    T1 = (W * M_XY).sum(dim=1)             # Sum of w_i w_j D^X_ij D^Y_ij
    T2 = -2.0 * (W * MX * MY).sum(dim=1)   # -2 Sum of w_i m^X_i m^Y_i
    T3 = gX * gY                           # + g^X g^Y

    num = T1 + T2 + T3  # (n,) - local numerators

    # --- Compute Variance V(X,X|Z), see paper for proof ---
    DXX = DX.pow(2)
    T1_X = (W * (W @ DXX)).sum(dim=1)
    T2_X = -2.0 * (W * MX * MX).sum(dim=1)
    T3_X = gX * gX
    varX = T1_X + T2_X + T3_X  # (n,)

    # --- Correlation ---
    rho2_locals = num / torch.sqrt((varX * varY + 1e-12).clamp(min=eps))
    rho2_locals = rho2_locals.clamp(min=0.0)

    rho_locals = torch.sqrt(rho2_locals + 1e-12)

    if method in ["max", "standard"]:
        rho_global = rho_locals.max()
    elif method == "mean":
        rho_global = rho_locals.mean()

    return rho_global


# debug and print
if __name__ == "__main__":
    X = torch.randn(1000, 3)
    Y = torch.randn(1000, 3)
    Z = torch.randn(1000, 3)

    print(sdisco_metric(X, Y, Z, h=0.5,))