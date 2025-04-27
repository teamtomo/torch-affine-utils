"""4x4 matrices for rotations and translations.

Functions in this module generate matrices which left-multiply column vectors containing
`xyzw` or `zyxw` homogenous coordinates.
"""

import torch
import einops


def Rx(
    angles: torch.Tensor,
    zyx: bool = False,
    device: torch.device | None = None
) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates around the X-axis.

    Matrix structure (xyzw):
    ┌             ┐
    │ 1  0   0  0 │
    │ 0  c  -s  0 │
    │ 0  s   c  0 │
    │ 0  0   0  1 │
    └             ┘

    Matrix structure (zyxw):
    ┌             ┐
    │ 1  0   0  0 │
    │ 0  c   s  0 │
    │ 0 -s   c  0 │
    │ 0  0   0  1 │
    └             ┘
    where c=cos(θ), s=sin(θ)

    Parameters
    ----------
    angles: torch.Tensor
        `(..., )` array of angles in degrees
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    # shape (...) -> (n, )
    angles = torch.as_tensor(angles, dtype=torch.float32, device=device)
    angles = torch.atleast_1d(angles)
    angles_packed, ps = einops.pack([angles], pattern='*')  # to 1d
    n = angles_packed.shape[0]

    # calculate useful values
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)

    # construct matrices
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, 1, 1] = c
    matrices[:, 1, 2] = -s
    matrices[:, 2, 1] = s
    matrices[:, 2, 2] = c

    # operating on zyx coordinates?
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))

    # shape (n, 4, 4) -> (..., 4, 4)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def Ry(
    angles: torch.Tensor,
    zyx: bool = False,
    device: torch.device | None = None
) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates around the Y-axis.

    Matrix structure (xyzw):
    ┌             ┐
    │ c  0  s  0  │
    │ 0  1  0  0  │
    │-s  0  c  0  │
    │ 0  0  0  1  │
    └             ┘

    Matrix structure (zyxw):
    ┌             ┐
    │ c  0 -s  0  │
    │ 0  1  0  0  │
    │ s  0  c  0  │
    │ 0  0  0  1  │
    └             ┘
    where c=cos(θ), s=sin(θ)

    Parameters
    ----------
    angles: torch.Tensor
        `(..., )` array of angles in degrees
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    # shape (...) -> (n, )
    angles = torch.as_tensor(angles, dtype=torch.float32, device=device)
    angles = torch.atleast_1d(angles)
    angles_packed, ps = einops.pack([angles], pattern='*')  # to 1d
    n = angles_packed.shape[0]

    # calculate useful values
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)

    # construct matrices
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 2] = s
    matrices[:, 2, 0] = -s
    matrices[:, 2, 2] = c

    # operating on zyx coordinates?
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))

    # shape (n, 4, 4) -> (..., 4, 4)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def Rz(
    angles: torch.Tensor,
    zyx: bool = False,
    device: torch.device | None = None
) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates around the Z-axis.

    Matrix structure (xyzw):
    ┌             ┐
    │ c -s  0  0  │
    │ s  c  0  0  │
    │ 0  0  1  0  │
    │ 0  0  0  1  │
    └             ┘

    Matrix structure (zyxw):
    ┌             ┐
    │ c  s  0  0  │
    │-s  c  0  0  │
    │ 0  0  1  0  │
    │ 0  0  0  1  │
    └             ┘
    where c=cos(θ), s=sin(θ)

    Parameters
    ----------
    angles: torch.Tensor
        `(..., )` array of angles in degrees
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    # shape (...) -> (n, )
    angles = torch.as_tensor(angles, dtype=torch.float32, device=device)
    angles = torch.atleast_1d(angles)
    angles_packed, ps = einops.pack([angles], pattern='*')  # to 1d
    n = angles_packed.shape[0]

    # calculate useful values
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))

    # shape (n, 4, 4) -> (..., 4, 4)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def T(shifts: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """4x4 matrices for translations.

    Matrix structure:
    ┌             ┐
    │ 1  0  0  t0 │
    │ 0  1  0  t1 │
    │ 0  0  1  t2 │
    │ 0  0  0  1  │
    └             ┘
    where t0, t1, t2 are translation components at indices 0, 1, 2 in the last
    dimension of `shifts`.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 3)` array of shifts.
    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 3)` array of shifts.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 4, 4)` array of 4x4 shift matrices.
    """
    # shape (...) -> (n, )
    shifts = torch.as_tensor(shifts, dtype=torch.float32, device=device)
    shifts = torch.atleast_1d(shifts)
    shifts, ps = einops.pack([shifts], pattern='* coords')  # to 2d
    n = shifts.shape[0]

    # construct matrices
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, :3, 3] = shifts

    # shape (n, 4, 4) -> (..., 4, 4)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def S(
    scale_factors: torch.Tensor,
    device: torch.device | None = None
) -> torch.Tensor:
    """4x4 matrices for scaling.

     Matrix structure:
    ┌             ┐
    │ s1 0  0  0  │
    │ 0  s2 0  0  │
    │ 0  0  s3 0  │
    │ 0  0  0  1  │
    └             ┘
    where s1, s2, s3 are the scaling factors at indices 0, 1, 2 in the last
    dimension of `scale_factors`.

    Parameters
    ----------
    scale_factors: torch.Tensor
        `(..., 3)` array of scale factors.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 4, 4)` array of 4x4 shift matrices.
    """
    # shape (...) -> (n, )
    scale_factors = torch.as_tensor(scale_factors, dtype=torch.float32, device=device)
    scale_factors = torch.atleast_1d(scale_factors)
    scale_factors, ps = einops.pack([scale_factors], pattern='* coords')  # to 2d
    n = scale_factors.shape[0]

    # construct matrices
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, [0, 1, 2], [0, 1, 2]] = scale_factors

    # shape (n, 4, 4) -> (..., 4, 4)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices
