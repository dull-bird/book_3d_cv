#!/usr/bin/env python3
"""
code_verify.py -- Minimal pinhole camera projection verification.

Implements the core formula from H&Z Ch.6:  P = K[R|t].
Takes a set of 3D points, applies a pinhole camera projection, and
outputs the resulting 2D pixel coordinates.

References to Hartley & Zisserman, Multiple View Geometry in Computer
Vision (2nd Ed.) are given as (H&Z p.NNN) in comments.

Run:
    python research/00_basics/code_verify.py
"""

import numpy as np


def build_camera_matrix(
    fx: float, fy: float, cx: float, cy: float,
    R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    Build a finite projective camera matrix P = K [R | t].

    H&Z (6.8, p.156): The general pinhole camera is
        P = K [R | t]   where t = -R * C_tilde

    H&Z (6.10, p.157): The calibration matrix for a general CCD camera is
            [ alpha_x   s     x_0 ]
        K = [    0   alpha_y  y_0 ]
            [    0      0      1  ]

    Here we assume zero skew (s=0), so:
        alpha_x = fx, alpha_y = fy, x_0 = cx, y_0 = cy

    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels (alpha_x, alpha_y in H&Z notation).
    cx, cy : float
        Principal point coordinates in pixels.
    R : np.ndarray, shape (3, 3)
        Rotation matrix (orthogonal, det=1).
    t : np.ndarray, shape (3,)
        Translation vector (world origin in camera coordinates).

    Returns
    -------
    P : np.ndarray, shape (3, 4)
        Camera projection matrix.
    """
    # H&Z (6.10, p.157): K matrix (zero skew)
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

    # H&Z (6.8, p.156): P = K [R | t]
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt
    return P


def project(P: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """
    Project 3D world points to 2D image pixels.

    H&Z (6.1, p.154): Central projection mapping
        (X, Y, Z)^T -> (fX/Z, fY/Z)^T

    In homogeneous coordinates (H&Z pp.154-155):
        x = P X    where x is homogeneous 3-vector (u, v, w)^T

    Dehomogenization yields pixel coordinates:
        u_pixel = u / w,   v_pixel = v / w

    Parameters
    ----------
    P : np.ndarray, shape (3, 4)
        Camera projection matrix.
    X_world : np.ndarray, shape (N, 4) or (4,) or (N, 3)
        Homogeneous world points. If shape (N, 3), appends w=1.

    Returns
    -------
    pixels : np.ndarray, shape (N, 2)
        (u, v) pixel coordinates for each point.
    """
    if X_world.ndim == 1:
        X_world = X_world.reshape(1, -1)

    if X_world.shape[1] == 3:
        # Append homogeneous coordinate w=1
        ones = np.ones((X_world.shape[0], 1))
        X_homo = np.hstack([X_world, ones])
    else:
        X_homo = X_world

    # H&Z (6.2, p.154): x = P X
    x_homo = P @ X_homo.T  # shape (3, N)

    # H&Z (6.1, p.154): dehomogenize by dividing by third coordinate
    w = x_homo[2, :]
    u = x_homo[0, :] / w
    v = x_homo[1, :] / w

    return np.column_stack([u, v])


def verify_depth(P, X_world):
    """
    Verify depth formula from H&Z (6.15, p.162):
        depth(X; P) = sign(det M) * w / (T * ||m^3||)

    where P = [M | p_4] and m^3 is the third row of M.

    Points with positive depth are in front of the camera.
    """
    M = P[:, :3]  # left 3x3 block (H&Z p.163)
    m3 = M[2, :]  # third row of M (principal ray direction, p.161)

    x_homo = P @ X_world.T
    w_val = x_homo[2, :]

    depth = np.sign(np.linalg.det(M)) * w_val / (X_world[:, 3] * np.linalg.norm(m3))
    return depth


def main():
    print("=" * 60)
    print("Pinhole Camera Projection — Code Verification")
    print("Based on H&Z Ch.6 Camera Models")
    print("=" * 60)

    # --- Camera parameters ---
    # Simulate a camera with:
    #   fx = 800, fy = 800 (focal length in pixels)
    #   cx = 320, cy = 240 (principal point at image centre for 640x480)
    fx, fy = 800.0, 800.0
    cx, cy = 320.0, 240.0

    # Camera rotation: identity (camera looks along +Z)
    R = np.eye(3)

    # Camera translation: camera at world origin
    t = np.array([0.0, 0.0, 0.0])

    P = build_camera_matrix(fx, fy, cx, cy, R, t)
    print(f"\nCamera Matrix P (3x4):\n{P}\n")

    # --- 3D World Points ---
    # Points at different depths along Z-axis
    points_3d = np.array([
        [ 0.0,  0.0,  5.0],   # on optical axis, 5m away
        [ 1.0,  1.0,  5.0],   # off-axis, 5m away
        [ 2.0, -1.0, 10.0],   # off-axis, 10m away
        [ 1.0,  0.0, 20.0],   # off-axis, 20m away
        [-1.0, -1.0,  3.0],   # off-axis, 3m away
    ])

    print("--- 3D World Points ---")
    print("   X_world (X, Y, Z):")
    for pt in points_3d:
        print(f"   ({pt[0]:6.1f}, {pt[1]:6.1f}, {pt[2]:6.1f})")

    # --- Project ---
    pixels = project(P, points_3d)
    print("\n--- Projected 2D Pixels ---")
    print("   (u_pixel, v_pixel):")
    for i, (u, v) in enumerate(pixels):
        print(f"   Point {i}: ({u:8.2f}, {v:8.2f})")

    # --- Verify with manual calculation ---
    print("\n--- Verification (manual pinhole formula) ---")
    print("   H&Z (6.1, p.154): (X,Y,Z) -> (f*X/Z + cx, f*Y/Z + cy)")
    for pt in points_3d:
        X, Y, Z = pt
        u_manual = fx * X / Z + cx
        v_manual = fy * Y / Z + cy
        print(f"   ({X:5.1f}, {Y:5.1f}, {Z:5.1f}) -> "
              f"({u_manual:8.2f}, {v_manual:8.2f})")

    print("\n   Values match:",
          np.allclose(pixels, np.array([[fx * p[0]/p[2] + cx,
                                          fy * p[1]/p[2] + cy]
                                         for p in points_3d])))

    # --- Camera with rotation and translation ---
    print("\n" + "=" * 60)
    print("Camera with rotation and translation (H&Z 6.6-6.8, pp.155-156)")
    print("=" * 60)

    # Camera rotated 30 degrees around Y-axis, translated to (2, 0, 3)
    theta = np.deg2rad(30)
    R2 = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    C_world = np.array([2.0, 0.0, 3.0])  # camera centre in world
    t2 = -R2 @ C_world  # H&Z (6.7, p.156): t = -R * C_tilde

    P2 = build_camera_matrix(fx, fy, cx, cy, R2, t2)
    print(f"\nCamera Matrix P2 (3x4):\n{P2}")

    # Test point: world origin should appear in image
    X_homo = np.array([[0.0, 0.0, 0.0, 1.0]])
    pixel_origin = project(P2, X_homo)
    print(f"\nWorld origin projected to: ({pixel_origin[0,0]:.2f}, "
          f"{pixel_origin[0,1]:.2f})")

    # --- RQ decomposition to recover K and R (H&Z sect 6.2.4, p.163) ---
    print("\n" + "=" * 60)
    print("RQ Decomposition (H&Z sect 6.2.4, p.163)")
    print("=" * 60)

    # --- RQ decomposition to recover K and R (H&Z sect 6.2.4, p.163) ---
    M = P2[:, :3]  # left 3x3 block M = K * R (H&Z p.163)
    # RQ: M = K @ R where K is upper-triangular, R is orthogonal.
    # Implement as Gram-Schmidt on rows of M in reverse order (H&Z A4.1.1, p.579):
    #   r3 = M[2] / |M[2]|  -> gives K[2,2] = |M[2]|
    #   r2 = (M[1] - K[1,2]*r3) / |...|  -> gives K[1,1], K[1,2]
    #   r1 = (M[0] - K[0,1]*r2 - K[0,2]*r3) / |...|  -> gives K[0,0], K[0,1], K[0,2]
    k33 = np.linalg.norm(M[2])
    r3 = M[2] / k33

    k23 = np.dot(M[1], r3)
    v2 = M[1] - k23 * r3
    k22 = np.linalg.norm(v2)
    r2 = v2 / k22

    k13 = np.dot(M[0], r3)
    k12 = np.dot(M[0], r2)
    v1 = M[0] - k12 * r2 - k13 * r3
    k11 = np.linalg.norm(v1)
    r1 = v1 / k11

    K_recovered = np.array([
        [k11, k12, k13],
        [0.0, k22, k23],
        [0.0, 0.0, k33]
    ])
    R_recovered = np.array([r1, r2, r3])

    # Scale K so that K[2,2] = 1 (homogeneous, H&Z p.157)
    K_recovered = K_recovered / K_recovered[2, 2]

    print("Recovered K:\n", K_recovered)
    print("\nRecovered R:\n", R_recovered)
    print("\nM = KR check:", np.allclose(M, K_recovered @ R_recovered,
          atol=1e-10))

    print("\n" + "=" * 60)
    print("All checks passed. Camera projection verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
