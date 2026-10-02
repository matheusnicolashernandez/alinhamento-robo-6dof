
import math
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Alinhamento Robô 6 DOF",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

URDF_PATH = Path(__file__).with_name("normal.urdf")
URDF_TO_MM = 1000.0

INITIAL_Q_DEG = np.array(
    [-27.0, 20.0, -26.0, 20.0, -80.0, 90.0],
    dtype=float,
)

DEFAULTS = {
    "tube_diameter": 500.0,
    "tube_length": 1300.0,
    "tube_x": 0.0,
    "tube_y": 0.0,
    "tube_z": -500.0,
    "base_x": 0.0,
    "base_y": -500.0,
    "base_z": -2054.0,
    "target_distance": 120.0,
    "dist_tol": 0.50,
    "align_tol_deg": 0.10,
    "frame_width": 670.0,
    "frame_height": 130.0,
    "gain": 0.35,
    "damping": 1e-3,
    "jac_step_deg": 0.05,
    "max_dq_deg": 1.0,
}

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

JOINT_LABELS = ["q1", "q2", "q3", "q4", "q5", "q6"]


# ============================================================
# MATEMÁTICA
# ============================================================

def rpy_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    rx = np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]],
        dtype=float,
    )

    ry = np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]],
        dtype=float,
    )

    rz = np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]],
        dtype=float,
    )

    return rz @ ry @ rx


def make_t(R=None, p=None):
    T = np.eye(4)

    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)

    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float)

    return T


def origin_t(xyz, rpy):
    return make_t(
        rpy_matrix(*rpy),
        np.asarray(xyz, dtype=float) * URDF_TO_MM,
    )


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)

    if n < 1e-12:
        return np.eye(3)

    x, y, z = axis / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    return np.array(
        [
            [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
        ],
        dtype=float,
    )


def ray_cylinder_intersection(p, d, radius, tube_center):
    """Interseção com cilindro infinito paralelo ao eixo Z."""

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)
    c0 = np.asarray(tube_center, dtype=float)

    px = p[0] - c0[0]
    py = p[1] - c0[1]

    a = d[0] ** 2 + d[1] ** 2
    b = 2.0 * (px * d[0] + py * d[1])
    c = px**2 + py**2 - radius**2

    if abs(a) < 1e-12:
        return None

    disc = b*b - 4.0*a*c

    if disc < 0.0:
        return None

    root = math.sqrt(max(0.0, disc))

    t1 = (-b - root) / (2.0*a)
    t2 = (-b + root) / (2.0*a)

    valid = [t for t in (t1, t2) if t >= 0.0]
    return min(valid) if valid else None
