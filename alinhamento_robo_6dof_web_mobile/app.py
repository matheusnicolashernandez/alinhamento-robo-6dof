
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


def cylinder_mesh_z(center, radius, height, n_theta=48, n_z=18):
    """Malha cilíndrica para Plotly."""

    cx, cy, cz = center

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    z = np.linspace(cz - height/2.0, cz + height/2.0, n_z)

    tt, zz = np.meshgrid(theta, z)

    xx = cx + radius * np.cos(tt)
    yy = cy + radius * np.sin(tt)

    return xx, yy, zz


def cylinder_wireframe_trace(center, radius, height, color="rgba(189,189,189,0.55)", width=2, n_theta=64, n_z=24, n_vertical=16, name="Tubo"):
    """Cilindro como malha Scatter3d estática, robusta durante restyle."""
    cx, cy, cz = map(float, center)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    levels = np.linspace(cz - height / 2.0, cz + height / 2.0, n_z)
    x, y, z = [], [], []

    for zz in levels:
        for th in theta:
            x.append(cx + radius * math.cos(th))
            y.append(cy + radius * math.sin(th))
            z.append(zz)
        x.append(None); y.append(None); z.append(None)

    for k in range(n_vertical):
        th = 2.0 * math.pi * k / n_vertical
        xx = cx + radius * math.cos(th)
        yy = cy + radius * math.sin(th)
        x.extend([xx, xx, None])
        y.extend([yy, yy, None])
        z.extend([cz - height / 2.0, cz + height / 2.0, None])

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
        name=name,
        connectgaps=False,
    )


def cylinder_mesh_z_mesh3d(center, radius, height, n_theta=64):
    """Malha lateral de um cilindro para go.Mesh3d.

    O tubo permanece oco e é um trace estático. Usar Mesh3d aqui evita
    que uma go.Surface transparente perca a renderização quando os traces
    móveis são atualizados durante o Play em navegadores móveis.
    """
    cx, cy, cz = map(float, center)
    n_theta = max(int(n_theta), 16)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    z0 = cz - float(height) / 2.0
    z1 = cz + float(height) / 2.0

    bottom = np.column_stack((
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        np.full(n_theta, z0),
    ))
    top = np.column_stack((
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        np.full(n_theta, z1),
    ))

    vertices = np.vstack((bottom, top))
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    i = []
    j = []
    k = []
    for n in range(n_theta):
        m = (n + 1) % n_theta
        i.extend([n, n])
        j.extend([m, n_theta + m])
        k.extend([n_theta + m, n_theta + n])

    return x, y, z, i, j, k


def cylinder_mesh_between(p1, p2, radius, n_theta=18):
    """Malha cilíndrica de um elo entre dois pontos."""

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    axis = p2 - p1
    L = np.linalg.norm(axis)

    if L < 1e-9:
        return None

    w = axis / L

    ref = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(w, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(w, ref)
    u /= np.linalg.norm(u)

    v = np.cross(w, u)
    v /= np.linalg.norm(v)

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)

    ring1 = np.array(
        [
            p1 + radius * (math.cos(t) * u + math.sin(t) * v)
            for t in theta
        ]
    )

    ring2 = np.array(
        [
            p2 + radius * (math.cos(t) * u + math.sin(t) * v)
            for t in theta
        ]
    )

    vertices = np.vstack([ring1, ring2])

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    i = []
    j = []
    k = []

    for n in range(n_theta):
        m = (n + 1) % n_theta

        i.extend([n, m])
        j.extend([m, n_theta + m])
        k.extend([n_theta + n, n_theta + n])

    # Tampas simples
    center1 = len(x)
    center2 = center1 + 1

    x = np.r_[x, p1[0], p2[0]]
    y = np.r_[y, p1[1], p2[1]]
    z = np.r_[z, p1[2], p2[2]]

    for n in range(n_theta):
        m = (n + 1) % n_theta

        i.append(center1)
        j.append(m)
        k.append(n)

        i.append(center2)
        j.append(n_theta + n)
        k.append(n_theta + m)

    return x, y, z, i, j, k


# ============================================================
# URDF
# ============================================================

class RobotURDF:
    def __init__(self, path):

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Arquivo normal.urdf não encontrado: {self.path}"
            )

        self.root = ET.parse(self.path).getroot()
        self.joints = {}
        self._read()

    @staticmethod
    def vec(element, attr, default=(0, 0, 0)):
        if element is None:
            return np.array(default, dtype=float)

        txt = element.attrib.get(attr)

        if not txt:
            return np.array(default, dtype=float)

        return np.array(
            [float(v) for v in txt.split()],
            dtype=float,
        )

    def _read(self):

        names = {
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        }

        for e in self.root.findall("joint"):

            name = e.attrib["name"]

            if name not in names:
                continue

            origin = e.find("origin")

            xyz = self.vec(origin, "xyz")
            rpy = self.vec(origin, "rpy")

            axis_e = e.find("axis")
            axis = self.vec(
                axis_e,
                "xyz",
                (0, 0, 1),
            )

            lower = -np.inf
            upper = np.inf

            limit = e.find("limit")

            if limit is not None:

                if "lower" in limit.attrib:
                    lower = float(limit.attrib["lower"])

                if "upper" in limit.attrib:
                    upper = float(limit.attrib["upper"])

            self.joints[name] = {
                "origin": origin_t(xyz, rpy),
                "axis": axis,
                "lower": lower,
                "upper": upper,
            }

    def limits(self):
        return np.array(
            [
                [
                    self.joints[n]["lower"],
                    self.joints[n]["upper"],
                ]
                for n in JOINT_NAMES
            ],
            dtype=float,
        )

    def base_pose(self, cfg):
        return make_t(
            np.eye(3),
            np.array(
                [
                    cfg["base_x"],
                    cfg["base_y"],
                    cfg["base_z"],
                ]
            ),
        )

    def fk(self, q, cfg):

        q = np.asarray(q, dtype=float)
        qd = dict(zip(JOINT_NAMES, q))

        base_t = self.base_pose(cfg)

        # J1 diretamente sobre a base, sem o elo adicional do URDF.
        j1_mount = (
            base_t
            @ make_t(
                np.eye(3),
                np.array([0.0, 0.0, 90.0]),
            )
        )

        T = {
            "base": base_t,
            "joint_1": None,
            "joint_2": None,
            "joint_3": None,
            "joint_4": None,
            "joint_5": None,
            "joint_6": None,
        }

        parent_t = j1_mount

        for index, name in enumerate(JOINT_NAMES):

            joint = self.joints[name]

            if index == 0:
                Tj = parent_t @ make_t(
                    axis_angle(
                        joint["axis"],
                        qd[name],
                    )
                )
            else:
                Tj = (
                    parent_t
                    @ joint["origin"]
                    @ make_t(
                        axis_angle(
                            joint["axis"],
                            qd[name],
                        )
                    )
                )

            T[name] = Tj
            parent_t = Tj

        T["end_effector"] = T["joint_6"].copy()
        return T


# ============================================================
# LASERS
# ============================================================

class FourLasers:
    def __init__(self, robot):
        self.robot = robot

    def sensor_positions(self, q, cfg):

        T = self.robot.fk(q, cfg)["end_effector"]
        center = T[:3, 3]

        ex = T[:3, 0]
        ey = T[:3, 1]

        width_axis = -ex
        height_axis = ey

        return {
            "A": center - width_axis*cfg["frame_width"]/2 - height_axis*cfg["frame_height"]/2,
            "B": center + width_axis*cfg["frame_width"]/2 - height_axis*cfg["frame_height"]/2,
            "C": center - width_axis*cfg["frame_width"]/2 + height_axis*cfg["frame_height"]/2,
            "D": center + width_axis*cfg["frame_width"]/2 + height_axis*cfg["frame_height"]/2,
        }

    def readings(self, q, cfg):

        T = self.robot.fk(q, cfg)["end_effector"]
        sensors = self.sensor_positions(q, cfg)

        ex = T[:3, 0]
        ey = T[:3, 1]

        width_axis = -ex
        height_axis = ey

        normal = np.cross(width_axis, height_axis)
        normal /= max(np.linalg.norm(normal), 1e-12)

        radius = cfg["tube_diameter"] / 2.0

        tube_center = np.array(
            [
                cfg["tube_x"],
                cfg["tube_y"],
                cfg["tube_z"],
            ]
        )

        distances = []
        data = []

        for label in ["A", "B", "C", "D"]:

            p = sensors[label].copy()
            ray = normal.copy()

            radial = np.array(
                [
                    p[0] - cfg["tube_x"],
                    p[1] - cfg["tube_y"],
                    0.0,
                ]
            )

            if np.dot(ray, -radial) < 0.0:
                ray = -ray

            t = ray_cylinder_intersection(
                p,
                ray,
                radius,
                tube_center,
            )

            if t is None:
                distances.append(np.nan)
                data.append((p, ray, None))
            else:
                hit = p + t*ray
                distances.append(t)
                data.append((p, ray, hit))

        return np.asarray(distances, dtype=float), data


# ============================================================
# CONTROLADOR
# ============================================================

class Controller:
    def __init__(self, robot, lasers, cfg):
        self.robot = robot
        self.lasers = lasers
        self.cfg = cfg

    def error(self, q):
        d, _ = self.lasers.readings(q, self.cfg)
        return d - self.cfg["target_distance"]

    def jacobian(self, q):

        J = np.zeros((4, 6))
        h = math.radians(self.cfg["jac_step_deg"])

        for j in range(6):

            qp = q.copy()
            qm = q.copy()

            qp[j] += h
            qm[j] -= h

            ep = self.error(qp)
            em = self.error(qm)

            valid = np.isfinite(ep) & np.isfinite(em)

            J[valid, j] = (
                ep[valid] - em[valid]
            ) / (2.0*h)

        return J

    def step(self, q):

        d, _ = self.lasers.readings(q, self.cfg)

        if not np.all(np.isfinite(d)):
            return q.copy(), False

        e = d - self.cfg["target_distance"]

        J = self.jacobian(q)

        A = (
            J.T @ J
            + self.cfg["damping"] * np.eye(6)
        )

        b = J.T @ e

        try:
            dq = -self.cfg["gain"] * np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            dq = -self.cfg["gain"] * np.linalg.pinv(J) @ e

        max_step = math.radians(
            self.cfg["max_dq_deg"]
        )

        dq = np.clip(
            dq,
            -max_step,
            max_step,
        )

        qnew = q + dq

        limits = self.robot.limits()

        for i in range(6):
            qnew[i] = np.clip(
                qnew[i],
                limits[i, 0],
                limits[i, 1],
            )

        return qnew, True


# ============================================================
# PLOT 3D
# ============================================================

def scene_bounds(cfg, robot):
    tube_r = cfg["tube_diameter"]/2.0

    tube_min = np.array(
        [
            cfg["tube_x"] - tube_r,
            cfg["tube_y"] - tube_r,
            cfg["tube_z"] - cfg["tube_length"]/2.0,
        ]
    )

    tube_max = np.array(
        [
            cfg["tube_x"] + tube_r,
            cfg["tube_y"] + tube_r,
            cfg["tube_z"] + cfg["tube_length"]/2.0,
        ]
    )

    base = np.array(
        [
            cfg["base_x"],
            cfg["base_y"],
            cfg["base_z"],
        ]
    )

    base_pad = np.array([180.0, 180.0, 180.0])

    lo = np.minimum(tube_min, base - base_pad)
    hi = np.maximum(tube_max, base + base_pad)

    # margem visual
    span = hi - lo
    lo -= span * 0.08
    hi += span * 0.08

    return lo, hi


def make_scene_figure(q, cfg, robot, lasers):

    T = robot.fk(q, cfg)

    fig = go.Figure()

    # ----------------------------
    # TUBO
    # ----------------------------

    tx, ty, tz, ti, tj, tk = cylinder_mesh_z_mesh3d(
        center=(
            cfg["tube_x"],
            cfg["tube_y"],
            cfg["tube_z"],
        ),
        radius=cfg["tube_diameter"]/2.0,
        height=cfg["tube_length"],
    )

    fig.add_trace(
        go.Mesh3d(
            x=tx,
            y=ty,
            z=tz,
            i=ti,
            j=tj,
            k=tk,
            opacity=0.22,
            color="#BDBDBD",
            hoverinfo="skip",
            name="Tubo",
            flatshading=False,
            lighting=dict(ambient=0.75, diffuse=0.25, specular=0.05),
        )
    )

    # ----------------------------
    # EIXO DO TUBO
    # ----------------------------

    z1 = cfg["tube_z"] - cfg["tube_length"]/2
    z2 = cfg["tube_z"] + cfg["tube_length"]/2

    fig.add_trace(
        go.Scatter3d(
            x=[cfg["tube_x"], cfg["tube_x"]],
            y=[cfg["tube_y"], cfg["tube_y"]],
            z=[z1, z2],
            mode="lines",
            line=dict(
                color="#E53935",
                width=5,
            ),
            name="Eixo do tubo",
            hoverinfo="skip",
        )
    )

    # ----------------------------
    # BASE
    # ----------------------------

    bx, by, bz = cylinder_mesh_z(
        center=(
            cfg["base_x"],
            cfg["base_y"],
            cfg["base_z"],
        ),
        radius=120,
        height=180,
        n_theta=40,
        n_z=8,
    )

    fig.add_trace(
        go.Surface(
            x=bx,
            y=by,
            z=bz,
            opacity=1.0,
            colorscale=[
                [0, "#555555"],
                [1, "#555555"],
            ],
            showscale=False,
            hoverinfo="skip",
            name="Base",
        )
    )

    # ----------------------------
    # ROBÔ: J1 -> ... -> J6
    # ----------------------------

    joint_points = [
        T["joint_1"][:3, 3],
        T["joint_2"][:3, 3],
        T["joint_3"][:3, 3],
        T["joint_4"][:3, 3],
        T["joint_5"][:3, 3],
        T["joint_6"][:3, 3],
    ]

    # Links como linhas grossas em vez de Mesh3d.
    # Isso evita um problema de desaparecimento dos elos durante a
    # animação WebGL/Plotly no navegador móvel.
    link_widths = [14, 16, 15, 14, 13]

    for i in range(5):
        p1 = joint_points[i]
        p2 = joint_points[i+1]

        fig.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(
                    color="#4682B4",
                    width=link_widths[i],
                ),
                hoverinfo="skip",
                name=f"Link {i+1}",
            )
        )

    # ----------------------------
    # JUNTAS
    # ----------------------------

    jp = np.asarray(joint_points)

    fig.add_trace(
        go.Scatter3d(
            x=jp[:,0],
            y=jp[:,1],
            z=jp[:,2],
            mode="markers",
            marker=dict(
                size=7,
                color="#0B2E59",
            ),
            name="Juntas",
            hovertemplate=(
                "Junta %{text}<br>"
                "X=%{x:.1f}<br>"
                "Y=%{y:.1f}<br>"
                "Z=%{z:.1f}<extra></extra>"
            ),
            text=JOINT_LABELS,
        )
    )

    # ----------------------------
    # TOOL / RETÂNGULO
    # ----------------------------

    ee = T["end_effector"]

    center = ee[:3, 3]

    ex = ee[:3, 0]
    ey = ee[:3, 1]
    ez = ee[:3, 2]

    width_axis = -ex
    height_axis = ey

    sensors = {
        "A": center - width_axis*cfg["frame_width"]/2 - height_axis*cfg["frame_height"]/2,
        "B": center + width_axis*cfg["frame_width"]/2 - height_axis*cfg["frame_height"]/2,
        "C": center - width_axis*cfg["frame_width"]/2 + height_axis*cfg["frame_height"]/2,
        "D": center + width_axis*cfg["frame_width"]/2 + height_axis*cfg["frame_height"]/2,
    }

    corners = np.array(
        [
            sensors["A"],
            sensors["B"],
            sensors["D"],
            sensors["C"],
            sensors["A"],
        ]
    )

    fig.add_trace(
        go.Scatter3d(
            x=corners[:,0],
            y=corners[:,1],
            z=corners[:,2],
            mode="lines",
            line=dict(
                color="#00B8D9",
                width=8,
            ),
            name="Retângulo",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers",
            marker=dict(
                size=6,
                color="white",
                line=dict(
                    color="#333333",
                    width=1,
                ),
            ),
            name="Centro / J6",
            hoverinfo="skip",
        )
    )

    # Normal do retângulo = Z_EE.
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], center[0] + ez[0]*130],
            y=[center[1], center[1] + ez[1]*130],
            z=[center[2], center[2] + ez[2]*130],
            mode="lines",
            line=dict(
                color="#AB47BC",
                width=5,
            ),
            name="Normal / Laser",
            hoverinfo="skip",
        )
    )

    # ----------------------------
    # LASERS
    # ----------------------------

    distances, data = lasers.readings(q, cfg)

    for label, (p, ray, hit) in zip(
        ["A","B","C","D"],
        data,
    ):

        fig.add_trace(
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                marker=dict(
                    size=6,
                    color="#FFB300",
                ),
                text=[label],
                textposition="top center",
                textfont=dict(
                    size=13,
                    color="#111111",
                ),
                name=f"Sensor {label}",
                hovertemplate=(
                    f"{label}<br>"
                    "X=%{x:.1f}<br>"
                    "Y=%{y:.1f}<br>"
                    "Z=%{z:.1f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        if hit is not None:

            fig.add_trace(
                go.Scatter3d(
                    x=[p[0], hit[0]],
                    y=[p[1], hit[1]],
                    z=[p[2], hit[2]],
                    mode="lines",
                    line=dict(
                        color="#FF6D00",
                        width=5,
                    ),
                    name=f"Laser {label}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[hit[0]],
                    y=[hit[1]],
                    z=[hit[2]],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="#00C853",
                    ),
                    name=f"Impacto {label}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # ----------------------------
    # CAMERA / ESCALA FIXA
    # ----------------------------

    lo, hi = scene_bounds(cfg, robot)

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0,
        ),
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision="fixed_scene",
        scene=dict(
            xaxis=dict(
                title="X (mm)",
                range=[float(lo[0]), float(hi[0])],
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title="Y (mm)",
                range=[float(lo[1]), float(hi[1])],
                showgrid=True,
                zeroline=False,
            ),
            zaxis=dict(
                title="Z (mm)",
                range=[float(lo[2]), float(hi[2])],
                showgrid=True,
                zeroline=False,
            ),
            aspectmode="data",
            dragmode="orbit",
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(
                    x=1.55,
                    y=1.55,
                    z=1.15,
                ),
            ),
        ),
        showlegend=False,
    )

    return fig, distances


# ============================================================
# ESTADO
# ============================================================

def initialize_state(robot):

    if "q" not in st.session_state:
        st.session_state.q = np.radians(
            INITIAL_Q_DEG.copy()
        )

    if "history" not in st.session_state:
        st.session_state.history = {
            "iteration": [],
            "A": [],
            "B": [],
            "C": [],
            "D": [],
        }

    if "status" not in st.session_state:
        st.session_state.status = "Pronto"

    if "run_id" not in st.session_state:
        st.session_state.run_id = 0

    if "manual_nonce" not in st.session_state:
        st.session_state.manual_nonce = 0

    if "trajectory" not in st.session_state:
        st.session_state.trajectory = None

    if "trajectory_cfg" not in st.session_state:
        st.session_state.trajectory_cfg = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def reset_history():
    st.session_state.history = {
        "iteration": [],
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }


def add_history(iteration, d):

    h = st.session_state.history

    h["iteration"].append(int(iteration))

    for i, label in enumerate(["A","B","C","D"]):
        h[label].append(
            float(d[i]) if np.isfinite(d[i]) else np.nan
        )


def config_from_widgets():

    return {
        "tube_diameter": float(st.session_state["tube_diameter"]),
        "tube_length": float(st.session_state["tube_length"]),
        "tube_x": float(st.session_state["tube_x"]),
        "tube_y": float(st.session_state["tube_y"]),
        "tube_z": float(st.session_state["tube_z"]),
        "base_x": float(st.session_state["base_x"]),
        "base_y": float(st.session_state["base_y"]),
        "base_z": float(st.session_state["base_z"]),
        "target_distance": float(st.session_state["target_distance"]),
        "dist_tol": float(st.session_state["dist_tol"]),
        "align_tol_deg": float(st.session_state["align_tol_deg"]),
        "frame_width": DEFAULTS["frame_width"],
        "frame_height": DEFAULTS["frame_height"],
        "gain": DEFAULTS["gain"],
        "damping": DEFAULTS["damping"],
        "jac_step_deg": DEFAULTS["jac_step_deg"],
        "max_dq_deg": DEFAULTS["max_dq_deg"],
    }


def alignment_angle_deg(robot, q, cfg):

    T = robot.fk(q, cfg)["end_effector"]

    # Lado longo do retângulo = -X_EE.
    axis = -T[:3, 0]

    target = np.array([0.0, 0.0, 1.0])

    c = abs(np.dot(axis, target))
    c = np.clip(
        c / max(np.linalg.norm(axis), 1e-12),
        -1.0,
        1.0,
    )

    return math.degrees(math.acos(c))


def current_metrics(robot, lasers, q, cfg):

    d, _ = lasers.readings(q, cfg)

    angle = alignment_angle_deg(
        robot,
        q,
        cfg,
    )

    finite = np.isfinite(d)

    if np.any(finite):
        max_dist_error = float(
            np.max(
                np.abs(
                    d[finite] -
                    cfg["target_distance"]
                )
            )
        )
    else:
        max_dist_error = float("nan")

    return d, angle, max_dist_error


# ============================================================
# TRAJETÓRIA PRÉ-CALCULADA + ANIMAÇÃO NO NAVEGADOR
# ============================================================

def solve_trajectory(q0, cfg, robot, lasers, max_iterations=300):

    controller = Controller(
        robot,
        lasers,
        cfg,
    )

    q = np.asarray(q0, dtype=float).copy()

    states = [q.copy()]
    history = {
        "iteration": [0],
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }

    aligned = False
    reason = None

    for iteration in range(max_iterations + 1):

        d, angle, max_dist_error = current_metrics(
            robot,
            lasers,
            q,
            cfg,
        )

        for i, label in enumerate(["A", "B", "C", "D"]):
            history[label].append(
                float(d[i])
                if np.isfinite(d[i])
                else np.nan
            )

        if not np.all(np.isfinite(d)):
            reason = "Um ou mais lasers não atingem o tubo."
            break

        if (
            max_dist_error <= cfg["dist_tol"]
            and
            angle <= cfg["align_tol_deg"]
        ):
            aligned = True
            break

        if iteration >= max_iterations:
            reason = "Limite de 300 iterações atingido."
            break

        qnew, ok = controller.step(q)

        if not ok:
            reason = "Falha na atualização da Jacobiana."
            break

        q = qnew
        states.append(q.copy())
        history["iteration"].append(iteration + 1)

    metrics = {
        "final_q": q.copy(),
        "aligned": aligned,
        "reason": reason,
        "iterations": len(states) - 1,
        "history": history,
    }

    return states, metrics


def _dynamic_snapshot(q, cfg, robot, lasers):
    """
    Retorna somente os traces móveis.
    A ordem é fixa para permitir animação por frames.
    """

    T = robot.fk(q, cfg)

    joint_points = [
        T["joint_1"][:3, 3],
        T["joint_2"][:3, 3],
        T["joint_3"][:3, 3],
        T["joint_4"][:3, 3],
        T["joint_5"][:3, 3],
        T["joint_6"][:3, 3],
    ]

    traces = []

    # 5 links móveis como linhas grossas.
    # Scatter3d anima de forma mais robusta que Mesh3d no WebGL móvel.
    link_widths = [14, 16, 15, 14, 13]

    for i in range(5):
        p1 = joint_points[i]
        p2 = joint_points[i+1]

        traces.append(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(
                    color="#4682B4",
                    width=link_widths[i],
                ),
                hoverinfo="skip",
                name=f"Link {i+1}",
            )
        )

    # Juntas.
    jp = np.asarray(joint_points)

    traces.append(
        go.Scatter3d(
            x=jp[:,0],
            y=jp[:,1],
            z=jp[:,2],
            mode="markers",
            marker=dict(
                size=7,
                color="#0B2E59",
            ),
            name="Juntas",
            hoverinfo="skip",
        )
    )

    # Retângulo direto na J6.
    ee = T["end_effector"]

    center = ee[:3, 3]
    ex = ee[:3, 0]
    ey = ee[:3, 1]
    ez = ee[:3, 2]

    width_axis = -ex
    height_axis = ey

    sensors = {
        "A": center - width_axis*cfg["frame_width"]/2 - height_axis*cfg["frame_height"]/2,
        "B": center + width_axis*cfg["frame_width"]/2 - height_axis*cfg["frame_height"]/2,
        "C": center - width_axis*cfg["frame_width"]/2 + height_axis*cfg["frame_height"]/2,
        "D": center + width_axis*cfg["frame_width"]/2 + height_axis*cfg["frame_height"]/2,
    }

    corners = np.array([
        sensors["A"],
        sensors["B"],
        sensors["D"],
        sensors["C"],
        sensors["A"],
    ])

    traces.append(
        go.Scatter3d(
            x=corners[:,0],
            y=corners[:,1],
            z=corners[:,2],
            mode="lines",
            line=dict(
                color="#00B8D9",
                width=8,
            ),
            name="Retângulo",
            hoverinfo="skip",
        )
    )

    traces.append(
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers",
            marker=dict(
                size=6,
                color="white",
                line=dict(
                    color="#333333",
                    width=1,
                ),
            ),
            name="Centro / J6",
            hoverinfo="skip",
        )
    )

    # Normal = Z_EE = direção dos lasers.
    traces.append(
        go.Scatter3d(
            x=[center[0], center[0] + ez[0]*130],
            y=[center[1], center[1] + ez[1]*130],
            z=[center[2], center[2] + ez[2]*130],
            mode="lines",
            line=dict(
                color="#AB47BC",
                width=5,
            ),
            name="Normal / Laser",
            hoverinfo="skip",
        )
    )

    distances, data = lasers.readings(q, cfg)

    for label, (p, ray, hit) in zip(
        ["A", "B", "C", "D"],
        data,
    ):

        traces.append(
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                marker=dict(
                    size=6,
                    color="#FFB300",
                ),
                text=[label],
                textposition="top center",
                textfont=dict(
                    size=13,
                    color="#111111",
                ),
                name=f"Sensor {label}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if hit is None:
            hx = [np.nan, np.nan]
            hy = [np.nan, np.nan]
            hz = [np.nan, np.nan]
            mx = [np.nan]
            my = [np.nan]
            mz = [np.nan]
        else:
            hx = [p[0], hit[0]]
            hy = [p[1], hit[1]]
            hz = [p[2], hit[2]]
            mx = [hit[0]]
            my = [hit[1]]
            mz = [hit[2]]

        traces.append(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="lines",
                line=dict(
                    color="#FF6D00",
                    width=5,
                ),
                name=f"Laser {label}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        traces.append(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="markers",
                marker=dict(
                    size=5,
                    color="#00C853",
                ),
                name=f"Impacto {label}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def _clean_json_value(value):
    """Converte valores NumPy/não finitos em tipos JSON nativos."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return [_clean_json_value(v) for v in value.tolist()]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (list, tuple)):
        return [_clean_json_value(v) for v in value]

    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None

    if isinstance(value, int):
        return int(value)

    return value


def _dynamic_snapshot_payload(q, cfg, robot, lasers):
    """
    Extrai somente X/Y/Z dos traces móveis.
    A ordem é a mesma em todos os frames.

    IMPORTANTE:
    - Não inclui tubo, eixo ou base.
    - Esses três traces são criados uma vez e nunca são alterados
      durante a animação.
    """
    dynamic = _dynamic_snapshot(q, cfg, robot, lasers)

    payload = []

    for trace in dynamic:
        payload.append(
            {
                "x": _clean_json_value(trace.x),
                "y": _clean_json_value(trace.y),
                "z": _clean_json_value(trace.z),
            }
        )

    return payload


def _interpolate_states(states, frame_count):
    """
    Interpola as poses para tornar o movimento visual mais lento e suave.

    A trajetória do controlador continua sendo a original.
    A interpolação é apenas visual e não altera o cálculo.
    """
    state_array = np.asarray(states, dtype=float)

    if len(state_array) == 1:
        return state_array

    frame_count = max(int(frame_count), 2)

    positions = np.linspace(
        0.0,
        float(len(state_array) - 1),
        frame_count,
    )

    samples = []

    for pos in positions:
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, len(state_array) - 1)
        a = pos - i0

        q = (
            (1.0 - a) * state_array[i0]
            + a * state_array[i1]
        )

        samples.append(q)

    return np.asarray(samples, dtype=float)


def make_animated_scene_figure(
    states,
    cfg,
    robot,
    lasers,
    frame_count=60,
):
    """
    Cria a cena inicial e a sequência de estados para animação
    no navegador.

    Estratégia:
    - Os traces estáticos (tubo, eixo e base) são criados uma única vez.
    - O tubo usa Mesh3d estático para maior estabilidade no WebGL móvel.
    - Os traces móveis são atualizados exclusivamente por Plotly.restyle().
    - Não usamos Plotly Frames, Plotly.animate() ou redraw da cena 3D.
    """
    if not states:
        return go.Figure(), []

    # --------------------------------------------------------
    # Interpolação APENAS visual.
    # --------------------------------------------------------
    samples = _interpolate_states(
        states,
        frame_count=frame_count,
    )

    # --------------------------------------------------------
    # Geometria fixa.
    # --------------------------------------------------------
    fig = go.Figure()

    # Trace 0: tubo estático em malha Scatter3d.
    # Não entra em nenhuma restyle da animação.
    fig.add_trace(
        cylinder_wireframe_trace(
            center=(cfg["tube_x"], cfg["tube_y"], cfg["tube_z"]),
            radius=cfg["tube_diameter"] / 2.0,
            height=cfg["tube_length"],
            color="rgba(0,0,0,0.55)",
            width=1,
            n_theta=24,
            n_z=8,
            n_vertical=8,
            name="Tubo",
        )
    )

    z1 = cfg["tube_z"] - cfg["tube_length"]/2
    z2 = cfg["tube_z"] + cfg["tube_length"]/2

    fig.add_trace(
        go.Scatter3d(
            x=[cfg["tube_x"], cfg["tube_x"]],
            y=[cfg["tube_y"], cfg["tube_y"]],
            z=[z1, z2],
            mode="lines",
            line=dict(
                color="#E53935",
                width=5,
            ),
            name="Eixo",
            hoverinfo="skip",
        )
    )

    bx, by, bz = cylinder_mesh_z(
        center=(
            cfg["base_x"],
            cfg["base_y"],
            cfg["base_z"],
        ),
        radius=120,
        height=180,
        n_theta=40,
        n_z=8,
    )

    fig.add_trace(
        go.Surface(
            x=bx,
            y=by,
            z=bz,
            opacity=1.0,
            colorscale=[
                [0, "#555555"],
                [1, "#555555"],
            ],
            showscale=False,
            hoverinfo="skip",
            name="Base",
        )
    )

    # --------------------------------------------------------
    # Estado inicial móvel.
    # --------------------------------------------------------
    dynamic0 = _dynamic_snapshot(
        samples[0],
        cfg,
        robot,
        lasers,
    )

    for trace in dynamic0:
        fig.add_trace(trace)

    # --------------------------------------------------------
    # Camera / escala: mantidas iguais à versão original.
    # --------------------------------------------------------
    lo, hi = scene_bounds(cfg, robot)

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=5,
            b=0,
        ),
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        uirevision="fixed_scene",
        scene=dict(
            xaxis=dict(
                title="X (mm)",
                range=[
                    float(lo[0]),
                    float(hi[0]),
                ],
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title="Y (mm)",
                range=[
                    float(lo[1]),
                    float(hi[1]),
                ],
                showgrid=True,
                zeroline=False,
            ),
            zaxis=dict(
                title="Z (mm)",
                range=[
                    float(lo[2]),
                    float(hi[2]),
                ],
                showgrid=True,
                zeroline=False,
            ),
            aspectmode="data",
            dragmode="orbit",
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(
                    x=1.55,
                    y=1.55,
                    z=1.15,
                ),
            ),
        ),
    )

    # --------------------------------------------------------
    # Snapshots somente dos traces móveis.
    # --------------------------------------------------------
    snapshots = [
        _dynamic_snapshot_payload(
            q,
            cfg,
            robot,
            lasers,
        )
        for q in samples
    ]

    return fig, snapshots


def make_animated_html(
    fig,
    snapshots,
    height=650,
    autoplay=False,
    frame_delay_ms=220,
):
    """
    Animação client-side sem Plotly Frames.

    Em cada passo:
      Plotly.restyle() atualiza apenas os traces móveis.
      Tubo, eixo e base não são tocados.
    """
    fig_json = fig.to_json()

    snapshots_json = json.dumps(
        _clean_json_value(snapshots),
        separators=(",", ":"),
        allow_nan=False,
    )

    autoplay_js = "startAnimation();" if autoplay else ""

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
html, body {{
    margin: 0;
    padding: 0;
    background: white;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: Arial, sans-serif;
}}
#controls {{
    height: 38px;
    display: flex;
    align-items: center;
    gap: 6px;
    padding-left: 4px;
    box-sizing: border-box;
}}
button {{
    border: 1px solid #b8b8b8;
    background: #ffffff;
    border-radius: 4px;
    padding: 5px 10px;
    cursor: pointer;
    font-size: 13px;
}}
button:active {{
    background: #eeeeee;
}}
#plot {{
    width: 100%;
    height: calc(100% - 38px);
}}
</style>
</head>
<body>
<div id="controls">
    <button id="play">▶ Play</button>
    <button id="stop">■ Parar</button>
</div>
<div id="plot"></div>

<script>
const fig = {fig_json};
const snapshots = {snapshots_json};
const gd = document.getElementById("plot");

const STATIC_COUNT = 3;
const DYNAMIC_COUNT = fig.data.length - STATIC_COUNT;
const DYNAMIC_INDICES = Array.from(
    {{length: DYNAMIC_COUNT}},
    (_, i) => STATIC_COUNT + i
);

const FRAME_DELAY = {int(frame_delay_ms)};
let running = false;
let currentFrame = 0;
let animationToken = 0;

function sleep(ms) {{
    return new Promise(resolve => setTimeout(resolve, ms));
}}

let userCamera = null;
let userIsInteracting = false;

function copyCamera() {{
    if (gd.layout && gd.layout.scene && gd.layout.scene.camera) {{
        return JSON.parse(JSON.stringify(gd.layout.scene.camera));
    }}
    return null;
}}

async function applySnapshot(snapshot) {{
    const x = snapshot.map(item => item.x);
    const y = snapshot.map(item => item.y);
    const z = snapshot.map(item => item.z);

    // Atualiza somente os traces dinâmicos.
    // NÃO fazemos relayout da câmera aqui: isso era o que fazia uma
    // rotação longa voltar para a posição anterior.
    await Plotly.restyle(
        gd,
        {{
            x: x,
            y: y,
            z: z
        }},
        DYNAMIC_INDICES
    );
}}

async function startAnimation() {{
    if (snapshots.length === 0) {{
        return;
    }}

    // Play sempre reinicia a trajetória imediatamente.
    // Se já estiver rodando, cancela a execução atual e começa novamente.
    running = false;
    animationToken += 1;

    running = true;
    const myToken = animationToken;

    // Volta imediatamente para a pose inicial, sem recalcular a trajetória.
    currentFrame = 0;
    await applySnapshot(snapshots[0]);

    while (
        running &&
        myToken === animationToken &&
        currentFrame < snapshots.length - 1
    ) {{
        await sleep(FRAME_DELAY);

        if (!running || myToken !== animationToken) {{
            break;
        }}

        // Durante um gesto longo de rotação/zoom, deixamos o Plotly
        // ter controle exclusivo da câmera. A trajetória pausa por
        // alguns instantes e continua exatamente de onde parou.
        if (userIsInteracting) {{
            continue;
        }}

        currentFrame += 1;
        await applySnapshot(snapshots[currentFrame]);
    }}

    if (myToken === animationToken) {{
        running = false;
    }}
}}

function stopAnimation() {{
    running = false;
    animationToken += 1;
}}

document.getElementById("play").addEventListener(
    "click",
    startAnimation
);

document.getElementById("stop").addEventListener(
    "click",
    stopAnimation
);

Plotly.newPlot(
    gd,
    fig.data,
    fig.layout,
    {{
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        displayModeBar: true,
        modeBarButtonsToAdd: [
            "resetCameraDefault",
            "resetCameraLastSave"
        ]
    }}
).then(function () {{
    // Detecta o gesto diretamente no elemento do gráfico.
    // Isso impede que a atualização dos traces roube o controle da câmera.
    gd.addEventListener('mousedown', function() {{
        userIsInteracting = true;
    }});
    gd.addEventListener('touchstart', function() {{
        userIsInteracting = true;
    }}, {{passive: true}});

    window.addEventListener('mouseup', function() {{
        userIsInteracting = false;
        setTimeout(function() {{ userCamera = copyCamera(); }}, 0);
    }});
    window.addEventListener('touchend', function() {{
        userIsInteracting = false;
        setTimeout(function() {{ userCamera = copyCamera(); }}, 0);
    }}, {{passive: true}});

    gd.on('plotly_relayout', function(evt) {{
        if (evt) {{
            const keys = Object.keys(evt);
            if (keys.some(k => k === 'scene.camera' || k.startsWith('scene.camera.'))) {{
                userCamera = copyCamera();
            }}
        }}
    }});

    userCamera = copyCamera();

    {autoplay_js}
}});
</script>
</body>
</html>
"""

    return html


def make_static_camera_html_v32(fig, height=620):
    """Cena inicial no DOM principal do Streamlit, preservando a câmera."""
    fig_json = fig.to_json()
    camera_key = "robot_scene_camera_v32"

    html = f"""
<div id="robot-scene-v32" style="width:100%;height:{int(height)}px;background:#ffffff;overflow:hidden;"></div>
<script>
(function() {{
    const container = document.getElementById("robot-scene-v32");
    if (!container) return;

    const CAMERA_KEY = "{camera_key}";
    const FIG = {fig_json};

    function getSavedCamera() {{
        try {{
            const raw = window.localStorage.getItem(CAMERA_KEY);
            return raw ? JSON.parse(raw) : null;
        }} catch (e) {{
            return null;
        }}
    }}

    function saveCamera(gd) {{
        try {{
            const camera = gd && gd.layout && gd.layout.scene && gd.layout.scene.camera;
            if (camera) window.localStorage.setItem(CAMERA_KEY, JSON.stringify(camera));
        }} catch (e) {{}}
    }}

    function render() {{
        if (!window.Plotly) return;

        const savedCamera = getSavedCamera();
        if (savedCamera) {{
            FIG.layout = FIG.layout || {{}};
            FIG.layout.scene = FIG.layout.scene || {{}};
            FIG.layout.scene.camera = savedCamera;
        }}

        Plotly.newPlot(
            container,
            FIG.data,
            FIG.layout,
            {{
                responsive: true,
                displaylogo: false,
                scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToAdd: [
                    "resetCameraDefault",
                    "resetCameraLastSave"
                ]
            }}
        ).then(function(gd) {{
            gd.on("plotly_relayout", function(evt) {{
                if (!evt) return;
                const keys = Object.keys(evt);
                if (keys.some(k => k === "scene.camera" || k.startsWith("scene.camera."))) {{
                    saveCamera(gd);
                }}
            }});

            gd.addEventListener("mouseup", function() {{
                setTimeout(function() {{ saveCamera(gd); }}, 0);
            }});
            gd.addEventListener("touchend", function() {{
                setTimeout(function() {{ saveCamera(gd); }}, 0);
            }}, {{passive:true}});

            saveCamera(gd);
        }});
    }}

    if (window.Plotly) {{
        render();
    }} else {{
        const script = document.createElement("script");
        script.src = "https://cdn.plot.ly/plotly-latest.min.js";
        script.onload = render;
        document.head.appendChild(script);
    }}
}})();
</script>
"""
    return html

# ============================================================
# GRÁFICO
# ============================================================

def graph_figure(cfg):

    h = st.session_state.history

    fig = go.Figure()

    x = h["iteration"]

    for label in ["A","B","C","D"]:

        fig.add_trace(
            go.Scatter(
                x=x,
                y=h[label],
                mode="lines+markers",
                name=label,
                line=dict(width=2),
                marker=dict(size=4),
            )
        )

    target = cfg["target_distance"]
    tol = cfg["dist_tol"]

    if x:
        xmin = min(x)
        xmax = max(x)

        fig.add_trace(
            go.Scatter(
                x=[xmin, xmax],
                y=[target, target],
                mode="lines",
                line=dict(
                    dash="dash",
                    width=2,
                ),
                name=f"Alvo {target:.1f} mm",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[xmin, xmax, xmax, xmin, xmin],
                y=[
                    target-tol,
                    target-tol,
                    target+tol,
                    target+tol,
                    target-tol,
                ],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(100,100,100,0.12)",
                name=f"± {tol:.2f} mm",
            )
        )

    fig.update_layout(
        height=330,
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10,
        ),
        title="Distâncias dos 4 lasers",
        xaxis_title="Iteração",
        yaxis_title="Distância (mm)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        uirevision="distance_graph",
    )

    return fig


# ============================================================
# APLICAÇÃO
# ============================================================

@st.cache_resource
def load_robot():
    return RobotURDF(URDF_PATH)


robot = load_robot()
lasers = FourLasers(robot)
initialize_state(robot)

st.title("Alinhamento automático — Robô 6 DOF + 4 lasers")
st.caption(
    "Versão web para celular/tablet. "
    "O cálculo continua baseado no normal.urdf."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

def update_joint_from_widgets():
    """Atualiza a pose imediatamente quando qualquer junta é alterada."""
    q_values = [
        float(st.session_state[f"q_deg_{i}"])
        for i in range(6)
    ]
    st.session_state.q = np.radians(np.asarray(q_values, dtype=float))
    st.session_state.trajectory = None
    st.session_state.trajectory_cfg = None
    st.session_state.last_result = None
    st.session_state.status = "Pose manual atualizada"


with st.sidebar:

    st.header("Configuração")

    st.number_input(
        "Diâmetro do tubo (mm)",
        min_value=50.0,
        max_value=5000.0,
        value=DEFAULTS["tube_diameter"],
        step=10.0,
        key="tube_diameter",
    )

    st.number_input(
        "Comprimento do tubo (mm)",
        min_value=200.0,
        max_value=10000.0,
        value=DEFAULTS["tube_length"],
        step=50.0,
        key="tube_length",
    )

    st.subheader("Posição do tubo")

    st.number_input(
        "Tubo X (mm)",
        min_value=-5000.0,
        max_value=5000.0,
        value=DEFAULTS["tube_x"],
        step=10.0,
        key="tube_x",
    )

    st.number_input(
        "Tubo Y (mm)",
        min_value=-5000.0,
        max_value=5000.0,
        value=DEFAULTS["tube_y"],
        step=10.0,
        key="tube_y",
    )

    st.number_input(
        "Tubo Z (mm)",
        min_value=-5000.0,
        max_value=5000.0,
        value=DEFAULTS["tube_z"],
        step=10.0,
        key="tube_z",
    )

    st.subheader("Posição da base")

    st.number_input(
        "Base X (mm)",
        min_value=-5000.0,
        max_value=5000.0,
        value=DEFAULTS["base_x"],
        step=10.0,
        key="base_x",
    )

    st.number_input(
        "Base Y (mm)",
        min_value=-5000.0,
        max_value=5000.0,
        value=DEFAULTS["base_y"],
        step=10.0,
        key="base_y",
    )

    st.number_input(
        "Base Z (mm)",
        min_value=-5000.0,
        max_value=5000.0,
        value=DEFAULTS["base_z"],
        step=10.0,
        key="base_z",
    )

    st.subheader("Alinhamento")

    st.number_input(
        "Distância alvo (mm)",
        min_value=1.0,
        max_value=2000.0,
        value=DEFAULTS["target_distance"],
        step=1.0,
        key="target_distance",
    )

    st.number_input(
        "Tolerância distância (mm)",
        min_value=0.01,
        max_value=100.0,
        value=DEFAULTS["dist_tol"],
        step=0.1,
        key="dist_tol",
    )

    st.number_input(
        "Tolerância angular (°)",
        min_value=0.01,
        max_value=30.0,
        value=DEFAULTS["align_tol_deg"],
        step=0.01,
        key="align_tol_deg",
    )

    st.subheader("Juntas")

    limits = robot.limits()

    q_deg = []

    for i, label in enumerate(JOINT_LABELS):

        value = st.number_input(
            f"{label} (°)",
            min_value=float(math.degrees(limits[i,0])),
            max_value=float(math.degrees(limits[i,1])),
            value=float(
                st.session_state.q[i] * 180.0 / math.pi
            ),
            step=1.0,
            key=f"q_deg_{i}",
            on_change=update_joint_from_widgets,
        )

        q_deg.append(value)

    # As juntas são aplicadas automaticamente pelo callback de cada campo.

    if st.button(
        "↺ Resetar pose",
        use_container_width=True,
    ):
        st.session_state.q = np.radians(
            INITIAL_Q_DEG.copy()
        )
        for i in range(6):
            st.session_state[f"q_deg_{i}"] = float(INITIAL_Q_DEG[i])
        reset_history()
        st.session_state.trajectory = None
        st.session_state.trajectory_cfg = None
        st.session_state.last_result = None
        st.session_state.status = "Pose inicial restaurada"
        st.rerun()

    st.divider()

    align_clicked = st.button(
        "▶ ALINHAR AUTOMATICAMENTE",
        use_container_width=True,
        type="primary",
    )

    stop_clicked = st.button(
        "■ PARAR",
        use_container_width=True,
    )

    if stop_clicked:
        st.session_state.stop_requested = True
        st.session_state.status = "Parada solicitada"

cfg = config_from_widgets()

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if align_clicked:

    st.session_state.stop_requested = False

    # Se já existe uma trajetória para a mesma configuração,
    # reproduz a partir da pose inicial da trajetória.
    previous_states = st.session_state.get("trajectory")
    previous_cfg = st.session_state.get("trajectory_cfg")

    can_replay = (
        previous_states is not None
        and len(previous_states) > 1
        and previous_cfg == cfg
    )

    if can_replay:
        q0 = np.asarray(previous_states[0], dtype=float).copy()
    else:
        q0 = st.session_state.q.copy()

    with st.spinner("Calculando trajetória de alinhamento..."):
        states, result = solve_trajectory(
            q0, cfg, robot, lasers, max_iterations=300
        )

    st.session_state.trajectory = states
    st.session_state.trajectory_cfg = cfg
    st.session_state.q = result["final_q"]
    st.session_state.last_result = result
    st.session_state.history = result["history"]

# ------------------------------------------------------------
# ÚLTIMA SIMULAÇÃO — FICA PERSISTENTE APÓS O CLIQUE
# ------------------------------------------------------------

last_result = st.session_state.get("last_result")
trajectory = st.session_state.get("trajectory")
trajectory_cfg = st.session_state.get("trajectory_cfg")

if (
    last_result is not None
    and trajectory is not None
    and trajectory_cfg == cfg
    and len(trajectory) > 0
):

    # Recria o componente em cada rerun. Assim o Play e o gráfico
    # não desaparecem quando o Streamlit atualiza a página.
    animation_fig, animation_snapshots = make_animated_scene_figure(
        trajectory, cfg, robot, lasers, frame_count=60
    )

    components.html(
        make_animated_html(
            animation_fig,
            animation_snapshots,
            height=650,
            autoplay=True,
            frame_delay_ms=220,
        ),
        height=650,
        scrolling=False,
    )

    final_q = np.asarray(last_result["final_q"], dtype=float)
    d_final, angle_final, max_dist_error_final = current_metrics(
        robot, lasers, final_q, cfg
    )

    if last_result["aligned"]:
        st.success(
            f"✓ ALINHADO em {last_result['iterations']} iterações • "
            f"erro de distância máx.: {max_dist_error_final:.3f} mm • "
            f"erro angular: {angle_final:.4f}°"
        )
    else:
        message = last_result["reason"] or "Trajetória encerrada."
        st.warning(
            message + " "
            + f"Erro de distância máx.: {max_dist_error_final:.3f} mm • "
            + f"erro angular: {angle_final:.4f}°"
        )

    c1, c2, c3, c4 = st.columns(4)
    for c, label, value in zip(
        [c1, c2, c3, c4], ["A", "B", "C", "D"], d_final
    ):
        with c:
            st.metric(
                f"Laser {label}",
                f"{value:.2f} mm" if np.isfinite(value) else "—",
            )

    st.plotly_chart(
        graph_figure(cfg),
        width="stretch",
        config={"displaylogo": False},
    )

    st.caption(
        "A animação é reproduzida no navegador. "
        "O tubo e a base permanecem estáticos; apenas os elementos móveis "
        "são atualizados durante o movimento."
    )

else:



        q = st.session_state.q

        d, angle, max_dist_error = current_metrics(
            robot,
            lasers,
            q,
            cfg,
        )

        scene_fig, _ = make_scene_figure(
            q,
            cfg,
            robot,
            lasers,
        )

        st.html(
            make_static_camera_html_v32(scene_fig, height=620),
            width="stretch",
            unsafe_allow_javascript=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        labels = ["A", "B", "C", "D"]

        for i, label in enumerate(labels):

            value = (
                f"{d[i]:.2f} mm"
                if np.isfinite(d[i])
                else "—"
            )

            with [col1, col2, col3, col4][i]:
                st.metric(
                    f"Laser {label}",
                    value,
                )

        st.write(
            f"**Status:** {st.session_state.status}  \n"
            f"**Erro angular:** {angle:.4f}°  \n"
            f"**Erro máximo de distância:** "
            f"{max_dist_error:.3f} mm"
        )

        st.plotly_chart(
            graph_figure(cfg),
            width="stretch",
            config={
                "displaylogo": False,
            },
        )

        st.caption(
            "O retângulo está acoplado diretamente à J6; "
            "o Z do end-effector é a normal/perpendicular dos lasers."
        )
