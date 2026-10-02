
import math
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



def cylinder_wireframe_z(center, radius, height, n_theta=48, n_rings=5, n_generators=16):
    """Cilindro desenhado apenas com linhas 3D.

    Em navegadores móveis, a malha Surface/WebGL pode desaparecer quando a cena
    recebe muitas atualizações durante a animação. O wireframe é muito mais
    estável e mantém tubo/base presentes durante todo o movimento.
    """
    cx, cy, cz = center
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=True)
    z_levels = np.linspace(cz - height/2.0, cz + height/2.0, n_rings)

    xs, ys, zs = [], [], []

    # Anéis horizontais.
    for z0 in z_levels:
        for t in theta:
            xs.append(cx + radius * math.cos(t))
            ys.append(cy + radius * math.sin(t))
            zs.append(z0)
        xs.append(None); ys.append(None); zs.append(None)

    # Geratrizes verticais.
    theta_g = np.linspace(0.0, 2.0 * math.pi, n_generators, endpoint=False)
    z0 = cz - height/2.0
    z1 = cz + height/2.0
    for t in theta_g:
        x0 = cx + radius * math.cos(t)
        y0 = cy + radius * math.sin(t)
        xs += [x0, x0, None]
        ys += [y0, y0, None]
        zs += [z0, z1, None]

    return xs, ys, zs

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

    # Mesmo intervalo numérico nos três eixos: evita qualquer sensação de
    # escala diferente entre X/Y/Z. A elipse aparente de um círculo em uma
    # vista 3D oblíqua é apenas efeito de projeção, não deformação da escala.
    center = (lo + hi) / 2.0
    half = float(np.max(hi - lo)) / 2.0
    half *= 1.08

    lo = center - half
    hi = center + half

    return lo, hi


def make_scene_figure(q, cfg, robot, lasers):

    T = robot.fk(q, cfg)

    fig = go.Figure()

    # ----------------------------
    # TUBO
    # ----------------------------

    tx, ty, tz = cylinder_wireframe_z(
        center=(
            cfg["tube_x"],
            cfg["tube_y"],
            cfg["tube_z"],
        ),
        radius=cfg["tube_diameter"]/2.0,
        height=cfg["tube_length"],
        n_theta=48,
        n_rings=5,
        n_generators=16,
    )

    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz,
        mode="lines",
        line=dict(color="#9E9E9E", width=2),
        opacity=0.38,
        hoverinfo="skip",
        name="Tubo",
    ))

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

    bx, by, bz = cylinder_wireframe_z(
        center=(
            cfg["base_x"],
            cfg["base_y"],
            cfg["base_z"],
        ),
        radius=120,
        height=180,
        n_theta=48,
        n_rings=4,
        n_generators=20,
    )

    fig.add_trace(
        go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            line=dict(color="#3F3F3F", width=7),
            opacity=0.95,
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
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(
                    x=1.55,
                    y=1.55,
                    z=1.75,
                ),
                projection=dict(type="orthographic"),
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


def _dynamic_compact_snapshot(q, cfg, robot, lasers):
    """Retorna a cena móvel compactada em apenas 7 traces.

    A animação atualiza somente estes traces com Plotly.restyle().
    Assim, tubo e base permanecem intocados durante o movimento.
    """

    T = robot.fk(q, cfg)

    joint_points = np.asarray([
        T["joint_1"][:3, 3],
        T["joint_2"][:3, 3],
        T["joint_3"][:3, 3],
        T["joint_4"][:3, 3],
        T["joint_5"][:3, 3],
        T["joint_6"][:3, 3],
    ])

    # Links: uma única linha com separadores None.
    link_x, link_y, link_z = [], [], []
    for i in range(5):
        p1 = joint_points[i]
        p2 = joint_points[i + 1]
        link_x += [float(p1[0]), float(p2[0]), None]
        link_y += [float(p1[1]), float(p2[1]), None]
        link_z += [float(p1[2]), float(p2[2]), None]

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

    corners = np.asarray([
        sensors["A"],
        sensors["B"],
        sensors["D"],
        sensors["C"],
        sensors["A"],
    ])

    # Lasers: quatro segmentos em uma única linha.
    laser_x, laser_y, laser_z = [], [], []
    impact_points = []
    distances, data = lasers.readings(q, cfg)

    labels = ["A", "B", "C", "D"]
    sensor_points = []
    for label, (p, ray, hit) in zip(labels, data):
        sensor_points.append(p)
        if hit is None:
            laser_x += [float(p[0]), float(p[0]), None]
            laser_y += [float(p[1]), float(p[1]), None]
            laser_z += [float(p[2]), float(p[2]), None]
        else:
            laser_x += [float(p[0]), float(hit[0]), None]
            laser_y += [float(p[1]), float(hit[1]), None]
            laser_z += [float(p[2]), float(hit[2]), None]
            impact_points.append(hit)

    if impact_points:
        impact = np.asarray(impact_points)
        impact_x = impact[:, 0].tolist()
        impact_y = impact[:, 1].tolist()
        impact_z = impact[:, 2].tolist()
    else:
        impact_x = [None]
        impact_y = [None]
        impact_z = [None]

    return {
        "links": {"x": link_x, "y": link_y, "z": link_z},
        "joints": {
            "x": joint_points[:, 0].astype(float).tolist(),
            "y": joint_points[:, 1].astype(float).tolist(),
            "z": joint_points[:, 2].astype(float).tolist(),
        },
        "rectangle": {
            "x": corners[:, 0].astype(float).tolist(),
            "y": corners[:, 1].astype(float).tolist(),
            "z": corners[:, 2].astype(float).tolist(),
        },
        "center": {
            "x": [float(center[0])],
            "y": [float(center[1])],
            "z": [float(center[2])],
        },
        "normal": {
            "x": [float(center[0]), float(center[0] + ez[0]*130)],
            "y": [float(center[1]), float(center[1] + ez[1]*130)],
            "z": [float(center[2]), float(center[2] + ez[2]*130)],
        },
        "sensors": {
            "x": [float(p[0]) for p in sensor_points],
            "y": [float(p[1]) for p in sensor_points],
            "z": [float(p[2]) for p in sensor_points],
        },
        "sensor_labels": labels,
        "lasers": {"x": laser_x, "y": laser_y, "z": laser_z},
        "impacts": {"x": impact_x, "y": impact_y, "z": impact_z},
    }


def _sample_states(states, frame_count):
    """Interpola a trajetória para exatamente ``frame_count`` frames.

    O solver normalmente produz poucos estados (por exemplo, 12 iterações).
    A versão anterior só interpolava quando havia estados demais e, portanto,
    uma trajetória curta continuava com poucos frames, deixando o movimento
    visual muito brusco. Aqui sempre interpolamos para a quantidade pedida.
    """
    state_array = np.asarray(states, dtype=float)

    if state_array.ndim != 2 or state_array.shape[0] == 0:
        return np.empty((0, 6), dtype=float)

    if frame_count <= 1 or state_array.shape[0] == 1:
        return state_array[[0]].copy()

    frame_count = int(max(frame_count, 2))

    positions = np.linspace(0.0, state_array.shape[0] - 1.0, frame_count)
    samples = np.empty((frame_count, state_array.shape[1]), dtype=float)

    for k, pos in enumerate(positions):
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, state_array.shape[0] - 1)
        alpha = pos - i0
        samples[k] = (1.0 - alpha) * state_array[i0] + alpha * state_array[i1]

    return samples


def make_animated_scene_figure(states, cfg, robot, lasers, frame_count=80):
    """Cria uma cena com 3 traces estáticos e 8 traces móveis.

    A animação é feita com Frames do próprio Plotly e com ``traces`` explícitos.
    Isso é importante: cada frame atualiza SOMENTE os objetos móveis e nunca
    substitui ou recria tubo, eixo e base.

    Além disso, todos os pontos que pertencem ao conjunto móvel (robô,
    retângulo, sensores e lasers) são calculados a partir da MESMA pose dentro
    de cada frame. Assim A/B/C/D não podem ficar em uma pose diferente da
    moldura durante a reprodução.
    """

    if not states:
        return go.Figure(), []

    samples = _sample_states(states, frame_count)
    snapshots = [
        _dynamic_compact_snapshot(q, cfg, robot, lasers)
        for q in samples
    ]

    first = snapshots[0]
    fig = go.Figure()

    # --------------------------------------------------------
    # ESTÁTICOS: nunca entram nos frames
    # --------------------------------------------------------

    tx, ty, tz = cylinder_wireframe_z(
        center=(cfg["tube_x"], cfg["tube_y"], cfg["tube_z"]),
        radius=cfg["tube_diameter"]/2.0,
        height=cfg["tube_length"],
        n_theta=64,
        n_rings=7,
        n_generators=20,
    )

    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz,
        mode="lines",
        line=dict(color="#9E9E9E", width=2),
        opacity=0.42,
        hoverinfo="skip",
        name="Tubo",
        showlegend=False,
    ))

    z1 = cfg["tube_z"] - cfg["tube_length"]/2
    z2 = cfg["tube_z"] + cfg["tube_length"]/2
    fig.add_trace(go.Scatter3d(
        x=[cfg["tube_x"], cfg["tube_x"]],
        y=[cfg["tube_y"], cfg["tube_y"]],
        z=[z1, z2],
        mode="lines",
        line=dict(color="#E53935", width=5),
        hoverinfo="skip",
        name="Eixo",
        showlegend=False,
    ))

    bx, by, bz = cylinder_wireframe_z(
        center=(cfg["base_x"], cfg["base_y"], cfg["base_z"]),
        radius=120,
        height=180,
        n_theta=64,
        n_rings=5,
        n_generators=24,
    )
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color="#303030", width=8),
        opacity=1.0,
        hoverinfo="skip",
        name="Base",
        showlegend=False,
    ))

    # Índices fixos dos traces móveis.
    # 3 = links, 4 = juntas, 5 = retângulo, 6 = centro,
    # 7 = normal, 8 = lasers, 9 = impactos, 10 = sensores.
    fig.add_trace(go.Scatter3d(
        x=first["links"]["x"], y=first["links"]["y"], z=first["links"]["z"],
        mode="lines",
        line=dict(color="#4682B4", width=15),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["joints"]["x"], y=first["joints"]["y"], z=first["joints"]["z"],
        mode="markers",
        marker=dict(size=7, color="#0B2E59"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["rectangle"]["x"], y=first["rectangle"]["y"], z=first["rectangle"]["z"],
        mode="lines",
        line=dict(color="#00B8D9", width=8),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["center"]["x"], y=first["center"]["y"], z=first["center"]["z"],
        mode="markers",
        marker=dict(size=6, color="white", line=dict(color="#333333", width=1)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["normal"]["x"], y=first["normal"]["y"], z=first["normal"]["z"],
        mode="lines",
        line=dict(color="#AB47BC", width=5),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["lasers"]["x"], y=first["lasers"]["y"], z=first["lasers"]["z"],
        mode="lines",
        line=dict(color="#FF6D00", width=5),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["impacts"]["x"], y=first["impacts"]["y"], z=first["impacts"]["z"],
        mode="markers",
        marker=dict(size=5, color="#00C853"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=first["sensors"]["x"], y=first["sensors"]["y"], z=first["sensors"]["z"],
        mode="markers+text",
        marker=dict(size=6, color="#FFB300"),
        text=first["sensor_labels"],
        textposition="top center",
        textfont=dict(size=13, color="#111111"),
        hoverinfo="skip", showlegend=False,
    ))

    # --------------------------------------------------------
    # FRAMES: apenas os 8 traces móveis
    # --------------------------------------------------------

    mobile_indices = [3, 4, 5, 6, 7, 8, 9, 10]
    frames = []

    for k, s in enumerate(snapshots):
        frame_data = [
            go.Scatter3d(x=s["links"]["x"], y=s["links"]["y"], z=s["links"]["z"]),
            go.Scatter3d(x=s["joints"]["x"], y=s["joints"]["y"], z=s["joints"]["z"]),
            go.Scatter3d(x=s["rectangle"]["x"], y=s["rectangle"]["y"], z=s["rectangle"]["z"]),
            go.Scatter3d(x=s["center"]["x"], y=s["center"]["y"], z=s["center"]["z"]),
            go.Scatter3d(x=s["normal"]["x"], y=s["normal"]["y"], z=s["normal"]["z"]),
            go.Scatter3d(x=s["lasers"]["x"], y=s["lasers"]["y"], z=s["lasers"]["z"]),
            go.Scatter3d(x=s["impacts"]["x"], y=s["impacts"]["y"], z=s["impacts"]["z"]),
            go.Scatter3d(
                x=s["sensors"]["x"],
                y=s["sensors"]["y"],
                z=s["sensors"]["z"],
                text=s["sensor_labels"],
            ),
        ]
        frames.append(
            go.Frame(
                name=f"frame_{k}",
                data=frame_data,
                traces=mobile_indices,
            )
        )

    fig.frames = frames

    lo, hi = scene_bounds(cfg, robot)

    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        uirevision="fixed_scene_v3",
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
            # Igualdade física entre X/Y/Z. O modo cube evita que o browser
            # altere a razão visual entre as unidades.
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.10, y=1.10, z=3.60),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
                projection=dict(type="orthographic"),
            ),
        ),
    )

    return fig, frames


def make_canvas_animation_html(snapshots, cfg, robot, height=680, autoplay=True, frame_delay_ms=550):
    """Animação 3D desenhada em Canvas 2D, sem WebGL/Plotly.

    Esta é uma escolha deliberada para a reprodução no celular. A cena inteira
    é redesenhada a cada frame: tubo, base e robô não dependem de traces WebGL
    persistentes. Assim nenhum objeto pode desaparecer durante o Play.
    A projeção é ortográfica e usa a mesma escala para X/Y/Z.
    """
    import json

    if not snapshots:
        return ""

    lo, hi = scene_bounds(cfg, robot)
    scene_center = ((lo + hi) / 2.0).astype(float)
    half = float(np.max(hi - lo)) / 2.0 * 1.08

    static_cfg = {
        "tube_x": float(cfg["tube_x"]),
        "tube_y": float(cfg["tube_y"]),
        "tube_z": float(cfg["tube_z"]),
        "tube_radius": float(cfg["tube_diameter"] / 2.0),
        "tube_length": float(cfg["tube_length"]),
        "base_x": float(cfg["base_x"]),
        "base_y": float(cfg["base_y"]),
        "base_z": float(cfg["base_z"]),
        "base_radius": 120.0,
        "base_height": 180.0,
    }

    payload = {
        "frames": snapshots,
        "static": static_cfg,
        "scene_center": scene_center.tolist(),
        "half": half,
    }

    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=True)
    autoplay_js = "setTimeout(startAnimation, 500);" if autoplay else ""

    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
html,body {{ margin:0; padding:0; background:#fff; width:100%; height:100%; overflow:hidden; font-family:Arial,sans-serif; }}
#wrap {{ position:relative; width:100%; height:100%; background:#fff; }}
#canvas {{ width:100%; height:100%; display:block; touch-action:none; }}
#controls {{ position:absolute; top:10px; left:10px; z-index:10; display:flex; gap:7px; }}
button {{ border:1px solid #aaa; background:#fff; color:#222; border-radius:7px; padding:8px 13px; font-size:15px; box-shadow:0 1px 4px rgba(0,0,0,.12); }}
#progress {{ position:absolute; left:12px; bottom:10px; z-index:10; background:rgba(255,255,255,.86); padding:5px 8px; border-radius:6px; font-size:13px; color:#444; }}
</style>
</head>
<body>
<div id="wrap">
<canvas id="canvas"></canvas>
<div id="controls">
  <button id="play">▶ Play</button>
  <button id="stop">■ Parar</button>
</div>
<div id="progress"></div>
</div>
<script>
const DATA = {data_json};
const frames = DATA.frames || [];
const S = DATA.static;
const sceneCenter = DATA.scene_center;
const half = DATA.half;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const progress = document.getElementById('progress');
let playing = false;
let token = 0;
let current = 0;
let W = 1, H = 1, scale = 1;

// Câmera ortográfica fixa. Unidades iguais nos 3 eixos.
const eye = normalize([1.10, 1.10, 3.60]);
const worldUp = [0,0,1];
const right = normalize(cross(eye, worldUp));
const up = normalize(cross(right, eye));

function normalize(a) {{
  const n = Math.hypot(a[0],a[1],a[2]) || 1;
  return [a[0]/n,a[1]/n,a[2]/n];
}}
function cross(a,b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }}
function dot(a,b) {{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }}
function sub(a,b) {{ return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }}

function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  W = Math.max(1, rect.width);
  H = Math.max(1, rect.height);
  canvas.width = Math.round(W*dpr);
  canvas.height = Math.round(H*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  scale = Math.min((W-50)/(2*half), (H-60)/(2*half));
  drawFrame(current);
}}

function project(p) {{
  const r = sub(p, sceneCenter);
  const px = dot(r,right);
  const py = dot(r,up);
  const depth = dot(r,eye);
  return [W/2 + px*scale, H/2 - py*scale, depth];
}}

function ellipsePoints(center, radius, z, n=64) {{
  const out = [];
  for (let i=0;i<=n;i++) {{
    const a = 2*Math.PI*i/n;
    out.push(project([center[0]+radius*Math.cos(a), center[1]+radius*Math.sin(a), z]));
  }}
  return out;
}}

function drawPolygon(points, fill) {{
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0],points[0][1]);
  for (let i=1;i<points.length;i++) ctx.lineTo(points[i][0],points[i][1]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}}

function drawPolyline(points, stroke, width, close=false) {{
  if (!points.length) return;
  ctx.beginPath();
  let started=false;
  for (let i=0;i<points.length;i++) {{
    if (points[i] === null) {{
      if (started) {{
        ctx.strokeStyle=stroke; ctx.lineWidth=width; ctx.stroke();
      }}
      ctx.beginPath(); started=false; continue;
    }}
    if (!started) {{ ctx.moveTo(points[i][0],points[i][1]); started=true; }}
    else ctx.lineTo(points[i][0],points[i][1]);
  }}
  if (started) {{
    if (close) ctx.closePath();
    ctx.strokeStyle=stroke;
    ctx.lineWidth=width;
    ctx.lineCap='round';
    ctx.lineJoin='round';
    ctx.stroke();
  }}
}}

function drawStaticCylinder(center, radius, height) {{
  const rings = 7;
  const levels = [];
  for (let k=0;k<rings;k++) levels.push(center[2]-height/2 + k*height/(rings-1));

  ctx.globalAlpha = 0.10;
  for (let k=0;k<rings-1;k++) {{
    const a = ellipsePoints(center,radius,levels[k]);
    const b = ellipsePoints(center,radius,levels[k+1]);
    for (let i=0;i<a.length-1;i++) drawPolygon([a[i],a[i+1],b[i+1],b[i]], '#8A9299');
  }}
  ctx.globalAlpha = 1;

  for (const z of levels) drawPolyline(ellipsePoints(center,radius,z), '#9E9E9E', 1.1, true);
  for (let i=0;i<16;i++) {{
    const a = 2*Math.PI*i/16;
    const p0 = project([center[0]+radius*Math.cos(a), center[1]+radius*Math.sin(a), levels[0]]);
    const p1 = project([center[0]+radius*Math.cos(a), center[1]+radius*Math.sin(a), levels[levels.length-1]]);
    drawPolyline([p0,p1], '#9E9E9E', 0.9);
  }}
}}

function drawLine3D(a,b,color,width) {{ drawPolyline([project(a),project(b)], color, width); }}

function drawRobot(s) {{
  const lx=s.links.x, ly=s.links.y, lz=s.links.z;
  for (let i=0;i<lx.length;i+=3) {{
    if (lx[i] == null || lx[i+1] == null) continue;
    drawLine3D([lx[i],ly[i],lz[i]],[lx[i+1],ly[i+1],lz[i+1]],'#4682B4',8);
  }}

  for (let i=0;i<s.joints.x.length;i++) {{
    const p=project([s.joints.x[i],s.joints.y[i],s.joints.z[i]]);
    ctx.beginPath(); ctx.arc(p[0],p[1],6,0,2*Math.PI); ctx.fillStyle='#0B2E59'; ctx.fill();
  }}

  const c=[];
  for (let i=0;i<s.rectangle.x.length;i++) c.push(project([s.rectangle.x[i],s.rectangle.y[i],s.rectangle.z[i]]));
  drawPolyline(c,'#00B8D9',4,true);

  const cp=project([s.center.x[0],s.center.y[0],s.center.z[0]]);
  ctx.beginPath(); ctx.arc(cp[0],cp[1],4,0,2*Math.PI); ctx.fillStyle='#fff'; ctx.fill(); ctx.strokeStyle='#333'; ctx.lineWidth=1; ctx.stroke();
  drawLine3D([s.normal.x[0],s.normal.y[0],s.normal.z[0]],[s.normal.x[1],s.normal.y[1],s.normal.z[1]],'#AB47BC',3);

  const lxs=s.lasers.x, lys=s.lasers.y, lzs=s.lasers.z;
  for (let i=0;i<lxs.length;i+=3) {{
    if (lxs[i] == null || lxs[i+1] == null) continue;
    drawLine3D([lxs[i],lys[i],lzs[i]],[lxs[i+1],lys[i+1],lzs[i+1]],'#FF6D00',3);
  }}

  for (let i=0;i<s.impacts.x.length;i++) {{
    if (s.impacts.x[i] == null) continue;
    const p=project([s.impacts.x[i],s.impacts.y[i],s.impacts.z[i]]);
    ctx.beginPath(); ctx.arc(p[0],p[1],4,0,2*Math.PI); ctx.fillStyle='#00C853'; ctx.fill();
  }}

  for (let i=0;i<s.sensors.x.length;i++) {{
    const p=project([s.sensors.x[i],s.sensors.y[i],s.sensors.z[i]]);
    ctx.beginPath(); ctx.arc(p[0],p[1],5,0,2*Math.PI); ctx.fillStyle='#FFB300'; ctx.fill();
    ctx.font='bold 14px Arial'; ctx.fillStyle='#111'; ctx.fillText(s.sensor_labels[i],p[0]+7,p[1]-7);
  }}
}}

function drawAxes() {{
  const L = half*0.55;
  drawLine3D([-L,0,sceneCenter[2]],[L,0,sceneCenter[2]],'#D0D0D0',1);
  drawLine3D([0,-L,sceneCenter[2]],[0,L,sceneCenter[2]],'#D0D0D0',1);
  drawLine3D([0,0,sceneCenter[2]-L],[0,0,sceneCenter[2]+L],'#D0D0D0',1);
  drawLine3D([S.tube_x,S.tube_y,S.tube_z-S.tube_length/2],[S.tube_x,S.tube_y,S.tube_z+S.tube_length/2],'#E53935',3);
}}

function drawFrame(k) {{
  if (!frames.length) return;
  current=Math.max(0,Math.min(k,frames.length-1));
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  drawAxes();
  drawStaticCylinder([S.tube_x,S.tube_y,S.tube_z],S.tube_radius,S.tube_length);
  drawStaticCylinder([S.base_x,S.base_y,S.base_z],S.base_radius,S.base_height);
  drawRobot(frames[current]);
  progress.textContent = 'Frame ' + (current+1) + ' / ' + frames.length;
}}

async function startAnimation() {{
  if (!frames.length || playing) return;
  const my = ++token;
  playing=true;
  if (current >= frames.length-1) current=0;
  while (playing && my===token) {{
    drawFrame(current);
    if (current >= frames.length-1) break;
    await new Promise(r=>setTimeout(r,{frame_delay_ms}));
    if (my!==token) break;
    current++;
  }}
  playing=false;
}}

function stopAnimation() {{ playing=false; token++; drawFrame(current); }}
document.getElementById('play').onclick=startAnimation;
document.getElementById('stop').onclick=stopAnimation;
window.addEventListener('resize', resize);
resize();
{autoplay_js}
</script>
</body>
</html>'''


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

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

st.title("Alinhamento automático — Robô 6 DOF + 4 lasers")
st.caption(
    "Versão web para celular/tablet. "
    "O cálculo continua baseado no normal.urdf."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

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
        )

        q_deg.append(value)

    manual_q = np.radians(
        np.asarray(q_deg, dtype=float)
    )

    if st.button(
        "Aplicar juntas",
        use_container_width=True,
    ):
        st.session_state.q = manual_q
        st.session_state.trajectory = None
        st.session_state.trajectory_cfg = None
        st.session_state.status = "Pose manual aplicada"
        st.rerun()

    if st.button(
        "↺ Resetar pose",
        use_container_width=True,
    ):
        st.session_state.q = np.radians(
            INITIAL_Q_DEG.copy()
        )
        reset_history()
        st.session_state.trajectory = None
        st.session_state.trajectory_cfg = None
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

if align_clicked:

    st.session_state.stop_requested = False

    q0 = st.session_state.q.copy()

    with st.spinner("Calculando trajetória de alinhamento..."):

        states, result = solve_trajectory(
            q0,
            cfg,
            robot,
            lasers,
            max_iterations=300,
        )

    st.session_state.trajectory = states
    st.session_state.trajectory_cfg = cfg
    st.session_state.q = result["final_q"]

    # A trajetória é calculada uma única vez no servidor.
    # A movimentação é reproduzida pelo navegador de forma fluida.
    animation_fig, animation_snapshots = make_animated_scene_figure(
        states,
        cfg,
        robot,
        lasers,
        frame_count=60,
    )

    # Reproduzimos a simulação em Canvas 2D. Isso elimina a dependência do
    # WebGL/Plotly durante o movimento e impede que tubo/base desapareçam.
    components.html(
        make_canvas_animation_html(
            animation_snapshots,
            cfg,
            robot,
            height=680,
            autoplay=True,
            frame_delay_ms=550,
        ),
        height=680,
        scrolling=False,
    )

    final_q_deg = np.degrees(
        result["final_q"]
    )

    d_final, angle_final, max_dist_error_final = current_metrics(
        robot,
        lasers,
        result["final_q"],
        cfg,
    )

    # ----------------------------
    # RESULTADO
    # ----------------------------

    if result["aligned"]:

        st.success(
            f"✓ ALINHADO em {result['iterations']} iterações • "
            f"erro de distância máx.: "
            f"{max_dist_error_final:.3f} mm • "
            f"erro angular: {angle_final:.4f}°"
        )

        st.session_state.status = "✓ ALINHADO"

    else:

        message = result["reason"] or "Trajetória encerrada."

        st.warning(
            message
            + " "
            + f"Erro de distância máx.: "
            + f"{max_dist_error_final:.3f} mm • "
            + f"erro angular: {angle_final:.4f}°"
        )

        st.session_state.status = message

    # ----------------------------
    # LEITURAS FINAIS
    # ----------------------------

    c1, c2, c3, c4 = st.columns(4)

    for c, label, value in zip(
        [c1, c2, c3, c4],
        ["A", "B", "C", "D"],
        d_final,
    ):

        with c:
            st.metric(
                f"Laser {label}",
                (
                    f"{value:.2f} mm"
                    if np.isfinite(value)
                    else "—"
                ),
            )

    # ----------------------------
    # GRÁFICO DA TRAJETÓRIA
    # ----------------------------

    history = result["history"]

    # Usa o histórico calculado diretamente, sem depender de reruns.
    original_history = st.session_state.history
    st.session_state.history = history

    st.plotly_chart(
        graph_figure(cfg),
        width="stretch",
        config={
            "displaylogo": False,
        },
    )

    st.session_state.history = original_history

    st.info(
        f"Trajetória calculada: {result['iterations']} iterações • "
        f"{len(animation_snapshots)} frames visuais. "
        "O cálculo é feito uma única vez no servidor e a animação é reproduzida em Canvas no navegador."
    )

# ------------------------------------------------------------
# VISUALIZAÇÃO NORMAL
# ------------------------------------------------------------

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

    st.plotly_chart(
        scene_fig,
        width="stretch",
        config={
            "scrollZoom": False,
            "displaylogo": False,
        },
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
