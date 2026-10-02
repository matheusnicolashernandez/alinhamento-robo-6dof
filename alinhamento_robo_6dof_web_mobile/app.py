
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Alinhamento Robô 6 DOF",
    page_icon=str(Path(__file__).with_name("favicon.png")),
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
                    x=0.65,
                    y=0.65,
                    z=3.80,
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
# TRAJETÓRIA PRÉ-CALCULADA
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
            and angle <= cfg["align_tol_deg"]
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


def _sample_states(states, frame_count):
    """Interpola os estados para uma reprodução visual mais suave."""
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


def animate_server_side(states, cfg, robot, lasers, frame_count=20, delay_ms=500):
    """Reproduz a trajetória usando o próprio Streamlit, sem JavaScript/WebGL frames.

    A cada frame a cena COMPLETA é recriada. Portanto tubo, base, robô,
    retângulo e sensores são sempre calculados e enviados juntos.
    Essa abordagem é mais lenta que uma animação client-side, mas é muito
    mais previsível dentro de um iframe do Streamlit Cloud em desktop/celular.
    """
    import time

    samples = _sample_states(states, frame_count)
    if samples.size == 0:
        return 0

    scene_placeholder = st.empty()
    progress_placeholder = st.empty()

    total = len(samples)
    delay = max(0.05, float(delay_ms) / 1000.0)

    for index, q in enumerate(samples):
        scene_fig, _ = make_scene_figure(q, cfg, robot, lasers)

        scene_placeholder.plotly_chart(
            scene_fig,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

        progress_placeholder.caption(
            f"Simulação: quadro {index + 1}/{total}"
        )

        if index < total - 1:
            time.sleep(delay)

    progress_placeholder.empty()
    return total


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

    st.number_input(
        "Tempo por quadro da simulação (ms)",
        min_value=100,
        max_value=1500,
        value=500,
        step=50,
        key="animation_delay_ms",
        help="Aumente este valor para deixar a simulação mais lenta.",
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

cfg = config_from_widgets()

if align_clicked:

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

    # --------------------------------------------------------
    # SIMULAÇÃO
    # --------------------------------------------------------
    # Sem Canvas, sem Plotly Frames e sem JavaScript.
    # O Streamlit atualiza o mesmo placeholder com a cena COMPLETA.
    frame_count = max(12, min(24, len(states) * 2))
    rendered_frames = animate_server_side(
        states,
        cfg,
        robot,
        lasers,
        frame_count=frame_count,
        delay_ms=int(st.session_state["animation_delay_ms"]),
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
        f"{rendered_frames} quadros visuais. "
        "A simulação atualiza a cena completa a cada quadro para manter "
        "tubo, base, robô e sensores sincronizados."
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
