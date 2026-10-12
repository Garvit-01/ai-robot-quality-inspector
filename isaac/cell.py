"""The work cell as a USD stage: built once, then re-posed for every scene.

Import this only *after* SimulationApp has started (it needs `pxr` from Isaac Sim).

Layout (metres, Z up, table top at z = 0, robot base at the origin):

                 +y
                  |        [ fixture ]  (0.45, +0.22)
     robot base   |
        (0,0) ----+-----------------------[ reject bin ] (0.74, 0)  ---- +x
                  |
                  |        [ carrier ]  (0.45, -0.22)

Everything here is simple primitives on purpose: the parts in this cell are simple
geometry, and every prop must be re-posable in microseconds between frames.

The gripper is a two-finger *stand-in* (palm, two fingers, a wrist link), not the
real UR5e + Robotiq. The pre-existing cell never had the arm inside Isaac Sim, and a
checkpoint photo only needs to show whether the fingers hold the part. Stated as a
limitation in the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from geometry import chipped_cube_mesh, crack_ribbon

# All dimensions and placement maths live in layout.py (pure Python, unit-tested).
import layout as L  # noqa: E402
from layout import (  # noqa: E402
    BIN_CENTER, BIN_SIZE, CARRIER_CENTER, CARRIER_SIZE, CARRIER_SLOT, CARRIER_TAG,
    CARRIER_TOP, FINGER, FIXTURE_BASE_TOP, FIXTURE_CENTER, FIXTURE_OUTER, FIXTURE_POCKET,
    FIXTURE_WALL_TOP, FLOOR_Z, GRIPPER_GAP_EMPTY, GRIPPER_GAP_OPEN, NEST_INNER, NEST_LINE,
    PART_EDGE, TABLE_CENTER, TABLE_SIZE, TAG_SIZE,
)


# ---------------------------------------------------------------------------------------
# Small USD helpers
# ---------------------------------------------------------------------------------------
def _material(stage: Usd.Stage, path: str, rgb, roughness=0.6, metallic=0.0) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _set_color(mat: UsdShade.Material, rgb) -> None:
    shader = UsdShade.Shader(mat.GetPrim().GetChild("Shader"))
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*rgb))


def _textured_material(stage: Usd.Stage, path: str, texture: Path) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, f"{path}/Shader")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


def _bind(prim: Usd.Prim, mat: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _box(stage, path, size, center, mat) -> Usd.Prim:
    """An axis-aligned box: a unit cube scaled. `center` is (x, y, z) of its centre."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(*center))
    api.SetScale(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), mat)
    return cube.GetPrim()


def _pose(prim: Usd.Prim, xyz, rot_xyz_deg=(0.0, 0.0, 0.0)) -> None:
    """Set translate + rotate. USD's XYZ order rotates about x first, then y, then z —
    so a seat tilt (about x) is applied in the part's own frame before its yaw (about z)."""
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*xyz))
    api.SetRotate(Gf.Vec3f(*rot_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def _set_matrix(prim: Usd.Prim, R, t) -> None:
    """Full rigid pose from a 3x3 rotation and a translation (see layout.usd_matrix_rows).
    Needed for parts: a seat tilt toward an arbitrary wall isn't an XYZ Euler triple."""
    rows = L.usd_matrix_rows(R, t)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*[v for row in rows for v in row]))


def _visible(prim: Usd.Prim, on: bool) -> None:
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if on else img.MakeInvisible()


# ---------------------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------------------
@dataclass
class Views:
    overview: str
    closeup: str


class Cell:
    """Builds the stage once; `apply()` re-poses it for one SceneSpec."""

    def __init__(self, stage: Usd.Stage, tag_texture: Path, resolution=(1024, 768)):
        self.stage = stage
        self.resolution = resolution
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        s = stage

        # Materials. Colours that domain randomisation touches keep a handle.
        self.m_floor = _material(s, "/World/Looks/Floor", (0.18, 0.18, 0.19), 0.9)
        self.m_table = _material(s, "/World/Looks/Table", (0.55, 0.56, 0.58), 0.7)
        self.m_carrier = _material(s, "/World/Looks/Carrier", (0.72, 0.74, 0.78), 0.5, 0.3)
        self.m_fixture = _material(s, "/World/Looks/Fixture", (0.12, 0.16, 0.26), 0.35, 0.6)
        self.m_bin = _material(s, "/World/Looks/Bin", (0.82, 0.22, 0.12), 0.6)
        self.m_part = _material(s, "/World/Looks/Part", (0.62, 0.62, 0.64), 0.45, 0.4)
        self.m_grip = _material(s, "/World/Looks/Gripper", (0.10, 0.10, 0.11), 0.5, 0.2)
        self.m_wrist = _material(s, "/World/Looks/Wrist", (0.78, 0.80, 0.82), 0.4, 0.1)
        self.m_robot = _material(s, "/World/Looks/RobotBase", (0.70, 0.72, 0.75), 0.4, 0.1)
        # Bare metal exposed by a chip. Machined parts in a cell like this are usually
        # painted or anodised, so a knocked corner shows bright raw metal — that is what
        # an inspector actually sees, not a shortcut for the model.
        # Satin, not mirror: at metallic 0.95 the facet just reflected the grey sky and
        # was indistinguishable from the paint. Freshly exposed aluminium reads brighter.
        self.m_chip = _material(s, "/World/Looks/ChipBareMetal", (0.94, 0.94, 0.96), 0.35, 0.3)
        self.m_tag = _textured_material(s, "/World/Looks/Tag", tag_texture)
        # A crack shows as a dark line: the gap is in shadow and the broken edges are rough.
        self.m_crack = _material(s, "/World/Looks/Crack", (0.05, 0.05, 0.055), 0.95, 0.0)
        self.m_print = _material(s, "/World/Looks/NestPrint", (0.16, 0.17, 0.19), 0.8, 0.0)

        # Floor and table.
        _box(s, "/World/Floor", (6.0, 6.0, 0.02), (0.4, 0.0, FLOOR_Z - 0.01), self.m_floor)
        tx, ty, tz = TABLE_SIZE
        _box(s, "/World/Table", TABLE_SIZE, (TABLE_CENTER[0], TABLE_CENTER[1], -tz / 2), self.m_table)
        base = UsdGeom.Cylinder.Define(s, "/World/RobotBase")
        base.CreateRadiusAttr(0.075)
        base.CreateHeightAttr(0.10)
        UsdGeom.XformCommonAPI(base).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.05))
        _bind(base.GetPrim(), self.m_robot)

        # Carrier tile with its AprilTag. The tag localises the carrier, not the part.
        cx, cy = CARRIER_CENTER
        _box(s, "/World/Carrier", CARRIER_SIZE, (cx, cy, CARRIER_TOP / 2), self.m_carrier)
        self._tag_quad("/World/CarrierTag", CARRIER_TAG, CARRIER_TOP + 0.0005)
        # A printed nest outline around the slot: what a real carrier has, and what makes
        # "displaced" visible at all. Without it the only cue was distance to the tag.
        sx, sy = CARRIER_SLOT
        o = NEST_INNER / 2 + NEST_LINE / 2
        ln = NEST_INNER + 2 * NEST_LINE
        zt = CARRIER_TOP + 0.0002
        for name, size, ctr in (("N", (ln, NEST_LINE, 0.0004), (sx, sy + o, zt)),
                                ("S", (ln, NEST_LINE, 0.0004), (sx, sy - o, zt)),
                                ("E", (NEST_LINE, NEST_INNER, 0.0004), (sx + o, sy, zt)),
                                ("W", (NEST_LINE, NEST_INNER, 0.0004), (sx - o, sy, zt))):
            _box(s, f"/World/CarrierNest/{name}", size, ctr, self.m_print)

        # Fixture: a base plate plus four walls forming a square pocket.
        fx, fy = FIXTURE_CENTER
        _box(s, "/World/Fixture/Base", (FIXTURE_OUTER, FIXTURE_OUTER, FIXTURE_BASE_TOP),
             (fx, fy, FIXTURE_BASE_TOP / 2), self.m_fixture)
        wall_h = FIXTURE_WALL_TOP - FIXTURE_BASE_TOP
        wall_t = (FIXTURE_OUTER - FIXTURE_POCKET) / 2
        wz = FIXTURE_BASE_TOP + wall_h / 2
        off = FIXTURE_POCKET / 2 + wall_t / 2
        _box(s, "/World/Fixture/WallN", (FIXTURE_OUTER, wall_t, wall_h), (fx, fy + off, wz), self.m_fixture)
        _box(s, "/World/Fixture/WallS", (FIXTURE_OUTER, wall_t, wall_h), (fx, fy - off, wz), self.m_fixture)
        _box(s, "/World/Fixture/WallE", (wall_t, FIXTURE_POCKET, wall_h), (fx + off, fy, wz), self.m_fixture)
        _box(s, "/World/Fixture/WallW", (wall_t, FIXTURE_POCKET, wall_h), (fx - off, fy, wz), self.m_fixture)

        # Reject bin: an open box.
        bx, by = BIN_CENTER
        bw, bd, bh = BIN_SIZE
        t = 0.006
        _box(s, "/World/Bin/Floor", (bw, bd, t), (bx, by, t / 2), self.m_bin)
        _box(s, "/World/Bin/WallN", (bw, t, bh), (bx, by + bd / 2 - t / 2, bh / 2), self.m_bin)
        _box(s, "/World/Bin/WallS", (bw, t, bh), (bx, by - bd / 2 + t / 2, bh / 2), self.m_bin)
        _box(s, "/World/Bin/WallE", (t, bd, bh), (bx + bw / 2 - t / 2, by, bh / 2), self.m_bin)
        _box(s, "/World/Bin/WallW", (t, bd, bh), (bx - bw / 2 + t / 2, by, bh / 2), self.m_bin)

        self._build_parts()
        self._build_gripper()
        self._build_lights()
        self.views = self._build_cameras()

    # -- builders -----------------------------------------------------------------------
    def _tag_quad(self, path, xy, z):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        s = TAG_SIZE / 2
        mesh.CreatePointsAttr([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0), Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
        UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(xy[0], xy[1], z))
        _bind(mesh.GetPrim(), self.m_tag)

    def _build_parts(self):
        s = self.stage
        self.parts = {}

        def holder(name):
            return UsdGeom.Xform.Define(s, f"/World/Parts/{name}").GetPrim()

        p = holder("correct")
        c = UsdGeom.Cube.Define(s, "/World/Parts/correct/geo")
        c.CreateSizeAttr(PART_EDGE)
        _bind(c.GetPrim(), self.m_part)
        self.parts[("correct", None)] = p

        p = holder("cylinder")
        cyl = UsdGeom.Cylinder.Define(s, "/World/Parts/cylinder/geo")
        cyl.CreateRadiusAttr(0.025)
        cyl.CreateHeightAttr(0.05)
        _bind(cyl.GetPrim(), self.m_part)
        self.parts[("wrong", "cylinder")] = p

        p = holder("small_cube")
        c = UsdGeom.Cube.Define(s, "/World/Parts/small_cube/geo")
        c.CreateSizeAttr(0.035)
        _bind(c.GetPrim(), self.m_part)
        self.parts[("wrong", "small_cube")] = p

        p = holder("tall_block")
        c = UsdGeom.Cube.Define(s, "/World/Parts/tall_block/geo")
        c.CreateSizeAttr(1.0)
        UsdGeom.XformCommonAPI(c).SetScale(Gf.Vec3f(0.05, 0.05, 0.075))
        _bind(c.GetPrim(), self.m_part)
        self.parts[("wrong", "tall_block")] = p

        # Defective, crack variant: a plain cube with a thin dark ribbon on its top face,
        # rebuilt per scene (crack length, angle and path vary).
        p = holder("cracked")
        c = UsdGeom.Cube.Define(s, "/World/Parts/cracked/geo")
        c.CreateSizeAttr(PART_EDGE)
        _bind(c.GetPrim(), self.m_part)
        self.crack_mesh = UsdGeom.Mesh.Define(s, "/World/Parts/cracked/crack")
        self.crack_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdGeom.XformCommonAPI(self.crack_mesh).SetTranslate(Gf.Vec3d(0, 0, PART_EDGE / 2 + 0.0003))
        _bind(self.crack_mesh.GetPrim(), self.m_crack)
        self.parts[("defective", "crack")] = p

        # Defective, chip variant: mesh rebuilt per scene (chip size and corner vary).
        p = holder("defective")
        self.defect_mesh = UsdGeom.Mesh.Define(s, "/World/Parts/defective/geo")
        self.defect_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        _bind(self.defect_mesh.GetPrim(), self.m_part)
        # chipped_cube_mesh() always emits the chip triangle as the last of 7 faces.
        chip_subset = UsdGeom.Subset.CreateGeomSubset(
            self.defect_mesh, "chip", UsdGeom.Tokens.face, Vt.IntArray([6]), "materialBind")
        UsdGeom.Subset.SetFamilyType(self.defect_mesh, "materialBind", UsdGeom.Tokens.nonOverlapping)
        _bind(chip_subset.GetPrim(), self.m_chip)
        self.parts[("defective", "chip")] = p

        for prim in self.parts.values():
            _visible(prim, False)

    def _build_gripper(self):
        s = self.stage
        self.gripper = UsdGeom.Xform.Define(s, "/World/Gripper").GetPrim()
        # Origin = the grasp point between the finger pads.
        fx, fy, fz = FINGER
        self.finger_l = _box(s, "/World/Gripper/FingerL", FINGER, (0, 0, 0), self.m_grip)
        self.finger_r = _box(s, "/World/Gripper/FingerR", FINGER, (0, 0, 0), self.m_grip)
        palm_z0 = fz / 2 + 0.008
        _box(s, "/World/Gripper/Palm", (0.045, 0.11, 0.035), (0, 0, palm_z0 + 0.0175), self.m_grip)
        wrist = UsdGeom.Cylinder.Define(s, "/World/Gripper/Wrist")
        wrist.CreateRadiusAttr(0.032)
        wrist.CreateHeightAttr(0.12)  # a short flange; a tall link read as a pillar on the table
        UsdGeom.XformCommonAPI(wrist).SetTranslate(Gf.Vec3d(0, 0, palm_z0 + 0.035 + 0.06))
        _bind(wrist.GetPrim(), self.m_wrist)

    def _set_gap(self, gap: float):
        fx, fy, fz = FINGER
        y = gap / 2 + fy / 2
        UsdGeom.XformCommonAPI(self.finger_l).SetTranslate(Gf.Vec3d(0, +y, fz / 2 - 0.02))
        UsdGeom.XformCommonAPI(self.finger_r).SetTranslate(Gf.Vec3d(0, -y, fz / 2 - 0.02))

    def _build_lights(self):
        s = self.stage
        self.dome = UsdLux.DomeLight.Define(s, "/World/Lights/Dome")
        self.dome.CreateIntensityAttr(800.0)
        self.sun = UsdLux.DistantLight.Define(s, "/World/Lights/Key")
        self.sun.CreateIntensityAttr(2500.0)
        self.sun.CreateAngleAttr(1.5)
        for light in (self.dome, self.sun):
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(5500.0)

    def _camera(self, path, focal_mm, w, h) -> UsdGeom.Camera:
        cam = UsdGeom.Camera.Define(self.stage, path)
        cam.CreateFocalLengthAttr(focal_mm)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        return cam

    def _build_cameras(self) -> Views:
        w, h = self.resolution
        self.cam_overview = self._camera("/World/Cameras/Overview", 15.0, w, h)
        self.cam_closeup = self._camera("/World/Cameras/Closeup", 28.0, w, h)
        return Views("/World/Cameras/Overview", "/World/Cameras/Closeup")

    @staticmethod
    def _look_at(cam: UsdGeom.Camera, eye, target):
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(view.GetInverse())

    # -- per-scene ----------------------------------------------------------------------
    def apply(self, spec, dr) -> dict:
        """Pose everything for `spec` and apply domain randomisation `dr`.
        Returns the concrete world-frame values used, for the label file."""
        # Part: show exactly one variant.
        if spec.part_kind == "wrong":
            key = ("wrong", spec.wrong_shape)
        elif spec.part_kind == "defective":
            key = ("defective", spec.defect_type)
        else:
            key = ("correct", None)
        for k, prim in self.parts.items():
            _visible(prim, k == key)
        part = self.parts[key]
        if key == ("defective", "chip"):
            pts, counts, idx = chipped_cube_mesh(PART_EDGE, spec.chip_mm / 1000.0, spec.chip_corner)
            self.defect_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            self.defect_mesh.GetFaceVertexCountsAttr().Set(counts)
            self.defect_mesh.GetFaceVertexIndicesAttr().Set(idx)
        elif key == ("defective", "crack"):
            pts, counts, idx = crack_ribbon(spec.crack_mm / 1000.0, spec.crack_angle_deg,
                                            PART_EDGE / 2, spec.seed)
            self.crack_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            self.crack_mesh.GetFaceVertexCountsAttr().Set(counts)
            self.crack_mesh.GetFaceVertexIndicesAttr().Set(idx)
            self.crack_mesh.CreateNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(pts)))

        # Gripper.
        ax, ay, az = L.gripper_anchor(spec)
        g_xyz = (ax, ay, az + spec.gripper_height_m)
        _pose(self.gripper, g_xyz, (0, 0, 0))
        if not spec.gripper_closed:
            self._set_gap(GRIPPER_GAP_OPEN)
        elif spec.gripper_holding:
            self._set_gap(PART_EDGE)
        else:
            self._set_gap(GRIPPER_GAP_EMPTY)

        if spec.part_location == "gripper":
            p_xyz, p_rot = g_xyz, L.rot_z(0.0)
        else:
            p_xyz, p_rot = L.part_pose(spec)
        _set_matrix(part, p_rot, p_xyz)

        # Domain randomisation.
        _set_color(self.m_table, dr["table_rgb"])
        _set_color(self.m_part, dr["part_rgb"])
        self.dome.GetIntensityAttr().Set(dr["dome_intensity"])
        self.sun.GetIntensityAttr().Set(dr["key_intensity"])
        for light in (self.dome, self.sun):
            light.GetColorTemperatureAttr().Set(dr["color_temp_k"])
        _pose(self.sun.GetPrim(), (0, 0, 0), (dr["key_elevation_deg"] - 90.0, 0, dr["key_azimuth_deg"]))

        # Cameras: a fixed-ish overview, and a close-up on this checkpoint's region.
        j = dr["cam_jitter"]
        ov_eye = (0.98 + j[0], 0.0 + j[1], 1.05 + j[2])
        ov_target = (0.46, 0.0, 0.0)
        self._look_at(self.cam_overview, ov_eye, ov_target)

        roi = self._roi(spec, g_xyz, p_xyz)
        cu_eye = (roi[0] + 0.34 + j[0] * 0.5, roi[1] - 0.12 + j[1] * 0.5, roi[2] + 0.28 + j[2] * 0.5)
        self._look_at(self.cam_closeup, cu_eye, roi)

        return {
            "part_world_xyz": [round(v, 5) for v in p_xyz],
            "part_world_rotation": [[round(v, 6) for v in row] for row in p_rot],
            "gripper_world_xyz": [round(v, 5) for v in g_xyz],
            "overview_eye": [round(v, 4) for v in ov_eye],
            "closeup_eye": [round(v, 4) for v in cu_eye],
            "closeup_target": [round(v, 4) for v in roi],
        }

    @staticmethod
    def _roi(spec, g_xyz, p_xyz):
        cp = spec.checkpoint
        if cp == "CP_PRE_GRASP":
            return (CARRIER_SLOT[0], CARRIER_SLOT[1], CARRIER_TOP + 0.02)
        if cp == "CP_POST_PLACE":
            return (FIXTURE_CENTER[0], FIXTURE_CENTER[1], FIXTURE_BASE_TOP + 0.02)
        # Lift/transport: frame the gripper and whatever is (or isn't) under it.
        return (g_xyz[0], g_xyz[1], g_xyz[2] - 0.04)
