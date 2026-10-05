"""Build the MuJoCo scene from pinned official robot and scanned YCB assets."""

from __future__ import annotations
import warnings
import numpy as np
from pathlib import Path

# FRUIT_SPECS = (
#     ("apple_0", "apple", 1.05, 0.075),
#     ("apple_1", "apple", 0.95, 0.065),
#     ("apple_2", "apple", 1.0, 0.055),
#     ("banana_0", "banana", 1.05, 0.055),
#     ("banana_1", "banana", 0.95, 0.065),
#     ("banana_2", "banana", 1.0, 0.075),
#     ("pear_0", "pear", 1.05, 0.065),
#     ("pear_1", "pear", 0.95, 0.075),
#     ("pear_2", "pear", 1.0, 0.055),
# )
FRUIT_SPECS = (
    ("apple_0", "apple", 1.0, 0.075),
    ("banana_0", "banana", 1.0, 0.075),
    ("pear_0", "pear", 1.0, 0.075),
)
# FRUIT_SPECS = (("apple_0", "apple", 1.0, 0.075),)
TEXTURE_SUFFIXES = {"apple": "026", "banana": "024", "pear": "029"}


def _fruit_material(spec, assets: Path, kind: str):
    import mujoco

    suffix = TEXTURE_SUFFIXES[kind]
    texture = spec.add_texture(
        name=f"ycb_{kind}_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(assets / "ycb" / kind / f"material_0.{suffix}.png"),
    )
    material = spec.add_material(
        name=f"ycb_{kind}_material",
        rgba=[1, 1, 1, 1],
        specular=0.08,
        shininess=0.08,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    return material


def _add_bin(
    world,
    name: str,
    center: tuple[float, float, float],
    color,
    half_size: tuple[float, float] = (0.13, 0.11),
):
    import mujoco

    body = world.add_body(name=name, pos=center)
    darker = [max(0.0, value * 0.65) for value in color[:3]] + [1.0]
    half_x, half_y = half_size
    wall_half_height = 0.045
    body.add_geom(
        name=f"{name}_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_x, half_y, 0.012],
        rgba=darker,
        friction=[1.2, 0.01, 0.002],
    )
    for index, (pos, size) in enumerate(
        (
            ((half_x, 0, wall_half_height), (0.012, half_y, wall_half_height)),
            ((-half_x, 0, wall_half_height), (0.012, half_y, wall_half_height)),
            ((0, half_y, wall_half_height), (half_x, 0.012, wall_half_height)),
            ((0, -half_y, wall_half_height), (half_x, 0.012, wall_half_height)),
        )
    ):
        body.add_geom(
            name=f"{name}_wall_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=color,
            friction=[1.2, 0.01, 0.002],
        )


def build_official_scene(timestep: float):
    """Return a compiled UR5e + Robotiq 2F-85 + YCB fruit model."""
    import mujoco

    assets = Path(__file__).with_name("assets")
    robot_root = assets / "menagerie"
    ur5e = mujoco.MjSpec.from_file(
        str(robot_root / "universal_robots_ur5e" / "ur5e.xml")
    )
    gripper = mujoco.MjSpec.from_file(str(robot_root / "robotiq_2f85" / "2f85.xml"))
    # Resolve option inheritance explicitly before attaching the child spec.
    # The scene applies its final solver settings immediately afterwards.
    gripper.option.impratio = ur5e.option.impratio
    gripper.option.cone = ur5e.option.cone
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict.*")
        ur5e.attach(gripper, site=ur5e.site("attachment_site"), prefix="gripper_")
    ur5e.option.timestep = timestep
    ur5e.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
    ur5e.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    ur5e.option.impratio = 10
    ur5e.body("base").pos = np.asarray([-0.30, 0.0, 0.05])

    world = ur5e.worldbody
    sky = ur5e.add_texture(
        name="bright_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.72, 0.80, 0.90],
        rgb2=[0.48, 0.57, 0.68],
        width=512,
        height=3072,
    )
    del sky  # The skybox role is established by its texture type.
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[15, 15, 0.1],
        rgba=[0.62, 0.67, 0.73, 1],
    )
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.38, 0, 0.035],
        size=[0.68, 0.66, 0.035],
        rgba=[0.72, 0.76, 0.80, 1],
        friction=[1.3, 0.01, 0.002],
    )
    world.add_light(
        name="key_light",
        pos=[0.2, -0.7, 2.0],
        dir=[0.1, 0.2, -1],
        diffuse=[0.78, 0.78, 0.78],
        ambient=[0.18, 0.18, 0.18],
        specular=[0.12, 0.12, 0.12],
        castshadow=False,
    )
    world.add_light(
        name="fill_light",
        pos=[1.2, 0.8, 1.4],
        dir=[-0.5, -0.3, -1],
        diffuse=[0.38, 0.42, 0.48],
        ambient=[0.10, 0.10, 0.12],
        castshadow=False,
    )
    # Feature 1 prioritises reliable instance separation and bin verification.
    # A true overhead camera removes perspective shortening and keeps the
    # complete tabletop visible; the identity camera orientation looks along
    # MuJoCo local -Z, straight down at the table.
    world.add_camera(
        name="top_camera",
        pos=[0.38, 0.0, 1.82],
        quat=[1.0, 0.0, 0.0, 0.0],
        fovy=50,
    )
    # front_camera_position = np.asarray([1.2, -1.2, 1.3])
    # front_camera_target = np.asarray([0.3, 0.0, 0.1])
    # front_camera_position = np.asarray([1.5, -1.3, 1.5])
    # front_camera_target = np.asarray([0.3, 0.0, 0.3])
    front_camera_position = np.asarray([1.5, -1.3, 1.5])
    front_camera_target = np.asarray([0.3, 0.0, 0.2])
    front_camera = world.add_camera(
        name="front_camera",
        pos=front_camera_position,
        fovy=48,
    )
    front_camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_ZAXIS
    front_camera.alt.zaxis = front_camera_position - front_camera_target

    # 获取夹爪根体并添加相机
    gripper_body = ur5e.body("gripper_base")
    camera = gripper_body.add_camera()
    camera.name = "wrist_camera"
    camera.pos = [0.05, 0.0, 0.05]
    # camera.quat = [0.3536, 0.6124, 0.6124, 0.3536]
    # camera.quat = [0.2706, 0.6533, 0.6533, 0.2706]
    # camera.quat = [0.1830, 0.6830, 0.6830, 0.1830]
    camera.quat = [0.0923, 0.7011, 0.7011, 0.0923]
    # camera.quat = [0.0, 0.7071, 0.7071, 0.0]
    camera.fovy = 60

    _add_bin(world, "blue_bin", (0.27, -0.50, 0.082), [0.06, 0.28, 0.90, 1])
    _add_bin(
        world,
        "green_bin",
        (0.27, 0.50, 0.082),
        [0.08, 0.72, 0.14, 1],
        half_size=(0.17, 0.14),
    )
    materials = {
        kind: _fruit_material(ur5e, assets, kind)
        for kind in ("apple", "banana", "pear")
    }
    for body_name, kind, scale, spawn_z in FRUIT_SPECS:
        visual_mesh = ur5e.add_mesh(
            name=f"{body_name}_visual_mesh",
            file=str(assets / "ycb" / kind / "visual.obj"),
            scale=[scale, scale, scale],
        )
        collision_mesh = ur5e.add_mesh(
            name=f"{body_name}_collision_mesh",
            file=str(assets / "ycb" / kind / "collision.obj"),
            scale=[scale, scale, scale],
        )
        body = world.add_body(name=body_name, pos=[0.50, 0, spawn_z])
        body.add_freejoint(name=f"{body_name}_free")
        body.add_geom(
            name=f"{body_name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=visual_mesh.name,
            material=materials[kind].name,
            contype=0,
            conaffinity=0,
            group=2,
        )
        nominal_mass = {"apple": 0.068, "banana": 0.066, "pear": 0.075}[kind]
        body.add_geom(
            name=f"{body_name}_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=collision_mesh.name,
            mass=nominal_mass * scale**3,
            friction=[1.4, 0.01, 0.003],
            group=3,
            rgba=[0, 0, 0, 0],
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict.*")
        model = ur5e.compile()
    # The stock UR gains are tuned for the bare wrist.  The attached 2F-85
    # adds enough mass and lever arm that the wrist otherwise settles about
    # 2--3 cm away from its command.  Preserve the official controller form
    # while scaling its stiffness/damping for the payload.
    model.actuator_gainprm[:6, 0] *= 4.0
    model.actuator_biasprm[:6, 1] *= 4.0
    model.actuator_biasprm[:6, 2] *= 2.0
    return model
