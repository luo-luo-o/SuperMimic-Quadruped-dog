"""Build dog MJCF/URDF from the original URDF's assembled CAD mesh poses.

CAD coordinates: X right, Y rear, Z up. Simulation: X forward, Y left, Z up.
Each visual has a link-local origin and each link has a joint transform. Apply
that entire chain at source zero, then express the mesh in the new link frame
at the nominal standing pose. This also handles the anomalous right rear leg.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src/supermimic/robots/assets/dog'
SOURCE = Path('/home/hcl/workspace/rlsar/src/rl_sar_zoo/dog_description')
C = np.array([-.019782, -.0554, -.0345])  # CAD trunk frame near the shoulder axes
R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], float)


def stl_vertices(path: Path):
    b = path.read_bytes()
    n = struct.unpack_from('<I', b, 80)[0]
    if 84 + 50*n != len(b):
        raise ValueError(f'{path}: expected binary STL')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    return np.frombuffer(b, dtype=dtype, offset=84, count=n).copy()


def pose(origin: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.eye(3), np.zeros(3)
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    return Rotation.from_euler('xyz', rpy).as_matrix(), xyz


def source_link_poses(src: ET.Element) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    children: dict[str, list[ET.Element]] = {}
    for joint in src.findall('joint'):
        children.setdefault(joint.find('parent').get('link'), []).append(joint)
    out = {}

    def visit(name: str, rot: np.ndarray, pos: np.ndarray):
        out[name] = rot, pos
        for joint in children.get(name, []):
            jr, jp = pose(joint.find('origin'))
            visit(joint.find('child').get('link'), rot @ jr, pos + rot @ jp)

    visit('base_link', np.eye(3), np.zeros(3))
    return out


def source_point(poses: dict[str, tuple[np.ndarray, np.ndarray]], name: str,
                 local: np.ndarray | None = None) -> np.ndarray:
    rot, pos = poses[name]
    return (rot @ (np.zeros(3) if local is None else local) + pos - C) @ R.T


def write_mesh(name: str, source: ET.Element, source_pose: tuple[np.ndarray, np.ndarray],
               target_pose: tuple[np.ndarray, np.ndarray]):
    """Preserve the source URDF's assembled visual pose at the new standing pose."""
    arr = stl_vertices(SOURCE / 'meshes' / f'{name}.stl')
    verts = arr['vertices'].astype(float)
    normals = arr['normal'].astype(float)
    visual = source.find('visual')
    vr, vp = pose(visual.find('origin'))
    sr, sp = source_pose
    tr, tp = target_pose
    world = (verts @ vr.T + vp) @ sr.T + sp
    standard = (world - C) @ R.T
    arr['vertices'] = ((standard - tp) @ tr).astype('<f4')
    arr['normal'] = (normals @ vr.T @ sr.T @ R.T @ tr).astype('<f4')
    data = b'CAD axes corrected; joint-local mesh'.ljust(80, b' ') + struct.pack('<I', len(arr)) + arr.tobytes()
    (OUT / 'meshes' / f'{name}.stl').write_bytes(data)


def add_shank_collisions(calf: ET.Element, leg: str, source: ET.Element,
                         source_pose: tuple[np.ndarray, np.ndarray],
                         target_pose: tuple[np.ndarray, np.ndarray]):
    """Carry over the CAD shank's two fitted cylinders in the same frame as its mesh."""
    sr, sp = source_pose
    tr, tp = target_pose
    for i, collision in enumerate(source.findall('collision')):
        cylinder = collision.find('geometry/cylinder')
        if cylinder is None:
            continue
        cr, cp = pose(collision.find('origin'))
        half = float(cylinder.get('length')) / 2
        endpoints = np.array([[0, 0, -half], [0, 0, half]])
        world = (endpoints @ cr.T + cp) @ sr.T + sp
        standard = (world - C) @ R.T
        local = (standard - tp) @ tr
        add(calf, 'geom', name=f'{leg}_calf{i}_collision', type='capsule',
            fromto=fmt(local.ravel()), size=cylinder.get('radius'), rgba='0.5 0.5 0.5 0.2')


def transform_inertial(link: ET.Element, source_pose, target_pose):
    """Express the CAD center of mass and its inertia tensor in the new link frame."""
    tr, tp = target_pose
    sr, sp = source_pose
    inertial = link.find('inertial')
    mass = float(inertial.find('mass').get('value'))
    ir, ip = pose(inertial.find('origin'))
    com = tr.T @ (R @ (sr @ ip + sp - C) - tp)
    values = {key: float(value) for key, value in inertial.find('inertia').attrib.items()}
    source_inertia = np.array([
        [values['ixx'], values['ixy'], values['ixz']],
        [values['ixy'], values['iyy'], values['iyz']],
        [values['ixz'], values['iyz'], values['izz']],
    ])
    rotation = tr.T @ R @ sr @ ir
    return mass, com, rotation @ source_inertia @ rotation.T


def add_full_inertial(body: ET.Element, mass: float, center: np.ndarray, inertia: np.ndarray):
    # The CAD axes are rounded; remove numerical residue in effectively zero cross terms.
    full = (*inertia.diagonal(), inertia[0, 1], inertia[0, 2], inertia[1, 2])
    add(body, 'inertial', pos=fmt(center), mass=f'{mass:.7g}',
        fullinertia=fmt(0.0 if abs(value) < 1e-12 else value for value in full))


def merged_calf_inertial(source_links: dict[str, ET.Element],
                         source_poses: dict[str, tuple[np.ndarray, np.ndarray]],
                         shank_name: str, foot_name: str,
                         target_pose: tuple[np.ndarray, np.ndarray]):
    """Transform CAD shank/foot inertias to the new calf frame and merge them."""
    parts = [
        transform_inertial(source_links[name], source_poses[name], target_pose)
        for name in (shank_name, foot_name)
    ]
    total_mass = sum(mass for mass, _, _ in parts)
    center = sum(mass * com for mass, com, _ in parts) / total_mass
    inertia = np.zeros((3, 3))
    for mass, com, own_inertia in parts:
        delta = com - center
        inertia += own_inertia + mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return total_mass, center, inertia


def add_thigh_collisions(thigh: ET.Element, leg: str, side_sign: int):
    """Approximate the two CAD side plates and their thin transverse brace."""
    rgba = '0.5 0.5 0.5 0.2'
    # Measured in the corrected thigh STL: 8 mm plates centered at y=3.8 and
    # -58 mm on the left, with x half-widths of 10–15 mm and z midpoint -60 mm.
    for name, y in (('outer', 0.0038), ('inner', -0.058)):
        add(thigh, 'geom', name=f'{leg}_thigh_{name}_collision', type='box',
            pos=fmt((0, side_sign * y, -0.060)), size='0.012 0.004 0.070', rgba=rgba)
    add(thigh, 'geom', name=f'{leg}_thigh_brace_collision', type='cylinder',
        fromto=fmt((0, side_sign * -0.058, -0.060, 0, side_sign * 0.0038, -0.060)),
        size='0.004', rgba=rgba)


def fmt(v):
    return ' '.join(f'{x:.7g}' for x in v)


def add(parent, tag, **kwargs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kwargs.items()})


def build():
    (OUT / 'meshes').mkdir(parents=True, exist_ok=True)
    src = ET.parse(SOURCE / 'urdf' / 'Robotic_dog.urdf').getroot()
    source_links = {link.get('name'): link for link in src.findall('link')}
    source_poses = source_link_poses(src)
    model = ET.Element('mujoco', model='dog')
    add(model, 'compiler', angle='radian', autolimits='true')
    asset = add(model, 'asset')
    names = ['main_body_1'] + [f'{s}_{p}_{part}_1' for p in ('front','after') for s in ('left','right') for part in ('shoulder','thigh','shank')]
    for name in names:
        add(asset, 'mesh', name=name, file=f'meshes/{name}.stl')
    world = add(model, 'worldbody')
    trunk = add(world, 'body', name='trunk', pos='0 0 0')
    add(trunk, 'freejoint')
    # CAD base_link is a nonphysical marker. Omit its mesh and 1.4068 g mass.
    add(trunk, 'inertial', pos='0 0 0.024', mass='0.757692', diaginertia='0.0032 0.009 0.011')
    add(trunk, 'geom', name='trunk_collision', type='box', size='0.184 0.112 0.031', pos='0 0 0.0115', contype='1', conaffinity='1', rgba='0.7 0.7 0.7 0.25')
    add(trunk, 'geom', name='trunk_visual', type='mesh', mesh='main_body_1', contype='0', conaffinity='0', rgba='0.7 0.7 0.7 1')
    contact = add(model, 'contact')
    # The coarse trunk box fills the hip cutouts. Exclude contact there between
    # connected assembly parts.
    for leg, side, fore in [('FL','left','front'),('FR','right','front'),('RL','left','after'),('RR','right','after')]:
        sign = 1 if side == 'left' else -1
        shoulder_name = f'{side}_{fore}_shoulder_1'
        thigh_name = f'{side}_{fore}_thigh_1'
        shank_name = f'{side}_{fore}_shank_1'
        foot_name = f'{side}_{fore}_foot_link'
        hip_pos = source_point(source_poses, shoulder_name)
        thigh_pos = source_point(source_poses, thigh_name)
        knee_pos = source_point(source_poses, shank_name)
        _, foot_local = pose(source_links[foot_name].find('collision/origin'))
        foot_pos = source_point(source_poses, foot_name, foot_local)
        thigh_offset = thigh_pos - hip_pos
        thigh_to_knee = knee_pos - thigh_pos
        knee_to_foot = foot_pos - knee_pos
        upper = float(np.linalg.norm(thigh_to_knee))
        lower = float(np.linalg.norm(knee_to_foot[[0, 2]]))
        thigh_angle = float(np.arctan2(-thigh_to_knee[0], -thigh_to_knee[2]))
        calf_angle = float(np.arctan2(-knee_to_foot[0], -knee_to_foot[2]) - thigh_angle)
        thigh_pose = Rotation.from_euler('y', thigh_angle).as_matrix(), thigh_pos
        # The CAD left/right foot offsets differ by 0.7 mm. Use their mean to
        # preserve exact reflection for mirrored motions (0.35 mm per side).
        foot_local_target = (0.0, -sign * 0.027151, -lower)
        add(contact, 'exclude', body1='trunk', body2=f'{leg}_thigh')
        hip = add(trunk, 'body', name=f'{leg}_hip', pos=fmt(hip_pos))
        add(hip, 'joint', name=f'{leg}_hip_joint', type='hinge', axis='1 0 0', range='-0.8 0.8', damping='0.02', armature='0.01')
        add(hip, 'inertial', pos='0 0 -0.012', mass='0.1065874', diaginertia='0.000052 0.000052 0.000047')
        add(hip, 'geom', name=f'{leg}_hip_visual', type='mesh', mesh=f'{side}_{fore}_shoulder_1', contype='0', conaffinity='0')
        thigh = add(hip, 'body', name=f'{leg}_thigh', pos=fmt(thigh_offset))
        add(thigh, 'joint', name=f'{leg}_thigh_joint', type='hinge', axis='0 1 0', range='-1.5 2.5', damping='0.02', armature='0.01')
        add_full_inertial(thigh, *transform_inertial(source_links[thigh_name], source_poses[thigh_name], thigh_pose))
        add(thigh, 'geom', name=f'{leg}_thigh_visual', type='mesh', mesh=f'{side}_{fore}_thigh_1', contype='0', conaffinity='0')
        add_thigh_collisions(thigh, leg, sign)
        calf = add(thigh, 'body', name=f'{leg}_calf', pos=fmt((0,0,-upper)))
        add(calf, 'joint', name=f'{leg}_calf_joint', type='hinge', axis='0 1 0', range='-2.6 -0.05', damping='0.02', armature='0.01')
        add(calf, 'geom', name=f'{leg}_calf_visual', type='mesh', mesh=f'{side}_{fore}_shank_1', contype='0', conaffinity='0')
        add(calf, 'geom', name=f'{leg}_foot_collision', type='sphere', size='0.0135', pos=fmt(foot_local_target), friction='0.8 0.005 0.0005', condim='6')
        add(calf, 'site', name=leg, pos=fmt(foot_local_target), size='0.004')
        # Target body poses at the standing pose, in trunk coordinates.
        hip_pose = np.eye(3), hip_pos
        calf_pose = Rotation.from_euler('y', thigh_angle + calf_angle).as_matrix(), knee_pos
        add_full_inertial(calf, *merged_calf_inertial(source_links, source_poses, shank_name, foot_name, calf_pose))
        add_shank_collisions(calf, leg, source_links[shank_name], source_poses[shank_name], calf_pose)
        for part, target in [('shoulder', hip_pose), ('thigh', thigh_pose), ('shank', calf_pose)]:
            name = f'{side}_{fore}_{part}_1'
            write_mesh(name, source_links[name], source_poses[name], target)
    write_mesh('main_body_1', source_links['main_body_1'], source_poses['main_body_1'], (np.eye(3), np.zeros(3)))
    ET.indent(model, space='  ')
    (OUT / 'dog.xml').write_text(ET.tostring(model, encoding='unicode') + '\n')
    write_urdf(model)


def write_urdf(model):
    """Companion URDF with the same corrected frames, meshes, masses and joint limits."""
    robot = ET.Element('robot', name='dog')
    root = model.find('./worldbody/body')

    def write_trunk_collisions(link, geom):
        # URDF has no link-pair exclusions. Leave the four CAD hip openings
        # clear while keeping the trunk box's overall extents.
        hx, hy, hz = np.fromstring(geom.get('size'), sep=' ')
        x, y, z = np.fromstring(geom.get('pos'), sep=' ')
        cutout_x, cutout_y = 0.126, 0.046
        pieces = [('center', y, 2 * hx, 2 * cutout_y)]
        for name, sign in (('left', 1), ('right', -1)):
            pieces.append((name, y + sign * (cutout_y + hy) / 2, 2 * cutout_x, hy - cutout_y))
        for name, center_y, length, width in pieces:
            collision = add(link, 'collision', name=f'trunk_{name}_collision')
            add(collision, 'origin', xyz=fmt((x, center_y, z)), rpy='0 0 0')
            add(add(collision, 'geometry'), 'box', size=fmt((length, width, 2 * hz)))

    def visit(body, parent=None):
        name = body.get('name')
        link = add(robot, 'link', name=name)
        inertial = body.find('inertial')
        ins = add(link, 'inertial')
        add(ins, 'origin', xyz=inertial.get('pos'), rpy='0 0 0')
        add(ins, 'mass', value=inertial.get('mass'))
        if inertial.get('fullinertia'):
            xx, yy, zz, xy, xz, yz = inertial.get('fullinertia').split()
        else:
            xx, yy, zz = inertial.get('diaginertia').split()
            xy = xz = yz = '0'
        add(ins, 'inertia', ixx=xx, ixy=xy, ixz=xz, iyy=yy, iyz=yz, izz=zz)
        for geom in body.findall('geom'):
            if geom.get('type') == 'mesh':
                visual = add(link, 'visual')
                add(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
                geometry = add(visual, 'geometry')
                add(geometry, 'mesh', filename=f"meshes/{geom.get('mesh')}.stl")
            elif geom.get('name') == 'trunk_collision':
                write_trunk_collisions(link, geom)
            elif geom.get('name','').endswith('_collision'):
                collision = add(link, 'collision', name=geom.get('name'))
                if geom.get('type') in ('capsule', 'cylinder'):
                    points = np.fromstring(geom.get('fromto'), sep=' ').reshape(2,3)
                    center = points.mean(0)
                    length = float(np.linalg.norm(points[1]-points[0]))
                    direction = (points[1] - points[0]) / length
                    rotation, _ = Rotation.align_vectors([direction], [[0, 0, 1]])
                    add(collision, 'origin', xyz=fmt(center), rpy=fmt(rotation.as_euler('xyz')))
                    geometry = add(collision, 'geometry')
                    add(geometry, 'cylinder', radius=geom.get('size'), length=f'{length:.7g}')
                else:
                    add(collision, 'origin', xyz=geom.get('pos','0 0 0'), rpy='0 0 0')
                    geometry = add(collision, 'geometry')
                    if geom.get('type') == 'box':
                        add(geometry, 'box', size=fmt(2*np.fromstring(geom.get('size'),sep=' ')))
                    elif geom.get('type') == 'sphere':
                        add(geometry, 'sphere', radius=geom.get('size'))
        if parent is not None:
            mjjoint = body.find('joint')
            joint = add(robot, 'joint', name=mjjoint.get('name'), type='revolute')
            add(joint, 'parent', link=parent)
            add(joint, 'child', link=name)
            add(joint, 'origin', xyz=body.get('pos'), rpy='0 0 0')
            add(joint, 'axis', xyz=mjjoint.get('axis'))
            lo, hi = mjjoint.get('range').split()
            add(joint, 'limit', lower=lo, upper=hi, effort='2.2', velocity='4.717106086471161')
            add(joint, 'dynamics', damping=mjjoint.get('damping'), friction='0')
        for child in body.findall('body'):
            visit(child,name)
    visit(root)
    ET.indent(robot, space='  ')
    (OUT/'dog.urdf').write_text(ET.tostring(robot,encoding='unicode')+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE, help='source dog_description directory')
    SOURCE = parser.parse_args().source
    build()
