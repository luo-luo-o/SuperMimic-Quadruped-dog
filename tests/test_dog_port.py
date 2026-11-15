"""Check the CAD correction and conservative actuator settings stay consistent."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quadmotion.robot import load_robot
from supermimic.robots.dog import XML, dog_spec


def test_dog_physics_and_urdf_agree():
    model = mujoco.MjModel.from_xml_path(str(XML))
    urdf = ET.parse(XML.with_suffix('.urdf')).getroot()
    masses = [float(link.find('inertial/mass').get('value')) for link in urdf.findall('link')]
    assert len(masses) == 13
    assert sum(masses) == pytest.approx(model.body_mass.sum())
    assert sum(masses) == pytest.approx(2.0142088)
    assert np.all(model.body_inertia[1:] > 0)
    assert np.all(np.isclose(model.dof_armature[6:], 0.01))
    assert len(urdf.findall('joint')) == 12
    assert all(float(j.find('limit').get('effort')) == 2.2 for j in urdf.findall('joint'))


def test_curved_calf_center_of_mass_uses_cad_offset():
    model = mujoco.MjModel.from_xml_path(str(XML))
    urdf = ET.parse(XML.with_suffix('.urdf')).getroot()
    for leg in ('FL', 'FR', 'RL', 'RR'):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_calf')
        com = model.body_ipos[bid]
        urdf_link = urdf.find(f".//link[@name='{leg}_calf']")
        urdf_com = np.fromstring(urdf_link.find('inertial/origin').get('xyz'), sep=' ')
        assert np.allclose(com, urdf_com, atol=1e-6)
        assert 0.005 < com[0] < 0.025
        assert 0.026 < abs(com[1]) < 0.028
        assert -0.065 < com[2] < -0.05
        assert float(urdf_link.find('inertial/inertia').get('ixz')) != 0


def test_thigh_cad_inertia_and_fork_collisions_match_in_both_formats():
    for path in (XML, XML.with_suffix('.urdf')):
        model = mujoco.MjModel.from_xml_path(str(path))
        for leg in ('FL', 'FR', 'RL', 'RR'):
            side = 1 if leg.endswith('L') else -1
            thigh = model.body(f'{leg}_thigh')
            np.testing.assert_allclose(thigh.ipos, [0.000046, -side * 0.02727, -0.06005], atol=5e-6)
            axes = Rotation.from_quat(thigh.iquat, scalar_first=True).as_matrix()
            inertia = axes @ np.diag(thigh.inertia) @ axes.T
            np.testing.assert_allclose(inertia, np.diag([0.000171, 0.000117, 0.000061]), atol=1e-9)

            for plate, y in (('outer', 0.0038), ('inner', -0.058)):
                geom = model.geom(f'{leg}_thigh_{plate}_collision')
                assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
                np.testing.assert_allclose(geom.pos, [0, side * y, -0.060], atol=1e-7)
                np.testing.assert_allclose(geom.size, [0.012, 0.004, 0.070], atol=1e-7)

            brace = model.geom(f'{leg}_thigh_brace_collision')
            assert brace.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
            np.testing.assert_allclose(brace.pos, [0, -side * 0.0271, -0.060], atol=1e-7)
            np.testing.assert_allclose(brace.size[:2], [0.004, 0.0309], atol=1e-7)
            axis = Rotation.from_quat(brace.quat, scalar_first=True).apply([0, 0, 1])
            np.testing.assert_allclose(np.abs(axis), [0, 1, 0], atol=1e-6)


def test_dog_stance_and_actuation():
    quad = load_robot('dog')
    assert quad.nominal_root_height() == pytest.approx(0.2235682548163149)
    spec = dog_spec()
    assert spec.joint_names == quad.joint_names
    assert spec.stand_height == pytest.approx(quad.nominal_root_height())
    actuator = spec.entity_cfg_fn(False).articulation.actuators[0]
    assert actuator.effort_limit == 2.2
    assert actuator.stiffness == 40.0 and actuator.damping == 1.0
    assert dog_spec(stiffness=80).entity_cfg_fn(False).articulation.actuators[0].stiffness == 80


def test_mesh_feet_match_collision_feet_at_home():
    from supermimic.robots.dog import HOME, JOINT_NAMES, STAND_Z

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, STAND_Z, 1, 0, 0, 0]
    data.qpos[7:] = [HOME[name] for name in JOINT_NAMES]
    mujoco.mj_forward(model, data)
    for leg in ('FL', 'FR', 'RL', 'RR'):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{leg}_calf_visual')
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        mesh = model.mesh_vert[start:start + count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        bottom = mesh[mesh[:, 2] < np.quantile(mesh[:, 2], 0.01)].mean(axis=0)
        foot = data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{leg}_foot_collision')]
        assert np.linalg.norm(bottom[:2] - foot[:2]) < 0.01, leg
        assert abs(bottom[2] - (foot[2] - 0.0135)) < 0.01, leg


def test_joint_centers_match_cad_assembly_at_home():
    from supermimic.robots.dog import HOME, JOINT_NAMES, STAND_Z

    expected = {
        'FL': ((.175033, .095, -.002658), (.158323, .11171, -.025748), (.09774147, .11171, -.12933312)),
        'FR': ((.175033, -.095, -.002658), (.158323, -.11171, -.025748), (.09774147, -.11171, -.12933312)),
        'RL': ((-.174967, .095, -.002658), (-.158257, .11171, -.025748), (-.24674425, .11171, -.10680358)),
        'RR': ((-.174967, -.095, -.002658), (-.158257, -.11171, -.025748), (-.25003806, -.11171, -.10305412)),
    }
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, STAND_Z, 1, 0, 0, 0]
    data.qpos[7:] = [HOME[name] for name in JOINT_NAMES]
    mujoco.mj_forward(model, data)
    for leg, points in expected.items():
        for part, point in zip(('hip', 'thigh', 'calf'), points):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_{part}')
            assert np.allclose(data.xpos[bid] - [0, 0, STAND_Z], point, atol=2e-6)


@pytest.mark.parametrize('at_home', [False, True])
def test_urdf_and_mjcf_joint_centers_match(at_home):
    from supermimic.robots.dog import HOME, JOINT_NAMES, STAND_Z

    mjcf = mujoco.MjModel.from_xml_path(str(XML))
    mjdata = mujoco.MjData(mjcf)
    mjdata.qpos[:7] = [0, 0, STAND_Z, 1, 0, 0, 0]
    mjdata.qpos[7:] = [HOME[name] for name in JOINT_NAMES] if at_home else 0
    mujoco.mj_forward(mjcf, mjdata)
    urdf = mujoco.MjModel.from_xml_path(str(XML.with_suffix('.urdf')))
    urdata = mujoco.MjData(urdf)
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(urdf, mujoco.mjtObj.mjOBJ_JOINT, name)
        urdata.qpos[urdf.jnt_qposadr[jid]] = HOME[name] if at_home else 0
    mujoco.mj_forward(urdf, urdata)
    for leg in ('FL', 'FR', 'RL', 'RR'):
        for part in ('hip', 'thigh', 'calf'):
            name = f'{leg}_{part}'
            mjid = mujoco.mj_name2id(mjcf, mujoco.mjtObj.mjOBJ_BODY, name)
            urid = mujoco.mj_name2id(urdf, mujoco.mjtObj.mjOBJ_BODY, name)
            assert np.allclose(mjdata.xpos[mjid] - [0, 0, STAND_Z], urdata.xpos[urid], atol=1e-8)


@pytest.mark.parametrize('suffix', ['.xml', '.urdf'])
def test_dog_home_has_no_self_collision(suffix):
    from supermimic.robots.dog import HOME, JOINT_NAMES, STAND_Z

    model = mujoco.MjModel.from_xml_path(str(XML.with_suffix(suffix)))
    data = mujoco.MjData(model)
    if suffix == '.xml':
        data.qpos[:7] = [0, 0, STAND_Z, 1, 0, 0, 0]
    for name in JOINT_NAMES:
        data.qpos[model.joint(name).qposadr[0]] = HOME[name]
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_urdf_shank_collision_axes_match_mjcf():
    mjcf = ET.parse(XML).getroot()
    urdf = ET.parse(XML.with_suffix('.urdf')).getroot()
    for leg in ('FL', 'FR', 'RL', 'RR'):
        body = mjcf.find(f".//body[@name='{leg}_calf']")
        link = urdf.find(f".//link[@name='{leg}_calf']")
        for i in (0, 1):
            name = f'{leg}_calf{i}_collision'
            geom = body.find(f"geom[@name='{name}']")
            collision = link.find(f"collision[@name='{name}']")
            endpoints = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
            center = endpoints.mean(axis=0)
            direction = endpoints[1] - endpoints[0]
            origin = collision.find('origin')
            urdf_center = np.fromstring(origin.get('xyz'), sep=' ')
            urdf_axis = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).apply([0, 0, 1])
            assert np.allclose(urdf_center, center, atol=1e-6)
            assert np.allclose(urdf_axis, direction / np.linalg.norm(direction), atol=1e-6)
