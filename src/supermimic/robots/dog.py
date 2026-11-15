"""Corrected CAD dog model and training settings.

Physical values and motor data: assets/dog/SOURCE.md. The model uses the
continuous torque limit, since mjlab's position actuator has no motor-speed curve.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

from supermimic.robots.rover import ROVER_SPEC
from supermimic.robots.spec import DomainRandomization, ResetNoise, StateScales

XML = Path(__file__).resolve().parent / 'assets' / 'dog' / 'dog.xml'
LEGS = ('FL', 'FR', 'RL', 'RR')
JOINT_NAMES = tuple(f'{leg}_{part}_joint' for leg in LEGS for part in ('hip','thigh','calf'))
LEG_HOME = {
    'FL': (0.5292036732051035, -1.481744357468247),
    'FR': (0.5292036732051035, -1.481744357468247),
    'RL': (0.8292036732051032, -1.5817443574682473),
    'RR': (0.8707963267948965, -1.62333701105804),
}
HOME = {f'{leg}_{part}_joint': angle for leg in LEGS for part, angle in (('hip', 0.0), ('thigh', LEG_HOME[leg][0]), ('calf', LEG_HOME[leg][1]))}
STAND_Z = 0.2235682548163149
MIRROR_PERM = tuple(3*(i^1)+k for i in range(4) for k in range(3))
MIRROR_SIGN = (-1.0,1.0,1.0)*4


def get_spec():
    import mujoco
    return mujoco.MjSpec.from_file(str(XML))


def get_entity_cfg(domain_randomization: bool = True, stiffness: float = 40.0):
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    del domain_randomization
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,STAND_Z+0.01),joint_pos=dict(HOME),joint_vel={'.*':0.0}),
        collisions=(CollisionCfg(geom_names_expr=(r'.*_collision',),contype=1,conaffinity=1,
                   solref=(0.01,1),condim={r'.*_foot_collision':6,r'.*':1},
                   priority={r'.*_foot_collision':1,r'.*':0},
                   friction={r'.*_foot_collision':(0.8,0.005,0.0005)}),),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(BuiltinPositionActuatorCfg(target_names_expr=(r'.*_joint',),
                stiffness=stiffness,damping=stiffness/40.0,effort_limit=2.2),),
            soft_joint_pos_limit_factor=1.0),
    )


def dog_spec(stiffness: float | None = None):
    base = ROVER_SPEC
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,model.nbody)]
    swap = {'FL_':'FR_','FR_':'FL_','RL_':'RR_','RR_':'RL_'}
    body_map = {n:next((n.replace(a,b) for a,b in swap.items() if n.startswith(a)),n) for n in names}
    kp = 40.0 if stiffness is None else float(stiffness)
    return replace(base,
        name='dog',xml_path=XML,joint_names=JOINT_NAMES,root_body='trunk',anchor_body='trunk',
        tracked_bodies=('trunk',)+tuple(f'{leg}_{part}' for part in ('thigh','calf') for leg in LEGS),
        foot_geom_regex=r'^(FL|FR|RL|RR)_foot_collision$',com_body='trunk',
        entity_cfg_fn=partial(get_entity_cfg,stiffness=kp),action_scale=0.25,
        root_height_threshold=0.085,stand_height=STAND_Z,viewer_distance=1.0,
        state_scales=StateScales(
            err_scale=(0.01,0.05,0.1,1.0,0.1,3.0),
            out_scale=(0.05,1.0,3.0,0.001,0.005,0.02),
            in_scale=(0.2,1.0,0.5,1.0,1.0,2.5),
            xi_friction=(0.75,0.45),xi_bias=(0.0,0.01),xi_com=(0.0,0.015)),
        dr=DomainRandomization(friction_range=(0.3,1.2),encoder_bias=0.01,
            com_offset=((-0.015,0.015),(-0.01,0.01),(-0.01,0.01)),
            action_delay_substeps=(0,0),eval_friction=0.8),
        reset_noise=ResetNoise(pose_range={'x':(-0.015,0.015),'y':(-0.015,0.015),
            'z':(-0.003,0.003),'roll':(-0.1,0.1),'pitch':(-0.1,0.1),'yaw':(-0.2,0.2)},
            velocity_range={'x':(-0.25,0.25),'y':(-0.25,0.25),'z':(-0.1,0.1),
            'roll':(-0.52,0.52),'pitch':(-0.52,0.52),'yaw':(-0.78,0.78)},
            joint_position_range=(-0.1,0.1)),
        reward_std_body_pos=0.08,reward_std_root_pos=0.1,
        lie_height=(0.04,0.11),push_speed=(1.4,2.1),push_down_speed=(0.5,1.0),
        mirror_joint_perm=MIRROR_PERM,mirror_joint_sign=MIRROR_SIGN,mirror_body_map=body_map)
