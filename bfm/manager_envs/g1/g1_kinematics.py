from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ETree

import numpy as np
import torch
from easydict import EasyDict
from scipy.ndimage import gaussian_filter1d

from bfm.utils.torch_utils import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quat_angle_axis,
    quat_identity_like,
    quat_inverse,
    quat_mul_norm,
    quaternion_to_matrix,
    wxyz_to_xyzw,
)

from .spec import BFMZeroG1Spec


@dataclass(frozen=True)
class BFMZeroMJCFKinematicTree:
    body_names: tuple[str, ...]
    parent_indices: torch.Tensor
    local_translation: torch.Tensor
    local_rotation_wxyz: torch.Tensor
    joint_ranges: torch.Tensor
    body_to_joint: OrderedDict[str, str]
    motor_names: tuple[str, ...]


def _float_array(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=np.float32)
    return np.fromstring(text, dtype=np.float32, sep=" ")


def parse_g1_mjcf_kinematic_tree(path: str | Path) -> BFMZeroMJCFKinematicTree:
    """Parse only the G1 kinematic tree needed for BFM-Zero motion FK."""

    xml_doc_root = ETree.parse(path).getroot()
    xml_world_body = xml_doc_root.find("worldbody")
    if xml_world_body is None:
        raise ValueError(f"MJCF has no worldbody: {path}")
    xml_body_root = xml_world_body.find("body")
    if xml_body_root is None:
        raise ValueError(f"MJCF has no root body: {path}")

    actuator = xml_doc_root.find("actuator")
    motor_names = tuple(motor.attrib["name"] for motor in actuator.findall("motor")) if actuator is not None else ()

    node_names: list[str] = []
    parent_indices: list[int] = []
    local_translation: list[np.ndarray] = []
    local_rotation: list[np.ndarray] = []
    joint_ranges: list[np.ndarray] = []
    body_to_joint: OrderedDict[str, str] = OrderedDict()

    def add_xml_node(xml_node, parent_index: int, node_index: int) -> int:
        node_name = xml_node.attrib.get("name")
        if node_name is None:
            raise ValueError("G1 MJCF body is missing a name")
        node_names.append(node_name)
        parent_indices.append(parent_index)
        local_translation.append(_float_array(xml_node.attrib.get("pos"), (0.0, 0.0, 0.0)))
        local_rotation.append(_float_array(xml_node.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)))

        curr_index = node_index
        node_index += 1
        joints = xml_node.findall("joint")
        for joint in joints:
            if joint.attrib.get("type") == "free":
                continue
            if joint.attrib.get("range") is None:
                joint_ranges.append(np.array([-np.pi, np.pi], dtype=np.float32))
            else:
                joint_ranges.append(_float_array(joint.attrib.get("range"), (-np.pi, np.pi)))
            body_to_joint[node_name] = joint.attrib.get("name", "")

        for next_node in xml_node.findall("body"):
            node_index = add_xml_node(next_node, curr_index, node_index)
        return node_index

    add_xml_node(xml_body_root, -1, 0)
    if len(joint_ranges) != len(motor_names):
        raise AssertionError(f"MJCF joint range count {len(joint_ranges)} != motor count {len(motor_names)}")

    return BFMZeroMJCFKinematicTree(
        body_names=tuple(node_names),
        parent_indices=torch.tensor(parent_indices, dtype=torch.long),
        local_translation=torch.tensor(np.array(local_translation), dtype=torch.float32),
        local_rotation_wxyz=torch.tensor(np.array(local_rotation), dtype=torch.float32),
        joint_ranges=torch.tensor(np.array(joint_ranges), dtype=torch.float32),
        body_to_joint=body_to_joint,
        motor_names=motor_names,
    )


class BFMZeroG1Kinematics:
    """Minimal G1 FK used by the manager light motion library.

    This intentionally only preserves the FK semantics used by the BFM-Zero pkl
    motions. Visual assets, shape metadata, and directory-based motion configs
    stay out of the manager/light path.
    """

    def __init__(self, spec: BFMZeroG1Spec, device: torch.device | str = torch.device("cpu")):
        self.spec = spec
        self.device = torch.device(device)
        tree = parse_g1_mjcf_kinematic_tree(spec.mjcf_path)
        if tree.body_names != spec.motion_body_names:
            raise AssertionError(f"Motion body order mismatch: {tree.body_names} != {spec.motion_body_names}")
        if tree.motor_names != spec.dof_names:
            raise AssertionError(f"MJCF motor order mismatch: {tree.motor_names} != {spec.dof_names}")

        self.body_names = list(tree.body_names)
        self.body_names_augment = list(tree.body_names)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(tree.motor_names)
        self.joints_range = tree.joint_ranges.to(self.device)
        self._parents = tree.parent_indices.to(self.device)
        self._offsets = tree.local_translation.reshape(1, -1, 3).to(self.device)
        self._local_rotation = tree.local_rotation_wxyz.reshape(1, -1, 4).to(self.device)

        for body_name, parent_name, pos, rot in zip(
            spec.extend_body_names,
            spec.extend_parent_names,
            spec.extend_pos,
            spec.extend_rot_wxyz,
            strict=True,
        ):
            self.body_names_augment.append(body_name)
            self._parents = torch.cat(
                [self._parents, torch.tensor([self.body_names.index(parent_name)], dtype=torch.long, device=self.device)]
            )
            self._offsets = torch.cat(
                [self._offsets, torch.tensor([[pos]], dtype=torch.float32, device=self.device)],
                dim=1,
            )
            self._local_rotation = torch.cat(
                [self._local_rotation, torch.tensor([[rot]], dtype=torch.float32, device=self.device)],
                dim=1,
            )

        self.num_bodies_augment = len(self.body_names_augment)
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float()

    def fk_batch(self, pose: torch.Tensor, trans: torch.Tensor, *, return_full: bool = False, dt: float = 1 / 30):
        device, dtype = pose.device, pose.dtype
        batch_size, seq_len = pose.shape[:2]
        pose = pose[..., : self.num_bodies, :]
        if pose.shape[2] < self.num_bodies_augment:
            pad = torch.zeros(
                batch_size,
                seq_len,
                self.num_bodies_augment - pose.shape[2],
                pose.shape[3],
                dtype=dtype,
                device=device,
            )
            pose = torch.cat([pose, pad], dim=2)

        pose_quat = axis_angle_to_quaternion(pose.clone())
        pose_mat = quaternion_to_matrix(pose_quat).reshape(batch_size, seq_len, -1, 3, 3)
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))

        result = EasyDict()
        if self.num_bodies_augment > self.num_bodies:
            if return_full:
                result.global_velocity_extend = self._compute_velocity(wbody_pos, dt)
                result.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
            result.global_translation_extend = wbody_pos.clone()
            result.global_rotation_mat_extend = wbody_mat.clone()
            result.global_rotation_extend = wbody_rot
            wbody_pos = wbody_pos[..., : self.num_bodies, :]
            wbody_mat = wbody_mat[..., : self.num_bodies, :, :]
            wbody_rot = wbody_rot[..., : self.num_bodies, :]

        result.global_translation = wbody_pos
        result.global_rotation_mat = wbody_mat
        result.global_rotation = wbody_rot
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            result.local_rotation = wxyz_to_xyzw(pose_quat)
            result.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            result.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            result.global_angular_velocity = rigidbody_angular_velocity
            result.global_velocity = rigidbody_linear_velocity
            result.dof_pos = pose.sum(dim=-1)[..., 1 : self.num_bodies]
            dof_vel = (result.dof_pos[:, 1:] - result.dof_pos[:, :-1]) / dt
            result.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
            result.fps = int(1 / dt)
        return result

    def forward_kinematics_batch(
        self,
        rotations: torch.Tensor,
        root_rotations: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = root_rotations.device, root_rotations.dtype
        batch_size, seq_len = rotations.shape[:2]
        num_bodies = self._offsets.shape[1]
        expanded_offsets = self._offsets[:, None].expand(batch_size, seq_len, num_bodies, 3).to(device).type(dtype)
        local_rotation_mat = self._local_rotation_mat.to(device).type(dtype)
        parents = self._parents.to(device)

        positions_world = []
        rotations_world = []
        for body_i in range(num_bodies):
            parent_i = int(parents[body_i].item())
            if parent_i == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
                continue
            jpos = (
                torch.matmul(rotations_world[parent_i][:, :, 0], expanded_offsets[:, :, body_i, :, None]).squeeze(-1)
                + positions_world[parent_i]
            )
            rot_mat = torch.matmul(
                rotations_world[parent_i],
                torch.matmul(local_rotation_mat[:, body_i : body_i + 1], rotations[:, :, body_i - 1 : body_i]),
            )
            positions_world.append(jpos)
            rotations_world.append(rot_mat)

        return torch.stack(positions_world, dim=2), torch.cat(rotations_world, dim=2)

    @staticmethod
    def _compute_velocity(p: torch.Tensor, time_delta: float, gaussian_filter: bool = True) -> torch.Tensor:
        velocity = np.gradient(p.detach().cpu().numpy(), axis=-3) / time_delta
        if gaussian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
        return torch.from_numpy(velocity).to(p)

    @staticmethod
    def _compute_angular_velocity(r: torch.Tensor, time_delta: float, gaussian_filter: bool = True) -> torch.Tensor:
        diff_quat_data = quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = quat_mul_norm(
            r[..., 1:, :, :],
            quat_inverse(r[..., :-1, :, :], w_last=True),
            w_last=True,
        )
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data, w_last=True)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if gaussian_filter:
            angular_velocity = torch.from_numpy(
                gaussian_filter1d(angular_velocity.detach().cpu().numpy(), 2, axis=-3, mode="nearest")
            ).to(r)
        return angular_velocity
