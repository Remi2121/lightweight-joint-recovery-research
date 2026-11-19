"""
Skeleton graph construction for the 25-joint Kinect v2 layout
(shared by NTU RGB+D and PKU-MMD).

Implements Yan et al.'s spatial configuration partitioning (K=3):
label 0 = root joint itself, label 1 = "centripetal" neighbours
(closer to the skeleton's gravity centre than the root), label 2 =
"centrifugal" neighbours (farther than the root).
"""

import numpy as np

NUM_JOINTS = 25

# NTU RGB+D / Kinect v2 bone connectivity (0-indexed), standard layout
# used across ST-GCN / 2s-AGCN / MS-AAGCN reference implementations.
BONES = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
    (23, 24), (24, 11),
]

CENTER_JOINT = 20  # "spine shoulder" / mid-spine, used as the skeleton's root reference


class Graph:
    """Builds the K=3 spatial-configuration-partitioned adjacency set A: (K, V, V)."""

    def __init__(self, num_joints=NUM_JOINTS, bones=None, center=CENTER_JOINT, max_hop=1):
        self.num_joints = num_joints
        self.bones = bones or BONES
        self.center = center
        self.max_hop = max_hop
        self.hop_dis = self._build_hop_distance()
        self.A = self._build_spatial_configuration_partition()

    def _build_hop_distance(self):
        A = np.zeros((self.num_joints, self.num_joints))
        for i, j in self.bones:
            A[i, j] = 1
            A[j, i] = 1
        hop_dis = np.full((self.num_joints, self.num_joints), np.inf)
        transfer = [np.eye(self.num_joints)]
        cur = np.eye(self.num_joints)
        reach = (A > 0).astype(int)
        for d in range(1, self.max_hop + 1):
            cur = cur @ reach
            transfer.append((cur > 0).astype(int))
        arrive = sum(transfer)
        hop_dis[arrive > 0] = 0
        for d in range(self.max_hop, 0, -1):
            hop_dis[transfer[d] > 0] = d
        np.fill_diagonal(hop_dis, 0)
        return hop_dis

    def _normalize(self, A):
        Dl = np.sum(A, axis=0)
        Dn = np.zeros_like(A)
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return A @ Dn

    def _build_spatial_configuration_partition(self):
        valid_hop = range(0, self.max_hop + 1)
        adjacency = np.zeros((self.num_joints, self.num_joints))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self._normalize(adjacency)

        # gravity centre = mean position is runtime-dependent; for the
        # *graph structure* we instead use hop-distance-to-center (a static
        # proxy used by several open ST-GCN implementations when a fixed,
        # data-independent partition is desired).
        center_hop = self.hop_dis[:, self.center]

        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_joints, self.num_joints))
            a_close = np.zeros((self.num_joints, self.num_joints))
            a_far = np.zeros((self.num_joints, self.num_joints))
            for i in range(self.num_joints):
                for j in range(self.num_joints):
                    if self.hop_dis[j, i] == hop:
                        if center_hop[j] == center_hop[i]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif center_hop[j] < center_hop[i]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_far[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_close)
                A.append(a_far)
        return np.stack(A, axis=0)  # (K, V, V), K=3 for max_hop=1
