import torch
import heapq


class PRM:
    def __init__(self, points_num, kin, nn_model, device, dtype):
        self.params = {"device": device, "dtype": dtype}
        self.kin = kin
        self.nn_model = nn_model
        self.points_num = points_num
        self.min_distance = 0.1
        self.edgemap = [(p, i) for p in range(self.points_num - 1) for i in range(p + 1, self.points_num)]
        self.edgemap = torch.tensor(self.edgemap, dtype=torch.int32, device=device)

    def costmap_compute(self, path):
        costmap = torch.cdist(path, path, p=2)

        segment_lengths = costmap[self.edgemap[:, 0], self.edgemap[:, 1]]
        num_segments = torch.clamp_min((segment_lengths / self.min_distance)
                                       .ceil().log2().floor().exp2().to(torch.int32), 1,)
        num_points = num_segments + 1
        index_offsets = torch.cumsum(
            torch.tensor([0] + num_points.tolist(), **self.params), dim=0
        )

        segment_starts = index_offsets[:-1]
        segment_ends = index_offsets[1:]

        max_num_segments = num_segments.max().item()
        edge_num = num_segments.size(0)
        t = torch.linspace(0, 1, steps=max_num_segments + 1, **self.params).view(
            1, -1, 1
        )

        A = path[self.edgemap[:, 0]]
        B = path[self.edgemap[:, 1]]
        expanded_A = A.unsqueeze(1).expand(-1, max_num_segments + 1, -1)
        expanded_B = B.unsqueeze(1).expand(-1, max_num_segments + 1, -1)

        indices = (
            torch.arange(max_num_segments + 1, **self.params)
            .unsqueeze(0)
            .expand(edge_num, max_num_segments + 1)
        )  # Shape (n, m) tensor with row-wise 0 to m-1

        B_expanded = (
            (max_num_segments / num_segments)
            .unsqueeze(-1)
            .expand(edge_num, max_num_segments + 1)
        )  # Shape (n, m) tensor where each row is B

        # Create a mask where True values correspond to positions that are multiples of B
        mask = (indices % B_expanded) == 0

        points = expanded_A + t * (expanded_B - expanded_A)
        points = points[mask]

        self.kin.forward_kinematics_batch(points)
        min_distances, _, _ = self.nn_model.get_distances_batch(
            points, self.nn_model.obs
        )
        distance_min = min_distances.min(dim=1, keepdim=True)[0]

        cost_min = 100000 * torch.square(torch.square(distance_min.abs()))
        cost_min[distance_min > 0] = 0

        segment_ranges = (
            torch.arange(cost_min.size(0), **self.params)
            .unsqueeze(0)
            .expand(segment_starts.size(0), -1)
        )

        mask = (segment_ranges >= segment_starts.unsqueeze(1)) & (
            segment_ranges < segment_ends.unsqueeze(1)
        )

        cost_increments = torch.sum(
            cost_min.squeeze(-1).unsqueeze(0) * mask.float(), dim=1
        )

        # Update costmap
        i_indices = self.edgemap[:, 0]
        j_indices = self.edgemap[:, 1]

        costmap[i_indices, j_indices] += cost_increments
        costmap[j_indices, i_indices] = costmap[i_indices, j_indices]

        return costmap

    def dijkstra_planning(self, costmap):
        openlist = [(0, 0)]
        camefrom = {}
        g_score = {0: 0}
        closedlist = set()

        while openlist:
            _, current = heapq.heappop(openlist)
            closedlist.add(current)

            if current == self.points_num - 1:
                path = []
                while current in camefrom:
                    path.append(current)
                    current = camefrom[current]
                path.append(current)
                path.reverse()
                return path, g_score[self.points_num - 1]

            for neighbor in range(costmap.size(0)):
                if neighbor in closedlist:
                    continue

                tentative_g = g_score[current] + costmap[current, neighbor].item()

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    camefrom[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(openlist, (tentative_g, neighbor))

        return [], 0

    def prm_planning(self, path: torch.Tensor):
        '''_summary_

        Args:
            path (torch.Tensor): n x 2 tensor, the path to plan

        Returns:
            torch.Tensor, float: path and cost
        '''
        costmap = self.costmap_compute(path)
        reconnect_path_indices, path_cost = self.dijkstra_planning(costmap)
        reconnect_path = path[reconnect_path_indices]
        return reconnect_path, path_cost
