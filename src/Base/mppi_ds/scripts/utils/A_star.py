import torch
import heapq


class Node:
    '''_summary_
    定义一个节点类，表示一个搜索节点
    '''

    def __init__(self, x, y, g, h, parent=None):
        self.x = x
        self.y = y
        self.g = g  # 从起点到当前节点的实际代价
        self.h = h  # 从当前节点到目标节点的估算代价
        self.f = g + h  # 总代价
        self.parent = parent  # 父节点，用于回溯路径

    def __lt__(self, other):
        # 优先队列的比较函数，按 f 值升序排序
        return self.f < other.f


class A_star:
    def __init__(self, q_0, q_f, obs):
        self.q_0 = q_0.tolist()
        self.q_f = q_f.tolist()
        self.obs = obs

        self.resolution = 0.1

    def update_state(self, obs):
        self.obs = obs

    def is_colliding(self, x, y):
        position = torch.tensor([x, y], dtype=torch.float32)
        dist = torch.norm(position - self.obs[:, :2], dim=1) - self.obs[:, 3]
        return torch.any(dist < 0)

    def distance(self, x, y):
        return abs(x - self.q_f[0]) + abs(y - self.q_f[1])

    def plan(self):
        # 创建起始节点
        open_set = []
        start_node = Node(self.q_0[0], self.q_0[1], 0, self.distance(self.q_0[0], self.q_0[1]))
        heapq.heappush(open_set, (start_node.f, start_node))

        # 已扩展的节点集合
        closed_set = set()

        while open_set:
            # 获取 f 值最小的节点
            _, current_node = heapq.heappop(open_set)

            # 如果到达目标点，构建路径
            if self.distance(current_node.x, current_node.y) < 2 * self.resolution:
                path = []
                while current_node:
                    path.append(torch.tensor([current_node.x, current_node.y], dtype=torch.float32))
                    current_node = current_node.parent
                return path[::-1]  # 反转路径，得到从起点到目标点的路径

            # 将当前节点加入已扩展集合
            closed_set.add((current_node.x, current_node.y))

            # 扩展邻居节点
            for dx, dy in [(-self.resolution, 0), (self.resolution, 0), (0, -self.resolution), (0, self.resolution)]:
                nx = current_node.x + dx
                ny = current_node.y + dy

                # 如果邻居在障碍物上或已扩展过，则跳过
                if self.is_colliding(nx, ny) or (nx, ny) in closed_set:
                    continue

                g = current_node.g + self.resolution  # 每次移动的代价为 1
                h = self.distance(nx, ny)  # 启发式函数，欧几里得距离
                neighbor_node = Node(nx, ny, g, h, current_node)
                if any(node[1].x == nx and node[1].y == ny for node in open_set):
                    for i, (_, node) in enumerate(open_set):
                        if node.x == nx and node.y == ny:
                            if g < node.g:
                                open_set[i] = (neighbor_node.f, neighbor_node)
                                heapq.heapify(open_set)
                else:
                    # 将邻居节点加入开放列表
                    heapq.heappush(open_set, (neighbor_node.f, neighbor_node))

        return None  # 如果没有找到路径
