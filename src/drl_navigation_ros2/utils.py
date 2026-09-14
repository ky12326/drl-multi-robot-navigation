from dataclasses import dataclass
import numpy as np


@dataclass
class pos_data:
    name = None
    x = None
    y = None
    angle = None


def check_position(x, y, element_positions, min_dist):
    """
    源代码式位置检查：
    不再使用结构化区域，也不额外限制 wall_margin。
    """
    for element in element_positions:
        distance_vector = [element[0] - x, element[1] - y]
        distance = np.linalg.norm(distance_vector)
        if distance < min_dist:
            return False
    return True


def set_random_position(name, element_positions):
    angle = np.random.uniform(-np.pi, np.pi)

    for _ in range(300):
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        if check_position(x, y, element_positions, 1.8):
            element_positions.append([x, y])

            eval_element = pos_data()
            eval_element.name = name
            eval_element.x = x
            eval_element.y = y
            eval_element.angle = angle
            return eval_element

    # 兜底，避免死循环
    eval_element = pos_data()
    eval_element.name = name
    eval_element.x = 0.0
    eval_element.y = 0.0
    eval_element.angle = angle
    return eval_element


def record_eval_positions(n_eval_scenarios=10):
    """
    回退到源代码式评估：
    - 4 个固定障碍物旧布局
    - 4 个随机障碍物 obstacle5~8
    - 机器人全图随机
    - 目标全图随机（通过 set_random_position 间接保证与已有元素最小间距）
    """
    scenarios = []

    for _ in range(n_eval_scenarios):
        eval_scenario = []
        element_positions = [
            [-2.93, 3.17],
            [2.86, -3.00],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]

        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            eval_element = set_random_position(name, element_positions)
            eval_scenario.append(eval_element)

        eval_element = set_random_position("turtlebot3_waffle", element_positions)
        eval_scenario.append(eval_element)

        eval_element = set_random_position("target", element_positions)
        eval_scenario.append(eval_element)

        scenarios.append(eval_scenario)

    return scenarios
