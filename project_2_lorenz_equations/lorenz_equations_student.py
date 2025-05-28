#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程参考答案
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def lorenz_system(state: np.ndarray, sigma: float, r: float, b: float) -> np.ndarray:
    x, y, z = state
    return np.array([
        sigma * (y - x),
        r * x - y - x * z,
        x * y - b * z
    ])


def solve_lorenz_equations(sigma: float=10.0, r: float=28.0, b: float=8/3,
                          x0: float=0.1, y0: float=0.1, z0: float=0.1,
                          t_span: tuple[float, float]=(0, 50), dt: float=0.01):
    """
    求解洛伦兹方程
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lambda t, state: lorenz_system(state, sigma, r, b), 
                   t_span, [x0, y0, z0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y


def plot_lorenz_attractor(t: np.ndarray, y: np.ndarray):
    """
    绘制洛伦兹吸引子
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0], y[1], y[2], lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz Attractor')
    plt.show()


def compare_initial_conditions(ic1: tuple[float, float, float], 
                              ic2: tuple[float, float, float], 
                              t_span: tuple[float, float]=(0, 50), dt: float=0.01):
    """
    比较不同初始条件的解
    """
    t1, y1 = solve_lorenz_equations(x0=ic1[0], y0=ic1[1], z0=ic1[2], t_span=t_span, dt=dt)
    t2, y2 = solve_lorenz_equations(x0=ic2[0], y0=ic2[1], z0=ic2[2], t_span=t_span, dt=dt)
    
    # 计算轨迹距离
    distance = np.sqrt((y1[0]-y2[0])**2 + (y1[1]-y2[1])**2 + (y1[2]-y2[2])**2)
    
    # 绘制比较图
    plt.figure(figsize=(12, 6))
    plt.plot(t1, y1[0], label=f'IC1: {ic1}')
    plt.plot(t2, y2[0], label=f'IC2: {ic2}')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.title('Comparison of X(t) with Different Initial Conditions')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t1, distance, label='Distance between trajectories')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.title('Distance between Trajectories over Time')
    plt.legend()
    plt.show()


def main():
    """
    主函数，执行所有任务
    """
    # 任务A: 求解洛伦兹方程
    t, y = solve_lorenz_equations()
    
    # 任务B: 绘制洛伦兹吸引子
    plot_lorenz_attractor(t, y)
    
    # 任务C: 比较不同初始条件
    ic1 = (0.1, 0.1, 0.1)
    ic2 = (0.10001, 0.1, 0.1)  # 微小变化
    compare_initial_conditions(ic1, ic2)


if __name__ == '__main__':
    main()
