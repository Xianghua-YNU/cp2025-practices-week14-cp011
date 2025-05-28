#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简谐与非谐振子数值模拟与分析
功能：
1. 实现简谐和非谐振子的数值求解
2. 分析振幅对周期的影响
3. 比较两种振子的相空间轨迹
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state, t, omega=1.0):
    """
    简谐振子微分方程
    参数：
        state: [位移x, 速度v]
        t: 时间变量
        omega: 自然角频率
    返回：
        [dx/dt, dv/dt]
    """
    x, v = state
    return np.array([v, -omega**2 * x])  # 简谐运动方程: d²x/dt² = -ω²x

def anharmonic_oscillator_ode(state, t, omega=1.0):
    """
    非谐振子微分方程（三次方恢复力）
    参数同简谐振子
    返回：
        [dx/dt, dv/dt]
    """
    x, v = state
    return np.array([v, -omega**2 * x**3])  # 非线性恢复力: d²x/dt² = -ω²x³

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    四阶龙格-库塔单步积分
    参数：
        ode_func: 微分方程函数
        state: 当前状态
        t: 当前时间
        dt: 时间步长
        **kwargs: 传递给ode_func的参数
    返回：
        下一时刻的状态
    """
    k1 = ode_func(state, t, **kwargs)  # 初始斜率
    k2 = ode_func(state + 0.5*dt*k1, t + 0.5*dt, **kwargs)  # 中点斜率1
    k3 = ode_func(state + 0.5*dt*k2, t + 0.5*dt, **kwargs)  # 中点斜率2
    k4 = ode_func(state + dt*k3, t + dt, **kwargs)  # 终点斜率
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)  # 加权平均

def solve_ode(ode_func: Callable, initial_state: np.ndarray, 
             t_span: Tuple[float, float], dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解ODE系统
    参数：
        ode_func: 微分方程函数
        initial_state: 初始状态
        t_span: (起始时间, 结束时间)
        dt: 时间步长
    返回：
        (时间数组, 状态矩阵)
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)  # 时间离散化
    states = np.zeros((len(t), len(initial_state)))  # 状态存储矩阵
    states[0] = initial_state  # 设置初始条件
    
    for i in range(1, len(t)):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """绘制状态量随时间变化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')  # 位移曲线
    plt.plot(t, states[:, 1], label='Velocity v(t)')  # 速度曲线
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """绘制相空间轨迹图"""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])  # x-v平面轨迹
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')  # 等比例坐标轴
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    通过峰值检测计算振动周期
    参数：
        t: 时间数组
        states: 状态矩阵
    返回：
        平均周期（如无法检测返回nan）
    """
    x = states[:, 0]
    peaks = []
    for i in range(1, len(x)-1):  # 寻找局部极大值
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(t[i])
    
    if len(peaks) < 2:  # 需要至少2个峰值计算周期
        return np.nan
    
    periods = np.diff(peaks)  # 计算峰值间隔
    return np.mean(periods)  # 返回平均周期

def main():
    """主执行函数"""
    # 基本参数设置
    omega = 1.0  # 自然频率
    t_span = (0, 50)  # 模拟时长
    dt = 0.01  # 时间步长
    
    # 任务1：简谐振子模拟
    initial_state = np.array([1.0, 0.0])  # 初始位移1，速度0
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t, states, 'Harmonic Oscillator Time Evolution')
    period = analyze_period(t, states)
    print(f'Harmonic Period: {period:.4f} (Theory: {2*np.pi/omega:.4f})')
    
    # 任务2：振幅对周期的影响（简谐振子）
    amplitudes = [0.5, 1.0, 2.0]  # 测试不同振幅
    print("\nHarmonic Oscillator Period vs Amplitude:")
    for A in amplitudes:
        t, states = solve_ode(harmonic_oscillator_ode, np.array([A, 0.0]), t_span, dt, omega=omega)
        period = analyze_period(t, states)
        print(f'Amplitude {A}: Period = {period:.4f}')  # 简谐振子周期应与振幅无关
    
    # 任务3：非谐振子分析
    print("\nAnharmonic Oscillator Period vs Amplitude:")
    for A in amplitudes:
        t, states = solve_ode(anharmonic_oscillator_ode, np.array([A, 0.0]), t_span, dt, omega=omega)
        period = analyze_period(t, states)
        print(f'Amplitude {A}: Period = {period:.4f}')  # 非谐振子周期随振幅变化
        plot_time_evolution(t, states, f'Anharmonic Oscillator (A={A})')
    
    # 任务4：相空间轨迹比较
    initial_state = np.array([1.0, 0.0])
    # 简谐振子相图（应为椭圆）
    t, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_harmonic, 'Harmonic Phase Space')
    # 非谐振子相图（非线性变形）
    t, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_anharmonic, 'Anharmonic Phase Space')

if __name__ == "__main__":
    main()
