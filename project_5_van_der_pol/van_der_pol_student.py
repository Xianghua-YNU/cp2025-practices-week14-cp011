#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, List

def van_der_pol_ode(t, state, mu=1.0, omega=1.0):
    """van der Pol振子的一阶微分方程组。"""
    x, v = state
    return np.array([v, mu*(1-x**2)*v - omega**2*x])

def solve_ode(ode_func, initial_state, t_span, dt, **kwargs):
    """使用solve_ivp求解常微分方程组"""
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(ode_func, t_span, initial_state, 
                   t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """Plot the time evolution of states."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """Plot the phase space trajectory."""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """分析极限环的特征（振幅和周期）。"""
    # 跳过初始瞬态
    skip = int(len(states)*0.5)
    x = states[skip:, 0]
    t = np.arange(len(x))
    
    # 计算振幅（取最大值的平均）
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(x[i])
    amplitude = np.mean(peaks) if peaks else np.nan
    
    # 计算周期（取相邻峰值点的时间间隔平均）
    if len(peaks) >= 2:
        periods = np.diff(t[1:-1][np.array([x[i] > x[i-1] and x[i] > x[i+1] for i in range(1, len(x)-1)])])
        period = np.mean(periods) if len(periods) > 0 else np.nan
    else:
        period = np.nan
    
    return amplitude, period

def main():
    # Set basic parameters
    mu = 1.0
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # Task 1 - Basic implementation
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
    
    # Task 2 - Parameter influence analysis
    mu_values = [1.0, 2.0, 4.0]
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
        amplitude, period = analyze_limit_cycle(states)
        print(f'μ = {mu}: Amplitude ≈ {amplitude:.3f}, Period ≈ {period*dt:.3f}')
    
    # Task 3 - Phase space analysis
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'Phase Space Trajectory of van der Pol Oscillator (μ={mu})')

if __name__ == "__main__":
    main()
