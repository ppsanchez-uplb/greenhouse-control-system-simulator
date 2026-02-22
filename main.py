import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

def simulate_pid_thermal(
    t: np.ndarray,
    Kplant: float,
    tau: float,
    Kp: float,
    Ki: float,
    Kd: float,
    N: float,
    r: np.ndarray,
    d: np.ndarray,
    T_init: float,
):
    """
    Thermal plant (first order):
      dT/dt = -(1/tau) T + (Kplant/tau) u

    PID with filtered derivative:
      e = r - T
      xi_dot = e
      xd_dot = -N*xd + N*e
      u = Kp*e + Ki*xi + Kd*N*(e - xd) + d

    N is the derivative filter speed (rad per second). Larger N means less filtering.
    """

    dt = float(t[1] - t[0])

    T = np.zeros_like(t, dtype=float)
    xi = 0.0
    xd = 0.0

    T[0] = T_init

    for k in range(1, len(t)):
        e = float(r[k - 1] - T[k - 1])

        xi_dot = e
        xd_dot = -N * xd + N * e

        xi = xi + xi_dot * dt
        xd = xd + xd_dot * dt

        u = (Kp * e) + (Ki * xi) + (Kd * N * (e - xd)) + float(d[k - 1])

        T_dot = -(1.0 / tau) * T[k - 1] + (Kplant / tau) * u
        T[k] = T[k - 1] + T_dot * dt

    return T

def main():
    # Time base
    t_end = 1200.0
    dt = 0.5
    t = np.arange(0.0, t_end + dt, dt)

    # Plant parameters (adjust for your process)
    Kplant = 1.0
    tau = 180.0

    # Derivative filter parameter
    # Typical choice: N between 5 and 30 for smooth behavior
    N_default = 10.0

    # Initial UI values
    Kp0 = 2.0
    Ki0 = 0.02
    Kd0 = 10.0
    f0 = 0.002
    A0 = 1.0
    Ttarget0 = 30.0
    Tinit0 = 25.0

    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plt.subplots_adjust(left=0.10, right=0.97, bottom=0.42, top=0.92)

    ax.set_title("Temperature response simulator with PID live tuning")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature")
    ax.grid(True)

    line_T, = ax.plot([], [], linewidth=2, label="Temperature T(t)")
    line_r, = ax.plot([], [], linewidth=1, alpha=0.7, label="Reference r(t)")
    ax.legend(loc="upper right")

    # UI layout
    ax_kp = plt.axes([0.10, 0.32, 0.70, 0.03])
    ax_ki = plt.axes([0.10, 0.27, 0.70, 0.03])
    ax_kd = plt.axes([0.10, 0.22, 0.70, 0.03])
    ax_f  = plt.axes([0.10, 0.17, 0.70, 0.03])
    ax_a  = plt.axes([0.10, 0.12, 0.70, 0.03])
    ax_tr = plt.axes([0.10, 0.07, 0.70, 0.03])
    ax_t0 = plt.axes([0.10, 0.02, 0.70, 0.03])

    ax_mode = plt.axes([0.83, 0.08, 0.14, 0.16])
    ax_reset = plt.axes([0.83, 0.26, 0.14, 0.05])

    s_kp = Slider(ax_kp, "Kp", valmin=0.0, valmax=50.0, valinit=Kp0, valstep=0.1)
    s_ki = Slider(ax_ki, "Ki", valmin=0.0, valmax=1.0,  valinit=Ki0, valstep=0.001)
    s_kd = Slider(ax_kd, "Kd", valmin=0.0, valmax=200.0, valinit=Kd0, valstep=0.5)

    s_f  = Slider(ax_f,  "Freq (Hz)",  valmin=0.0001, valmax=0.02, valinit=f0, valstep=0.0001)
    s_a  = Slider(ax_a,  "Amplitude",  valmin=0.0,    valmax=10.0, valinit=A0, valstep=0.05)

    s_tr = Slider(ax_tr, "Target Temp",  valmin=0.0, valmax=60.0, valinit=Ttarget0, valstep=0.1)
    s_t0 = Slider(ax_t0, "Initial Temp", valmin=0.0, valmax=60.0, valinit=Tinit0,   valstep=0.1)

    mode = RadioButtons(ax_mode, ("Reference sine", "Disturbance sine"), active=0)
    b_reset = Button(ax_reset, "Reset")

    def recompute():
        Kp = float(s_kp.val)
        Ki = float(s_ki.val)
        Kd = float(s_kd.val)

        f = float(s_f.val)
        A = float(s_a.val)

        Ttarget = float(s_tr.val)
        Tinit = float(s_t0.val)

        w = 2.0 * np.pi * f

        if mode.value_selected == "Reference sine":
            r = Ttarget + A * np.sin(w * t)
            d = np.zeros_like(t)
        else:
            r = np.full_like(t, Ttarget)
            d = A * np.sin(w * t)

        T = simulate_pid_thermal(
            t=t,
            Kplant=Kplant,
            tau=tau,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            N=N_default,
            r=r,
            d=d,
            T_init=Tinit,
        )
        return T, r

    def update_plot(_=None):
        T, r = recompute()
        line_T.set_data(t, T)
        line_r.set_data(t, r)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def on_reset(event):
        s_kp.reset()
        s_ki.reset()
        s_kd.reset()
        s_f.reset()
        s_a.reset()
        s_tr.reset()
        s_t0.reset()
        mode.set_active(0)
        update_plot()

    s_kp.on_changed(update_plot)
    s_ki.on_changed(update_plot)
    s_kd.on_changed(update_plot)
    s_f.on_changed(update_plot)
    s_a.on_changed(update_plot)
    s_tr.on_changed(update_plot)
    s_t0.on_changed(update_plot)
    mode.on_clicked(lambda _: update_plot())
    b_reset.on_clicked(on_reset)

    update_plot()
    plt.show()

if __name__ == "__main__":
    main()