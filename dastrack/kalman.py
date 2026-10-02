"""
Constant-velocity Kalman filter with RTS smoother (manuscript Eqs. 10-15, 18-19).

State ``x = [channel position, signed velocity (m/s)]``.  Velocity updates come
from the F-K slant-stack; position updates come from the de-slant centroid.
"""
import numpy as np


class KF:
    def __init__(self, ch0, v0, fs, dx, q_vel=0.4, pos_sigma=6.0, vel_sigma=2.5,
                 r_vel=6.25, r_pos=9.0, v_min=0.5, v_max=30.0,
                 contrast_r_lo=0.3, contrast_r_hi=8.0):
        self.fs, self.dx = fs, dx
        self.q, self.Rv, self.Rp = q_vel, r_vel, r_pos
        self.v_min, self.v_max = v_min, v_max
        self.cr_lo, self.cr_hi = contrast_r_lo, contrast_r_hi
        self.x = np.array([float(ch0), float(v0)])
        self.P = np.diag([pos_sigma ** 2, vel_sigma ** 2])
        self.x_prior, self.P_prior, self.x_post, self.P_post, self.F_log = [], [], [], [], []

    def predict(self, step_samples):
        """Propagate by ``step_samples`` (negative = backward in time)."""
        dt = float(step_samples) / self.fs
        F = np.array([[1.0, dt / self.dx], [0.0, 1.0]])
        adt = abs(dt)
        sdt = np.sign(dt)
        Q = self.q * np.array([[adt ** 3 / (3 * self.dx ** 2), sdt * adt ** 2 / (2 * self.dx)],
                               [sdt * adt ** 2 / (2 * self.dx), adt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x_prior.append(self.x.copy())
        self.P_prior.append(self.P.copy())
        self.F_log.append(F)
        return float(self.x[0])

    def gate_ms(self, gate_sigma, lo, hi):
        """Velocity gate half-width (m/s), clipped to [lo, hi]."""
        return float(np.clip(gate_sigma * np.sqrt(self.P[1, 1] + self.Rv), lo, hi))

    def update_velocity(self, v_meas, r_scale=1.0):
        H = np.array([[0.0, 1.0]])
        R = self.Rv * r_scale
        y = float(v_meas) - float((H @ self.x)[0])
        S = float((H @ self.P @ H.T)[0, 0]) + R
        K = (self.P @ H.T) / S
        self.x = self.x + K[:, 0] * y
        self.P = (np.eye(2) - K @ H) @ self.P
        s = np.sign(self.x[1]) if self.x[1] != 0 else 1.0
        self.x[1] = s * float(np.clip(abs(self.x[1]), self.v_min, self.v_max))
        return abs(y) / np.sqrt(S)

    def update_position(self, ch_meas, r_scale=1.0):
        H = np.array([[1.0, 0.0]])
        R = self.Rp * r_scale
        y = float(ch_meas) - float((H @ self.x)[0])
        S = float((H @ self.P @ H.T)[0, 0]) + R
        K = (self.P @ H.T) / S
        self.x = self.x + K[:, 0] * y
        self.P = (np.eye(2) - K @ H) @ self.P

    def commit(self):
        self.x_post.append(self.x.copy())
        self.P_post.append(self.P.copy())

    def rts(self):
        """Rauch-Tung-Striebel backward pass over the committed states."""
        n = len(self.x_post)
        if n == 0:
            return np.empty((0, 2)), np.empty((0, 2, 2))
        xs = [x.copy() for x in self.x_post]
        Ps = [p.copy() for p in self.P_post]
        for i in range(n - 2, -1, -1):
            F = self.F_log[i + 1]
            Pp = self.P_prior[i + 1]
            C = self.P_post[i] @ F.T @ np.linalg.inv(Pp)
            xs[i] = self.x_post[i] + C @ (xs[i + 1] - self.x_prior[i + 1])
            Ps[i] = self.P_post[i] + C @ (Ps[i + 1] - Pp) @ C.T
        return np.array(xs), np.array(Ps)

    @property
    def v_ms(self):
        return float(self.x[1])

    @property
    def ch(self):
        return float(self.x[0])
