import math
import numpy as np


#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2



class ExtendedKalmanFiler:
    def __init__(self, time_step=0.1) -> None:
        # covariance matrix of the process noise
        self.Q = np.diag([
            0.1,                # variance of location on x-axis
            0.1,                # variance of location on y-axis
            np.deg2rad(1.0),    # variance of yaw angle
            1.0                 # variance of velocity
        ]) ** 2
        # covariance matrix of the observation noise at time t
        self.R = np.diag([1.0, 1.0]) ** 2

        # covariance matrix of the state
        self.P = np.eye(4)

        self.DT = time_step     # time tick [s]

    def estimate(self, x_est, z, u):
        """
        Args:
            x_est:  state vector [x_t, y_t, theta_t, v_t]
            u:      input vector
            z:      observation vector (x_t, y_t)
        """
        # predict
        x_pred = self.motion_model(x_est, u)
        jF = self.jacob_f(x_est, u)
        p_pred = jF @ self.P @ jF.T + self.Q

        # update
        jH = self.jacob_h()
        z_pred = self.observation_model(x_pred)
        y = z - z_pred
        S = jH @ p_pred @ jH.T + self.R
        K = p_pred @ jH.T @ np.linalg.inv(S)

        # x_t2 = x_pred + Ky
        x_est = x_pred + K @ y
        self.P = (np.eye(len(x_est)) - K @ jH) @ p_pred
        return x_est
    
    def observation(self, x_true, xd, u):
        x_true = self.motion_model(x_true, u)

        # add noise to gps x-y
        z = self.observation_model(x_true) + GPS_NOISE @ np.random.randn(2, 1)

        # add noise to input
        ud = u + INPUT_NOISE @ np.random.randn(2, 1)
        
        xd = self.motion_model(xd, ud)
        return x_true, z, xd, ud
    
    def motion_model(self, x, u):
        """
        Args:
            x:
            u:
        Returns:

        """
        F = np.eye(4)
        F[-1, -1] = 0.0

        B = np.array([
            [self.DT * math.cos(x[2, 0]), 0],
            [self.DT * math.sin(x[2, 0]), 0],
            [0.0, self.DT],
            [1.0, 0.0]
        ])
        x = F @ x + B @ u
        # x = F @ x
        return x
    
    def observation_model(self, x):
        """
        Args:
            x:
        Returns:

        """
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        z = H @ x
        return z
    
    def jacob_f(self, x, u):
        """Jacobian of Motion Model

        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.DT * v * math.sin(yaw), self.DT * math.cos(yaw)],
            [0.0, 1.0, self.DT * v * math.cos(yaw), self.DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return jF
    
    def jacob_h(self):
        """Jacobian of Observation Model
        Args:
            x:
        Returns:

        """
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return jH
    

def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    time = 0.0
    DT = 0.1
    SIM_TIME = 50.0    # simulation time [s]

    # state vector [x y yaw v]
    x_est = np.zeros((4, 1))
    x_true = np.zeros((4, 1))
    x_dr = np.zeros((4, 1)) # dead reckoning

    # history
    hx_est = x_est
    hx_true = x_true
    hx_dr = x_true
    hz = np.zeros((2, 1))

    ekf_filter = ExtendedKalmanFiler(DT)

    u = calc_input()

    while SIM_TIME >= time:
        time += DT

        x_true, z, x_dr, ud = ekf_filter.observation(x_true, x_dr, u)
        x_est = ekf_filter.estimate(x_est, z, ud)

        # store data history
        hx_est = np.hstack([hx_est, x_est])
        hx_dr = np.hstack([hx_dr, x_dr])
        hx_true = np.hstack([hx_true, x_true])
        hz = np.hstack([hz, z])

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(hz[0, :], hz[1, :], ".g")
        plt.plot(hx_true[0, :].flatten(), hx_true[1, :].flatten(), "-b")
        plt.plot(hx_dr[0, :].flatten(), hx_dr[1, :].flatten(), "-k")
        plt.plot(hx_est[0, :].flatten(), hx_est[1, :].flatten(), "-r")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)