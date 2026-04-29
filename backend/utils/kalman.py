"""Per-landmark Kalman filter for temporal pose smoothing.

Each landmark (x, y, z) gets its own Kalman filter tracking
position + velocity. This gives:
  - Smooth trajectories (no jitter)
  - Velocity-based prediction during dropouts (1-3 frames)
  - Responsive tracking during fast movements
"""

import numpy as np


class LandmarkKalmanFilter:
    """Kalman filter for a single landmark's (x, y, z) coordinates.

    State vector: [x, y, z, vx, vy, vz] (position + velocity)
    Measurement:  [x, y, z]
    """

    def __init__(self, process_noise: float = 0.005, measurement_noise: float = 0.02):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=np.float64)

        # State covariance
        self.P = np.eye(6, dtype=np.float64) * 0.1

        # State transition (constant velocity model)
        # dt=1 (one frame step), updated per-call if needed
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = 1.0  # x += vx * dt
        self.F[1, 4] = 1.0  # y += vy * dt
        self.F[2, 5] = 1.0  # z += vz * dt

        # Measurement matrix (we observe position only)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        # Velocity components get more noise (less certain about acceleration)
        self.Q[3, 3] = process_noise * 4
        self.Q[4, 4] = process_noise * 4
        self.Q[5, 5] = process_noise * 4

        # Measurement noise
        self.R = np.eye(3, dtype=np.float64) * measurement_noise

        self._initialized = False

    def predict(self):
        """Predict next state based on velocity model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        """Update state with new measurement [x, y, z]."""
        if not self._initialized:
            self.x[:3] = measurement
            self._initialized = True
            return

        # Innovation
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        """Current estimated position [x, y, z]."""
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity [vx, vy, vz]."""
        return self.x[3:]


class PoseKalmanFilter:
    """Kalman filter bank for all 33 MediaPipe landmarks.

    Provides smooth, predictive tracking with velocity estimation.
    """

    def __init__(self, num_landmarks: int = 33,
                 process_noise: float = 0.003, measurement_noise: float = 0.015):
        self.filters = [
            LandmarkKalmanFilter(process_noise, measurement_noise)
            for _ in range(num_landmarks)
        ]
        self._num_landmarks = num_landmarks
        self._frames_since_update = [0] * num_landmarks
        self._max_predict_frames = 5  # predict up to 5 frames without measurement

    def update(self, landmarks: list[dict]) -> list[dict]:
        """Process a new set of landmarks through the Kalman bank.

        Args:
            landmarks: List of 33 dicts with x, y, z, v keys

        Returns:
            Filtered landmarks with smoothed positions
        """
        result = []

        for i, (lm, kf) in enumerate(zip(landmarks, self.filters)):
            if lm["v"] >= 0.3:
                # Good measurement — predict then update
                kf.predict()
                kf.update(np.array([lm["x"], lm["y"], lm["z"]]))
                self._frames_since_update[i] = 0

                pos = kf.position
                result.append({
                    "x": round(float(pos[0]), 4),
                    "y": round(float(pos[1]), 4),
                    "z": round(float(pos[2]), 4),
                    "v": lm["v"],
                })
            else:
                # No measurement — predict from velocity
                self._frames_since_update[i] += 1

                if self._frames_since_update[i] <= self._max_predict_frames:
                    kf.predict()
                    pos = kf.position
                    # Decay confidence based on frames without measurement
                    decay = max(0.1, 1.0 - self._frames_since_update[i] * 0.15)
                    result.append({
                        "x": round(float(pos[0]), 4),
                        "y": round(float(pos[1]), 4),
                        "z": round(float(pos[2]), 4),
                        "v": round(decay, 4),
                    })
                else:
                    # Too many frames without data — report as invisible
                    result.append(lm)

        return result
