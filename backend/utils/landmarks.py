"""Landmark serialization utilities for MediaPipe pose data."""


def landmarks_to_dict(landmark_list) -> list[dict]:
    """Convert MediaPipe NormalizedLandmark list to serializable dicts."""
    return [
        {
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),
            "v": round(lm.visibility, 4),
        }
        for lm in landmark_list
    ]


# MediaPipe Pose landmark connections for drawing stick figures
# Each tuple is (start_index, end_index)
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Right arm
    (12, 14), (14, 16),
    # Left arm
    (11, 13), (13, 15),
    # Right leg
    (24, 26), (26, 28),
    # Left leg
    (23, 25), (25, 27),
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Hands
    (16, 18), (16, 20), (16, 22),
    (15, 17), (15, 19), (15, 21),
    # Feet
    (28, 30), (28, 32),
    (27, 29), (27, 31),
]

# Body part groupings for color-coding
BODY_PARTS = {
    "torso": [(11, 12), (11, 23), (12, 24), (23, 24)],
    "right_arm": [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)],
    "left_arm": [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)],
    "right_leg": [(24, 26), (26, 28), (28, 30), (28, 32)],
    "left_leg": [(23, 25), (25, 27), (27, 29), (27, 31)],
}
