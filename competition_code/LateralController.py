import numpy as np
import math


def normalize_rad(rad: float):
    return rad % (2 * np.pi)


class LatController:
    def run(self, vehicle_location, vehicle_rotation, next_waypoint) -> float:
        """
        Calculates the steering command using the pure pursuit algorithm.

        Args:
            vehicle_location (np.array): Current vehicle location [x, y].
            vehicle_rotation (float): Current vehicle rotation (yaw) in radians.
            next_waypoint (Waypoint): Next waypoint to track.

        Returns:
            steering_command (float): Steering command in radians.
        """

        # Calculate vector pointing from vehicle to next waypoint
        waypoint_vector = np.array(next_waypoint.location) - np.array(vehicle_location)

        # Project waypoint vector onto heading vector to find lookahead point
        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        if distance_to_waypoint == 0:
            return 0  # Prevent division by zero

        waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

        # Calculate steering command
        alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(
            math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
        )

        steering_command = 1.50 * math.atan2(
            2.0 * 4.7 * math.sin(alpha) / distance_to_waypoint, 1.0
        )

        return float(steering_command)
