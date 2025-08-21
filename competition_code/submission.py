"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
import atexit

# from scipy.interpolate import interp1d

# Store the directory path at module load time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

useDebug = True
useDebugPrinting = False
debugData = {}


def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)


@atexit.register
def saveDebugData():
    if useDebug:
        print("Saving debug data")
        jsonData = json.dumps(debugData, indent=4)
        with open(
            os.path.join(SCRIPT_DIR, "debugData", "debugData.json"), "w+"
        ) as outfile:
            outfile.write(jsonData)
        print("Debug Data Saved")


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()
        self.section_indeces = []
        # Stable section IDs aligned with section_indeces; use these for tuning so inserting sections doesn't shift behavior
        self.section_ids = []
        self.num_ticks = 0
        self.section_start_ticks = 0
        # current_section is the physical index into section_indeces (legacy); current_section_id is the stable ID
        self.current_section = 0
        self.current_section_id = 0
        self.lapNum = 1

    async def initialize(self) -> None:
        # NOTE waypoints are changed through this line
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(os.path.join(SCRIPT_DIR, "waypoints", "waypointsPrimary.npz"))
            )[35:]
        )

        # Define sections with stable IDs. Keep these IDs unchanged even if you insert a new section later.
        section_metadata = [
            {"id": 0, "loc": [-278, 372]},   # Section 0 start location
            {"id": 1, "loc": [64, 890]},    # Section 1 start location
            {"id": 2, "loc": [511, 1037]},  # Section 2 start location
            {"id": 3, "loc": [762, 908]},   # Section 3 start location
            {"id": 10, "loc": [664, 667]}, # Section 10 start location (I added it)
            {"id": 4, "loc": [198, 307]},   # Section 4 start location
            {"id": 5, "loc": [-11, 60]},    # Section 5 start location
            {"id": 6, "loc": [-85, -339]},  # Section 6 start location
            {"id": 7, "loc": [-210, -1060]},# Section 7 start location 
            {"id": 8, "loc": [-318, -991]}, # Section 8 start location
            {"id": 9, "loc": [-352, -119]}, # Section 9 start location
        ]

        # for me testing lol
        # sectionLocations = [
        #     [-278, 372], # Section 0 start location
        #     [-144, 847], # Section 1 start location
        #     [256, 911], # Section 2 start location
        #     [657, 1071], # Section 3 start location
        #     [664, 667],  # I wanna add this but don't rly wanna change everything to match T_T
        #     [710, 704], # Section 4 start location
        #     [73, 204], # Section 5 start location
        #     [-84, -216], # Section 6 start location
        #     [-210, -1060], # Section 7 start location 
        #     [-318, -991], # Section 8 start location
        #     [-372, -619], # Section 9 start location
        # ]
        # Build section indices and stable IDs
        for s in section_metadata:
            self.section_indeces.append(
                findClosestIndex(np.array(s["loc"], dtype=float), self.maneuverable_waypoints)
            )
            self.section_ids.append(int(s["id"]))

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indeces}")
        print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

    async def step(self) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        self.num_ticks += 1

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        # compute and print section timing
        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                print(f"Section {i}: {self.num_ticks - self.section_start_ticks} ticks")
                self.section_start_ticks = self.num_ticks
                self.current_section = i
                # Set stable section ID for tuning/logic
                if i < len(self.section_ids):
                    self.current_section_id = self.section_ids[i]
                else:
                    self.current_section_id = i  # fallback

                if self.current_section_id == 0 and self.lapNum != 3:
                    self.lapNum += 1
                    print(f"\nLap {self.lapNum}\n")

        nextWaypointIndex = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh)

        # Pure pursuit controller to steer the vehicle
        steer_control = self.lat_controller.run(
            vehicle_location, vehicle_rotation, waypoint_to_follow
        )

        # Custom controller to control the vehicle's speed
        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            nextWaypointIndex : nextWaypointIndex + 300
        ]
        throttle, brake, gear = self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section_id,
        )

        steerMultiplier = round((current_speed_kmh + 0.001) / 120, 3)
        
        sid = self.current_section_id
        if sid == 2:
            steerMultiplier *= 1.2
        if sid in [3]:
            # steerMultiplier *= 0.9
            steerMultiplier = np.clip(steerMultiplier * 1.75, 2.3, 3.5)
        if sid == 4:
            steerMultiplier = min(1.45, steerMultiplier * 1.65)
        if sid == 5:
            steerMultiplier *= 1.1
        if sid in [6]:
            # steerMultiplier = min(steerMultiplier * 5, 5.35)
            steerMultiplier = np.clip(steerMultiplier * 5.5, 5.5, 7)
        if sid == 7:
            steerMultiplier *= 2
        if sid == 9:
            steerMultiplier = max(steerMultiplier, 1.6)

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": np.clip(steer_control * steerMultiplier, -1, 1),
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,  # Gears do not appear to have an impact on speed
        }
        
        if useDebug:
            debugData[self.num_ticks] = {}
            debugData[self.num_ticks]["loc"] = [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ]
            debugData[self.num_ticks]["throttle"] = round(float(control["throttle"]), 3)
            debugData[self.num_ticks]["brake"] = round(float(control["brake"]), 3)
            debugData[self.num_ticks]["steer"] = round(float(control["steer"]), 10)
            debugData[self.num_ticks]["speed"] = round(current_speed_kmh, 3)
            # The chosen recommended speed (km/h) from ThrottleController for this tick
            rec_speed = self.throttle_controller.last_recommended_speed_kmh
            if rec_speed is not None:
                debugData[self.num_ticks]["recommended_speed"] = round(float(rec_speed), 3)
            else:
                debugData[self.num_ticks]["recommended_speed"] = None
            debugData[self.num_ticks]["lap"] = self.lapNum
            # Add section info per tick
            debugData[self.num_ticks]["section"] = int(self.current_section)
            # Stable section ID that won't shift if you insert new sections
            debugData[self.num_ticks]["section_id"] = int(self.current_section_id)
            debugData[self.num_ticks]["section_ticks"] = int(self.num_ticks - self.section_start_ticks)

            if useDebugPrinting and self.num_ticks % 5 == 0:
                print(
                    f"- Target waypoint: ({waypoint_to_follow.location[0]:.2f}, {waypoint_to_follow.location[1]:.2f}) index {nextWaypointIndex} \n\
Current location: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) index {self.current_waypoint_idx} section {self.current_section} \n\
Distance to target waypoint: {math.sqrt((waypoint_to_follow.location[0] - vehicle_location[0]) ** 2 + (waypoint_to_follow.location[1] - vehicle_location[1]) ** 2):.3f}\n"
                )

                print(
                    f"--- Speed: {current_speed_kmh:.2f} kph \n\
Throttle: {control['throttle']:.3f} \n\
Brake: {control['brake']:.3f} \n\
Steer: {control['steer']:.10f} \n"
                )

        await self.vehicle.apply_action(control)
        return control

    def get_lookahead_value(self, speed):
        """
        Returns the number of waypoints to look ahead based on the speed the car is currently going
        """
        speed_to_lookahead_dict = {
            90: 9,
            110: 12,
            130: 15,
            160: 19,
            180: 23,
            200: 27,
            250: 32,
            300: 37,
        }

        # Interpolation method
        # NOTE does not work as well as the dictionary lookahead method, likely to cause crashes.

        # speedBoundList = [0, 90, 110, 130, 160, 180, 200, 250, 300]
        # lookaheadList = [5, 11, 13, 15, 18, 22, 25, 28, 32]

        # interpolationFunction = interp1d(speedBoundList, lookaheadList)
        # return int(interpolationFunction(speed))

        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
                return num_points       # number of waypoints to look ahead by
        return 8

    def get_lookahead_index(self, speed):
        """
        Adds the lookahead waypoint to the current waypoint and normalizes it so that the value does not go out of bounds
        """
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed)
        #       + " cur_ind " + str(self.current_waypoint_idx)
        #       + " num_points " + str(num_waypoints)
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(
            self.maneuverable_waypoints
        )

    # def get_lateral_pid_config(self):
    #     """
    #     Returns the PID values for the lateral (steering) PID
    #     """
    #     with open(
    #         f"{os.path.dirname(__file__)}\\configs\\LatPIDConfig.json", "r"
    #     ) as file:
    #         config = json.load(file)
    #     return config

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py (Summer 2023)
    def next_waypoint_smooth(self, current_speed: float):
        """
        If the speed is higher than 70, 'smooth out' the path that the car will take
        """
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def average_point(self, current_speed):
        """
        Returns a new averaged waypoint based on the location of a number of other waypoints
        """
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2

        # Section specific tuning (use stable section ID)
        sid = self.current_section_id
        if sid == 0:
            num_points = round(lookahead_value * 1.5)
        if sid == 3:
            next_waypoint_index = self.current_waypoint_idx + 22
            num_points = 35
        if sid == 4:
            num_points = lookahead_value + 5
            next_waypoint_index = self.current_waypoint_idx + 24
        if sid == 5:
            # num_points = round(lookahead_value * 1.1)
            num_points = lookahead_value
        if sid == 6:
            num_points = 5
            next_waypoint_index = self.current_waypoint_idx + 28
        if sid == 7:
            # Jolt between sections 6 and 7 likely due to the differences in lookahead values and steering multipliers. 
            num_points = round(lookahead_value * 1.25)
        if sid == 9:
            (self.current_waypoint_idx + 8) % len(self.maneuverable_waypoints)
            num_points = 0

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location

        sample_points = [
            (start_index_for_avg + i) % len(self.maneuverable_waypoints)
            for i in range(0, num_points)
        ]
        if num_points > 3:
            location_sum = reduce(
                lambda x, y: x + y,
                (self.maneuverable_waypoints[i].location for i in sample_points),
            )
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if sid == 1:
                max_shift_distance = 0.2
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv * max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(
                location=new_location,
                roll_pitch_yaw=np.ndarray([0, 0, 0]),
                lane_width=0.0,
            )
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location)
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint = self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint
