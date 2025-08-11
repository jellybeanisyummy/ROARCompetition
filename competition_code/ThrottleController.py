import numpy as np
import math
from collections import deque
from SpeedData import SpeedData
import roar_py_interface
from typing import List, Tuple


def distance_p_to_p(
    p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint
):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])


class ThrottleController:
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self):
        self.max_radius = 10000
        self.max_speed = 300
        self.intended_target_distance = [0, 30, 60, 90, 120, 140, 170]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")

    def run(
        self, waypoints, current_location, current_speed, current_section
    ) -> Tuple[float, float, int]:
        self.tick_counter += 1
        throttle, brake = self.get_throttle_and_brake(
            current_location, current_speed, current_section, waypoints
        )
        # gear = max(1, (int)(math.log(current_speed + 0.00001, 5)))
        gear = max(1, int(current_speed / 60))
        if throttle < 0:
            gear = -1

        # self.dprint("--- " + str(throttle) + " " + str(brake)
        #             + " steer " + str(steering)
        #             + "     loc x,z" + str(self.agent.vehicle.transform.location.x)
        #             + " " + str(self.agent.vehicle.transform.location.z))

        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        # throttle = 0.05 * (100 - current_speed)
        return throttle, brake, gear

    def get_throttle_and_brake(
        self, current_location, current_speed, current_section, waypoints
    ):
        """
        Returns throttle and brake values based off the car's current location and the radius of the approaching turn
        """

        nextWaypoint = self.get_next_interesting_waypoints(current_location, waypoints)
        r1 = self.get_radius(nextWaypoint[self.close_index : self.close_index + 3])
        r2 = self.get_radius(nextWaypoint[self.mid_index : self.mid_index + 3])
        r3 = self.get_radius(nextWaypoint[self.far_index : self.far_index + 3])

        target_speed1 = self.get_target_speed(r1, current_section)
        target_speed2 = self.get_target_speed(r2, current_section)
        target_speed3 = self.get_target_speed(r3, current_section)

        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]
        speed_data = []
        speed_data.append(
            self.speed_for_turn(close_distance, target_speed1, current_speed)
        )
        speed_data.append(
            self.speed_for_turn(mid_distance, target_speed2, current_speed)
        )
        speed_data.append(
            self.speed_for_turn(far_distance, target_speed3, current_speed)
        )

        if current_speed > 100:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            if current_section != 9:
                r4 = self.get_radius(
                    [
                        nextWaypoint[self.mid_index],
                        nextWaypoint[self.mid_index + 2],
                        nextWaypoint[self.mid_index + 4],
                    ]
                )
                target_speed4 = self.get_target_speed(r4, current_section)
                speed_data.append(
                    self.speed_for_turn(close_distance, target_speed4, current_speed)
                )

            r5 = self.get_radius(
                [
                    nextWaypoint[self.close_index],
                    nextWaypoint[self.close_index + 3],
                    nextWaypoint[self.close_index + 6],
                ]
            )
            target_speed5 = self.get_target_speed(r5, current_section)
            speed_data.append(
                self.speed_for_turn(close_distance, target_speed5, current_speed)
            )

        update = self.select_speed(speed_data)

        self.print_speed(
            " -- SPEED: ",
            speed_data[0].recommended_speed_now,
            speed_data[1].recommended_speed_now,
            speed_data[2].recommended_speed_now,
            (0 if len(speed_data) < 4 else speed_data[3].recommended_speed_now),
            current_speed,
        )

        throttle, brake = self.speed_data_to_throttle_and_brake(update)
        self.dprint("--- throt " + str(throttle) + " brake " + str(brake) + "---")
        return throttle, brake

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        """
        Converts speed data into throttle and brake values
        """

        # self.dprint("dist=" + str(round(speed_data.distance_to_section)) + " cs=" + str(round(speed_data.current_speed, 2))
        #             + " ts= " + str(round(speed_data.target_speed_at_distance, 2))
        #             + " maxs= " + str(round(speed_data.recommended_speed_now, 2)) + " pcnt= " + str(round(percent_of_max, 2)))

        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now
        avg_speed_change_per_tick = 2.4  # Speed decrease in kph per tick
        percent_change_per_tick = 0.075  # speed drop for one time-tick of braking
        true_percent_change_per_tick = round(
            avg_speed_change_per_tick / (speed_data.current_speed + 0.001), 5
        )
        speed_up_threshold = 0.9
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        brake_threshold_multiplier = 1.0
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        speed_change = round(speed_data.current_speed - self.previous_speed, 3)

        if percent_of_max > 1:
            # Consider slowing down
            # if speed_data.current_speed > 200:  # Brake earlier at higher speeds
            #     brake_threshold_multiplier = 0.9

            if percent_of_max > 1 + (
                brake_threshold_multiplier * true_percent_change_per_tick
            ):
                if self.brake_ticks > 0:
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: counter "
                        + str(self.brake_ticks)
                    )
                    return -1, 1

                # if speed is not decreasing fast, hit the brake.
                if self.brake_ticks <= 0 and speed_change < 2.5:
                    # start braking, and set for how many ticks to brake
                    self.brake_ticks = (
                        round(
                            (
                                speed_data.current_speed
                                - speed_data.recommended_speed_now
                            )
                            / 3
                        )
                    )
                    # self.brake_ticks = 1, or (1 or 2 but not more)
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: initiate counter "
                        + str(self.brake_ticks)
                    )
                    return -1, 1

                else:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: throttle early1: sp_ch="
                        + str(percent_speed_change)
                    )
                    self.brake_ticks = 0  # done slowing down. clear brake_ticks
                    return 1, 0
            else:
                if speed_change >= 2.5:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: throttle early2: sp_ch="
                        + str(percent_speed_change)
                    )
                    self.brake_ticks = 0  # done slowing down. clear brake_ticks
                    return 1, 0

                # TODO: Try to get rid of coasting. Unnecessary idle time that could be spent speeding up or slowing down
                throttle_to_maintain = self.get_throttle_to_maintain_speed(
                    speed_data.current_speed
                )

                if percent_of_max > 1.02 or percent_speed_change > (
                    -true_percent_change_per_tick / 2
                ):
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: throttle down: sp_ch="
                        + str(percent_speed_change)
                    )
                    return (
                        throttle_to_maintain * throttle_decrease_multiple,
                        0,
                    )  # coast, to slow down
                else:
                    # self.dprint("tb: tick " + str(self.tick_counter) + " brake: throttle maintain: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain, 0
        else:
            self.brake_ticks = 0  # done slowing down. clear brake_ticks
            # Speed up
            if speed_change >= 2.5:
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle: full speed drop: sp_ch="
                    + str(percent_speed_change)
                )
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle full: p_max="
                    + str(percent_of_max)
                )
                return 1, 0
            throttle_to_maintain = self.get_throttle_to_maintain_speed(
                speed_data.current_speed
            )
            if percent_of_max < 0.98 or true_percent_change_per_tick < -0.01:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle up: sp_ch="
                    + str(percent_speed_change)
                )
                return throttle_to_maintain * throttle_increase_multiple, 0
            else:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle maintain: sp_ch="
                    + str(percent_speed_change)
                )
                return throttle_to_maintain, 0

    # used to detect when speed is dropping due to brakes applied earlier. speed delta has a steep negative slope.
    def isSpeedDroppingFast(self, percent_change_per_tick: float, current_speed):
        """
        Detects if the speed of the car is dropping quickly.
        Returns true if the speed is dropping fast
        """
        percent_speed_change = (current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        return percent_speed_change < (-percent_change_per_tick / 2)

    # find speed_data with smallest recommended speed
    def select_speed(self, speed_data: List[SpeedData]):
        """
        Selects the smallest speed out of the speeds provided
        """
        min_speed = 1000
        index_of_min_speed = -1
        for i, sd in enumerate(speed_data):
            if sd.recommended_speed_now < min_speed:
                min_speed = sd.recommended_speed_now
                index_of_min_speed = i

        if index_of_min_speed != -1:
            return speed_data[index_of_min_speed]
        else:
            return speed_data[0]

    def get_throttle_to_maintain_speed(self, current_speed: float):
        """
        Returns a throttle value to maintain the current speed
        """
        throttle = 0.75 + current_speed / 500
        return throttle

    def speed_for_turn(
        self, distance: float, target_speed: float, current_speed: float
    ):
        """Generates a SpeedData object with the target speed for the far

        Args:
            distance (float): Distance from the start of the curve
            target_speed (float): Target speed of the curve
            current_speed (float): Current speed of the car

        Returns:
            SpeedData: A SpeedData object containing the distance to the corner, current speed, target speed, and max speed
        """
        # Takes in a target speed and distance and produces a speed that the car should target. Returns a SpeedData object

        d = (1 / 675) * (target_speed**2) + distance
        max_speed = math.sqrt(825 * d)
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def get_next_interesting_waypoints(self, current_location, more_waypoints):
        """Returns a list of waypoints that are approximately as far as specified in intended_target_distance from the current location

        Args:
            current_location (roar_py_interface.RoarPyWaypoint): The current location of the car
            more_waypoints ([roar_py_interface.RoarPyWaypoint]): A list of waypoints

        Returns:
            [roar_py_interface.RoarPyWaypoint]: A list of waypoints within specified distances of the car
        """
        # Returns a list of waypoints that are approximately as far as the given in intended_target_distance from the current location

        # return a list of points with distances approximately as given
        # in intended_target_distance[] from the current location.
        points = []
        dist = []  # for debugging
        start = roar_py_interface.RoarPyWaypoint(
            current_location, np.ndarray([0, 0, 0]), 0.0
        )
        # start = self.agent.vehicle.transform
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            # print("start " + str(start) + "\n- - - - -\n")
            # print("end " + str(end) +     "\n- - - - -\n")
            curr_dist += distance_p_to_p(start, end)
            # curr_dist += start.location.distance(end.location)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        self.dprint("wp dist " + str(dist))
        return points

    def get_radius(self, wp: List[roar_py_interface.RoarPyWaypoint]):
        """Returns the radius of a curve given 3 waypoints using the Menger Curvature Formula

        Args:
            wp (List[roar_py_interface.RoarPyWaypoint]): A list of 3 RoarPyWaypoints

        Returns:
            float: The radius of the curve made by the 3 given waypoints
        """

        point1 = (wp[0].location[0], wp[0].location[1])
        point2 = (wp[1].location[0], wp[1].location[1])
        point3 = (wp[2].location[0], wp[2].location[1])

        # Calculating length of all three sides
        len_side_1 = round(math.dist(point1, point2), 3)
        len_side_2 = round(math.dist(point2, point3), 3)
        len_side_3 = round(math.dist(point1, point3), 3)

        small_num = 2

        if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
            return self.max_radius

        # sp is semi-perimeter
        sp = (len_side_1 + len_side_2 + len_side_3) / 2

        # Calculating area using Herons formula
        area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
        if area_squared < small_num:
            return self.max_radius

        # Calculating curvature using Menger curvature formula
        radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))

        return radius

    def get_target_speed(self, radius: float, current_section: int):
        """Returns a target speed based on the radius of the turn and the section it is in

        Args:
            radius (float): The calculated radius of the turn
            current_section (int): The current section of the track the car is in

        Returns:
            float: The maximum speed the car can go around the corner at
        """

        mu = 2.75

        if radius >= self.max_radius:
            return self.max_speed

        if current_section == 0:
            mu = 4
        if current_section == 1:
            mu = 2.75
        if current_section == 2:
            mu = 3.35
        if current_section == 3:
            mu = 3.3
        if current_section == 4:
            mu = 2.85
        if current_section == 6:
            mu = 3.3
        if current_section == 9:
            mu = 2.1

        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6

        return max(
            20, min(target_speed, self.max_speed)
        )  # clamp between 20 and max_speed

    def print_speed(
        self, text: str, s1: float, s2: float, s3: float, s4: float, curr_s: float
    ):
        """
        Prints debug speed values
        """
        self.dprint(
            text
            + " s1= "
            + str(round(s1, 2))
            + " s2= "
            + str(round(s2, 2))
            + " s3= "
            + str(round(s3, 2))
            + " s4= "
            + str(round(s4, 2))
            + " cspeed= "
            + str(round(curr_s, 2))
        )

    # debug print
    def dprint(self, text):
        """
        Prints debug text
        """
        if self.display_debug:
            print(text)
            self.debug_strings.append(text)
