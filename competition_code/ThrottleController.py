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
        # Last chosen recommended speed (km/h) from select_speed; used for debug logging
        self.last_recommended_speed_kmh = None

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")

    def run(
        self, waypoints, current_location, current_speed, current_section_id
    ) -> Tuple[float, float, int]:
        self.tick_counter += 1
        throttle, brake = self.get_throttle_and_brake(
            current_location, current_speed, current_section_id, waypoints
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
        self, current_location, current_speed, current_section_id, waypoints
    ):
        """
        Returns throttle and brake values based off the car's current location and the radius of the approaching turn
        """

        # 1) Pick representative lookahead waypoints (close/mid/far)
        nextWaypoint = self.get_next_interesting_waypoints(current_location, waypoints)

        # 2) Estimate radii from 3-point windows at close/mid/far lookaheads
        r1 = self.get_radius(nextWaypoint[self.close_index : self.close_index + 3])  # close window
        r2 = self.get_radius(nextWaypoint[self.mid_index : self.mid_index + 3])      # mid window
        r3 = self.get_radius(nextWaypoint[self.far_index : self.far_index + 3])      # far window

        # 3) Convert radii to section-aware target speeds
        target_speed1 = self.get_target_speed(r1, current_section_id)  # for r1
        target_speed2 = self.get_target_speed(r2, current_section_id)  # for r2
        target_speed3 = self.get_target_speed(r3, current_section_id)  # for r3

        # Distances to those lookaheads (+3 bias on close to react slightly earlier)
        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]

        speed_data = []
        # 4) Compute recommended "speed now" for each lookahead
        speed_data.append(self.speed_for_turn(close_distance, target_speed1, current_speed))
        speed_data.append(self.speed_for_turn(mid_distance, target_speed2, current_speed))
        speed_data.append(self.speed_for_turn(far_distance, target_speed3, current_speed))

        # 5) At high speeds, add wider-spaced checks to detect long, sweeping turns
        if current_speed > 100:
            # Skip this extra mid-spaced check in section 9 (track-specific tuning)
            if current_section_id != 9:
                r4 = self.get_radius(
                    [
                        nextWaypoint[self.mid_index],
                        nextWaypoint[self.mid_index + 2],
                        nextWaypoint[self.mid_index + 4],
                    ]
                )
                target_speed4 = self.get_target_speed(r4, current_section_id)
                speed_data.append(
                    self.speed_for_turn(close_distance, target_speed4, current_speed)
                )

            # Wider spacing from close index as well (close, close+3, close+6)
            r5 = self.get_radius(
                [
                    nextWaypoint[self.close_index],
                    nextWaypoint[self.close_index + 3],
                    nextWaypoint[self.close_index + 6],
                ]
            )
            target_speed5 = self.get_target_speed(r5, current_section_id)
            speed_data.append(
                self.speed_for_turn(close_distance, target_speed5, current_speed)
            )

        # 6) Choose the most restrictive recommendation (slowest allowed now)
        update = self.select_speed(speed_data)
        # Cache recommended speed for external logging (km/h)
        try:
            self.last_recommended_speed_kmh = float(update.recommended_speed_now)
        except Exception:
            self.last_recommended_speed_kmh = None

        # 7) Debug print of per-lookahead recommended speeds and current speed
        self.print_speed(
            " -- SPEED: ",
            speed_data[0].recommended_speed_now,
            speed_data[1].recommended_speed_now,
            speed_data[2].recommended_speed_now,
            (0 if len(speed_data) < 4 else speed_data[3].recommended_speed_now),
            current_speed,
        )

        # 8) Map chosen recommendation to throttle/brake (considers recent speed and brake latch)
        throttle, brake = self.speed_data_to_throttle_and_brake(update)
        self.dprint("--- throt " + str(throttle) + " brake " + str(brake) + "---")
        return throttle, brake

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        """
        Converts speed data into throttle and brake values
        """
        # Trail braking (sections 1,4,6,9) with hysteresis, proportional+derivative, and max continuous window
        # Suggestion 1: hysteresis (separate start/stop thresholds)
        # Suggestion 2: proportional brake based on overspeed error
        # Suggestion 3: derivative relief if already decelerating fast
        # Suggestion 4: max continuous brake ticks safety release
        if not hasattr(self, 'trail_brake_active'):     # adding more attributes
            self.trail_brake_active = False
            self.trail_brake_ticks = 0
            self.filtered_recommended = None

        trail_brake_sections = {1, 4, 6, 9}
        section_id = getattr(speed_data, 'current_section_id', None)
        if section_id in trail_brake_sections:
            # Filter (low-pass) the recommended speed to avoid jitter triggering brakes repeatedly
            raw_rec = speed_data.recommended_speed_now
            if self.filtered_recommended is None:
                self.filtered_recommended = raw_rec
            else:
                self.filtered_recommended = 0.7 * self.filtered_recommended + 0.3 * raw_rec

            recommended = self.filtered_recommended
            percent_of_max = speed_data.current_speed / (recommended + 1e-6)
            overspeed_error = max(0.0, percent_of_max - 1.0)

            BRAKE_START = 1.03   # start hysteresis threshold
            BRAKE_STOP = 0.995   # release threshold
            MAX_CONTINUOUS_BRAKE = 12
            TARGET_DECEL_PER_TICK = 2.2  # kph per tick desired heavy braking
            K_ERROR = 8.0         # scales overspeed error to brake
            K_DECEL = 0.2         # adds extra if not decelerating fast enough

            # Activation / deactivation of trail brake state
            if not self.trail_brake_active:
                if percent_of_max >= BRAKE_START:
                    self.trail_brake_active = True
                    self.trail_brake_ticks = 0
            else:
                if percent_of_max <= BRAKE_STOP or overspeed_error < 0.005:
                    self.trail_brake_active = False

            if self.trail_brake_active:
                self.trail_brake_ticks += 1
                actual_decel = self.previous_speed - speed_data.current_speed  # kph per tick (positive when slowing)
                decel_deficit = max(0.0, TARGET_DECEL_PER_TICK - actual_decel)
                # Base proportional brake
                brake_amount = K_ERROR * overspeed_error
                # Add deficit term
                brake_amount += K_DECEL * decel_deficit
                # Derivative relief (already decelerating strongly)
                if actual_decel > TARGET_DECEL_PER_TICK * 1.1:
                    brake_amount *= 0.5
                # Taper near release band
                if percent_of_max < 1.01:
                    # scale down smoothly as we approach BRAKE_STOP
                    denom = (1.01 - BRAKE_STOP + 1e-6)
                    brake_amount *= max(0.0, (percent_of_max - BRAKE_STOP) / denom)
                brake_amount = max(0.0, min(1.0, brake_amount))
                # Max continuous safety release
                if self.trail_brake_ticks > MAX_CONTINUOUS_BRAKE and overspeed_error < 0.02:
                    self.trail_brake_active = False
                    return 0.0, 0.0
                if brake_amount < 0.03:
                    # Too small to matter; release
                    self.trail_brake_active = False
                    return 1.0, 0.0
                return 0.0, brake_amount
            else:
                # Accelerate / neutral outside brake state
                return 1.0, 0.0

        # self.dprint("dist=" + str(round(speed_data.distance_to_section)) + " cs=" + str(round(speed_data.current_speed, 2))
        #             + " ts= " + str(round(speed_data.target_speed_at_distance, 2))
        #             + " maxs= " + str(round(speed_data.recommended_speed_now, 2)) + " pcnt= " + str(round(percent_of_max, 2)))

        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now
        avg_speed_change_per_tick = 2.4  # Speed decrease in kph per tick
        percent_change_per_tick = 0.075  # speed drop for one time-tick of braking
        true_percent_change_per_tick = round(
            avg_speed_change_per_tick / (speed_data.current_speed + 0.001), 5
        )
        speed_up_threshold = 0.8        # changed from 0.9
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        brake_threshold_multiplier = 1.5        # changed from 1.0, won't break as much
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        speed_change = round(speed_data.current_speed - self.previous_speed, 3)

        if percent_of_max > 1:
            # Consider slowing down because current speed > recommended speed
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
            else:       # not SUPER over speed scenario
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
                ):      # just a bit over, still use throttle but gentler throttle
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
        else:       # if must speed up to reach recommended speed
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
                # too slow or got slower
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

    def get_target_speed(self, radius: float, current_section_id: int):
        """Returns a target speed based on the radius of the turn and the section (stable ID) it is in

        Args:
            radius (float): The calculated radius of the turn
            current_section_id (int): Stable section ID of the track the car is in

        Returns:
            float: The maximum speed the car can go around the corner at
        """

        mu = 2.75       # base aggressiveness (bigger == more aggressive)

        if radius >= self.max_radius:
            return self.max_speed

        # Per-section tuning by stable ID (unchanged if you insert new physical sections)
        mu_by_id = {
            0: 4,
            2: 3.37,     # changed from 3.35
            3: 3.35,
            10: 4.0,
            4: 2.85,
            5: 2.9,
            6: 3.3,
            9: 2.2,     # changed from 2.1
        }
        mu = mu_by_id.get(current_section_id, mu)

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
