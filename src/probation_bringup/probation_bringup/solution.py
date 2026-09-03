#!/usr/bin/env python3
"""Autonomous gate navigation for the probation task.

The node runs a small state machine driven by a fixed-rate control loop:

    INIT -> DESCEND -> SEARCH -> ALIGN -> APPROACH -> THROUGH -> DONE
                          ^         |         |          ^
                          |         |         |          | (squared up)
                          |         |         +-> SQUARE_UP
                          +---------+---------+---------+  (detection lost / stale)

SQUARE_UP is the guard against driving into the gate frame: at commit distance
the gate's width/height ratio tells us whether we are looking at it head-on or
from an angle, and we orbit until that ratio is inside the acceptance window
before committing to the blind run through.

Detections arrive asynchronously on the bounding-box topic and are stored as
node state, so every control method reads the latest values through the
`get_gate()` / `get_red_flare()` accessors rather than touching the topic.
"""
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from vision_msgs.msg import BoundingBoxArray
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# Labels published by the detector
GATE = 'gate'
RED_FLARE = 'red_flare'
TRACKED_LABELS = (GATE, RED_FLARE)

# Image centre in normalised bounding-box coordinates (0.0 = edge, 0.5 = centre)
IMAGE_CENTER_X = 0.33
IMAGE_CENTER_Y = 0.33

# Mission states
INIT = 'INIT'
DESCEND = 'DESCEND'
SEARCH = 'SEARCH'
ALIGN = 'ALIGN'
APPROACH = 'APPROACH'
SQUARE_UP = 'SQUARE_UP'
THROUGH = 'THROUGH'
DONE = 'DONE'


def clamp(value, limit):
    """Clamp to +/- limit."""
    return max(-limit, min(limit, value))


@dataclass
class Detection:
    """One detected object, with the derived values the control logic needs."""

    label_name: str
    label_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    stamp: Time          # when this detection was received

    @property
    def x_offset(self):
        """Horizontal error from image centre. Negative = object is left of centre."""
        return self.x - IMAGE_CENTER_X

    @property
    def y_offset(self):
        """Vertical error from image centre. Negative = object is above centre."""
        return self.y - IMAGE_CENTER_Y

    @property
    def d_range(self):
        """Rough inverse-size distance proxy. Bigger box => closer => smaller value."""
        return 1.0 / self.w if self.w > 0.0 else float('inf')

    @property
    def aspect_ratio(self):
        """Box width / height. Tells a head-on gate from one seen at an angle."""
        return self.w / self.h if self.h != 0.0 else float('inf')

    @property
    def area(self):
        return self.w * self.h


class GateNavigator(Node):

    def __init__(self):
        super().__init__('gate_navigator')

        self.declare_parameters(namespace='', parameters=[
            # --- detection handling ---
            # The detector misses frames, so a detection stays usable this long.
            ('detection_timeout', 1.0),
            ('smoothing_alpha', 0.5),        # EMA factor; 1.0 disables smoothing
            ('min_confidence', 0.4),

            # --- depth ---
            ('target_depth', -1.5),          # metres, matches /mavros/global_position/rel_alt
            ('depth_tolerance', 0.3),
            ('alt_gain', 0.5),
            ('max_vertical_speed', 0.4),

            # --- visual servoing ---
            ('yaw_gain', 1.5),
            ('depth_gain', 0.6),             # from the gate's vertical offset in frame
            ('max_yaw_rate', 0.6),
            ('center_tolerance', 0.08),      # |offset| below this counts as centred

            # --- forward motion ---
            ('forward_speed', 0.4),
            ('search_yaw_rate', 0.3),
            ('commit_width', 0.55),          # gate box width at which we commit and drive through
            # Seen from an angle the gate box narrows but keeps its height, so
            # width alone can never reach commit_width. Height is the angle
            # independent distance proxy that stops us creeping into the frame.
            ('commit_height', 0.55),
            ('through_duration', 6.0),       # seconds of blind forward travel through the gate

            # --- squaring up on the gate ---
            # Acceptance window for the gate box w/h. Outside it we are looking
            # at the gate from too steep an angle to fit through the opening.
            ('gate_ratio_min', 0.4),
            ('gate_ratio_max', 1.0),
            ('square_strafe_speed', 0.25),
            ('probe_duration', 2.0),         # seconds of strafing before judging a direction
            ('probe_min_improvement', 0.01), # ratio error must drop by this much to keep going
            ('square_timeout', 25.0),

            # --- obstacle avoidance (optional bonus) ---
            ('avoid_flare', True),
            ('flare_avoid_width', 0.15),     # flare box wider than this counts as close
            ('flare_avoid_x', 0.25),         # and within this |x_offset| counts as in the way
            ('strafe_speed', 0.3),

            # --- timeouts ---
            ('search_timeout', 60.0),
            ('align_timeout', 30.0),
            ('control_rate', 10.0),
        ])

        # Latest detection per label, e.g. {'gate': Detection, 'red_flare': Detection}.
        # This is the shared state every control method reads from.
        self.detections = {}

        # Vehicle telemetry
        self.vehicle_state = None            # mavros_msgs/State
        self.rel_alt = None                  # metres
        self.heading = None                  # degrees

        # Mission state
        self.state = INIT
        self.state_entered = self.get_clock().now()
        self.last_request = self.get_clock().now()
        # Was the gate near-filling the frame when we last saw it?
        self.was_close = False
        # SQUARE_UP bookkeeping: which way we are currently orbiting, and the
        # ratio error at the start of the current probe window.
        self.square_dir = 1.0
        self.probe_stamp = self.get_clock().now()
        self.probe_error = float('inf')
        self.probe_samples = []

        # --- interfaces ---
        self.create_subscription(
            BoundingBoxArray, '/main_camera/detection/bounding_boxes',
            self.bounding_boxes_callback, 10)
        self.create_subscription(State, '/mavros/state', self.state_callback, 10)
        self.create_subscription(
            Float64, '/mavros/global_position/rel_alt', self.rel_alt_callback, 10)
        self.create_subscription(
            Float64, '/mavros/global_position/compass_hdg', self.heading_callback, 10)

        self.publisher_ = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        period = 1.0 / self.param('control_rate')
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info('Gate navigator started')

    def param(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Sensor input
    # ------------------------------------------------------------------

    def bounding_boxes_callback(self, msg):
        if not msg.bounding_boxes:
            self.get_logger().debug('No bounding boxes received')
            return

        # Keep only tracked labels, and only the most confident box per label
        # (the detector can report the same object more than once in a frame).
        best = {}
        for box in msg.bounding_boxes:
            if box.label_name not in TRACKED_LABELS:
                continue
            if box.label_name not in best or box.conf > best[box.label_name].conf:
                best[box.label_name] = box

        now = self.get_clock().now()
        for label_name, box in best.items():
            self.detections[label_name] = self.build_detection(box, now)

    def state_callback(self, msg):
        self.vehicle_state = msg

    def rel_alt_callback(self, msg):
        self.rel_alt = msg.data

    def heading_callback(self, msg):
        self.heading = msg.data

    def build_detection(self, box, stamp):
        """Turn a raw BoundingBox into a Detection, smoothed against the previous one."""
        previous = self.detections.get(box.label_name)
        alpha = self.param('smoothing_alpha')

        # Only smooth against a previous value that is still fresh, otherwise a
        # detection returning after a long gap gets dragged towards a stale one.
        if previous is not None and self.is_fresh(previous, stamp):
            x = self.ema_smoothing(box.x, previous.x, alpha)
            y = self.ema_smoothing(box.y, previous.y, alpha)
            w = self.ema_smoothing(box.w, previous.w, alpha)
            h = self.ema_smoothing(box.h, previous.h, alpha)
        else:
            x, y, w, h = box.x, box.y, box.w, box.h

        return Detection(
            label_name=box.label_name, label_id=box.label_id,
            x=x, y=y, w=w, h=h, conf=box.conf, stamp=stamp)

    @staticmethod
    def ema_smoothing(current_value, previous_value, alpha=0.5):
        return alpha * current_value + (1.0 - alpha) * previous_value

    # ------------------------------------------------------------------
    # Detection access — call these from any other method
    # ------------------------------------------------------------------

    def is_fresh(self, detection, now=None):
        """True if the detection is recent enough to act on."""
        now = now if now is not None else self.get_clock().now()
        age = (now - detection.stamp).nanoseconds * 1e-9
        return age <= self.param('detection_timeout')

    def get_detection(self, label_name, min_conf=None):
        """Latest fresh detection for a label, or None if missing/stale/low confidence."""
        min_conf = self.param('min_confidence') if min_conf is None else min_conf
        detection = self.detections.get(label_name)
        if detection is None or detection.conf < min_conf or not self.is_fresh(detection):
            return None
        return detection

    def get_gate(self, min_conf=None):
        return self.get_detection(GATE, min_conf)

    def get_red_flare(self, min_conf=None):
        return self.get_detection(RED_FLARE, min_conf)

    def clear_detections(self):
        """Drop all stored detections, e.g. when restarting a search."""
        self.detections.clear()

    # ------------------------------------------------------------------
    # Arming / mode
    # ------------------------------------------------------------------

    def ensure_guided_and_armed(self):
        """Request GUIDED mode and arming until the vehicle reports both. Non-blocking."""
        if self.vehicle_state is None:
            self.get_logger().info('Waiting for /mavros/state...', throttle_duration_sec=5.0)
            return False

        # Rate-limit service calls so we do not spam the flight controller
        now = self.get_clock().now()
        if (now - self.last_request).nanoseconds * 1e-9 < 2.0:
            return self.vehicle_state.mode == 'GUIDED' and self.vehicle_state.armed
        self.last_request = now

        if self.vehicle_state.mode != 'GUIDED':
            if self.set_mode_client.service_is_ready():
                request = SetMode.Request(base_mode=0, custom_mode='GUIDED')
                self.set_mode_client.call_async(request)
                self.get_logger().info('Requesting GUIDED mode')
            else:
                self.get_logger().warn('/mavros/set_mode not available yet')
            return False

        if not self.vehicle_state.armed:
            if self.arming_client.service_is_ready():
                self.arming_client.call_async(CommandBool.Request(value=True))
                self.get_logger().info('Requesting arm')
            else:
                self.get_logger().warn('/mavros/cmd/arming not available yet')
            return False

        return True

    # ------------------------------------------------------------------
    # Control primitives
    # ------------------------------------------------------------------

    def keep_gate_in_center(self, vel_cmd, gate):
        """Set yaw and vertical rate to drive the gate towards the frame centre.

        Returns True once the gate is centred within the tolerance.
        """
        # Gate right of centre (x_offset > 0) => yaw right => negative angular.z
        vel_cmd.angular.z = clamp(
            -self.param('yaw_gain') * gate.x_offset, self.param('max_yaw_rate'))
        # Gate below centre (y_offset > 0) => move down => negative linear.z
        vel_cmd.linear.z = clamp(
            -self.param('depth_gain') * gate.y_offset, self.param('max_vertical_speed'))

        tolerance = self.param('center_tolerance')
        return abs(gate.x_offset) < tolerance and abs(gate.y_offset) < tolerance

    def hold_depth(self, vel_cmd):
        """Set vertical rate to hold `target_depth`. Returns True once within tolerance."""
        if self.rel_alt is None:
            return False
        error = self.param('target_depth') - self.rel_alt
        vel_cmd.linear.z = clamp(
            self.param('alt_gain') * error, self.param('max_vertical_speed'))
        return abs(error) < self.param('depth_tolerance')

    def avoid_flare(self, vel_cmd):
        """Strafe away from the flare if it is close and in our path.

        Returns True if an avoidance manoeuvre was applied.
        """
        if not self.param('avoid_flare'):
            return False
        flare = self.get_red_flare()
        if flare is None:
            return False
        if flare.w < self.param('flare_avoid_width'):
            return False                      # still far away
        if abs(flare.x_offset) > self.param('flare_avoid_x'):
            return False                      # off to the side, not in the way

        # Flare on the right (x_offset > 0) => strafe left => positive linear.y
        direction = 1.0 if flare.x_offset > 0 else -1.0
        vel_cmd.linear.y = direction * self.param('strafe_speed')
        self.get_logger().info(
            'Avoiding flare: x_offset=%.2f w=%.2f -> strafe %s'
            % (flare.x_offset, flare.w, 'left' if direction > 0 else 'right'))
        return True

    def ratio_error(self, gate):
        """How far the gate's w/h falls outside the acceptance window. 0.0 = inside."""
        ratio = gate.aspect_ratio
        low = self.param('gate_ratio_min')
        high = self.param('gate_ratio_max')
        if ratio == float('inf'):
            return float('inf')        # degenerate box, treat as unusable
        if ratio < low:
            return low - ratio
        if ratio > high:
            return ratio - high
        return 0.0

    def is_squared(self, gate):
        """True if the gate is seen head-on enough to fit the vehicle through."""
        return self.ratio_error(gate) == 0.0

    def is_at_commit_distance(self, gate):
        """True once the gate box is large enough that we are at the gate face."""
        return (gate.w >= self.param('commit_width')
                or gate.h >= self.param('commit_height'))

    def square_up(self, vel_cmd, gate):
        """Orbit the gate until its w/h ratio is inside the acceptance window.

        w/h is symmetric: it says the view is off-axis but not which side we are
        off from. So we strafe one way for `probe_duration`, keep the direction
        while the ratio error keeps dropping, and flip when it does not.

        Box jitter is about as large as the improvement we are probing for, so
        each window is judged on the mean of its samples rather than on the one
        reading that happens to land at the end of it.

        No forward motion here — the whole point is not to enter the gate at an
        angle. Yaw keeps the gate centred while we translate, which turns the
        strafe into an orbit around it.
        """
        self.keep_gate_in_center(vel_cmd, gate)
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = self.square_dir * self.param('square_strafe_speed')

        error = self.ratio_error(gate)
        if error != float('inf'):
            self.probe_samples.append(error)

        now = self.get_clock().now()
        if (now - self.probe_stamp).nanoseconds * 1e-9 < self.param('probe_duration'):
            return
        if not self.probe_samples:
            self.probe_stamp = now      # nothing usable seen; start a fresh window
            return

        error = sum(self.probe_samples) / len(self.probe_samples)
        self.probe_samples.clear()
        if error > self.probe_error - self.param('probe_min_improvement'):
            self.square_dir = -self.square_dir
            self.get_logger().info(
                'Squaring up: no improvement (%.3f -> %.3f), strafing %s instead'
                % (self.probe_error, error, 'left' if self.square_dir > 0 else 'right'))
        self.probe_stamp = now
        self.probe_error = error

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def transition(self, new_state):
        if new_state == self.state:
            return
        self.get_logger().info('State: %s -> %s' % (self.state, new_state))
        self.state = new_state
        self.state_entered = self.get_clock().now()
        if new_state == SEARCH:
            self.was_close = False
        elif new_state == SQUARE_UP:
            # Start a fresh probe; the infinite baseline keeps the first window
            # from flipping direction before we have measured anything.
            self.probe_stamp = self.state_entered
            self.probe_error = float('inf')
            self.probe_samples = []

    def time_in_state(self):
        return (self.get_clock().now() - self.state_entered).nanoseconds * 1e-9

    def control_loop(self):
        vel_cmd = Twist()

        if self.state == DONE:
            self.publisher_.publish(vel_cmd)   # keep streaming zeros
            return

        # The vehicle only accepts velocity commands in GUIDED mode
        if not self.ensure_guided_and_armed():
            self.publisher_.publish(vel_cmd)
            return

        if self.state == INIT:
            self.transition(DESCEND)

        elif self.state == DESCEND:
            at_depth = self.hold_depth(vel_cmd)
            # If the gate is already visible there is no need to finish descending
            if at_depth or self.get_gate() is not None:
                self.transition(SEARCH)
            elif self.rel_alt is None and self.time_in_state() > 5.0:
                self.get_logger().warn('No rel_alt data; skipping descend')
                self.transition(SEARCH)

        elif self.state == SEARCH:
            self.hold_depth(vel_cmd)
            if self.get_gate() is not None:
                self.transition(ALIGN)
            elif self.time_in_state() > self.param('search_timeout'):
                self.get_logger().error('Gate not found within search timeout')
                self.transition(DONE)
            else:
                # Yaw on the spot to sweep the scene
                vel_cmd.angular.z = self.param('search_yaw_rate')
                self.get_logger().info('Searching for gate...', throttle_duration_sec=2.0)

        elif self.state == ALIGN:
            gate = self.get_gate()
            if gate is None:
                self.transition(SEARCH)
            elif self.keep_gate_in_center(vel_cmd, gate):
                self.transition(APPROACH)
            elif self.time_in_state() > self.param('align_timeout'):
                self.get_logger().warn('Align timed out; re-searching')
                self.transition(SEARCH)

        elif self.state == APPROACH:
            gate = self.get_gate()
            if gate is None:
                # Lost the gate. If it filled the frame we are close enough to
                # commit; otherwise go back to searching.
                self.transition(THROUGH if self.was_close else SEARCH)
            else:
                # Only worth committing blind on a lost gate if the last good
                # view was both near the gate face and square to it.
                near = (gate.w >= self.param('commit_width') * 0.8
                        or gate.h >= self.param('commit_height') * 0.8)
                self.was_close = near and self.is_squared(gate)
                self.keep_gate_in_center(vel_cmd, gate)
                # Slow down as we approach, and stop turning hard while moving
                vel_cmd.linear.x = self.param('forward_speed')
                self.avoid_flare(vel_cmd)

                self.get_logger().info(
                    'Approach: x_off=%.2f y_off=%.2f w=%.2f d_range=%.2f conf=%.2f'
                    % (gate.x_offset, gate.y_offset, gate.w, gate.d_range, gate.conf),
                    throttle_duration_sec=1.0)

                if self.is_at_commit_distance(gate):
                    # Only drive through a gate we are square to, otherwise we
                    # clip the frame on the way in.
                    if self.is_squared(gate):
                        self.transition(THROUGH)
                    else:
                        self.get_logger().info(
                            'At gate but off-axis (w/h=%.2f); squaring up'
                            % gate.aspect_ratio)
                        self.transition(SQUARE_UP)

        elif self.state == SQUARE_UP:
            gate = self.get_gate()
            if gate is None:
                # Never commit blind from an angle we know is bad; find it again.
                self.transition(SEARCH)
            elif self.is_squared(gate):
                self.get_logger().info(
                    'Gate squared up: w/h=%.2f' % gate.aspect_ratio)
                self.transition(THROUGH)
            elif self.time_in_state() > self.param('square_timeout'):
                self.get_logger().warn('Could not square up on the gate; re-searching')
                self.transition(SEARCH)
            else:
                self.square_up(vel_cmd, gate)
                self.avoid_flare(vel_cmd)
                self.get_logger().info(
                    'Squaring up: w/h=%.2f err=%.3f dir=%s'
                    % (gate.aspect_ratio, self.ratio_error(gate),
                       'left' if self.square_dir > 0 else 'right'),
                    throttle_duration_sec=1.0)

        elif self.state == THROUGH:
            # The gate leaves the camera view before we are through it, so this
            # last stretch is open-loop: hold heading and drive straight.
            self.hold_depth(vel_cmd)
            vel_cmd.linear.x = self.param('forward_speed')
            if self.time_in_state() >= self.param('through_duration'):
                self.get_logger().info('Gate traversal complete')
                self.transition(DONE)

        self.publisher_.publish(vel_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Leave the vehicle stationary rather than with the last command latched
        try:
            node.publisher_.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
