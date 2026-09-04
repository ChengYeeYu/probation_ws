#!/usr/bin/env python3
# ---------- Imports ----------
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time

from vision_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# ---------- Constants ----------
IMAGE_CENTER_X = 0.33               # image centre in normalised box coordinates
IMAGE_CENTER_Y = 0.33

TRACKED_LABELS = ['gate', 'red_flare']      # labels the control logic reacts to

# ---------- Detection Data ----------
# One detected object, with the values the control logic derives from it.
@dataclass
class Detected_Object_Data:
    label_name: str
    x: float
    y: float
    w: float
    h: float
    conf: float
    stamp: Time                     # when this detection was received

    # Horizontal error from image centre. Negative means left of centre.
    @property
    def x_offset(self):
        return self.x - IMAGE_CENTER_X

    # Vertical error from image centre. Negative means above centre.
    @property
    def y_offset(self):
        return self.y - IMAGE_CENTER_Y

    # Box width over height. Separates a head-on gate from an angled one.
    @property
    def w_h_ratio(self):
        return self.w / self.h if self.h != 0.0 else float('inf')

# ---------- Mission States ----------
INIT = 'INIT'
DESCEND = 'DESCEND'
SEARCH = 'SEARCH'
APPROACH = 'APPROACH'
ALIGN = 'ALIGN'
THROUGH = 'THROUGH'
DONE = 'DONE'

# Keep a value inside [-limit, limit].
def clamp(value, limit):
    return max(-limit, min(limit, value))

# ---------- Gate Navigator Node ----------
class GateNavigator(Node):

    def __init__(self):
        super().__init__('gate_navigator')

        # ---------- State ----------
        self.detected_objects = {}

        self.vehicle_state = None       # None until the first message arrives
        self.rel_alt = None

        self.state = INIT
        self.state_entered = self.get_clock().now()
        self.last_request = self.get_clock().now()

        self.was_close = False          # last view was close and square, safe to commit blind

        self.align_dir = 1.0            # which way the sweep is currently strafing
        self.probe_samples = []         # recent ratio errors, judged as a sliding window

        self.align_leg = 1              # sweep leg number; each leg runs longer than the last
        self.align_leg_ticks = 0

        self.depth_settle_alt = None    # depth we last saw real progress at
        self.depth_settle_stamp = self.get_clock().now()

        self.centred_ticks = 0          # consecutive ticks squared and centred
        self.squared_ticks = 0          # consecutive ticks meeting the ratio requirement

        self.align_peak_ratio = 0.0     # best w/h this alignment has seen
        self.align_peak_ticks = 0       # how long that peak has stood unbeaten

        self.clear_ticks = 1000         # starts high so a clear approach begins at full speed

        # ---------- Parameters ----------
        self.declare_parameters(namespace='', parameters=[
            # --- detection ---
            ('detection_timeout', 1.0),
            ('smoothing_alpha', 0.5),        # EMA factor; 1.0 disables smoothing
            ('min_confidence', 0.4),

            # --- depth ---
            ('target_depth', -1.6),          # metres, matches /mavros/global_position/rel_alt
            ('depth_tolerance', 0.1),
            ('depth_settle_band', 0.02),     # depth moving less than this counts as settled
            ('depth_settle_time', 0.5),      # held that still for this long ends the descent
            ('alt_gain', 0.5),
            ('max_vertical_speed', 0.8),

            # --- visual servoing ---
            ('yaw_gain', 1.5),
            ('depth_gain', 0.6),             # drives depth from the gate's vertical offset
            ('max_yaw_rate', 0.6),
            ('center_tolerance', 0.1),       # |offset| below this counts as centred

            # --- forward motion ---
            ('forward_speed', 0.6),
            ('search_yaw_rate', 0.8),
            ('commit_width', 0.5),           # box size at which we stop approaching
            ('commit_height', 0.5),
            ('through_duration', 3.0),       # seconds of blind travel through the gate

            # --- aligning with the gate ---
            ('gate_ratio_min', 0.60),        # acceptance window for the gate box w/h
            ('gate_ratio_max', 1.0),
            ('gate_ratio_tolerance', 0.02),  # jitter allowance on that window
            ('align_strafe_speed', 0.3),
            ('probe_duration', 1.0),         # seconds of strafing before judging a direction
            ('probe_min_improvement', 0.01), # error must drop by this much to hold course
            ('align_max_leg', 4),            # legs stop growing here, capping one sweep
            ('align_confirm_ticks', 5),      # ticks the requirement must hold to count
            ('align_peak_fraction', 0.95),   # accept within this fraction of the peak w/h
            ('align_peak_hold_ticks', 1),    # ticks the peak must stand to count as a maximum
            ('commit_hold_ticks', 1),        # ticks squared and centred before committing

            # --- obstacle avoidance ---
            ('avoid_obstacles', True),       # master switch for the whole avoidance path
            ('flare_avoid_width', 0.008),    # box wider than this counts as close
            ('flare_clearance', 0.15),       # berth wanted from a barely-close flare
            ('flare_clearance_gain', 2.0),   # extra berth per unit of box width
            ('strafe_speed', 1.0),           # sideways speed while dodging
            ('avoid_slow_factor', 0.5),      # forward speed retained when avoidance triggers
            ('flare_stop_width', 0.02),      # box width at which forward motion stops entirely
            ('clear_hold_ticks', 1),         # ticks the path must stay clear before full speed

            # --- timeouts ---
            ('search_timeout', 30.0),
            ('approach_timeout', 30.0),
            ('align_timeout', 15.0),
            ('control_rate', 10.0),
        ])

        # ---------- Topics and Services ----------
        self.create_subscription(
            BoundingBoxArray,
            '/main_camera/detection/bounding_boxes',
            self.bounding_boxes_callback,
            10)

        self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10)

        self.create_subscription(
            Float64,
            '/mavros/global_position/rel_alt',
            self.rel_alt_callback,
            10)

        self.publisher_ = self.create_publisher(
            Twist,
            '/mavros/setpoint_velocity/cmd_vel_unstamped',
            10)

        self.set_mode_client = self.create_client(
            SetMode,
            '/mavros/set_mode')

        self.arming_client = self.create_client(
            CommandBool,
            '/mavros/cmd/arming')

        timer_period = 1.0 / self.param('control_rate')
        self.timer = self.create_timer(timer_period, self.control_loop)

    # Read a declared parameter by name.
    def param(self, name):
        return self.get_parameter(name).value

    # ---------- Callbacks ----------
    def state_callback(self, msg):
        self.vehicle_state = msg

    def rel_alt_callback(self, msg):
        self.rel_alt = msg.data

    def bounding_boxes_callback(self, msg):
        if not msg.bounding_boxes:
            self.get_logger().debug('No bounding boxes received')
            return

        # Keep the most confident box per tracked label; the detector can report
        # the same object more than once in a frame.
        most_confident_objects = {}
        for box in msg.bounding_boxes:
            if box.label_name not in TRACKED_LABELS:
                continue
            if box.label_name not in most_confident_objects or box.conf > most_confident_objects[box.label_name].conf:
                most_confident_objects[box.label_name] = box

        now = self.get_clock().now()
        for label_name, box in most_confident_objects.items():
            self.detected_objects[label_name] = self.smooth_detected_objects(box, now)

    # ---------- Detection Smoothing ----------
    # Smooth the box coordinates to reduce detector jitter.
    def smooth_detected_objects(self, box, timestamp):
        previous = self.detected_objects.get(box.label_name)
        alpha = self.param('smoothing_alpha')

        # Only smooth against a fresh previous value, or a detection returning
        # after a gap gets dragged towards a stale one.
        if previous is not None and self.is_fresh(previous, timestamp):
            x = self.ema_smoothing(box.x, previous.x, alpha)
            y = self.ema_smoothing(box.y, previous.y, alpha)
            w = self.ema_smoothing(box.w, previous.w, alpha)
            h = self.ema_smoothing(box.h, previous.h, alpha)
        else:
            x, y, w, h = box.x, box.y, box.w, box.h

        return Detected_Object_Data(
            label_name=box.label_name,
            x=x, y=y, w=w, h=h, conf=box.conf, stamp=timestamp)

    # True while a detection is younger than detection_timeout.
    def is_fresh(self, detected_object, now=None):
        now = now if now is not None else self.get_clock().now()
        age = (now - detected_object.stamp).nanoseconds * 1e-9
        return age <= self.param('detection_timeout')

    # Exponential moving average of one coordinate.
    @staticmethod
    def ema_smoothing(current_value, previous_value, alpha=0.5):
        return alpha * current_value + (1.0 - alpha) * previous_value

    # Latest usable detection for a label, or None if missing, stale or low confidence.
    def get_detected_object(self, label_name):
        min_conf = self.param('min_confidence')
        detected_object = self.detected_objects.get(label_name)
        if detected_object is None or detected_object.conf < min_conf or not self.is_fresh(detected_object):
            return None
        return detected_object

    # ---------- Vehicle Control ----------
    # Put the vehicle in GUIDED and arm it. True once both hold.
    def ensure_guided_and_armed(self):
        if self.vehicle_state is None:
            self.get_logger().info('Waiting for /mavros/state...', throttle_duration_sec=5.0)
            return False

        # Rate-limit service calls so we do not spam the flight controller.
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

    # Hold the target depth with a P controller. True once the descent is done,
    # either inside depth_tolerance or settled as deep as it will get.
    def hold_depth(self, vel_cmd):
        if self.rel_alt is None:
            return False

        error = self.param('target_depth') - self.rel_alt
        vel_cmd.linear.z = clamp(
            self.param('alt_gain') * error, self.param('max_vertical_speed'))

        if abs(error) < self.param('depth_tolerance'):
            return True
        return self.descent_has_settled()

    # True once depth stops changing. A P controller settles where thrust balances
    # buoyancy, and that point can sit outside depth_tolerance forever.
    def descent_has_settled(self):
        if self.rel_alt is None:
            return False

        now = self.get_clock().now()
        # Still making progress, so restart the window.
        if (self.depth_settle_alt is None
                or abs(self.rel_alt - self.depth_settle_alt) > self.param('depth_settle_band')):
            self.depth_settle_alt = self.rel_alt
            self.depth_settle_stamp = now
            return False

        return (now - self.depth_settle_stamp).nanoseconds * 1e-9 > self.param('depth_settle_time')

    # Yaw and dive to bring the gate towards the image centre.
    # True once it is centred within center_tolerance.
    def keep_gate_in_center(self, vel_cmd, gate):
        # Gate right of centre yaws right, which is a negative angular.z.
        vel_cmd.angular.z = clamp(
            -self.param('yaw_gain') * gate.x_offset, self.param('max_yaw_rate'))
        # Gate below centre moves down, which is a negative linear.z.
        vel_cmd.linear.z = clamp(
            -self.param('depth_gain') * gate.y_offset, self.param('max_vertical_speed'))

        tolerance = self.param('center_tolerance')
        return abs(gate.x_offset) < tolerance and abs(gate.y_offset) < tolerance

    # How far the gate's w/h falls outside the acceptance window. 0.0 means inside.
    def ratio_error(self, detected_object):
        ratio = detected_object.w_h_ratio
        low = self.param('gate_ratio_min')
        high = self.param('gate_ratio_max')

        if ratio == float('inf'):
            return float('inf')     # degenerate box, treat it as unusable
        if ratio < low:
            return low - ratio
        if ratio > high:
            return ratio - high
        return 0.0

    # True if we are square enough to the gate to fit through the opening.
    # The tolerance absorbs box jitter, since w/h also drifts with range.
    def can_pass_through(self, detected_object):
        return self.ratio_error(detected_object) <= self.param('gate_ratio_tolerance')

    # True once the requirement has held long enough that one noisy frame cannot
    # have caused it. There are two ways to meet it: the absolute window, and a
    # peak-relative test. w/h is largest head-on, so the sweep's best reading
    # measures head-on without us knowing its value in advance, which keeps the
    # requirement from being set out of reach.
    def alignment_confirmed(self, detected_object):
        ratio = detected_object.w_h_ratio
        if ratio != float('inf'):
            if ratio > self.align_peak_ratio:
                self.align_peak_ratio = ratio
                self.align_peak_ticks = 0
            else:
                self.align_peak_ticks = self.align_peak_ticks + 1

        met = self.can_pass_through(detected_object)

        # The peak only counts once it has stood unbeaten. While the sweep is still
        # climbing every reading is the best yet, so comparing against the running
        # best would pass at any angle.
        if (not met
                and ratio != float('inf')
                and self.align_peak_ratio > 0.0
                and self.align_peak_ticks >= self.param('align_peak_hold_ticks')):
            met = ratio >= self.align_peak_ratio * self.param('align_peak_fraction')

        if met:
            self.squared_ticks = self.squared_ticks + 1
        else:
            self.squared_ticks = 0
        return self.squared_ticks >= self.param('align_confirm_ticks')

    # True once the gate box is big enough that we have reached the gate face.
    def is_at_commit_distance(self, detected_object):
        return (detected_object.w >= self.param('commit_width')
                or detected_object.h >= self.param('commit_height'))

    # Orbit the gate until its w/h meets the requirement. w/h is symmetric, so it
    # gives the size of the error but not its sign. We strafe one way, hold while
    # the error drops, and turn around when it does not. Each leg runs longer than
    # the last, since equal legs would only retrace the previous arc.
    def align_with_gate(self, vel_cmd, detected_object):
        # Yaw holds the gate centred while we translate, turning the strafe into an
        # orbit. No forward motion, the point is not to enter the gate at an angle.
        self.keep_gate_in_center(vel_cmd, detected_object)
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = self.align_dir * self.param('align_strafe_speed')

        error = self.ratio_error(detected_object)
        if error != float('inf'):
            self.probe_samples.append(error)

        # Keep only the most recent probe_duration seconds of readings.
        window = int(self.param('probe_duration') * self.param('control_rate'))
        if window < 2:
            window = 2
        if len(self.probe_samples) > window:
            self.probe_samples = self.probe_samples[-window:]

        # A leg runs its full length before we may turn around, which also gives a
        # new direction time to show an effect.
        self.align_leg_ticks = self.align_leg_ticks + 1
        if self.align_leg_ticks < self.align_leg * window:
            return

        # The two halves only mean something once the window is full.
        if len(self.probe_samples) < window:
            return

        # Compare the newer half against the older half. Averaging rides out the box
        # jitter, and the sliding window re-judges every tick rather than at a fixed
        # boundary, so a direction that stops helping is caught as the trend turns.
        half = window // 2
        older = sum(self.probe_samples[:half]) / half
        newer = sum(self.probe_samples[half:]) / (window - half)

        improvement = older - newer
        if improvement < self.param('probe_min_improvement'):
            self.align_dir = -self.align_dir
            self.align_leg = min(self.align_leg + 1, self.param('align_max_leg'))
            self.align_leg_ticks = 0
            self.probe_samples = []
            if self.align_dir > 0:
                side = 'left'
            else:
                side = 'right'
            self.get_logger().info(
                'Align: no improvement (%.3f -> %.3f, gain %+.3f), strafing %s for %.1fs'
                % (older, newer, improvement, side,
                   self.align_leg * self.param('probe_duration')))
        else:
            self.get_logger().info(
                'Align: improving (%.3f -> %.3f, gain %+.3f), holding course'
                % (older, newer, improvement),
                throttle_duration_sec=1.0)

    # ---------- Obstacle Avoidance ----------
    # Clear water between our heading and the object's near edge, in normalised
    # image widths. Negative means our heading falls inside the box.
    @staticmethod
    def edge_gap(detected_object):
        return abs(detected_object.x_offset) - detected_object.w / 2.0

    # How much of that clear water we insist on. Box width stands in for distance,
    # so a wider box is closer and earns a wider berth.
    def required_clearance(self, detected_object):
        return (self.param('flare_clearance')
                + self.param('flare_clearance_gain') * detected_object.w)

    # How much forward speed survives while an obstacle is near. Fades from
    # avoid_slow_factor at the trigger width to a dead stop at flare_stop_width.
    def avoid_speed_factor(self):
        flare = self.get_detected_object('red_flare')
        if flare is None:
            return self.param('avoid_slow_factor')

        near = self.param('flare_avoid_width')
        stop = self.param('flare_stop_width')
        if flare.w >= stop:
            return 0.0
        # Guards the far end and a degenerate window that would divide by zero.
        if flare.w <= near or stop <= near:
            return self.param('avoid_slow_factor')
        return self.param('avoid_slow_factor') * (stop - flare.w) / (stop - near)

    # Strafe around the red flare when it is close and roughly ahead.
    # True if a manoeuvre was applied. Only touches linear.y, leaving yaw free.
    def avoid_flare(self, vel_cmd):
        # Record that avoidance was consulted, so the watch can tell "looked and saw
        # nothing" apart from "this state never asked".
        self.avoid_ran = True
        self.avoid_active = False

        # Switched off, report nothing in the way and leave linear.y untouched.
        if not self.param('avoid_obstacles'):
            return False

        flare = self.get_detected_object('red_flare')
        if flare is None:
            return False

        if flare.w < self.param('flare_avoid_width'):
            return False            # still far away

        # What matters is the near edge, not the centre: a narrow flare needs only a
        # nudge to clear, a wide one needs room. Both sides of this scale with the
        # box, so one pair of parameters covers near and far.
        if self.edge_gap(flare) > self.required_clearance(flare):
            return False            # far enough to the side, not in the way

        # Flare on the right means strafe left, which is a positive linear.y.
        if flare.x_offset > 0:
            direction = 1.0
        else:
            direction = -1.0
        vel_cmd.linear.y = direction * self.param('strafe_speed')

        if direction > 0:
            side = 'left'
        else:
            side = 'right'
        self.get_logger().info(
            'Avoiding flare: x_offset=%.2f w=%.2f -> strafe %s'
            % (flare.x_offset, flare.w, side))
        self.avoid_active = True
        return True

    # ---------- Avoidance Watch ----------
    # One short line per tick: the state, what the sub was told to do, and what
    # avoidance decided. Uses the stored detection so a flare that is seen but
    # rejected still shows its numbers.
    def avoidance_watch(self, vel_cmd, tick_state):
        flare = self.detected_objects.get('red_flare')
        if flare is None:
            seen = 'no flare'
        else:
            seen = ('flare w=%.3f gap=%+.2f/%.2f'
                    % (flare.w, self.edge_gap(flare), self.required_clearance(flare)))

        if not self.param('avoid_obstacles'):
            decision = 'off'
        elif not self.avoid_ran:
            decision = 'not checked'
        elif self.avoid_active:
            decision = 'DODGING'
        else:
            decision = 'clear'

        self.get_logger().info(
            '[%s] vx=%.2f vy=%+.2f wz=%+.2f | %s | %s'
            % (tick_state, vel_cmd.linear.x, vel_cmd.linear.y, vel_cmd.angular.z,
               seen, decision),
            throttle_duration_sec=0.5)

    # ---------- State Machine ----------
    # Move to a new state and reset whatever that state needs on entry.
    def transition(self, new_state):
        if new_state == self.state:
            return

        self.get_logger().info('State: %s -> %s' % (self.state, new_state))
        self.state = new_state
        self.state_entered = self.get_clock().now()

        if new_state == SEARCH:
            self.was_close = False
        elif new_state == DESCEND:
            self.depth_settle_alt = None
            self.depth_settle_stamp = self.state_entered
        elif new_state == APPROACH:
            self.clear_ticks = 1000     # assume clear until an obstacle says otherwise
        elif new_state == ALIGN:
            self.probe_samples = []
            self.align_leg = 1
            self.align_leg_ticks = 0
            self.centred_ticks = 0
            self.squared_ticks = 0
            self.align_peak_ratio = 0.0
            self.align_peak_ticks = 0

    # Seconds spent in the current state.
    def time_in_state(self):
        return (self.get_clock().now() - self.state_entered).nanoseconds * 1e-9

    # ---------- Control Loop ----------
    def control_loop(self):
        vel_cmd = Twist()

        # Reset each tick; avoid_flare sets these when a state calls it.
        self.avoid_ran = False
        self.avoid_active = False
        tick_state = self.state         # self.state may change before we log

        if self.state == DONE:
            self.publisher_.publish(vel_cmd)   # keep streaming zeros
            return

        # The vehicle only accepts velocity commands in GUIDED mode.
        if not self.ensure_guided_and_armed():
            self.publisher_.publish(vel_cmd)
            return

        # Wait for the first depth reading, then start descending.
        if self.state == INIT:
            self.transition(DESCEND)

        # Descend to the target depth, and move on once the descent finishes.
        elif self.state == DESCEND:
            finished = self.hold_depth(vel_cmd)

            # Show what the depth controller sees, so a descent that never finishes
            # can be told apart from one that never started.
            self.get_logger().info(
                'Descend: rel_alt=%s target=%.2f cmd_z=%.2f'
                % (self.rel_alt, self.param('target_depth'), vel_cmd.linear.z),
                throttle_duration_sec=1.0)

            if finished:
                self.get_logger().info(
                    'Descend finished at rel_alt=%.2f (target %.2f), searching'
                    % (self.rel_alt, self.param('target_depth')))
                self.transition(SEARCH)

        # Hold depth and yaw on the spot until the gate appears.
        elif self.state == SEARCH:
            self.hold_depth(vel_cmd)
            if self.get_detected_object('gate') is not None:
                self.transition(APPROACH)
            elif self.time_in_state() > self.param('search_timeout'):
                self.get_logger().error('Gate not found within search timeout')
                self.transition(DONE)
            else:
                vel_cmd.angular.z = self.param('search_yaw_rate')
                self.get_logger().info('Searching for gate...', throttle_duration_sec=2.0)

        # Drive at the gate, keeping it centred and dodging the flare.
        elif self.state == APPROACH:
            gate = self.get_detected_object('gate')
            if gate is None:
                # Lost it. If the last view was close and square we are at the face
                # and should commit, otherwise go back to searching.
                if self.was_close:
                    self.transition(THROUGH)
                else:
                    self.transition(SEARCH)
            elif self.time_in_state() > self.param('approach_timeout'):
                self.get_logger().warn('Approach timed out, re-searching')
                self.transition(SEARCH)
            else:
                # Would committing blind still be safe if we lost the gate next tick?
                near = (gate.w >= self.param('commit_width') * 0.8
                        or gate.h >= self.param('commit_height') * 0.8)
                self.was_close = near and self.can_pass_through(gate)

                # Centring runs every tick, otherwise the gate drifts out of frame.
                blocked = self.avoid_flare(vel_cmd)
                self.keep_gate_in_center(vel_cmd, gate)

                # One clear frame is usually jitter, so full speed only returns
                # after several in a row.
                if blocked:
                    self.clear_ticks = 0
                else:
                    self.clear_ticks = self.clear_ticks + 1

                # Forward speed for this tick. While actually strafing the factor
                # falls to zero, so what is set here is the ramp back in.
                if self.clear_ticks >= self.param('clear_hold_ticks'):
                    vel_cmd.linear.x = self.param('forward_speed')
                else:
                    factor = self.avoid_speed_factor()
                    vel_cmd.linear.x = self.param('forward_speed') * factor
                    self.get_logger().info(
                        'Obstacle in the way, easing off to %.0f%% (clear for %d/%d ticks)'
                        % (factor * 100.0, self.clear_ticks, self.param('clear_hold_ticks')),
                        throttle_duration_sec=1.0)

                self.get_logger().info(
                    'Approach: x_off=%.2f y_off=%.2f w=%.2f h=%.2f w/h=%.2f conf=%.2f'
                    % (gate.x_offset, gate.y_offset, gate.w, gate.h,
                       gate.w_h_ratio, gate.conf),
                    throttle_duration_sec=1.0)

                # Close enough for w/h to be trustworthy, so stop and square up.
                if self.is_at_commit_distance(gate):
                    self.transition(ALIGN)

        # Orbit until square to the gate, then centre it and commit.
        elif self.state == ALIGN:
            gate = self.get_detected_object('gate')
            if gate is None:
                # Never commit from an angle we know is bad.
                self.transition(SEARCH)
            elif self.time_in_state() > self.param('align_timeout'):
                self.get_logger().warn('Could not align with the gate, re-searching')
                self.transition(SEARCH)
            elif self.alignment_confirmed(gate):
                # Square now, but orbiting has left us off to one side. Stop strafing
                # and centre before committing, or the blind run clips the frame.
                # Avoidance runs here too, since this is the branch we always take.
                blocked = self.avoid_flare(vel_cmd)
                centred = self.keep_gate_in_center(vel_cmd, gate)

                # An obstacle resets the hold, so the blind run never starts with
                # something still in front of us.
                if centred and not blocked:
                    self.centred_ticks = self.centred_ticks + 1
                else:
                    self.centred_ticks = 0

                self.get_logger().info(
                    'Squared, centring: w/h=%.2f x_off=%.2f y_off=%.2f held=%d blocked=%s'
                    % (gate.w_h_ratio, gate.x_offset, gate.y_offset,
                       self.centred_ticks, blocked),
                    throttle_duration_sec=1.0)

                # Hold it there so we do not commit on one lucky frame.
                if self.centred_ticks >= self.param('commit_hold_ticks'):
                    self.get_logger().info(
                        'Gate aligned and centred: w/h=%.2f' % gate.w_h_ratio)
                    self.transition(THROUGH)
            else:
                self.centred_ticks = 0
                self.align_with_gate(vel_cmd, gate)
                self.avoid_flare(vel_cmd)

                if self.align_dir > 0:
                    side = 'left'
                else:
                    side = 'right'
                self.get_logger().info(
                    'Align: w/h=%.2f err=%.3f peak=%.2f need>=%.2f met=%d/%d dir=%s'
                    % (gate.w_h_ratio, self.ratio_error(gate), self.align_peak_ratio,
                       self.align_peak_ratio * self.param('align_peak_fraction'),
                       self.squared_ticks, self.param('align_confirm_ticks'), side),
                    throttle_duration_sec=1.0)

        # Open loop on purpose. Passing the gate plane pushes the box to the frame
        # edge, and a detection stays usable for detection_timeout after the last
        # sighting, so steering here would yaw hard at whatever offset it froze on.
        elif self.state == THROUGH:
            self.hold_depth(vel_cmd)
            vel_cmd.linear.x = self.param('forward_speed')
            # Open loop on the gate, but not blind to obstacles.
            self.avoid_flare(vel_cmd)
            if self.time_in_state() >= self.param('through_duration'):
                self.get_logger().info('Gate traversal complete')
                self.transition(DONE)

        # Report what avoidance saw and did on this tick, in every state.
        self.avoidance_watch(vel_cmd, tick_state)
        self.publisher_.publish(vel_cmd)

# ---------- Main ----------
def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Leave the vehicle stopped rather than with the last command latched.
        try:
            node.publisher_.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
