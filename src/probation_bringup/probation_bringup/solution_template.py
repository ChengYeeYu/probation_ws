#!/usr/bin/env python3
# ------------------------------------ Imports ------------------------------------ #
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time

# Import necessary message types
from vision_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# ------------------------------------ Constants ------------------------------------ #
# Image centre in normalised bounding-box coordinates (0.0 = edge, 0.5 = centre)
IMAGE_CENTER_X = 0.33
IMAGE_CENTER_Y = 0.33

# Labels to track; can be extended to include other objects
TRACKED_LABELS = ['gate', 'red_flare']

# ------------------------------------ Bounding Boxes Data Class ------------------------------------ #
@dataclass
class Detected_Object_Data:
    """One detected object, with the derived values the control logic needs."""
    label_name: str
    label_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    stamp: Time          # when this detection was received
    raw: object          # the unsmoothed BoundingBox, kept for the debug printout

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
        return 1.0 / self.h if self.h > 0.0 else float('inf')

    @property
    def w_h_ratio(self):
        """Box width / height. Tells a head-on gate from one seen at an angle."""
        return self.w / self.h if self.h != 0.0 else float('inf')

# ------------------------------------ Mission states ------------------------------------ #
# APPROACH drives up to the gate, then ALIGN squares us up with it before we commit.
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

# ------------------------------------ Gate Navigator Node ------------------------------------ #
class GateNavigator(Node):

    def __init__(self):
        super().__init__('gate_navigator')
        
        # ---------------- State Variables ----------------
        self.detected_objects = {}

        # Vehicle telemetry (None until the first message arrives)
        self.vehicle_state = None
        self.rel_alt = None
        self.heading = None

        # ---------------- Mission State ----------------
        self.state = INIT
        self.state_entered = self.get_clock().now()
        self.last_request = self.get_clock().now()

        # True when the gate last looked both close and square to us, so losing
        # the detection now means we are at the face and should commit.
        self.was_close = False

        # ALIGN bookkeeping: which way we are strafing, and the ratio error
        # measured over the last probe window.
        self.align_dir = 1.0
        self.probe_stamp = self.get_clock().now()
        self.probe_error = float('inf')
        self.probe_samples = []

        # DESCEND bookkeeping: the depth we last saw real progress at, and when.
        self.depth_settle_alt = None
        self.depth_settle_stamp = self.get_clock().now()

        # How many ticks in a row the gate has been both squared and centred.
        self.centred_ticks = 0
        
        
        # ---------------- Parameters Setup ----------------
        self.declare_parameters(namespace='', parameters=[
            # --- detection ---
            ('detection_timeout', 1.0),
            ('smoothing_alpha', 0.5),        # EMA factor; 1.0 disables smoothing
            ('min_confidence', 0.4),
            
            # --- depth ---
            ('target_depth', -1.6),          # metres, matches /mavros/global_position/rel_alt
            ('depth_tolerance', 0.1),
            ('depth_settle_band', 0.02),     # depth moving less than this counts as settled
            ('depth_settle_time', 1.0),      # ...for this long, and the descent is finished
            ('alt_gain', 0.5),
            ('max_vertical_speed', 0.5),
            
            # --- visual servoing ---
            ('yaw_gain', 1.5),
            ('depth_gain', 0.6),             # from the gate's vertical offset in frame
            ('max_yaw_rate', 0.6),
            ('center_tolerance', 0.1),       # |offset| below this counts as centred

            # --- forward motion ---
            ('forward_speed', 0.5),
            ('search_yaw_rate', 0.3),
            ('commit_width', 0.5),           # gate box size at which we stop approaching
            ('commit_height', 0.5),          # height still grows when the gate looks narrow
            ('through_duration', 6.0),       # seconds of blind forward travel through the gate

            # --- aligning with the gate ---
            # Acceptance window for the gate box w/h. Outside it we are looking at
            # the gate from too steep an angle to fit through the opening.
            ('gate_ratio_min', 0.75),        # head-on measured at ~0.85 in flight
            ('gate_ratio_max', 1.0),
            ('align_strafe_speed', 0.25),
            ('probe_duration', 2.0),         # seconds of strafing before judging a direction
            ('probe_min_improvement', 0.01), # ratio error must drop by this much to keep going
            ('commit_hold_ticks', 5),        # ticks squared AND centred before committing

            # --- obstacle avoidance ---
            ('flare_avoid_width', 0.05),     # flare box wider than this counts as close
            ('flare_avoid_x', 0.2),         # and within this |x_offset| counts as in the way
            ('strafe_speed', 0.5),

            # --- timeouts ---
            ('descend_timeout', 60.0),   # backstop only; descent_has_settled is the normal exit
            ('search_timeout', 60.0),
            ('approach_timeout', 45.0),
            ('align_timeout', 25.0),
            ('control_rate', 10.0),
        ])

        # ---------------- Establish connections to the necessary topics and services ----------------
        # Create a subscription for bounding boxes
        self.create_subscription(
            BoundingBoxArray, 
            '/main_camera/detection/bounding_boxes',
            self.bounding_boxes_callback, 
            10)
        
        # Create subscriptions for state
        self.create_subscription(
            State, 
            '/mavros/state', 
            self.state_callback, 
            10)
        
        # Create subscriptions for relative altitude
        self.create_subscription(
            Float64, 
            '/mavros/global_position/rel_alt', 
            self.rel_alt_callback, 
            10)
        
        # Create subscription for heading
        self.create_subscription(
            Float64, 
            '/mavros/global_position/compass_hdg', 
            self.heading_callback, 
            10)

        # Create a publisher for velocity commands
        self.publisher_ = self.create_publisher(
            Twist, 
            '/mavros/setpoint_velocity/cmd_vel_unstamped', 
            10)

        # Create clients for setting mode
        self.set_mode_client = self.create_client(
            SetMode, 
            '/mavros/set_mode')
        
        # Create a client for arming the drone
        self.arming_client = self.create_client(
            CommandBool, 
            '/mavros/cmd/arming')
       
        timer_period = 1.0 / self.param('control_rate')
        self.timer = self.create_timer(timer_period, self.control_loop)
        

    # ------------------------------------ Parameters Callback ------------------------------------ #
    def param(self, name):
        return self.get_parameter(name).value
        
    # ------------------------------------ Callback Functions ------------------------------------ #
    def state_callback(self, msg):
        self.vehicle_state = msg

    def rel_alt_callback(self, msg):
        self.rel_alt = msg.data

    def heading_callback(self, msg):
        self.heading = msg.data
        
    def bounding_boxes_callback(self, msg):
        if not msg.bounding_boxes:
            self.get_logger().debug('No bounding boxes received')
            return

        # Keep only tracked labels, and only the most confident box per label
        # (the detector can report the same object more than once in a frame).
        most_confident_objects = {}
        for box in msg.bounding_boxes:
            if box.label_name not in TRACKED_LABELS:
                continue
            if box.label_name not in most_confident_objects or box.conf > most_confident_objects[box.label_name].conf:
                most_confident_objects[box.label_name] = box

        # Build detected_objects objects for each tracked label
        now = self.get_clock().now()
        for label_name, box in most_confident_objects.items():
            self.detected_objects[label_name] = self.smooth_detected_objects(box, now)

    # ------------------------------------ Detection Smoothing ------------------------------------ #
    # Smoothing is applied to the bounding box coordinates to reduce jitter in the detection results.
    def smooth_detected_objects(self, box, timestamp):
        previous = self.detected_objects.get(box.label_name)
        alpha = self.param('smoothing_alpha')

        # Only smooth against a previous value that is still fresh, otherwise a detection returning after a long gap gets dragged towards a stale one.
        if previous is not None and self.is_fresh(previous, timestamp):
            x = self.ema_smoothing(box.x, previous.x, alpha)
            y = self.ema_smoothing(box.y, previous.y, alpha)
            w = self.ema_smoothing(box.w, previous.w, alpha)
            h = self.ema_smoothing(box.h, previous.h, alpha)
        else:
            x, y, w, h = box.x, box.y, box.w, box.h

        return Detected_Object_Data(
            label_name=box.label_name, label_id=box.label_id,
            x=x, y=y, w=w, h=h, conf=box.conf, stamp=timestamp, raw=box)
    
    # Check if a detection is still fresh based on the detection timeout parameter.
    def is_fresh(self, detected_object, now=None):
        now = now if now is not None else self.get_clock().now()
        age = (now - detected_object.stamp).nanoseconds * 1e-9
        return age <= self.param('detection_timeout')

    # EMA smoothing function to reduce jitter in the detection results.
    @staticmethod
    def ema_smoothing(current_value, previous_value, alpha=0.5):
        return alpha * current_value + (1.0 - alpha) * previous_value

    # Retrieve the latest fresh detection for a given label, or None if the detection is missing, stale, or has low confidence.
    def get_detected_object(self, label_name):
        min_conf = self.param('min_confidence') 
        detected_object = self.detected_objects.get(label_name)
        if detected_object is None or detected_object.conf < min_conf or not self.is_fresh(detected_object):
            return None
        return detected_object    

    # ------------------------------------ Vehicle Control ------------------------------------ #
    # TODO: Clean this up
    # Ensure the vehicle is in GUIDED mode and armed. Returns True if both conditions are met, False otherwise.
    def ensure_guided_and_armed(self):
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
    
    # Hold a target depth with a proportional controller. Returns True once the
    # descent is finished, which means either we are inside depth_tolerance, or the
    # depth has stopped changing and this is as deep as a P controller will get us.
    def hold_depth(self, vel_cmd):
        if self.rel_alt is None:
            return False

        error = self.param('target_depth') - self.rel_alt
        vel_cmd.linear.z = clamp(
            self.param('alt_gain') * error, self.param('max_vertical_speed'))

        # Reached the target, nothing left to wait for
        if abs(error) < self.param('depth_tolerance'):
            return True
        return self.descent_has_settled()
    
    # True once the depth has stopped changing, meaning we are as deep as this
    # controller is going to get us. hold_depth is a plain P controller, so it
    # settles wherever thrust balances buoyancy, and that resting point can sit
    # outside depth_tolerance forever.
    def descent_has_settled(self):
        if self.rel_alt is None:
            return False

        now = self.get_clock().now()
        # Still making real progress, so restart the window
        if (self.depth_settle_alt is None
                or abs(self.rel_alt - self.depth_settle_alt) > self.param('depth_settle_band')):
            self.depth_settle_alt = self.rel_alt
            self.depth_settle_stamp = now
            return False

        return (now - self.depth_settle_stamp).nanoseconds * 1e-9 > self.param('depth_settle_time')
    
    # Yaw and rise/dive to bring the gate towards the centre of the image.
    # Returns True once the gate is centred within center_tolerance.
    def keep_gate_in_center(self, vel_cmd, gate):
        # Gate right of centre (x_offset > 0) => yaw right => negative angular.z
        vel_cmd.angular.z = clamp(
            -self.param('yaw_gain') * gate.x_offset, self.param('max_yaw_rate'))
        # Gate below centre (y_offset > 0) => move down => negative linear.z
        vel_cmd.linear.z = clamp(
            -self.param('depth_gain') * gate.y_offset, self.param('max_vertical_speed'))

        tolerance = self.param('center_tolerance')
        return abs(gate.x_offset) < tolerance and abs(gate.y_offset) < tolerance

    # How far the gate's w/h ratio falls outside the acceptance window. 0.0 means inside it.
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

    # TODO: Make a tolerance
    # True if we are looking at the gate head-on enough to fit through the opening.
    def can_pass_through(self, detected_object):
        return self.ratio_error(detected_object) == 0.0

    # True once the gate box is big enough that we have reached the gate face.
    def is_at_commit_distance(self, detected_object):
        return (detected_object.w >= self.param('commit_width')
                or detected_object.h >= self.param('commit_height'))

    # Orbit the gate until its w/h ratio is inside the acceptance window.
    # w/h is symmetric, so it tells us we are off-axis but not which side we are off from.
    # We strafe one way for probe_duration, keep that direction while the error keeps
    # dropping, and flip when it does not. Box jitter is about as big as the improvement
    # we are looking for, so each window is judged on the mean of its samples instead of
    # the single reading that happens to land at the end of it.
    def align_with_gate(self, vel_cmd, detected_object):
        # Yaw keeps the gate centred while we translate, which turns the strafe into
        # an orbit around it. No forward motion here, the whole point is not to enter
        # the gate at an angle.
        self.keep_gate_in_center(vel_cmd, detected_object)
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = self.align_dir * self.param('align_strafe_speed')

        error = self.ratio_error(detected_object)
        if error != float('inf'):
            self.probe_samples.append(error)

        # Keep collecting until the probe window is over
        now = self.get_clock().now()
        if (now - self.probe_stamp).nanoseconds * 1e-9 < self.param('probe_duration'):
            return
        if not self.probe_samples:
            self.probe_stamp = now      # nothing usable seen, start a fresh window
            return

        # Judge the window on its average, then compare it with the previous one
        error = sum(self.probe_samples) / len(self.probe_samples)
        self.probe_samples = []

        if error > self.probe_error - self.param('probe_min_improvement'):
            self.align_dir = -self.align_dir
            if self.align_dir > 0:
                side = 'left'
            else:
                side = 'right'
            self.get_logger().info(
                'Align: no improvement (%.3f -> %.3f), strafing %s instead'
                % (self.probe_error, error, side))

        self.probe_stamp = now
        self.probe_error = error
    
    # Strafe sideways to get around the red flare when it is close and roughly ahead of us.
    # Returns True if an avoidance manoeuvre was applied.
    # Note this only touches linear.y, so yaw is still free to keep the gate centred.
    def avoid_flare(self, vel_cmd):
        flare = self.get_detected_object('red_flare')
        if flare is None:
            return False
        if flare.w < self.param('flare_avoid_width'):
            return False            # still far away
        if abs(flare.x_offset) > self.param('flare_avoid_x'):
            return False            # off to the side, not in the way

        # Flare on the right (x_offset > 0) => strafe left => positive linear.y
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
        return True


    # ------------------------------------ State Machine ------------------------------------ #
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
        elif new_state == ALIGN:
            # Start a fresh probe. The infinite baseline stops the first window
            # from flipping direction before we have measured anything.
            self.probe_stamp = self.state_entered
            self.probe_error = float('inf')
            self.probe_samples = []
            self.centred_ticks = 0

    # How long we have been in the current state, in seconds.
    def time_in_state(self):
        return (self.get_clock().now() - self.state_entered).nanoseconds * 1e-9

    # ------------------------------------ Control Loop ------------------------------------ #
    def control_loop(self):
        vel_cmd = Twist()

        if self.state == DONE:
            self.publisher_.publish(vel_cmd)   # keep streaming zeros
            return

        # The vehicle only accepts velocity commands in GUIDED mode
        if not self.ensure_guided_and_armed():
            self.publisher_.publish(vel_cmd)
            return

        # The INIT state is where the vehicle waits for the first depth reading to arrive, then transitions to DESCEND.
        if self.state == INIT:
            self.transition(DESCEND)

        # The DESCEND state is where the vehicle descends to a target depth. It uses a proportional controller to hold the target depth. If the descent has settled (i.e., the depth has stopped changing), it transitions to the SEARCH state.
        elif self.state == DESCEND:
            # hold_depth reports when the descent is done, either by reaching the
            # target or by settling as close to it as it can get.
            finished = self.hold_depth(vel_cmd)

            # Show what the depth controller is actually seeing, so a descend that
            # never finishes can be told apart from one that never started.
            self.get_logger().info(
                'Descend: rel_alt=%s target=%.2f cmd_z=%.2f'
                % (self.rel_alt, self.param('target_depth'), vel_cmd.linear.z),
                throttle_duration_sec=1.0)

            if finished:
                self.get_logger().info(
                    'Descend finished at rel_alt=%.2f (target %.2f), searching'
                    % (self.rel_alt, self.param('target_depth')))
                self.transition(SEARCH)
                
        # The SEARCH state is where the vehicle looks for the gate. It holds its depth and yaws on the spot to sweep the scene. If it detects the gate, it transitions to the APPROACH state. If it times out without finding the gate, it logs an error and transitions to DONE.
        elif self.state == SEARCH:
            self.hold_depth(vel_cmd)
            if self.get_detected_object('gate') is not None:
                self.transition(APPROACH)
            elif self.time_in_state() > self.param('search_timeout'):
                self.get_logger().error('Gate not found within search timeout')
                self.transition(DONE)
            else:
                # Yaw on the spot to sweep the scene
                vel_cmd.angular.z = self.param('search_yaw_rate')
                self.get_logger().info('Searching for gate...', throttle_duration_sec=2.0)

        elif self.state == APPROACH:
            gate = self.get_detected_object('gate')
            if gate is None:
                # Lost the gate. If the last good view was close and square we are at
                # the face already and should commit, otherwise go back to searching.
                if self.was_close:
                    self.transition(THROUGH)
                else:
                    self.transition(SEARCH)
            elif self.time_in_state() > self.param('approach_timeout'):
                self.get_logger().warn('Approach timed out, re-searching')
                self.transition(SEARCH)
            else:
                # Remember whether committing blind would still be safe if we lose
                # the gate on the next tick.
                near = (gate.w >= self.param('commit_width') * 0.8
                        or gate.h >= self.param('commit_height') * 0.8)
                self.was_close = near and self.can_pass_through(gate)

                # Centre the gate and drive forward, strafing around the flare if needed.
                # Centring runs every tick, otherwise the gate drifts out of frame.
                self.avoid_flare(vel_cmd)
                self.keep_gate_in_center(vel_cmd, gate)
                vel_cmd.linear.x = self.param('forward_speed')

                self.get_logger().info(
                    'Approach: x_off=%.2f y_off=%.2f w=%.2f h=%.2f w/h=%.2f conf=%.2f'
                    % (gate.x_offset, gate.y_offset, gate.w, gate.h,
                       gate.w_h_ratio, gate.conf),
                    throttle_duration_sec=1.0)

                # Close enough for the w/h ratio to be trustworthy, so stop and square up
                if self.is_at_commit_distance(gate):
                    self.transition(ALIGN)

        elif self.state == ALIGN:
            gate = self.get_detected_object('gate')
            # The timeout is checked before anything else, otherwise a gate that
            # squares up but never centres would keep us here forever.
            if gate is None:
                # Never commit from an angle we know is bad, go and find the gate again
                self.transition(SEARCH)
            elif self.time_in_state() > self.param('align_timeout'):
                self.get_logger().warn('Could not align with the gate, re-searching')
                self.transition(SEARCH)
            elif self.can_pass_through(gate):
                # Square to the gate, but the orbiting has almost certainly left us
                # off to one side. Stop strafing and centre the gate before we commit,
                # otherwise the blind run goes in at an offset and clips the frame.
                centred = self.keep_gate_in_center(vel_cmd, gate)
                if centred:
                    self.centred_ticks = self.centred_ticks + 1
                else:
                    self.centred_ticks = 0

                self.get_logger().info(
                    'Squared, centring: w/h=%.2f x_off=%.2f y_off=%.2f held=%d'
                    % (gate.w_h_ratio, gate.x_offset, gate.y_offset, self.centred_ticks),
                    throttle_duration_sec=1.0)

                # Hold it there for a moment so we do not commit on one lucky frame
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
                    'Align: w/h=%.2f err=%.3f dir=%s'
                    % (gate.w_h_ratio, self.ratio_error(gate), side),
                    throttle_duration_sec=1.0)

        elif self.state == THROUGH:
            # The gate leaves the camera view before we are through it, so this
            # last stretch is open loop: hold depth and drive straight.
            self.hold_depth(vel_cmd)
            vel_cmd.linear.x = self.param('forward_speed')
            if self.time_in_state() >= self.param('through_duration'):
                self.get_logger().info('Gate traversal complete')
                self.transition(DONE)

        self.publisher_.publish(vel_cmd)
            
# ------------------------------------ Main Function ------------------------------------ #
def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Leave the vehicle stopped rather than with the last command latched
        try:
            node.publisher_.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()
