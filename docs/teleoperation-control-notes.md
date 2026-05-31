# Teleoperation and ROV Control Notes

This note records the current design direction for `rov_lab` teleoperation and ROV base control.

## Teleoperation Device Direction

Keyboard control is useful only as a smoke-test device. It is not a good long-term demonstration source for underwater manipulation because it cannot provide smooth 6-DOF base commands and one or two arm commands at the same time.

Recommended path:

1. Use a 6-DOF haptic/master device for each manipulator arm.
2. Use a separate 6-DOF base command source for the ROV body, either a SpaceMouse-style input, joystick pair, or pedals plus joystick.
3. Keep the action schema device-agnostic:

```text
[rov_base_6d, left_arm_6d, left_gripper, right_arm_6d, right_gripper]
```

For the current single-arm setup, keep the same structure with only one arm active:

```text
[rov_base_6d, arm_6d, gripper]
```

For bi-manipulator teleoperation, the most scalable hardware setup is:

- two haptic/master arms for left/right manipulator end-effector commands;
- one low-rate base device for station keeping and gross ROV movement;
- optional foot pedals or mode switches for gripper and base/arm authority.

This avoids overloading a single device with too many degrees of freedom.

## ROV Base Controller Direction

OceanSim's original manual example injects force and torque directly into the ROV body. In `rov_lab`, `ROVVelocityAction` intentionally interprets the ROV command as body-frame target velocity and converts it to a wrench. This was introduced because direct force commands can excite the arm/base articulation and make the vehicle bend or oscillate.

Remaining bending can come from:

- applying aggressive base wrench while arm joints are compliant;
- low arm stiffness/damping or low joint effort limits;
- inaccurate ROV mass/inertia relative to the mounted arm;
- applying base control without compensating arm-induced moments;
- fixed joint or articulation-root setup in the USD not matching the intended rigid connection.

Recommended stabilization order:

1. Verify the USD joint between ROV body and arm is physically rigid enough for the intended setup.
2. Increase arm joint stiffness/damping and effort limits only to realistic values.
3. Tune ROV velocity gains down before increasing force/torque limits.
4. Add a body-rate damping term if oscillation persists.
5. Consider a station-keeping controller that regulates ROV pose/velocity while accounting for manipulator motion.

Do not switch back to raw force teleoperation for demonstrations unless the dataset explicitly needs direct thruster-command actions.
