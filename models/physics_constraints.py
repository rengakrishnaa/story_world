"""
Physics Constraint Definitions

Defines required evidence for evaluating physics constraints.
Constraint selection driven by intent classifier (problem_domain), not keywords.
"""

from typing import Optional
from models.epistemic import Constraint
from models.intent_classification import (
    get_problem_domain,
    PROBLEM_DOMAIN_VEHICLE,
    PROBLEM_DOMAIN_STATICS,
    PROBLEM_DOMAIN_STRUCTURAL,
    PROBLEM_DOMAIN_FLUID,
    PROBLEM_DOMAIN_GENERIC,
)


# Common physics evidence names
EVIDENCE_SPEED_PROFILE = "speed_profile"
EVIDENCE_TURN_RADIUS = "turn_radius"
EVIDENCE_FRICTION_ESTIMATE = "friction_estimate"
EVIDENCE_LATERAL_ACCELERATION = "lateral_acceleration"
EVIDENCE_YAW_RATE = "yaw_rate"
EVIDENCE_SLIP_ANGLE = "slip_angle"
EVIDENCE_ROLL_ANGLE = "roll_angle"
EVIDENCE_TIRE_GRIP = "tire_grip"
EVIDENCE_CENTER_OF_MASS = "center_of_mass"
EVIDENCE_VELOCITY_VECTOR = "velocity_vector"
EVIDENCE_ANGULAR_VELOCITY = "angular_velocity"
EVIDENCE_ACCELERATION_VECTOR = "acceleration_vector"
EVIDENCE_FORCE_VECTOR = "force_vector"
EVIDENCE_MASS = "mass"
EVIDENCE_MOMENT_OF_INERTIA = "moment_of_inertia"


def get_vehicle_dynamics_constraints() -> list[Constraint]:
    """
    Constraints for vehicle dynamics (turning, acceleration, stability).
    
    Example: "A vehicle takes a sharp turn at increasing speed"
    """
    return [
        Constraint(
            name="lateral_stability_limit",
            requires=[
                EVIDENCE_SPEED_PROFILE,
                EVIDENCE_TURN_RADIUS,
                EVIDENCE_FRICTION_ESTIMATE,
            ],
            description="Cannot evaluate lateral stability without speed, turn radius, and friction",
        ),
        Constraint(
            name="yaw_stability",
            requires=[
                EVIDENCE_YAW_RATE,
                EVIDENCE_SLIP_ANGLE,
                EVIDENCE_VELOCITY_VECTOR,
            ],
            description="Cannot evaluate yaw stability without yaw rate, slip angle, and velocity",
        ),
        Constraint(
            name="roll_stability",
            requires=[
                EVIDENCE_ROLL_ANGLE,
                EVIDENCE_LATERAL_ACCELERATION,
                EVIDENCE_CENTER_OF_MASS,
            ],
            description="Cannot evaluate roll stability without roll angle, lateral acceleration, and center of mass",
        ),
        Constraint(
            name="tire_grip_limit",
            requires=[
                EVIDENCE_TIRE_GRIP,
                EVIDENCE_LATERAL_ACCELERATION,
                EVIDENCE_FRICTION_ESTIMATE,
            ],
            description="Cannot evaluate tire grip limit without grip measurement, lateral acceleration, and friction",
        ),
    ]


def get_general_physics_constraints() -> list[Constraint]:
    """
    General physics constraints (gravity, energy conservation, etc.).
    """
    return [
        Constraint(
            name="gravity_violation",
            requires=[
                EVIDENCE_ACCELERATION_VECTOR,
                EVIDENCE_FORCE_VECTOR,
            ],
            description="Cannot check gravity violation without acceleration and force vectors",
        ),
        Constraint(
            name="energy_conservation",
            requires=[
                EVIDENCE_VELOCITY_VECTOR,
                EVIDENCE_MASS,
                EVIDENCE_FORCE_VECTOR,
            ],
            description="Cannot check energy conservation without velocity, mass, and force",
        ),
    ]


def get_kinematic_resolution_constraint() -> Constraint:
    """
    Constraint for kinematic resolution (already discovered by the system).
    """
    return Constraint(
        name="insufficient_kinematic_resolution",
        requires=[
            EVIDENCE_SPEED_PROFILE,
            EVIDENCE_TURN_RADIUS,
            EVIDENCE_ANGULAR_VELOCITY,
        ],
        description="Cannot evaluate kinematics without speed profile, turn radius, and angular velocity",
    )


def get_statics_constraints() -> list[Constraint]:
    """
    Constraints for static/stacking problems (no_tipping, stability).
    Evidence can come from intent spec (mass, geometry, friction, gravity).
    """
    return [
        Constraint(
            name="no_tipping",
            requires=[EVIDENCE_MASS, EVIDENCE_CENTER_OF_MASS],
            description="Cannot evaluate tipping without mass and center of mass",
        ),
    ]


def get_constraints_for_intent(
    intent: str,
    *,
    override_problem_domain: Optional[str] = None,
) -> list[Constraint]:
    """
    Get relevant constraints based on episode intent.
    Driven by LLM-classified problem_domain, not keywords.
    """
    domain = get_problem_domain(intent, override=override_problem_domain)
    constraints = []

    if domain == PROBLEM_DOMAIN_VEHICLE:
        constraints.extend(get_vehicle_dynamics_constraints())
        constraints.append(get_kinematic_resolution_constraint())
    elif domain == PROBLEM_DOMAIN_STATICS:
        constraints.extend(get_statics_constraints())
    elif domain == PROBLEM_DOMAIN_STRUCTURAL:
        constraints.extend(get_statics_constraints())
    elif domain == PROBLEM_DOMAIN_FLUID:
        constraints.append(get_kinematic_resolution_constraint())
    else:
        # generic: include both, evaluator uses evidence availability
        constraints.extend(get_statics_constraints())
        constraints.append(get_kinematic_resolution_constraint())

    constraints.extend(get_general_physics_constraints())

    # Deduplicate by constraint name
    seen = set()
    out = []
    for c in constraints:
        if c.name not in seen:
            seen.add(c.name)
            out.append(c)
    return out
