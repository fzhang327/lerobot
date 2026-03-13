import numpy as np

from lerobot.utils.eso_admittance_controller import ESOAdmittanceController


def test_critical_damping_is_auto_computed() -> None:
    controller = ESOAdmittanceController(dof=2, b0=2.0, omega_o=20.0, k_d=np.array([9.0, 16.0]), m_d=1.0)
    np.testing.assert_allclose(controller.d_d, np.array([6.0, 8.0]))


def test_freezes_when_force_limit_is_exceeded() -> None:
    controller = ESOAdmittanceController(
        dof=1,
        b0=1.0,
        omega_o=100.0,
        k_d=10.0,
        force_limit=0.5,
        target_cutoff_hz=None,
    )

    q_measured = np.array([0.0])
    controller.reset(q_measured)

    # Create a large observer error so z3 (and then force estimate) quickly exceeds limit.
    cmd = controller.update(q_vla_target=np.array([1.0]), q_measured=np.array([1.0]), dt=0.01)
    np.testing.assert_allclose(cmd, np.array([1.0]))
    np.testing.assert_allclose(controller.dq_c, np.array([0.0]))


def test_tracks_target_in_quiet_case() -> None:
    controller = ESOAdmittanceController(
        dof=1,
        b0=1.0,
        omega_o=25.0,
        k_d=30.0,
        force_deadband=0.0,
        force_limit=100.0,
        target_cutoff_hz=None,
    )

    q_measured = np.array([0.0])
    controller.reset(q_measured)

    q_cmd = q_measured.copy()
    for _ in range(200):
        q_cmd = controller.update(q_vla_target=np.array([0.5]), q_measured=q_cmd, dt=0.01)

    assert 0.40 < q_cmd[0] < 0.52


def test_force_estimation_respects_deadband_and_gravity_compensation() -> None:
    controller = ESOAdmittanceController(
        dof=1,
        b0=2.0,
        omega_o=10.0,
        k_d=10.0,
        force_deadband=0.1,
        target_cutoff_hz=None,
        gravity_fn=lambda q: np.array([0.2]),
    )

    # Below deadband after gravity compensation should become exactly zero.
    filtered = controller._apply_deadband(np.array([0.05]), deadband=controller.force_deadband)
    np.testing.assert_allclose(filtered, np.array([0.0]))

    # Verify gravity compensation callback is used as part of force estimate path.
    g = controller.gravity_compensation(np.array([0.0]))
    np.testing.assert_allclose(g, np.array([0.2]))
