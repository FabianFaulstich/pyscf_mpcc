from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pyscf import gto, scf, sylvester_cc
from pyscf.sylvester_cc import dfrpa as dense_dfrpa
from pyscf.sylvester_cc.dfrpa_sylvester import DFRPASylvester
from pyscf.sylvester_cc.laplace_quadrature import LaplaceQuadrature


def _bare_solver(eris=None, chebyshev_tol=1.0e-15):
    solver = DFRPASylvester.__new__(DFRPASylvester)
    solver._eris = eris
    solver.ll_chebyshev_tol = chebyshev_tol
    solver.ll_chebyshev_max_degree = 500
    solver._xvo_t2_timing = None
    solver._xvo_t2_timing_history = []
    solver.verbose = 0
    return solver


def _random_factors(seed=12):
    rng = np.random.default_rng(seed)
    naux, nlap, nvir, nocc = 3, 4, 5, 2
    Y = 0.1 * rng.standard_normal((naux, nlap, nvir, nocc))
    return rng, Y


@pytest.mark.parametrize(
    "option",
    (
        "ll_xvo_t2_mode",
        "ll_exchange_screen_tol",
        "ll_exchange_screen_mode",
        "ll_exchange_q_block_size",
        "ll_exchange_rank_block_size",
        "ll_exchange_check_interval",
        "ll_exchange_screen_refresh",
    ),
)
def test_retired_Xvo_t2_options_fail_clearly(option):
    with pytest.raises(ValueError, match="retired Xvo_t2 option"):
        DFRPASylvester(None, None, **{option: 0})


def test_dense_Xvo_t2_matches_reference_and_records_timing():
    rng, Y = _random_factors()
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(SimpleNamespace(Lov=Lov))

    t2 = solver.reconstruct_t2(Y)
    t2_antisym = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    reference = np.einsum("Lkc,ikac->Lai", Lov, t2_antisym)

    result = solver.get_Xvo_t2(Y)

    assert np.allclose(result, reference, atol=1.0e-13)
    assert set(solver.xvo_t2_timing) == {
        "direct",
        "reconstruct_t2",
        "exchange",
        "total",
        "dense_t2_bytes",
    }
    assert solver.xvo_t2_timing["dense_t2_bytes"] == t2.nbytes
    assert all(
        solver.xvo_t2_timing[name] >= 0.0
        for name in ("direct", "reconstruct_t2", "exchange", "total")
    )


def test_factorized_direct_Xvo_t2_matches_dense_direct_contraction():
    rng, Y = _random_factors(seed=13)
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(SimpleNamespace(Lov=Lov))
    Y_flat = solver._flatten_factors(Y)

    t2 = solver.reconstruct_t2(Y)
    reference_direct = 2.0 * np.einsum("Lkc,ikac->Lai", Lov, t2)
    overlap = np.einsum("Lkc,Rck->LR", Lov, Y_flat)
    factorized_direct = -2.0 * np.einsum("LR,Rai->Lai", overlap, Y_flat)

    assert np.allclose(factorized_direct, reference_direct, atol=2.0e-13)


def test_zero_factors_avoid_dense_reconstruction():
    rng, Y = _random_factors(seed=31)
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(
        SimpleNamespace(Lov=Lov, Lvo=Lov.transpose(0, 2, 1))
    )

    result = solver.get_Xvo_t2(np.zeros_like(Y))

    assert np.count_nonzero(result) == 0
    assert solver.xvo_t2_timing is None
    assert solver._xvo_t2_timing_history == []


def test_factorized_ladder_pert_energy_matches_dense_reference():
    rng, Y = _random_factors(seed=43)
    naux, _, nvir, nocc = Y.shape
    solver = _bare_solver()
    solver._Y = Y
    t2 = solver.reconstruct_t2(Y)
    Joo = rng.standard_normal((naux, nocc, nocc))
    Jvv = rng.standard_normal((naux, nvir, nvir))

    reference = dense_dfrpa.DFRPA.ladder_pert_energy(
        solver, t2, Joo, Jvv
    )
    result = solver.ladder_pert_energy(t2, Joo, Jvv)

    assert np.allclose(result, reference, atol=2.0e-12)


def test_factorized_ladder_requires_converged_factors():
    solver = _bare_solver()
    solver._Y = None
    with pytest.raises(RuntimeError, match="requires converged Y factors"):
        solver.ladder_pert_energy(
            np.zeros((1, 1, 1, 1)),
            np.zeros((1, 1, 1)),
            np.zeros((1, 1, 1)),
        )


def test_factorized_nonlinear_exchange_matches_dense_reference(monkeypatch):
    rng, Y = _random_factors(seed=44)
    naux, _, nvir, nocc = Y.shape
    eris = SimpleNamespace(
        Lov=rng.standard_normal((naux, nocc, nvir))
    )
    solver = _bare_solver(eris)
    t2 = solver.reconstruct_t2(Y)

    reference = dense_dfrpa.DFRPA.non_linear_exchange(solver, t2)

    def fail_if_t2_is_reconstructed(_):
        raise AssertionError("nonlinear exchange reconstructed dense t2")

    monkeypatch.setattr(solver, "reconstruct_t2", fail_if_t2_is_reconstructed)
    result = solver.non_linear_exchange(Y)

    assert np.allclose(result, reference, atol=2.0e-12)


def test_factorized_omega0_matches_dense_source():
    rng, Y = _random_factors(seed=14)
    naux, _, nvir, nocc = Y.shape
    eris = SimpleNamespace(
        Lov=rng.standard_normal((naux, nocc, nvir)),
        Lvo=rng.standard_normal((naux, nvir, nocc)),
        Loo=rng.standard_normal((naux, nocc, nocc)),
        Lvv=rng.standard_normal((naux, nvir, nvir)),
        fov=rng.standard_normal((nocc, nvir)),
    )
    solver = _bare_solver(eris)
    t1 = 0.1 * rng.standard_normal((nocc, nvir))
    Fov = rng.standard_normal((nocc, nvir))
    X, _, Xvo, Xvo_t2 = solver.get_X_factorized(t1, Y)

    t2 = solver.reconstruct_t2(Y)
    t2_antisym = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    reference = eris.fov.T.copy()
    reference -= np.einsum("Laj,Lji->ai", Xvo, eris.Loo)
    reference -= np.einsum("Laj,Lji->ai", Xvo_t2, eris.Loo)
    reference += np.einsum("Lai,L->ai", eris.Lvo, X)
    reference += np.einsum("Lae,Lei->ai", eris.Lvv, Xvo_t2)
    reference += np.einsum("ijab,jb->ai", t2_antisym, Fov)

    result = solver.get_omega0_factorized(X, Xvo, Xvo_t2, Fov, Y)
    assert np.allclose(result, reference, atol=2.0e-12)


def test_singles_and_doubles_reuse_iteration_intermediates(monkeypatch):
    rng = np.random.default_rng(52)
    naux, nlap, nvir, nocc = 2, 2, 3, 2
    eris = SimpleNamespace(
        Lov=0.02 * rng.standard_normal((naux, nocc, nvir)),
        Lvo=0.02 * rng.standard_normal((naux, nvir, nocc)),
        Loo=0.02 * rng.standard_normal((naux, nocc, nocc)),
        Lvv=0.02 * rng.standard_normal((naux, nvir, nvir)),
        foo=np.diag([-2.0, -1.5]),
        fvv=np.diag([0.5, 0.8, 1.2]),
        fov=0.001 * rng.standard_normal((nocc, nvir)),
    )
    solver = _bare_solver(eris)
    t1 = 0.01 * rng.standard_normal((nocc, nvir))
    Y = 0.01 * rng.standard_normal((naux, nlap, nvir, nocc))
    quad = LaplaceQuadrature(
        exponents=np.array([0.05, 0.2]),
        weights=np.array([0.4, 0.1]),
        ymin=0.1,
        ymax=20.0,
    )

    xvo_t2_calls = 0
    get_Xvo_t2 = solver.get_Xvo_t2

    def count_Xvo_t2(current_Y):
        nonlocal xvo_t2_calls
        xvo_t2_calls += 1
        return get_Xvo_t2(current_Y)

    monkeypatch.setattr(solver, "get_Xvo_t2", count_Xvo_t2)
    intermediates = solver.get_iteration_intermediates(t1, Y)
    base_fock = tuple(
        value.copy()
        for value in (intermediates.Foo, intermediates.Fvv, intermediates.Fov)
    )

    def fail_if_rebuilt(_, __):
        raise AssertionError("shared iteration intermediates were rebuilt")

    monkeypatch.setattr(
        solver, "get_iteration_intermediates", fail_if_rebuilt
    )
    Y_target = solver.solve_doubles(
        t1, Y, quad, intermediates=intermediates
    )
    t1_target = solver.solve_singles(
        t1, Y, quad, intermediates=intermediates
    )

    assert xvo_t2_calls == 1
    assert Y_target.shape == Y.shape
    assert t1_target.shape == t1.shape
    assert np.array_equal(intermediates.Foo, base_fock[0])
    assert np.array_equal(intermediates.Fvv, base_fock[1])
    assert np.array_equal(intermediates.Fov, base_fock[2])


def test_singles_laplace_uses_actual_nonsymmetric_actions():
    rng = np.random.default_rng(8)
    Foo = rng.standard_normal((2, 2)) - 4.0 * np.eye(2)
    Fvv = rng.standard_normal((4, 4)) + 2.0 * np.eye(4)
    omega0 = rng.standard_normal((4, 2))
    quad = LaplaceQuadrature(
        exponents=np.array([0.03, 0.2, 0.9]),
        weights=np.array([0.2, 0.4, 0.1]),
        ymin=0.1,
        ymax=20.0,
    )
    solver = _bare_solver()

    reference = -sum(
        weight * expm(-alpha * Fvv) @ omega0 @ expm(alpha * Foo)
        for alpha, weight in zip(quad.exponents, quad.weights)
    )
    result = solver.solve_singles_laplace(omega0, Foo, Fvv, quad)

    assert not np.allclose(Foo, Foo.T)
    assert not np.allclose(Fvv, Fvv.T)
    assert np.allclose(result, reference, atol=2.0e-12)


def test_singles_laplace_is_finite_for_wide_ghost_spectrum():
    Foo = np.diag([-20.0, -0.4])
    Fvv = np.diag([0.5, 71.0])
    omega0 = np.ones((2, 2))
    quad = LaplaceQuadrature(
        exponents=np.array([27.7]),
        weights=np.array([1.0]),
        ymin=0.1,
        ymax=100.0,
    )
    solver = _bare_solver()

    reference = -expm(-27.7 * Fvv) @ omega0 @ expm(27.7 * Foo)
    with np.errstate(over="raise", invalid="raise"):
        result = solver.solve_singles_laplace(omega0, Foo, Fvv, quad)

    assert np.isfinite(result).all()
    assert np.allclose(result, reference, rtol=2.0e-9, atol=2.0e-20)


def test_doubles_factors_match_explicit_matrix_exponentials():
    rng = np.random.default_rng(21)
    Foo = rng.standard_normal((2, 2)) - 3.0 * np.eye(2)
    Fvv = rng.standard_normal((4, 4)) + 2.0 * np.eye(4)
    Jvo = rng.standard_normal((3, 4, 2))
    quad = LaplaceQuadrature(
        exponents=np.array([0.04, 0.3, 0.7]),
        weights=np.array([0.3, 0.2, 0.05]),
        ymin=0.1,
        ymax=20.0,
    )
    solver = _bare_solver()

    reference = np.stack(
        [
            np.sqrt(weight)
            * np.einsum(
                "ab,Lbi,ij->Laj",
                expm(-alpha * Fvv),
                Jvo,
                expm(alpha * Foo),
            )
            for alpha, weight in zip(quad.exponents, quad.weights)
        ],
        axis=1,
    )
    result = solver.get_sylvester_laplace_matrix_factors(
        Jvo, Foo, Fvv, quad
    )

    assert np.allclose(result, reference, atol=2.0e-12)


def test_reconstructed_t2_has_pair_symmetry():
    _, Y = _random_factors(seed=17)
    t2 = DFRPASylvester.reconstruct_t2(Y)
    assert np.allclose(t2, t2.transpose(1, 0, 3, 2), atol=1.0e-14)


def test_nonpositive_dressed_gap_fails_clearly():
    solver = _bare_solver()
    quad = LaplaceQuadrature(
        exponents=np.array([0.1]),
        weights=np.array([1.0]),
        ymin=0.01,
        ymax=10.0,
    )
    with pytest.raises(ValueError, match="dressed orbital gaps must be positive"):
        solver._validate_dressed_gaps(
            np.array([[1.0]]), np.array([[0.0]]), quad, 1, "singles"
        )


def test_dressed_gap_outside_quadrature_fails_clearly():
    solver = _bare_solver()
    quad = LaplaceQuadrature(
        exponents=np.array([0.1]),
        weights=np.array([1.0]),
        ymin=1.0,
        ymax=2.0,
    )
    with pytest.raises(ValueError, match="does not cover required gaps"):
        solver._validate_dressed_gaps(
            np.array([[-1.0]]), np.array([[2.0]]), quad, 1, "singles"
        )


def test_canonical_water_energy_matches_dense_dfrpa():
    laplace_root = (
        Path(__file__).resolve().parents[3] / "external" / "laplace-minimax"
    )
    if not (laplace_root / "data" / "init_para.txt").is_file():
        pytest.skip("optional laplace-minimax tables are unavailable")

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-12)
    convergence = 1.0e-8

    reference = sylvester_cc.DFRPA(
        mf,
        ladder_mode=None,
        ll_con_tol=convergence,
        ll_max_its=100,
    )
    t1_reference, t2_reference = reference.kernel()
    reference_energy = reference.energy(t1_reference, t2_reference)

    factorized = sylvester_cc.DFRPASylvester(
        mf,
        ll_laplace_root=laplace_root,
        ll_laplace_npoints=16,
        ll_con_tol=convergence,
        ll_max_its=100,
        ll_chebyshev_tol=1.0e-14,
    )
    factorized.kernel()

    assert factorized.converged
    assert abs(factorized.e_corr - reference_energy) < 1.0e-7

    perturbative_reference = sylvester_cc.DFRPA(
        mf,
        ladder_mode="perturbative",
        ll_con_tol=convergence,
        ll_max_its=100,
    )
    t1_perturbative, t2_perturbative = perturbative_reference.kernel()
    perturbative_reference_energy = perturbative_reference.energy(
        t1_perturbative, t2_perturbative
    )

    perturbative_factorized = sylvester_cc.DFRPASylvester(
        mf,
        ladder_mode="perturbative",
        ll_laplace_root=laplace_root,
        ll_laplace_npoints=16,
        ll_con_tol=convergence,
        ll_max_its=100,
        ll_chebyshev_tol=1.0e-14,
    )
    perturbative_factorized.kernel()

    assert perturbative_factorized.converged
    assert abs(
        perturbative_factorized.e_corr - perturbative_reference_energy
    ) < 1.0e-7
    assert abs(
        perturbative_factorized.e_ladder_pert
        - perturbative_reference.e_ladder_pert
    ) < 1.0e-7

    exchange_reference = sylvester_cc.DFRPA(
        mf,
        ladder_mode="exchange_perturbative",
        ll_con_tol=convergence,
        ll_max_its=100,
    )
    t1_exchange_reference, t2_exchange_reference = (
        exchange_reference.kernel()
    )
    exchange_reference_energy = exchange_reference.energy(
        t1_exchange_reference, t2_exchange_reference
    )

    exchange_factorized = sylvester_cc.DFRPASylvester(
        mf,
        ladder_mode="exchange_perturbative",
        ll_laplace_root=laplace_root,
        ll_laplace_npoints=16,
        ll_con_tol=convergence,
        ll_max_its=100,
        ll_chebyshev_tol=1.0e-14,
    )
    exchange_factorized.kernel()

    assert exchange_factorized.converged
    assert abs(
        exchange_factorized.e_corr - exchange_reference_energy
    ) < 1.0e-7
    assert abs(
        exchange_factorized.e_ladder_pert
        - exchange_reference.e_ladder_pert
    ) < 1.0e-7
    assert abs(
        exchange_factorized.e_exchange_pert
        - exchange_reference.e_exchange_pert
    ) < 1.0e-7

    frozen_reference = sylvester_cc.DFRPA(
        mf,
        frozen=1,
        ladder_mode=None,
        ll_con_tol=convergence,
        ll_max_its=100,
    )
    t1_frozen_reference, t2_frozen_reference = frozen_reference.kernel()
    frozen_reference_energy = frozen_reference.energy(
        t1_frozen_reference, t2_frozen_reference
    )

    frozen_factorized = sylvester_cc.DFRPASylvester(
        mf,
        frozen=1,
        ll_laplace_root=laplace_root,
        ll_laplace_npoints=16,
        ll_con_tol=convergence,
        ll_max_its=100,
        ll_chebyshev_tol=1.0e-14,
    )
    t1_frozen, t2_frozen = frozen_factorized.kernel()

    assert frozen_factorized.converged
    assert t1_frozen.shape == (4, 2)
    assert t2_frozen.shape == (4, 4, 2, 2)
    assert abs(frozen_factorized.e_corr - frozen_reference_energy) < 1.0e-7


def test_benzene_with_ghost_carbon_has_finite_changes():
    laplace_root = (
        Path(__file__).resolve().parents[3] / "external" / "laplace-minimax"
    )
    if not (laplace_root / "data" / "init_para.txt").is_file():
        pytest.skip("optional laplace-minimax tables are unavailable")

    carbon_radius = 1.397
    hydrogen_radius = 2.479
    atoms = []
    for index in range(6):
        angle = index * np.pi / 3.0
        atoms.append(
            (
                "C",
                (
                    carbon_radius * np.cos(angle),
                    carbon_radius * np.sin(angle),
                    0.0,
                ),
            )
        )
        atoms.append(
            (
                "H",
                (
                    hydrogen_radius * np.cos(angle),
                    hydrogen_radius * np.sin(angle),
                    0.0,
                ),
            )
        )
    atoms.append(("ghost1", (0.0, 0.0, 3.5)))

    # A compact benzene basis keeps the test inexpensive.  The cc-pVQZ
    # functions on the ghost carbon reproduce the wide virtual spectrum that
    # previously caused 0*inf in the Chebyshev coefficients.
    mol = gto.M(
        atom=atoms,
        basis={
            "C": "sto-3g",
            "H": "sto-3g",
            "ghost1": gto.basis.load("cc-pvqz", "C"),
        },
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-10)
    factorized = sylvester_cc.DFRPASylvester(
        mf,
        ll_laplace_root=laplace_root,
        ll_laplace_npoints=16,
        ll_con_tol=1.0e-8,
        ll_max_its=1,
        diis=False,
    )
    quad_t1, _ = factorized._get_fixed_quadratures()
    lower_v, upper_v = factorized._symmetric_spectral_bounds(
        factorized._eris.fvv
    )

    # This exceeds the floating-point overflow threshold in the old direct
    # Bessel formula and ensures the molecular test exercises the regression.
    assert max(quad_t1.exponents) * (upper_v - lower_v) / 2.0 > 700.0
    with np.errstate(over="raise", invalid="raise"):
        t1, t2 = factorized.kernel()

    assert np.isfinite(t1).all()
    assert np.isfinite(t2).all()
    assert np.isfinite(factorized.e_corr)
