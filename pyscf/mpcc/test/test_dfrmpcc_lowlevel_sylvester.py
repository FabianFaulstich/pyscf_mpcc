import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.linalg
from pyscf import gto, lib, scf
from pyscf.mcscf import avas

from pyscf.mpcc import df_eri, dfrmpcc_lowlevel, laplace_quadrature
from pyscf.mpcc.dfrmpcc_lowlevel import MPCC_LL


def make_lowlevel_fixture():
    naux = 2
    nocc = 3
    nvir = 4

    rng = np.random.default_rng(12)
    eris = SimpleNamespace(
        Lvv=rng.normal(size=(naux, nvir, nvir)),
        Lov=rng.normal(size=(naux, nocc, nvir)),
        Loo=rng.normal(size=(naux, nocc, nocc)),
        foo=rng.normal(size=(nocc, nocc)),
        fvv=rng.normal(size=(nvir, nvir)),
        fov=rng.normal(size=(nocc, nvir)),
    )
    eris.Lvo = eris.Lov.transpose(0, 2, 1).copy()

    ll = MPCC_LL.__new__(MPCC_LL)
    ll._eris = eris
    ll.frags = []

    return ll, eris, rng


def laplace_minimax_root():
    root = Path(__file__).resolve().parents[3] / "external" / "laplace-minimax"
    if not root.exists():
        return None
    return root


def dense_sylvester_reference(Jvo, Foo, Fvv):
    naux, nvir, nocc = Jvo.shape
    nov = nocc * nvir
    pair_indices = [(p, q) for p in range(nov) for q in range(p, nov)]
    matrix = np.zeros((len(pair_indices), len(pair_indices)))
    source = np.einsum("Lai,Lbj->ijab", Jvo, Jvo)

    def split_ov(p):
        return divmod(p, nvir)

    def unpack_pair_symmetric(vector):
        x = np.zeros((nov, nov))
        for value, (p, q) in zip(vector, pair_indices):
            x[p, q] = value
            x[q, p] = value
        return x.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)

    def pack_pair_symmetric(tensor):
        out = np.empty(len(pair_indices))
        for idx, (p, q) in enumerate(pair_indices):
            i, a = split_ov(p)
            j, b = split_ov(q)
            out[idx] = tensor[i, j, a, b]
        return out

    def linear_residual(t2):
        tmp = np.einsum("bc,ijac->ijab", Fvv, t2)
        tmp -= np.einsum("mi,mjab->ijab", Foo, t2)
        return tmp + tmp.transpose(1, 0, 3, 2)

    for col in range(len(pair_indices)):
        basis = np.zeros(len(pair_indices))
        basis[col] = 1.0
        matrix[:, col] = pack_pair_symmetric(linear_residual(unpack_pair_symmetric(basis)))

    packed_t2 = np.linalg.solve(matrix, -pack_pair_symmetric(source))
    return unpack_pair_symmetric(packed_t2)


def mask_active_t2(shape, frags):
    mask = np.zeros(shape, dtype=bool)
    for act_hole, act_particle in frags:
        mask[np.ix_(act_hole, act_hole, act_particle, act_particle)] = True
    return mask


def sylvester_t2_residual(t2, Jvo, Foo, Fvv):
    tmp = lib.einsum("bc,ijac->ijab", Fvv, t2)
    tmp -= lib.einsum("mi,mjab->ijab", Foo, t2)
    res2 = tmp + tmp.transpose(1, 0, 3, 2)
    res2 += lib.einsum("Lai,Lbj->ijab", Jvo, Jvo)
    return res2


def sylvester_laplace_matrix_factors_expm_reference(Jvo, Foo, Fvv, quad):
    Foo = 0.5 * (Foo + Foo.T)
    Fvv = 0.5 * (Fvv + Fvv.T)
    factors = np.empty((Jvo.shape[0], quad.nlap, Jvo.shape[1], Jvo.shape[2]))
    for idx, (exponent, weight) in enumerate(zip(quad.exponents, quad.weights)):
        Gv = scipy.linalg.expm(-exponent * Fvv)
        Go = scipy.linalg.expm(exponent * Foo.T)
        factors[:, idx] = np.sqrt(weight) * np.einsum(
            "ac,Lck,ki->Lai", Gv, Jvo, Go
        )
    return factors


def compare_sylvester_residual_against_unfactorized(ll, t1, t2):
    Xoo, Xvo, X = ll.get_X(t1)
    _, Jvo = ll.get_J(Xoo, Xvo, t1)
    Foo, Fvv, Fov = ll.get_F(t1, X, Xoo, Xvo)

    res_unfactorized = ll.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)

    Foo, Fvv = ll.update_F(Foo, Fvv, Fov, t1)
    res_sylvester = sylvester_t2_residual(t2, Jvo, Foo, Fvv)

    diff = np.linalg.norm(res_sylvester - res_unfactorized)
    norm_ref = np.linalg.norm(res_unfactorized)
    rel = diff / max(norm_ref, 1e-16)

    return diff, rel, res_sylvester, res_unfactorized


def no_dc_laplace_t2(ll, t1):
    Xoo, Xvo, X = ll.get_X(t1)
    _, Jvo = ll.get_J(Xoo, Xvo, t1)
    Foo, Fvv, Fov = ll.get_F(t1, X, Xoo, Xvo)
    Foo_eff, Fvv_eff = ll.update_F(Foo.copy(), Fvv.copy(), Fov, t1)

    return ll.solve_t2_sylvester_laplace(Jvo, Foo_eff, Fvv_eff)


def projected_t2_error_norm(ll, t1, t2, t2_no_dc, frags):
    Xoo, Xvo, X = ll.get_X(t1)
    _, Jvo = ll.get_J(Xoo, Xvo, t1)
    Foo, Fvv, Fov = ll.get_F(t1, X, Xoo, Xvo)

    res_t2 = ll.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
    res_no_dc = ll.update_t2(t2_no_dc, Jvo, Foo, Fvv, Fov, t1)

    active_mask = mask_active_t2(t2.shape, frags)
    t2_error_residual = (res_t2 - res_no_dc) / ll._eris.D

    return np.linalg.norm(t2_error_residual[~active_mask])


def projected_t2_error_boundary_norm(ll, t1, t2, t2_no_dc, frags):
    Xoo, Xvo, X = ll.get_X(t1)
    _, Jvo = ll.get_J(Xoo, Xvo, t1)
    Foo, Fvv, Fov = ll.get_F(t1, X, Xoo, Xvo)

    res_t2 = ll.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
    res_no_dc = ll.update_t2(t2_no_dc, Jvo, Foo, Fvv, Fov, t1)
    residual = (res_t2 - res_no_dc) / ll._eris.D

    pieces = []
    nocc, _, nvir, _ = t2.shape
    for act_hole, act_particle in frags:
        inact_hole = np.delete(range(nocc), act_hole)
        inact_particle = np.delete(range(nvir), act_particle)
        pieces.append(
            residual[np.ix_(
                inact_hole, act_hole, act_particle, act_particle
            )].ravel()
        )
        pieces.append(
            residual[np.ix_(
                act_hole, act_hole, inact_particle, act_particle
            )].ravel()
        )

    return np.linalg.norm(np.concatenate(pieces))


class SylvesterIntermediateTest(unittest.TestCase):
    def test_get_sylvester_intermediates_matches_existing_pipeline(self):
        ll, _, rng = make_lowlevel_fixture()
        t1 = rng.normal(size=(3, 4))

        Xoo, Xvo, X = ll.get_X(t1)
        _, Jvo_ref = ll.get_J(Xoo, Xvo, t1)
        Foo_ref, Fvv_ref, Fov_ref = ll.get_F(t1, X, Xoo, Xvo)
        Foo_ref, Fvv_ref = ll.update_F(Foo_ref, Fvv_ref, Fov_ref, t1)

        Jvo, Foo, Fvv, Fov = ll.get_sylvester_intermediates(t1)

        np.testing.assert_allclose(Jvo, Jvo_ref)
        np.testing.assert_allclose(Foo, Foo_ref)
        np.testing.assert_allclose(Fvv, Fvv_ref)
        np.testing.assert_allclose(Fov, Fov_ref)

    def test_get_sylvester_intermediates_zero_t1_limit(self):
        ll, eris, _ = make_lowlevel_fixture()
        t1 = np.zeros((3, 4))

        Jvo, Foo, Fvv, Fov = ll.get_sylvester_intermediates(t1)

        np.testing.assert_allclose(Jvo, eris.Lvo)
        np.testing.assert_allclose(Foo, eris.foo)
        np.testing.assert_allclose(Fvv, eris.fvv)
        np.testing.assert_allclose(Fov, eris.fov)

    def test_get_sylvester_intermediates_shapes(self):
        ll, _, rng = make_lowlevel_fixture()
        t1 = rng.normal(size=(3, 4))

        Jvo, Foo, Fvv, Fov = ll.get_sylvester_intermediates(t1)

        self.assertEqual(Jvo.shape, (2, 4, 3))
        self.assertEqual(Foo.shape, (3, 3))
        self.assertEqual(Fvv.shape, (4, 4))
        self.assertEqual(Fov.shape, (3, 4))

    def test_get_sylvester_intermediates_does_not_mutate_inputs(self):
        ll, eris, rng = make_lowlevel_fixture()
        t1 = rng.normal(size=(3, 4))

        t1_ref = t1.copy()
        foo_ref = eris.foo.copy()
        fvv_ref = eris.fvv.copy()
        fov_ref = eris.fov.copy()
        Lov_ref = eris.Lov.copy()
        Lvo_ref = eris.Lvo.copy()

        ll.get_sylvester_intermediates(t1)

        np.testing.assert_allclose(t1, t1_ref)
        np.testing.assert_allclose(eris.foo, foo_ref)
        np.testing.assert_allclose(eris.fvv, fvv_ref)
        np.testing.assert_allclose(eris.fov, fov_ref)
        np.testing.assert_allclose(eris.Lov, Lov_ref)
        np.testing.assert_allclose(eris.Lvo, Lvo_ref)


class SylvesterResidualTest(unittest.TestCase):
    def test_sylvester_t2_residual_matches_unfactorized_update_t2(self):
        ll, _, rng = make_lowlevel_fixture()
        t1 = rng.normal(size=(3, 4))
        t2 = rng.normal(size=(3, 3, 4, 4))

        Xoo, Xvo, X = ll.get_X(t1)
        _, Jvo = ll.get_J(Xoo, Xvo, t1)
        Foo, Fvv, Fov = ll.get_F(t1, X, Xoo, Xvo)

        res_ref = ll.update_t2(t2, Jvo, Foo, Fvv, Fov, t1)
        Foo, Fvv = ll.update_F(Foo, Fvv, Fov, t1)
        res = sylvester_t2_residual(t2, Jvo, Foo, Fvv)

        np.testing.assert_allclose(res, res_ref)

    def test_sylvester_t2_residual_zero_t2_source_term(self):
        ll, _, rng = make_lowlevel_fixture()
        t2 = np.zeros((3, 3, 4, 4))
        Jvo = rng.normal(size=(2, 4, 3))
        Foo = rng.normal(size=(3, 3))
        Fvv = rng.normal(size=(4, 4))

        res = sylvester_t2_residual(t2, Jvo, Foo, Fvv)
        res_ref = np.einsum("Lai,Lbj->ijab", Jvo, Jvo)

        np.testing.assert_allclose(res, res_ref)

    def test_verify_sylvester_residual_against_unfactorized(self):
        ll, _, rng = make_lowlevel_fixture()
        t1 = rng.normal(size=(3, 4))
        t2 = rng.normal(size=(3, 3, 4, 4))

        diff, rel, res, res_ref = compare_sylvester_residual_against_unfactorized(
            ll, t1, t2
        )

        self.assertLess(diff, 1e-10)
        self.assertLess(rel, 1e-12)
        np.testing.assert_allclose(res, res_ref)


class LaplaceSylvesterSolverTest(unittest.TestCase):
    def test_solve_t2_sylvester_laplace_diagonal_limit(self):
        ll, _, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = np.diag(eo)
        Fvv = np.diag(ev)
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        denom = lib.direct_sum("a+b-i-j->ijab", ev, ev, eo, eo)
        approx_inverse = quad.approximate_inverse(denom)
        t2_ref = -np.einsum("Lai,Lbj->ijab", Jvo, Jvo) * approx_inverse

        t2 = ll.solve_t2_sylvester_laplace(Jvo, Foo, Fvv, quad)

        np.testing.assert_allclose(t2, t2_ref, rtol=1e-12, atol=1e-12)

    def test_solve_t2_sylvester_laplace_matches_semicanonical_reference(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        Jmo = np.einsum("aA,Lai,iI->LAI", qv, Jvo, qo)
        denom = lib.direct_sum("A+B-I-J->IJAB", ev, ev, eo, eo)
        t2_ref = -np.einsum("LAI,LBJ,IJAB->IJAB", Jmo, Jmo, quad.approximate_inverse(denom))
        t2_ref = np.einsum("iI,jJ,aA,bB,IJAB->ijab", qo, qo, qv, qv, t2_ref)

        t2 = ll.solve_t2_sylvester_laplace(Jvo, Foo, Fvv, quad)

        np.testing.assert_allclose(t2, t2_ref, rtol=1e-12, atol=1e-12)

    def test_solve_t2_sylvester_laplace_does_not_build_factors(self):
        ll, _, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = np.diag(eo)
        Fvv = np.diag(ev)
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        with mock.patch.object(
            ll, "get_sylvester_laplace_factors", side_effect=AssertionError
        ), mock.patch.object(
            ll, "get_sylvester_laplace_matrix_factors", side_effect=AssertionError
        ):
            t2 = ll.solve_t2_sylvester_laplace(Jvo, Foo, Fvv, quad)

        self.assertEqual(t2.shape, (3, 3, 4, 4))

    def test_sylvester_laplace_factors_reconstruct_dense_laplace_t2(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        Y = ll.get_sylvester_laplace_factors(Jvo, Foo, Fvv, quad)
        t2_from_factors = -np.einsum("LRai,LRbj->ijab", Y, Y)
        t2_dense = ll.solve_t2_sylvester_laplace(Jvo, Foo, Fvv, quad)

        self.assertEqual(Y.shape, (2, 3, 4, 3))
        np.testing.assert_allclose(t2_from_factors, t2_dense, rtol=1e-12, atol=1e-12)

    def test_sylvester_laplace_matrix_factors_match_semicanonical_for_diagonal_fock(self):
        ll, _, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = np.diag(eo)
        Fvv = np.diag(ev)
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        Y_matrix = ll.get_sylvester_laplace_matrix_factors(Jvo, Foo, Fvv, quad)
        Y_semicanonical = ll.get_sylvester_laplace_factors(Jvo, Foo, Fvv, quad)

        np.testing.assert_allclose(Y_matrix, Y_semicanonical, rtol=1e-12, atol=1e-12)

    def test_sylvester_laplace_matrix_factors_reconstruct_expm_reference(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        Y = ll.get_sylvester_laplace_matrix_factors(Jvo, Foo, Fvv, quad)
        t2 = -np.einsum("LRai,LRbj->ijab", Y, Y)
        t2_ref = np.zeros_like(t2)
        for exponent, weight in zip(quad.exponents, quad.weights):
            Gv = scipy.linalg.expm(-exponent * Fvv)
            Go = scipy.linalg.expm(exponent * Foo.T)
            Jhat = np.sqrt(weight) * np.einsum("ac,Lck,ki->Lai", Gv, Jvo, Go)
            t2_ref -= np.einsum("Lai,Lbj->ijab", Jhat, Jhat)

        np.testing.assert_allclose(t2, t2_ref, rtol=1e-12, atol=1e-12)

    def test_chebyshev_left_exp_action_matches_expm(self):
        ll, _, rng = make_lowlevel_fixture()
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        exponent = 1.3

        result = ll._chebyshev_exp_action_left(
            Fvv,
            Jvo,
            -exponent,
            ll._symmetric_spectral_bounds(Fvv),
            tol=1.0e-13,
        )
        ref = np.einsum("ac,Lci->Lai", scipy.linalg.expm(-exponent * Fvv), Jvo)

        np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-12)

    def test_chebyshev_right_exp_action_matches_expm(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        eo = np.array([-2.0, -1.0, -0.4])
        Foo = qo @ np.diag(eo) @ qo.T
        Jvo = rng.normal(size=(2, 4, 3))
        exponent = 1.3

        result = ll._chebyshev_exp_action_right(
            Foo,
            Jvo,
            exponent,
            ll._symmetric_spectral_bounds(Foo),
            tol=1.0e-13,
        )
        ref = np.einsum("Lak,ki->Lai", Jvo, scipy.linalg.expm(exponent * Foo.T))

        np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-12)

    def test_sylvester_laplace_matrix_factors_match_expm_reference(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        Y = ll.get_sylvester_laplace_matrix_factors(Jvo, Foo, Fvv, quad)
        Y_ref = sylvester_laplace_matrix_factors_expm_reference(
            Jvo, Foo, Fvv, quad
        )

        np.testing.assert_allclose(Y, Y_ref, rtol=1e-12, atol=1e-12)

    def test_sylvester_laplace_matrix_factors_do_not_call_expm(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        with mock.patch(
            "scipy.linalg.expm",
            side_effect=AssertionError("production matrix factors must not call expm"),
        ):
            Y = ll.get_sylvester_laplace_matrix_factors(Jvo, Foo, Fvv, quad)

        self.assertEqual(Y.shape, (2, 3, 4, 3))

    def test_sylvester_laplace_matrix_factors_do_not_fully_diagonalize(self):
        ll, _, rng = make_lowlevel_fixture()
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=10.0,
        )

        with mock.patch(
            "numpy.linalg.eigh",
            side_effect=AssertionError("production matrix factors must not call eigh"),
        ), mock.patch(
            "numpy.linalg.eigvalsh",
            side_effect=AssertionError("production matrix factors must not call eigvalsh"),
        ):
            Y = ll.get_sylvester_laplace_matrix_factors(Jvo, Foo, Fvv, quad)

        self.assertEqual(Y.shape, (2, 3, 4, 3))

    def test_sylvester_laplace_matrix_factors_do_not_diagonalize_for_quadrature(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        ll, eris, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        eris.eia = lib.direct_sum("-i+a->ia", eo, ev)
        ll.ll_laplace_quad = None
        ll.ll_laplace_quad_file = None
        ll.ll_laplace_root = root
        ll.ll_laplace_npoints = 12
        Foo = rng.normal(size=(3, 3))
        Foo = 0.5 * (Foo + Foo.T)
        Fvv = rng.normal(size=(4, 4))
        Fvv = 0.5 * (Fvv + Fvv.T)
        Jvo = rng.normal(size=(2, 4, 3))

        with mock.patch(
            "numpy.linalg.eigvalsh",
            side_effect=AssertionError("eigvalsh must not be used"),
        ):
            Y = ll.get_sylvester_laplace_matrix_factors(Jvo, Foo, Fvv)

        self.assertEqual(Y.shape, (2, 12, 4, 3))
        self.assertIn("laplace-minimax:init_para", ll.ll_laplace_quad.source)

    def test_solve_t2_sylvester_laplace_rejects_uncovered_denominators(self):
        ll, _, rng = make_lowlevel_fixture()
        Foo = np.diag(np.array([-2.0, -1.0, -0.4]))
        Fvv = np.diag(np.array([0.3, 0.8, 1.4, 2.1]))
        Jvo = rng.normal(size=(2, 4, 3))
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1]),
            weights=np.array([1.0]),
            ymin=0.1,
            ymax=1.0,
        )

        with self.assertRaisesRegex(ValueError, "does not cover denominators"):
            ll.solve_t2_sylvester_laplace(Jvo, Foo, Fvv, quad)

    def test_get_sylvester_laplace_quadrature_from_external_tables(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        ll, _, _ = make_lowlevel_fixture()
        ll.ll_laplace_quad = None
        ll.ll_laplace_quad_file = None
        ll.ll_laplace_root = root
        ll.ll_laplace_npoints = 12
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        denominators = lib.direct_sum("a+b-i-j->ijab", ev, ev, eo, eo)

        quad = ll.get_sylvester_laplace_quadrature(denominators)

        self.assertEqual(quad.nlap, 12)
        self.assertIn("laplace-minimax:init_para", quad.source)
        self.assertLessEqual(quad.ymin, np.min(denominators))
        self.assertGreaterEqual(quad.ymax, np.max(denominators))
        stats = quad.validate(ngrid=1000)
        self.assertLess(stats["max_rel"], 1.0e-8)

    def test_solve_t2_sylvester_laplace_external_tables_matches_dense_reference(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        ll, _, rng = make_lowlevel_fixture()
        ll.ll_laplace_quad = None
        ll.ll_laplace_quad_file = None
        ll.ll_laplace_root = root
        ll.ll_laplace_npoints = 12
        qo, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        qv, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        Foo = qo @ np.diag(eo) @ qo.T
        Fvv = qv @ np.diag(ev) @ qv.T
        Jvo = rng.normal(size=(2, 4, 3))

        t2_ref = dense_sylvester_reference(Jvo, Foo, Fvv)
        t2_laplace = ll.solve_t2_sylvester_laplace(Jvo, Foo, Fvv)

        np.testing.assert_allclose(t2_laplace, t2_ref, rtol=1e-10, atol=1e-10)

    def test_update_amps_sylvester_laplace_preserves_active_t2_block(self):
        ll, eris, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        eris.Lvv = np.zeros_like(eris.Lvv)
        eris.Loo = np.zeros_like(eris.Loo)
        eris.foo = np.diag(eo)
        eris.fvv = np.diag(ev)
        eris.fov = np.zeros((3, 4))
        eris.eia = lib.direct_sum("-i+a->ia", eo, ev)
        eris.D = lib.direct_sum("ia+jb->ijab", eris.eia, eris.eia)
        ll.ll_laplace_quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=100.0,
        )
        act_hole = np.array([0, 2])
        act_particle = np.array([1, 3])
        ll.frags = [[act_hole, act_particle]]
        t1 = np.zeros((3, 4))
        t2 = rng.normal(size=(3, 3, 4, 4))
        active_ref = t2[np.ix_(act_hole, act_hole, act_particle, act_particle)].copy()

        _, _, _, t2_new = ll.update_amps_sylvester_laplace(t1, t2.copy())

        np.testing.assert_allclose(
            t2_new[np.ix_(act_hole, act_hole, act_particle, act_particle)],
            active_ref,
        )

    def test_factorized_laplace_update_matches_dense_without_active_space(self):
        ll, eris, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        eris.Lvv = np.zeros_like(eris.Lvv)
        eris.Loo = np.zeros_like(eris.Loo)
        eris.Lov = rng.normal(size=(2, 3, 4))
        eris.Lvo = eris.Lov.transpose(0, 2, 1).copy()
        eris.foo = np.diag(eo)
        eris.fvv = np.diag(ev)
        eris.fov = rng.normal(size=(3, 4)) * 1.0e-3
        eris.eia = lib.direct_sum("-i+a->ia", eo, ev)
        eris.D = lib.direct_sum("ia+jb->ijab", eris.eia, eris.eia)
        ll.ll_laplace_quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=100.0,
        )

        t1 = np.zeros((3, 4))
        t2 = np.zeros((3, 3, 4, 4))

        _, _, t1_dense, t2_dense = ll.update_amps_sylvester_laplace(
            t1.copy(), t2.copy()
        )
        _, t1_factorized, dt2s_o, dt2s_v, Y = (
            ll.update_amps_sylvester_laplace_factorized(t1.copy(), [])
        )
        t2_factorized = ll.get_t2(Y, [], dt2s_o, dt2s_v)

        np.testing.assert_allclose(t1_factorized, t1_dense, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t2_factorized, t2_dense, rtol=1e-12, atol=1e-12)

    def test_sylvester_laplace_factorized_kernel_matches_dense_one_step(self):
        ll, eris, rng = make_lowlevel_fixture()
        eo = np.array([-2.0, -1.0, -0.4])
        ev = np.array([0.3, 0.8, 1.4, 2.1])
        eris.Lvv = np.zeros_like(eris.Lvv)
        eris.Loo = np.zeros_like(eris.Loo)
        eris.Lov = rng.normal(size=(2, 3, 4))
        eris.Lvo = eris.Lov.transpose(0, 2, 1).copy()
        eris.foo = np.diag(eo)
        eris.fvv = np.diag(ev)
        eris.fov = rng.normal(size=(3, 4)) * 1.0e-3
        eris.eia = lib.direct_sum("-i+a->ia", eo, ev)
        eris.D = lib.direct_sum("ia+jb->ijab", eris.eia, eris.eia)
        quad = laplace_quadrature.LaplaceQuadrature(
            exponents=np.array([0.1, 0.4, 1.3]),
            weights=np.array([0.2, 0.5, 0.7]),
            ymin=0.1,
            ymax=100.0,
        )
        ll.ll_laplace_quad = quad
        t1 = np.zeros((3, 4))
        t2 = np.zeros((3, 3, 4, 4))

        _, _, t1_dense, t2_dense = ll.update_amps_sylvester_laplace(
            t1.copy(), t2.copy()
        )

        ll.kernel_type = "sylvester_laplace_factorized"
        ll._kernels = {
            "sylvester_laplace_factorized": ll._sylvester_laplace_factorized_kernel
        }
        ll.ll_con_tol = 0.0
        ll.ll_max_its = 1
        ll.diis = False
        t1_factorized, t2_factorized = ll.kernel(t1.copy(), t2.copy())

        np.testing.assert_allclose(t1_factorized, t1_dense, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t2_factorized, t2_dense, rtol=1e-12, atol=1e-12)


class SylvesterLaplacePhysicalTest(unittest.TestCase):
    def test_physical_factorized_laplace_kernel_matches_dense_one_step(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        mol = gto.M(
            atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            basis="cc-pvtz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.verbose = 0
        mf.kernel()

        eris = df_eri.ERIs(mf, mf.mo_coeff)
        frags = []
        base = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
        )
        t1_0, t2_0 = base.init_amps()

        dense = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_method="sylvester_laplace",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
            ll_con_tol=0.0,
            ll_max_its=1,
        )
        dense.diis = False
        t1_dense, t2_dense = dense.kernel(t1_0.copy(), t2_0.copy())

        factorized = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="sylvester_laplace_factorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
            ll_con_tol=0.0,
            ll_max_its=1,
        )
        factorized.diis = False
        t1_factorized, t2_factorized = factorized.kernel(t1_0.copy(), t2_0.copy())

        e_dense = dense.get_energy(t1_dense, t2_dense)
        e_factorized = factorized.get_energy(t1_factorized, t2_factorized)
        print("\nPhysical factorized Laplace one-step comparison:")
        print(f"  ||dt1|| = {np.linalg.norm(t1_factorized - t1_dense):.3e}")
        print(f"  ||dt2|| = {np.linalg.norm(t2_factorized - t2_dense):.3e}")
        print(f"  dE      = {e_factorized - e_dense:.3e}")

        np.testing.assert_allclose(t1_factorized, t1_dense, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(t2_factorized, t2_dense, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(e_factorized, e_dense, rtol=0.0, atol=1e-10)

    def test_physical_factorized_laplace_active_space_one_step_diagnostic(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        mol = gto.M(
            atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            basis="cc-pvtz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.verbose = 0
        mf.kernel()

        avas_obj = avas.AVAS(mf, ["O 2p"], minao="sto-3g", openshell_option=3)
        avas_obj.with_iao = True
        avas_obj.threshold = 1e-7
        _, _, c_lo = avas_obj.kernel()
        act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        act_particle = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]
        frags = [[act_hole, act_particle]]

        eris = df_eri.ERIs(mf, c_lo)
        base = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
        )
        t1_0, t2_0 = base.init_amps()

        dense = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_method="sylvester_laplace",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
            ll_con_tol=0.0,
            ll_max_its=1,
        )
        dense.diis = False
        t1_dense, t2_dense = dense.kernel(t1_0.copy(), t2_0.copy())

        factorized = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="sylvester_laplace_factorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
            ll_con_tol=0.0,
            ll_max_its=1,
        )
        factorized.diis = False
        t1_factorized, t2_factorized = factorized.kernel(t1_0.copy(), t2_0.copy())

        active_mask = mask_active_t2(t2_0.shape, frags)
        dt1 = np.linalg.norm(t1_factorized - t1_dense)
        dt2 = np.linalg.norm(t2_factorized - t2_dense)
        dt2_active = np.linalg.norm((t2_factorized - t2_dense)[active_mask])
        dt2_env = np.linalg.norm((t2_factorized - t2_dense)[~active_mask])
        active_preservation = np.linalg.norm((t2_factorized - t2_0)[active_mask])
        e_dense = dense.get_energy(t1_dense, t2_dense)
        e_factorized = factorized.get_energy(t1_factorized, t2_factorized)

        print("\nPhysical active-corrected Laplace one-step comparison:")
        print(f"  ||dt1||              = {dt1:.3e}")
        print(f"  ||dt2||              = {dt2:.3e}")
        print(f"  ||dt2_active||       = {dt2_active:.3e}")
        print(f"  ||dt2_environment||  = {dt2_env:.3e}")
        print(f"  active preservation  = {active_preservation:.3e}")
        print(f"  dE                   = {e_factorized - e_dense:.3e}")

        self.assertLess(active_preservation, 1.0e-12)
        self.assertLess(dt2_active, 1.0e-12)
        self.assertLess(dt1, 1.0e-10)
        self.assertLess(dt2_env, 1.0e-7)
        self.assertAlmostEqual(e_factorized, e_dense, delta=1.0e-7)
        self.assertTrue(np.isfinite([dt1, dt2, e_dense, e_factorized]).all())

    def test_unfactorized_t1_transform_active_space_one_step_diagnostic(self):
        mol = gto.M(
            atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            basis="cc-pvtz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.verbose = 0
        mf.kernel()

        avas_obj = avas.AVAS(mf, ["O 2p"], minao="sto-3g", openshell_option=3)
        avas_obj.with_iao = True
        avas_obj.threshold = 1e-7
        _, _, c_lo = avas_obj.kernel()
        act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        act_particle = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]
        frags = [[act_hole, act_particle]]

        eris = df_eri.ERIs(mf, c_lo)
        ll = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_con_tol=1e-8,
            ll_max_its=20,
        )
        t1, t2 = ll.init_amps()

        _, _, _, t2_new = ll.update_amps_unfactorized(t1.copy(), t2.copy())
        active_mask = mask_active_t2(t2.shape, frags)
        active_preservation = np.linalg.norm((t2_new - t2)[active_mask])

        print("\nOne-step T1_transform active-space diagnostic:")
        print(f"  active preservation = {active_preservation:.12e}")

        self.assertLess(active_preservation, 1e-12)

    def test_t2_error_residual_for_t1_transform_and_dense_laplace(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        mol = gto.M(
            atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            basis="cc-pvdz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.verbose = 0
        mf.kernel()

        avas_obj = avas.AVAS(mf, ["O 2p"], minao="sto-3g", openshell_option=3)
        avas_obj.with_iao = True
        avas_obj.threshold = 1e-7
        _, _, c_lo = avas_obj.kernel()
        act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        act_particle = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]
        frags = [[act_hole, act_particle]]

        self.assertGreater(len(act_hole), 0)
        self.assertGreater(len(act_particle), 0)

        eris = df_eri.ERIs(mf, c_lo)
        base = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
        )
        t1_0, t2_0 = base.init_amps()

        common_kwargs = dict(
            ll_kernel_type="unfactorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
            ll_con_tol=1e-8,
            ll_max_its=50,
        )
        t1_transform = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_method="T1_transform",
            **common_kwargs,
        )
        t1_t1, t2_t1 = t1_transform.kernel(t1_0.copy(), t2_0.copy())
        t2_no_dc_t1 = no_dc_laplace_t2(t1_transform, t1_t1)
        t1_error = projected_t2_error_norm(
            t1_transform, t1_t1, t2_t1, t2_no_dc_t1, frags
        )

        dense_laplace = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_method="sylvester_laplace",
            **common_kwargs,
        )
        t1_dense, t2_dense = dense_laplace.kernel(t1_0.copy(), t2_0.copy())
        t2_no_dc_dense = no_dc_laplace_t2(dense_laplace, t1_dense)
        dense_error = projected_t2_error_norm(
            dense_laplace, t1_dense, t2_dense, t2_no_dc_dense, frags
        )

        print("\nEq:T2_error projected residual diagnostic:")
        print(f"  T1 transform   = {t1_error:.12e}")
        print(f"  Dense Laplace  = {dense_error:.12e}")

        self.assertLess(t1_error, 5.0e-8)
        self.assertLess(dense_error, 5.0e-8)

    def test_factorized_laplace_boundary_t2_error_residual(self):
        root = laplace_minimax_root()
        if root is None:
            self.skipTest("laplace-minimax submodule is not available")

        mol = gto.M(
            atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            basis="cc-pvdz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.verbose = 0
        mf.kernel()

        avas_obj = avas.AVAS(mf, ["O 2p"], minao="sto-3g", openshell_option=3)
        avas_obj.with_iao = True
        avas_obj.threshold = 1e-7
        _, _, c_lo = avas_obj.kernel()
        act_hole = np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        act_particle = np.where(avas_obj.vir_weights > avas_obj.threshold)[0]
        frags = [[act_hole, act_particle]]

        self.assertGreater(len(act_hole), 0)
        self.assertGreater(len(act_particle), 0)

        eris = df_eri.ERIs(mf, c_lo)
        base = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
        )
        t1_0, t2_0 = base.init_amps()

        factorized_laplace = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="sylvester_laplace_factorized",
            ll_laplace_root=root,
            ll_laplace_npoints=16,
            ll_con_tol=1e-8,
            ll_max_its=50,
        )
        t1_factorized, t2_factorized = factorized_laplace.kernel(
            t1_0.copy(), t2_0.copy()
        )

        factorized_laplace.ll_laplace_quad = None
        t2_no_dc = no_dc_laplace_t2(factorized_laplace, t1_factorized)
        boundary_error = projected_t2_error_boundary_norm(
            factorized_laplace, t1_factorized, t2_factorized, t2_no_dc, frags
        )

        print("\nFactorized Eq:T2_error boundary residual diagnostic:")
        print(f"  Factorized Laplace  = {boundary_error:.12e}")

        self.assertLess(boundary_error, 5.0e-8)

    def test_lowlevel_t1_transform_final_energy_is_finite_for_h2o_ccpvtz(self):
        mol = gto.M(
            atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            basis="cc-pvtz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.verbose = 0
        mf.kernel()

        eris = df_eri.ERIs(mf, mf.mo_coeff)
        frags = [[[], []]]
        base = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_con_tol=1e-8,
            ll_max_its=20,
        )
        t1_0, t2_0 = base.init_amps()

        ll = dfrmpcc_lowlevel.MPCC_LL(
            mf,
            eris,
            frags,
            ll_kernel_type="unfactorized",
            ll_method="T1_transform",
            ll_con_tol=1e-8,
            ll_max_its=20,
        )
        ll.diis = False
        t1, t2 = ll.kernel(t1_0.copy(), t2_0.copy())
        energy = ll.get_energy(t1, t2)

        self.assertTrue(np.isfinite(energy))


if __name__ == "__main__":
    unittest.main()
