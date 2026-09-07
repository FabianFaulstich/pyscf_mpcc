from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyscf import lib, df
from pyscf.ao2mo import _ao2mo

from pyscf.mpcc import mpcc_tools


@dataclass
class ActiveERIs:
    """Three-index integrals in one active orbital and auxiliary space."""

    act_hole: np.ndarray
    act_particle: np.ndarray
    Lpq: np.ndarray
    Loo: np.ndarray
    Lov: np.ndarray
    Lvo: np.ndarray
    Lvv: np.ndarray
    aux_transform: Optional[np.ndarray]
    eigenvalues: Optional[np.ndarray]
    threshold: Optional[float]
    original_naux: int
    metric_relative_error: float

    @property
    def naux(self):
        return self.Lpq.shape[0]

    @property
    def compression(self):
        return 1.0 - self.naux / self.original_naux


def _fragment_key(frag):
    return (
        tuple(np.asarray(frag[0], dtype=np.intp).tolist()),
        tuple(np.asarray(frag[1], dtype=np.intp).tolist()),
    )


def _make_active_eris_from_packed(
    active_packed, act_hole, act_particle, naf_threshold=None
):
    """Compress packed active ``Lpq`` integrals, then split orbital blocks.

    The closed-shell NAF metric is formed without first separating OO, OV,
    VO, and VV.  Off-diagonal packed pairs have weight ``sqrt(2)``, so

        2 L(active) L(active)^T = 2 Koo + 4 Kov + 2 Kvv.
    """
    active_packed = np.asarray(active_packed)
    act_hole = np.asarray(act_hole, dtype=np.intp)
    act_particle = np.asarray(act_particle, dtype=np.intp)
    nact_hole = len(act_hole)
    nact = nact_hole + len(act_particle)
    npair = nact * (nact + 1) // 2
    if active_packed.ndim != 2 or active_packed.shape[1] != npair:
        raise ValueError(
            "Packed active integrals have an inconsistent orbital-pair dimension"
        )
    original_naux = active_packed.shape[0]

    eigenvalues = None
    aux_transform = None
    metric_relative_error = 0.0
    transformed_packed = active_packed

    if naf_threshold is not None:
        naf_threshold = float(naf_threshold)
        if not 0.0 <= naf_threshold < 1.0:
            raise ValueError("naf_threshold must satisfy 0 <= threshold < 1")

        pair_p, pair_q = np.tril_indices(nact)
        pair_weights = np.ones(npair, dtype=active_packed.dtype)
        pair_weights[pair_p != pair_q] = np.sqrt(2.0)
        metric_factors = active_packed * pair_weights
        metric = 2.0 * lib.dot(metric_factors, metric_factors.T)
        del metric_factors
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        del metric
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        lambda_max = eigenvalues[0] if eigenvalues.size else 0.0
        if lambda_max <= 0.0:
            raise ValueError("Cannot construct NAFs from a zero active-integral metric")

        keep = eigenvalues / lambda_max > naf_threshold
        if not np.any(keep):
            raise ValueError(
                "naf_threshold discarded every auxiliary function in the active space"
            )
        aux_transform = eigenvectors[:, keep]
        transformed_packed = lib.dot(aux_transform.T, active_packed)

        discarded = eigenvalues[~keep]
        denominator = np.linalg.norm(eigenvalues)
        if denominator:
            metric_relative_error = np.linalg.norm(discarded) / denominator

    transformed_lpq = lib.unpack_tril(
        transformed_packed, filltriu=lib.SYMMETRIC
    )
    Loo = transformed_lpq[:, :nact_hole, :nact_hole]
    Lov = transformed_lpq[:, :nact_hole, nact_hole:]
    # Use the same storage as Lov so exact three-index permutational symmetry
    # is maintained after the auxiliary transformation.
    Lvo = Lov.transpose(0, 2, 1)
    Lvv = transformed_lpq[:, nact_hole:, nact_hole:]

    return ActiveERIs(
        act_hole=act_hole,
        act_particle=act_particle,
        Lpq=transformed_lpq,
        Loo=Loo,
        Lov=Lov,
        Lvo=Lvo,
        Lvv=Lvv,
        aux_transform=aux_transform,
        eigenvalues=eigenvalues,
        threshold=naf_threshold,
        original_naux=original_naux,
        metric_relative_error=metric_relative_error,
    )


def _make_active_eris(active_lpq, act_hole, act_particle, naf_threshold=None):
    """Convenience wrapper accepting a square active ``Lpq`` tensor."""
    active_lpq = np.asarray(active_lpq)
    if active_lpq.ndim != 3 or active_lpq.shape[1] != active_lpq.shape[2]:
        raise ValueError("The active Lpq tensor must have two equal orbital axes")
    pair_p, pair_q = np.tril_indices(active_lpq.shape[1])
    return _make_active_eris_from_packed(
        active_lpq[:, pair_p, pair_q],
        act_hole,
        act_particle,
        naf_threshold=naf_threshold,
    )


class ERIs:

    def __init__(
        self, mf, mo_coeff=None, active_spaces=None, naf_threshold=None
    ):

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.nao = mf.mol.nao
        self.nocc = mf.mol.nelec[0]
        self.nvir = mf.mol.nao - self.nocc
        self.naux = self.with_df.get_naoaux()
        if mo_coeff is None:
           self.mo_coeff = mf.mo_coeff
        else:
           self.mo_coeff = mo_coeff       
        self.mo_energy = mf.mo_energy

        if active_spaces is None:
            active_spaces = []
        self.active_spaces = self._validate_active_spaces(active_spaces)
        if naf_threshold is not None and len(self.active_spaces) != 1:
            raise ValueError(
                "NAF compression currently requires exactly one active fragment"
            )
        self.naf_threshold = naf_threshold
        self._active_eris = {}

        fock_mo = self.mo_coeff.T @ mf.get_fock() @ self.mo_coeff
        self.foo = fock_mo[: self.nocc, : self.nocc]
        self.fvv = fock_mo[self.nocc : self.nao, self.nocc : self.nao]
        self.fov = fock_mo[: self.nocc, self.nocc : self.nao]

        # NOTE change convetion to virtual, occupied 
        self.eia = lib.direct_sum(
            "-i+a->ia", np.diag(self.foo), np.diag(self.fvv)
        )

        # NOTE Shall we keep this call here?
        self._make_df_eris()
        self._make_df_denominator()

    def _validate_active_spaces(self, active_spaces):
        validated = []
        for frag in active_spaces:
            if len(frag) != 2:
                raise ValueError(
                    "Each active fragment must contain occupied and virtual indices"
                )
            act_hole = np.asarray(frag[0], dtype=np.intp)
            act_particle = np.asarray(frag[1], dtype=np.intp)
            if act_hole.ndim != 1 or act_particle.ndim != 1:
                raise ValueError("Active orbital indices must be one-dimensional")
            if len(np.unique(act_hole)) != len(act_hole):
                raise ValueError("Active occupied indices must be unique")
            if len(np.unique(act_particle)) != len(act_particle):
                raise ValueError("Active virtual indices must be unique")
            if np.any(act_hole < 0) or np.any(act_hole >= self.nocc):
                raise ValueError("Active occupied index is out of range")
            if np.any(act_particle < 0) or np.any(act_particle >= self.nvir):
                raise ValueError("Active virtual index is out of range")
            validated.append((act_hole, act_particle))
        return validated

    def get_active_eris(self, frag):
        """Return integral blocks prepared for a particular active fragment."""
        key = _fragment_key(frag)
        try:
            return self._active_eris[key]
        except KeyError as err:
            raise ValueError(
                "Active integral blocks were not prepared for this fragment"
            ) from err

    def _make_df_denominator(self):

        self.dD = mpcc_tools.piv_chol_tensor(self.eia)
        
        # NOTE REMOVE THIS LATER!!!
        self.D = lib.direct_sum("ia+jb->ijab", self.eia, self.eia)

          
    def _make_df_eris(self):
        
        Loo = np.empty((self.naux, self.nocc, self.nocc))
        Lov = np.empty((self.naux, self.nocc, self.nvir))
        Lvv = np.empty((self.naux, self.nvir, self.nvir))
        active_packed = []
        active_pair_specs = []
        for act_hole, act_particle in self.active_spaces:
            nact = len(act_hole) + len(act_particle)
            pair_p, pair_q = np.tril_indices(nact)
            active_mo = np.concatenate(
                (act_hole, self.nocc + act_particle)
            )
            active_packed.append(
                np.empty((self.naux, len(pair_p)), dtype=Loo.dtype)
            )
            active_pair_specs.append((active_mo, pair_p, pair_q))
        mo = np.asarray(self.mo_coeff, order='F')
        ijslice = (0, self.nao, 0, self.nao)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(self.with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1 - p0, self.nao, self.nao)
            Loo[p0:p1] = Lpq[:, :self.nocc, :self.nocc]
            Lov[p0:p1] = Lpq[:, :self.nocc, self.nocc:]
            Lvv[p0:p1] = Lpq[:, self.nocc:, self.nocc:]
            for frag_packed, (active_mo, pair_p, pair_q) in zip(
                active_packed, active_pair_specs
            ):
                # Scope the combined active tensor while the full transformed
                # Lpq chunk is already resident.  OO/OV/VV are split only
                # after the common auxiliary transformation has been applied.
                frag_packed[p0:p1] = Lpq[
                    :, active_mo[pair_p], active_mo[pair_q]
                ]
        Lpq = None
        Lvo = Lov.transpose(0,2,1).reshape(self.naux,self.nvir,self.nocc)
        self.Loo = Loo
        self.Lov = Lov
        self.Lvo = Lvo
        self.Lvv = Lvv       

        for frag_packed, (act_hole, act_particle) in zip(
            active_packed, self.active_spaces
        ):
            active_eris = _make_active_eris_from_packed(
                frag_packed,
                act_hole,
                act_particle,
                naf_threshold=self.naf_threshold,
            )
            self._active_eris[_fragment_key((act_hole, act_particle))] = active_eris
            if self.naf_threshold is not None:
                lib.logger.info(
                    self.with_df,
                    "Active NAF rank %d/%d (%.1f%% reduction), threshold %.3g, "
                    "metric error %.3g",
                    active_eris.naux,
                    active_eris.original_naux,
                    100.0 * active_eris.compression,
                    self.naf_threshold,
                    active_eris.metric_relative_error,
                )

        # NOTE 06/04 debate about cache 

    # hold all a-a, a-i, i-i tensor segments here 
    # compute Lvv on the fly *lazy?

    # NOTE Remove after discussion on 06/11
    # NOTE move contractions into the "class"
    # NOTE Now comes the second set of equations
    def get_Aia(self, u):

        tmp_eri = np.einsum("Lkc,Lad->kcad", self.Lov, self.Lvv)

        return np.einsum("kicd, kcad -> ia", u, tmp_eri)

    def get_Bia(self, u):

        tmp_eri = np.einsum("Lki,Llc->kilc", self.Loo, self.Lov)

        return np.einsum("klac, -kilc -> ia", u, tmp_eri)

    def get_Cia(self, u):

        return np.einsum("ikac, kc-> ia", u, self.fov)

    def get_Aijab(self, t):

        tmp_eri = np.einsum("Lac, Lbd->acbd", self.Lvv, self.Lvv)

        return np.einsum("ijcd, acbd -> ijab", t, tmp_eri)

    def get_Bijab(self, t):

        tmp_eri = np.einsum("Lkc, Lld -> kcld", self.self.Lov, self.self.Lov)
        B = np.einsum("ijcd, kcld -> ijkl", t, tmp_eri)
        tmp_eri = np.einsum("Lki, Ljl -> ijkl", self.Loo, self.Loo)
        B = tmp_eri + B

        return np.einsum("klab, ijkl -> ijab", t, B)

    def get_Cijab(self, t):

        # NOTE tmp_eri is the same as in get_Bijab
        tmp_eri = np.einsum("Lkc, Lld -> kcld", self.Lov, self.Lov)
        C = np.einsum("liad, kcld -> kiac", t, tmp_eri)
        tmp_eri = np.einsum("Lki, Lac -> kiac", self.Loo, self.Lvv)
        C = tmp_eri - 0.5 * C

        return np.einsum("kjbc, kiac -> ijab", t, C)

    def get_Dijab(self, t):

        L = self.make_L()
        u = make_u(t)
        B = L + 0.5 * np.eisum("ilad, dclk -> icak", u, L)

        return 0.5 * np.einsum("jkbc, icak -> ijab", u, B)

    def get_Eijab(self, t):

        u = make_u(t)
        tmp_eri = np.einsum("Lld, Lkc -> ldkc", self.Lov, self.Lov)
        E = self.fvv - np.einsum("klbd, ldkc -> bc", u, tmp_eri)

        return np.einsum("ijac, bc -> ijab", t, E)

    def get_Gijab(self, t):

        u = make_u(t)
        tmp_eri = np.einsum("Lkd, Llc -> kdlc", self.Lov, self.Lov)
        G = self.foo + np.einsum("ljcd, kdlc -> jk", u, tmp_eri)

        return np.einsum("-ikab, jk -> ijab", t, G)

    def make_u(t):

        return 2 * t - np.transpose(t, (0, 1, 3, 2))

    def make_L(self):

        tmp_eri1 = np.einsum("Lpr ,Lqs -> rspq", self.fov, self.fov)
        tmp_eri2 = np.einsum("Lps ,Lqr -> rspq", self.fov, self.fov)

        return 2 * tmp_eri1 - tmp_eri2

    #####
