from pyscf import df
from pyscf import lib
import numpy
from dataclasses import dataclass


@dataclass
class _MvoT2Blocks:
    inactive_particle_active_hole: numpy.ndarray
    active_particle_inactive_hole: numpy.ndarray
    active_particle_active_hole: numpy.ndarray


class screened:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
        self.add_DCA = kwargs.get('DCA', True)
        self.set_fragment(frags)
    @property
    def nvir(self):
        return self.mf.mol.nao - self.nocc
        
    @property
    def nocc(self):
        return self.mf.mol.nelec[0]

    @property
    def naux(self):
        return self.with_df.get_naoaux()

    @property
    def act_hole(self):
        return self._act_hole
            
    @property
    def act_particle(self):
        return self._act_particle
        
    @property
    def inact_hole(self):
        return self._inact_hole
        
    @property
    def inact_particle(self):
        return self._inact_particle

    @property
    def aux_idx(self):
        return self._aux_idx

    def set_fragment(self, frag):
        """Update fragment indices and all fragment-dependent integral blocks."""
        self.frag = frag
        self._act_hole = numpy.asarray(frag[0], dtype=numpy.intp)
        self._act_particle = numpy.asarray(frag[1], dtype=numpy.intp)
        self._inact_hole = numpy.setdiff1d(
            numpy.arange(self.nocc), self._act_hole, assume_unique=False
        )
        self._inact_particle = numpy.setdiff1d(
            numpy.arange(self.nvir), self._act_particle, assume_unique=False
        )
        self._aux_idx = numpy.arange(self.naux)
        self._set_integral_blocks()

    def _available_memory_bytes(self, fraction=0.8):
        max_memory = getattr(self.mf, "max_memory", lib.param.MAX_MEMORY)
        current_memory = lib.current_memory()[0]
        return max(0, int((max_memory - current_memory) * fraction * 1e6))

    def _aux_block_size(self, elements_per_aux, dtype):
        if self.naux == 0:
            return 1
        itemsize = numpy.dtype(dtype).itemsize
        available = self._available_memory_bytes(fraction=0.5)
        if elements_per_aux <= 0 or available <= 0:
            return 1
        return min(self.naux, max(1, available // (itemsize * elements_per_aux)))

    def _antisymmetrize_t2(self, t2):
        return 2.0 * t2 - t2.transpose(0, 1, 3, 2)

    @staticmethod
    def _antisymmetrized_t2_block(t2, h1, h2, p1, p2):
        direct = t2[numpy.ix_(h1, h2, p1, p2)]
        exchange = t2[numpy.ix_(h1, h2, p2, p1)].transpose(0, 1, 3, 2)
        return 2.0 * direct - exchange



    def _split_moo(self, tensor):
        return [
            tensor[numpy.ix_(self.aux_idx, self.inact_hole, self.inact_hole)],
            tensor[numpy.ix_(self.aux_idx, self.inact_hole, self.act_hole)],
            tensor[numpy.ix_(self.aux_idx, self.act_hole, self.inact_hole)],
            tensor[numpy.ix_(self.aux_idx, self.act_hole, self.act_hole)],
        ]

    def _split_mvo(self, tensor):
        return [
            tensor[numpy.ix_(self.aux_idx, self.inact_particle, self.inact_hole)],
            tensor[numpy.ix_(self.aux_idx, self.inact_particle, self.act_hole)],
            tensor[numpy.ix_(self.aux_idx, self.act_particle, self.inact_hole)],
            tensor[numpy.ix_(self.aux_idx, self.act_particle, self.act_hole)],
        ]

    def _build_M_t1_intermediates_pair(self, t1):
        """Build active-excluded and active-included T1 intermediates together."""
        t1_aa = t1[numpy.ix_(self.act_hole, self.act_particle)]

        M0_full = 2.0 * lib.einsum("Lkc,kc->L", self._eris.Lov, t1)
        Moo_full_tensor = lib.einsum("Lia,ja->Lij", self._eris.Lov, t1)
        Mvo_full_tensor = lib.einsum("Lac,ic->Lai", self._eris.Lvv, t1)

        Moo_full = self._split_moo(Moo_full_tensor)
        Mvo_full = self._split_mvo(Mvo_full_tensor)

        delta_m0 = 2.0 * lib.einsum("Lkc,kc->L", self.Lov_aa, t1_aa)
        delta_moo_ia = lib.einsum("Lic,jc->Lij", self.Lov_ia, t1_aa)
        delta_moo_aa = lib.einsum("Lic,jc->Lij", self.Lov_aa, t1_aa)

        Lvv_ia = self._eris.Lvv[
            numpy.ix_(self.aux_idx, self.inact_particle, self.act_particle)
        ]
        delta_mvo_ia = lib.einsum("Lac,ic->Lai", Lvv_ia, t1_aa)
        delta_mvo_aa = lib.einsum("Lac,ic->Lai", self.Lvv_aa, t1_aa)

        Moo_base = [
            Moo_full[0],
            Moo_full[1] - delta_moo_ia,
            Moo_full[2],
            Moo_full[3] - delta_moo_aa,
        ]
        Mvo_base = [
            Mvo_full[0],
            Mvo_full[1] - delta_mvo_ia,
            Mvo_full[2],
            Mvo_full[3] - delta_mvo_aa,
        ]

        base = (M0_full - delta_m0, Moo_base, Mvo_base)
        full = (M0_full, Moo_full, Mvo_full)
        return base, full

    def _build_M_t1_intermediates(self, t1, include_active_terms):
        base, full = self._build_M_t1_intermediates_pair(t1)
        return full if include_active_terms else base

    @staticmethod
    def _unpack_fock_blocks(Fov, Fvv, Foo):
        Fov_ii, Fov_aa, Fov_ai, Fov_ia = Fov
        Fvv_ai, Fvv_aa_t1, Fvv_aa_t2 = Fvv
        Foo_ia, Foo_aa_t1, Foo_aa_t2 = Foo
        return Fov_ii, Fov_aa, Fov_ai, Fov_ia, Fvv_ai, Fvv_aa_t1, Fvv_aa_t2, Foo_ia, Foo_aa_t1, Foo_aa_t2

        # at this point we will classify integralsi: 

    def _set_integral_blocks(self):

        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle
        naux_idx = self.aux_idx
        
        self.Loo_ii = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, inact_hole)]
        self.Loo_ia = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, act_hole)]
        self.Loo_ai = self._eris.Loo[numpy.ix_(naux_idx, act_hole, inact_hole)]
        self.Loo_aa = self._eris.Loo[numpy.ix_(naux_idx, act_hole, act_hole)]

        # Only active-row Lvv blocks are retained.  The dominant inactive-row
        # blocks are consumed directly from _eris.Lvv when Mvo is built.
        self.Lvv_ai = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, inact_particle)]
        self.Lvv_aa = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, act_particle)]

        self.Lov_ii = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, inact_particle)]
        self.Lov_ia = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, act_particle)]
        self.Lov_ai = self._eris.Lov[numpy.ix_(naux_idx, act_hole, inact_particle)]
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)]

        self.Lvo_aa = self._eris.Lvo[numpy.ix_(naux_idx, act_particle, act_hole)]



    def _set_t1_blocks(self, t1):

        t1_ii = t1[numpy.ix_(self.inact_hole, self.inact_particle)]
        t1_ia = t1[numpy.ix_(self.inact_hole, self.act_particle)]
        t1_ai = t1[numpy.ix_(self.act_hole, self.inact_particle)]
        t1_aa = t1[numpy.ix_(self.act_hole, self.act_particle)]

        return t1_ii, t1_ia, t1_ai, t1_aa

    @staticmethod
    def _accumulate_block_additions(base, additions, resolve_block):
        result = base.copy()
        for group_name, block_key, scale in additions:
            block = resolve_block(group_name, block_key)
            if block.size == 0:
                continue
            result += scale * block
        return result

    @staticmethod
    def _accumulate_block_contractions(base, specs, resolve_block):
        result = base.copy()
        for einsum_expr, block_refs, scale in specs:
            operands = [resolve_block(group_name, block_key) for group_name, block_key in block_refs]
            if any(operand.size == 0 for operand in operands):
                continue
            result += scale * lib.einsum(einsum_expr, *operands)
        return result

    def _build_t1_transform_J_blocks_rest(self, t1, Moo, Mvo):

        t1_blocks = dict(zip(("ii", "ia", "ai", "aa"), self._set_t1_blocks(t1)))
        moo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Moo))
        mvo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Mvo))
        def resolve_block(group_name, block_key):
            if group_name == "t1":
                return t1_blocks[block_key]
            if group_name == "Moo":
                return moo_blocks[block_key]
            if group_name == "Mvo":
                return mvo_blocks[block_key]
            return getattr(self, f"{group_name}_{block_key}")

        joo_ai = self._accumulate_block_additions(
            self.Loo_ai,
            [("Moo", "ai", 1.0)],
            resolve_block,
        )
        joo_ia = self._accumulate_block_additions(
            self.Loo_ia,
            [("Moo", "ia", 1.0)],
            resolve_block,
        )

        jvv_ai = self._accumulate_block_contractions(
            self.Lvv_ai,
            [
                ("Lkb,ka->Lab", [("Lov", "ii"), ("t1", "ia")], -1.0),
                ("Lkb,ka->Lab", [("Lov", "ai"), ("t1", "aa")], -1.0),
            ],
            resolve_block,
        )

        return joo_ai, joo_ia, jvv_ai

    def _build_t1_transform_J_blocks_aa(self, t1, Moo, Mvo, Mvo_t2, joo_ia, include_active_terms):

        t1_blocks = dict(zip(("ii", "ia", "ai", "aa"), self._set_t1_blocks(t1)))
        moo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Moo))
        mvo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Mvo))
        mvo_t2_blocks = {
            "aa": Mvo_t2.active_particle_active_hole,
        }

        def resolve_block(group_name, block_key):
            if group_name == "t1":
                return t1_blocks[block_key]
            if group_name == "Moo":
                return moo_blocks[block_key]
            if group_name == "Mvo":
                return mvo_blocks[block_key]
            if group_name == "Mvo_t2":
                return mvo_t2_blocks[block_key]
            return getattr(self, f"{group_name}_{block_key}")

        def contract_sum(specs, einsum_expr, block_resolver):
            result = None
            for block_refs, scale in specs:
                operands = [block_resolver(group_name, block_key) for group_name, block_key in block_refs]
                if any(operand.size == 0 for operand in operands):
                    continue
                term = scale * lib.einsum(einsum_expr, *operands)
                result = term if result is None else result + term
            if result is None:
                first_refs, _ = specs[0]
                first_ops = [block_resolver(group_name, block_key) for group_name, block_key in first_refs]
                result = 0.0 * lib.einsum(einsum_expr, *first_ops)
            return result

        joo_aa = self._accumulate_block_additions(
            self.Loo_aa,
            [("Moo", "aa", 1.0)],
            resolve_block,
        )

        jvv_aa = self.Lvv_aa.copy()
      # jvv_aa += contract_sum(
      #     [
      #         ([("Lov", "ia"), ("t1", "ia")], -1.0),
      #         ([("Lov", "aa"), ("t1", "aa")], -1.0),
      #     ],
      #     "Lkb,ka->Lab",
      #     resolve_block,
      # )

        jvv_aa += contract_sum(
            [
                ([("Lov", "ia"), ("t1", "ia")], -1.0),
            ],
            "Lkb,ka->Lab",
            resolve_block,
        )

        if include_active_terms:

            jvv_aa += contract_sum(
                [
                   ([("Lov", "aa"), ("t1", "aa")], -1.0),
                ],
                "Lkb,ka->Lab",
                resolve_block,
            )


        jvo_aa = self._accumulate_block_additions(
            self.Lvo_aa,
            [("Mvo", "aa", 1.0), ("Mvo_t2", "aa", 1.0)],
            resolve_block,
        )

        def resolve_j_block(group_name, block_key):
            if group_name == "Joo" and block_key == "ia":
                return joo_ia
            return resolve_block(group_name, block_key)

        jvo_aa += contract_sum(
            [([("Joo", "ia"), ("t1", "ia")], -1.0)],
            "Lki,ka->Lai",
            resolve_j_block,
        )

        return joo_aa, jvv_aa, jvo_aa

    
    def _build_t1_transform_fock_blocks_rest(self, t1, M, Moo, Mvo):

        t1_blocks = dict(zip(("ii", "ia", "ai", "aa"), self._set_t1_blocks(t1)))
        moo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Moo))
        mvo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Mvo))
        scalar_blocks = {"full": M}

        def resolve_block(group_name, block_key):
            if group_name == "t1":
                return t1_blocks[block_key]
            if group_name == "M":
                return scalar_blocks[block_key]
            if group_name == "Moo":
                return moo_blocks[block_key]
            if group_name == "Mvo":
                return mvo_blocks[block_key]
            return getattr(self, f"{group_name}_{block_key}")

        def contract_sum(specs, block_resolver):
            result = None
            for einsum_expr, block_refs, scale in specs:
                operands = [block_resolver(group_name, block_key) for group_name, block_key in block_refs]
                if any(operand.size == 0 for operand in operands):
                    continue
                term = scale * lib.einsum(einsum_expr, *operands)
                result = term if result is None else result + term
            if result is None:
                first_expr, first_refs, _ = specs[0]
                first_ops = [block_resolver(group_name, block_key) for group_name, block_key in first_refs]
                result = 0.0 * lib.einsum(first_expr, *first_ops)
            return result


        foo_ia = self._eris.foo[numpy.ix_(self.inact_hole, self.act_hole)].copy()
        foo_ia += contract_sum(
            [
                ("Lij,L->ij", [("Loo", "ia"), ("M", "full")], 1.0),
                ("Lmj,Lim->ij", [("Loo", "ia"), ("Moo", "ii")], -1.0),
                ("Lmj,Lim->ij", [("Loo", "aa"), ("Moo", "ia")], -1.0),
            ],
            resolve_block,
        )

        fvv_ai = self._eris.fvv[numpy.ix_(self.act_particle, self.inact_particle)].copy()
        fvv_ai += contract_sum(
            [
                ("Lab,L->ab", [("Lvv", "ai"), ("M", "full")], 1.0),
                ("Lmb,Lam->ab", [("Lov", "ii"), ("Mvo", "ai")], -1.0),
                ("Lmb,Lam->ab", [("Lov", "ai"), ("Mvo", "aa")], -1.0),
            ],
            resolve_block,
        )

        fov_ii = self._eris.fov[numpy.ix_(self.inact_hole, self.inact_particle)].copy()
        fov_ii += contract_sum(
            [
                ("Lia,L->ia", [("Lov", "ii"), ("M", "full")], 1.0),
                ("Lma,Lim->ia", [("Lov", "ii"), ("Moo", "ii")], -1.0),
                ("Lma,Lim->ia", [("Lov", "ai"), ("Moo", "ia")], -1.0),
            ],
            resolve_block,
        )

        fov_ia = self._eris.fov[numpy.ix_(self.inact_hole, self.act_particle)].copy()
        fov_ia += contract_sum(
            [
                ("Lia,L->ia", [("Lov", "ia"), ("M", "full")], 1.0),
                ("Lma,Lim->ia", [("Lov", "ia"), ("Moo", "ii")], -1.0),
                ("Lma,Lim->ia", [("Lov", "aa"), ("Moo", "ia")], -1.0),
            ],
            resolve_block,
        )

        fov_ai = self._eris.fov[numpy.ix_(self.act_hole, self.inact_particle)].copy()
        fov_ai += contract_sum(
            [
                ("Lia,L->ia", [("Lov", "ai"), ("M", "full")], 1.0),
                ("Lma,Lim->ia", [("Lov", "ii"), ("Moo", "ai")], -1.0),
                ("Lma,Lim->ia", [("Lov", "ai"), ("Moo", "aa")], -1.0),
            ],
            resolve_block,
        )

        return foo_ia, fvv_ai, fov_ii, fov_ia, fov_ai

    def _build_t1_transform_fock_blocks_aa(self, t1, M, Moo, Mvo, fov_ai, fov_ia):
        t1_blocks = dict(zip(("ii", "ia", "ai", "aa"), self._set_t1_blocks(t1)))
        moo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Moo))
        mvo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Mvo))
        scalar_blocks = {"full": M}

        def resolve_block(group_name, block_key):
            if group_name == "t1":
                return t1_blocks[block_key]
            if group_name == "M":
                return scalar_blocks[block_key]
            if group_name == "Moo":
                return moo_blocks[block_key]
            if group_name == "Mvo":
                return mvo_blocks[block_key]
            return getattr(self, f"{group_name}_{block_key}")

        def contract_sum(specs, block_resolver):
            result = None
            for einsum_expr, block_refs, scale in specs:
                operands = [block_resolver(group_name, block_key) for group_name, block_key in block_refs]
                if any(operand.size == 0 for operand in operands):
                    continue
                term = scale * lib.einsum(einsum_expr, *operands)
                result = term if result is None else result + term
            if result is None:
                first_expr, first_refs, _ = specs[0]
                first_ops = [block_resolver(group_name, block_key) for group_name, block_key in first_refs]
                result = 0.0 * lib.einsum(first_expr, *first_ops)
            return result



        foo_aa = self._eris.foo[numpy.ix_(self.act_hole, self.act_hole)].copy()
        foo_aa += contract_sum(
            [
                ("Lij,L->ij", [("Loo", "aa"), ("M", "full")], 1.0),
                ("Lmj,Lim->ij", [("Loo", "ia"), ("Moo", "ai")], -1.0),
                ("Lmj,Lim->ij", [("Loo", "aa"), ("Moo", "aa")], -1.0),
            ],
            resolve_block,
        )

        fvv_aa = self._eris.fvv[numpy.ix_(self.act_particle, self.act_particle)].copy()
        fvv_aa += contract_sum(
            [
                ("Lab,L->ab", [("Lvv", "aa"), ("M", "full")], 1.0),
                ("Lmb,Lam->ab", [("Lov", "ia"), ("Mvo", "ai")], -1.0),
                ("Lmb,Lam->ab", [("Lov", "aa"), ("Mvo", "aa")], -1.0),
            ],
            resolve_block,
        )

        fov_aa = self._eris.fov[numpy.ix_(self.act_hole, self.act_particle)].copy()
        fov_aa += contract_sum(
            [
                ("Lia,L->ia", [("Lov", "aa"), ("M", "full")], 1.0),
                ("Lma,Lim->ia", [("Lov", "ia"), ("Moo", "ai")], -1.0),
                ("Lma,Lim->ia", [("Lov", "aa"), ("Moo", "aa")], -1.0),
            ],
            resolve_block,
        )

        fov_blocks = {"ai": fov_ai, "ia": fov_ia}

        def resolve_fock_block(group_name, block_key):
            if group_name == "Fov":
                return fov_blocks[block_key]
            return resolve_block(group_name, block_key)

        foo_aa_t1 = foo_aa + contract_sum(
            [("ic,jc->ij", [("Fov", "ai"), ("t1", "ai")], 0.5)],
            resolve_fock_block,
        )
        foo_aa_t2 = foo_aa + contract_sum(
            [("ic,jc->ij", [("Fov", "ai"), ("t1", "ai")], 1.0)],
            resolve_fock_block,
        )
        fvv_aa_t1 = fvv_aa + contract_sum(
            [("lb,la->ab", [("Fov", "ia"), ("t1", "ia")], -0.5)],
            resolve_fock_block,
        )
        fvv_aa_t2 = fvv_aa + contract_sum(
            [("lb,la->ab", [("Fov", "ia"), ("t1", "ia")], -1.0)],
            resolve_fock_block,
        )

        return foo_aa_t1, foo_aa_t2, fvv_aa_t1, fvv_aa_t2, fov_aa

    def _contract_mvo_t2_block(
        self, t2, target_holes, target_particles, exclude_all_active=False
    ):
        """Contract Lov with antisymmetrized T2 without materializing either full tensor."""
        dtype = numpy.result_type(self._eris.Lov, t2)
        result = numpy.zeros(
            (self.naux, len(target_particles), len(target_holes)), dtype=dtype
        )
        if result.size == 0:
            return result

        hole_groups = ((self.inact_hole, False), (self.act_hole, True))
        particle_groups = ((self.inact_particle, False), (self.act_particle, True))
        for source_holes, holes_active in hole_groups:
            if source_holes.size == 0:
                continue
            for source_particles, particles_active in particle_groups:
                if source_particles.size == 0:
                    continue
                if exclude_all_active and holes_active and particles_active:
                    continue

                elements_per_particle = (
                    self.naux * len(source_holes)
                    + 2
                    * len(target_holes)
                    * len(source_holes)
                    * len(target_particles)
                )
                itemsize = numpy.dtype(dtype).itemsize
                available = self._available_memory_bytes(fraction=0.5)
                block_size = len(source_particles)
                if available <= 0:
                    block_size = 1
                elif elements_per_particle:
                    block_size = min(
                        block_size,
                        max(1, available // (itemsize * elements_per_particle)),
                    )

                for p0, p1 in lib.prange(0, len(source_particles), block_size):
                    contracted_particles = source_particles[p0:p1]
                    lov = self._eris.Lov[
                        numpy.ix_(self.aux_idx, source_holes, contracted_particles)
                    ]
                    direct = t2[
                        numpy.ix_(
                            target_holes,
                            source_holes,
                            target_particles,
                            contracted_particles,
                        )
                    ]
                    exchange = t2[
                        numpy.ix_(
                            target_holes,
                            source_holes,
                            contracted_particles,
                            target_particles,
                        )
                    ]
                    result += 2.0 * lib.einsum("Lkc,ikac->Lai", lov, direct)
                    result -= lib.einsum("Lkc,ikca->Lai", lov, exchange)
        return result

    def _build_Mvo_t2_blocks(self, t2):
        return _MvoT2Blocks(
            inactive_particle_active_hole=self._contract_mvo_t2_block(
                t2, self.act_hole, self.inact_particle
            ),
            active_particle_inactive_hole=self._contract_mvo_t2_block(
                t2, self.inact_hole, self.act_particle
            ),
            active_particle_active_hole=self._contract_mvo_t2_block(
                t2,
                self.act_hole,
                self.act_particle,
                exclude_all_active=True,
            ),
        )

    def create_M_intermediates(self, t1, t2, include_active_terms):
        M0, Moo, Mvo = self._build_M_t1_intermediates(t1, include_active_terms)
        return M0, Moo, Mvo, self._build_Mvo_t2_blocks(t2)


    def add_t2_to_fock(self, Fvv, Foo, Mvo_t2, t2):    

        #construct antisymmetrized t2:

        Foo_ia, Foo_aa_t1, Foo_aa_t2 = Foo
        Fvv_ai, Fvv_aa_t1, Fvv_aa_t2 = Fvv


        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle


        #calculate the additional contributions to Fvv_ai, Here we need to construct Mvo_t2 tensor from all active t2 amplitudes.
        t2_act = t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)]
        t2_act_antisym = self._antisymmetrize_t2(t2_act)
        Mvo_t2_additional = lib.einsum("Lkc, ikac -> Lai", self.Lov_aa, t2_act_antisym) 

        #generate the intermediate with full arrays 
        Foo_aa = lib.einsum(
            "Lie,Lej->ij", self.Lov_ai, Mvo_t2.inactive_particle_active_hole
        )
        Foo_aa += lib.einsum(
            "Lie,Lej->ij", self.Lov_aa, Mvo_t2.active_particle_active_hole
        )

        Foo_aa_t1 += Foo_aa
        Foo_aa_t2 += Foo_aa

        Foo_ia += lib.einsum(
            "Lie,Lej->ij", self.Lov_ii, Mvo_t2.inactive_particle_active_hole
        )
        Foo_ia += lib.einsum(
            "Lie,Lej->ij", self.Lov_ia, Mvo_t2.active_particle_active_hole
        )
        Foo_ia += lib.einsum("Lie,Lej->ij", self.Lov_ia, Mvo_t2_additional) #additional contribution from active t2 amplitudes
  
        Fvv_aa = -lib.einsum(
            "Lmb,Lam->ab", self.Lov_ia, Mvo_t2.active_particle_inactive_hole
        )
        Fvv_aa -= lib.einsum(
            "Lmb,Lam->ab", self.Lov_aa, Mvo_t2.active_particle_active_hole
        )

        Fvv_aa_t1 += Fvv_aa
        Fvv_aa_t2 += Fvv_aa

        Fvv_ai -= lib.einsum(
            "Lmb,Lam->ab", self.Lov_ii, Mvo_t2.active_particle_inactive_hole
        )
        Fvv_ai -= lib.einsum(
            "Lmb,Lam->ab", self.Lov_ai, Mvo_t2.active_particle_active_hole
        )
        Fvv_ai -= lib.einsum("Lmb,Lam->ab",self.Lov_ai,Mvo_t2_additional) #additional contribution from active t2 amplitudes

        Foo = [Foo_ia, Foo_aa_t1, Foo_aa_t2]
        Fvv = [Fvv_ai, Fvv_aa_t1, Fvv_aa_t2]

        return Foo, Fvv


    def R1_residue_active(self, t1, t2, Fov, Fvv, Foo, M0, Mvo, Mvo_t2):
         # Get index arrays for inactive holes and active particles
         inact_hole = self.inact_hole
         act_hole = self.act_hole
         inact_particle = self.inact_particle
         act_particle = self.act_particle
     

         t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)

         # You may need to define or pass Fov_aa, Fov_ii, Foo_ai, Fvv_ai, etc.
         # For now, let's assume they are slices of Fov, Fvv, Foo as in your t1_transform
         Fov_ii, Fov_aa, Fov_ai, Fov_ia, Fvv_ai, _, _, Foo_ia, _, _ = self._unpack_fock_blocks(Fov, Fvv, Foo)
         Mvo_ii, Mvo_ia, Mvo_ai, Mvo_aa = Mvo

         #R1 = Fov_aa.copy()
         R1 = self._eris.fov[numpy.ix_(act_hole, act_particle)].copy()
     
         Foo_ia_tmp = Foo_ia.copy()
         Fvv_ai_tmp = Fvv_ai.copy() 

         Foo_ia_tmp += lib.einsum("ic,jc->ij", Fov_ii, t1_ai)*0.5
         Foo_ia_tmp += lib.einsum("ic,jc->ij", Fov_ia, t1_aa)*0.5
     
         R1 -= lib.einsum("ki,ka->ia", Foo_ia_tmp, t1_ia)  
     
         Fvv_ai_tmp -= lib.einsum("lb,la -> ab", Fov_ii, t1_ia)*0.5
#new
         Fvv_ai_tmp -= lib.einsum("lb,la -> ab", Fov_ai, t1_aa)*0.5
     
         R1 += lib.einsum("ab,ib->ia", Fvv_ai_tmp, t1_ai)
         t2_aiai = self._antisymmetrized_t2_block(
             t2, act_hole, inact_hole, act_particle, inact_particle
         )
         R1 -= lib.einsum("jb, ijab -> ia", Fov_ii,
                          t2_aiai)

         t2_aaai = self._antisymmetrized_t2_block(
             t2, act_hole, act_hole, act_particle, inact_particle
         )
         R1 += lib.einsum("jb, ijab -> ia", Fov_ai, 
                          t2_aaai)

         t2_aiaa = self._antisymmetrized_t2_block(
             t2, act_hole, inact_hole, act_particle, act_particle
         )
         R1 += lib.einsum("jb, ijab -> ia", Fov_ia, 
                          t2_aiaa)
     
       
         R1 += lib.einsum("Lia, L -> ia", self.Lov_aa, M0)
         R1 -= lib.einsum("Lji,Laj->ia", self.Loo_ia, Mvo_ai)
         R1 -= lib.einsum("Lji,Laj->ia", self.Loo_aa, Mvo_aa)
         
         R1 -= lib.einsum(
             "Lji,Laj->ia", self.Loo_ia, Mvo_t2.active_particle_inactive_hole
         )
         R1 -= lib.einsum(
             "Lji,Laj->ia", self.Loo_aa, Mvo_t2.active_particle_active_hole
         )

         R1 += lib.einsum(
             "Lae,Lei->ia", self.Lvv_ai, Mvo_t2.inactive_particle_active_hole
         )
         R1 += lib.einsum(
             "Lae,Lei->ia", self.Lvv_aa, Mvo_t2.active_particle_active_hole
         )

        # Foo_ai -= lib.einsum("me,ie->im", Fov_ii, t1_ai)*0.5
        # Fvv_ai += lib.einsum("me, ma -> ae", Fov_ii, t1_ia)*0.5

         return R1

    def _ppl_contraction(self, left, right, t2, first_particles, second_particles):
        """Evaluate a PPL term, blocking only when its dense form exceeds memory."""
        nact_hole = len(self.act_hole)
        nleft = left.shape[1]
        nright = right.shape[1]
        nfirst = len(first_particles)
        nsecond = len(second_particles)
        dtype = numpy.result_type(left, right, t2)
        result_shape = (nact_hole, nact_hole, nleft, nright)
        result_elements = nact_hole * nact_hole * nleft * nright
        if nfirst == 0 or nsecond == 0 or result_elements == 0:
            return numpy.zeros(result_shape, dtype=dtype)

        itemsize = numpy.dtype(dtype).itemsize
        result_nbytes = itemsize * result_elements
        w_elements = nleft * nright * nfirst * nsecond
        t2_elements = nact_hole * nact_hole * nfirst * nsecond
        required_bytes = itemsize * (w_elements + t2_elements) + result_nbytes
        available_bytes = self._available_memory_bytes(fraction=0.8)

        if required_bytes <= available_bytes:
            w_abef = lib.einsum("Lae,Lbf->abef", left, right)
            t2_block = t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    first_particles,
                    second_particles,
                )
            ]
            return lib.einsum("abef,ijef->ijab", w_abef, t2_block)

        self._ppl_used_blocking = True
        result = numpy.zeros(result_shape, dtype=dtype)
        bytes_per_first = itemsize * nsecond * (
            nleft * nright + nact_hole * nact_hole
        )
        tile_budget = max(0, available_bytes - result_nbytes)
        block_size = max(1, tile_budget // max(1, bytes_per_first))
        block_size = min(nfirst, block_size)
        for p0, p1 in lib.prange(0, nfirst, block_size):
            particle_slice = first_particles[p0:p1]
            w_abef = lib.einsum("Lae,Lbf->abef", left[:, :, p0:p1], right)
            t2_block = t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    particle_slice,
                    second_particles,
                )
            ]
            result += lib.einsum("abef,ijef->ijab", w_abef, t2_block)
        return result

    def R2_residue_active(self, t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo):

        inact_hole = self.inact_hole
        act_hole = self.act_hole
        inact_particle = self.inact_particle
        act_particle = self.act_particle

        #get active hole dimensions:
        n_act_hole = len(act_hole)
        #get active particle dimensions:
        n_act_particle = len(act_particle)

        Joo_ai = Joo[0]
        Joo_ia = Joo[2]
        Jvv_ai = Jvv[0]
        Jvv_aa = Jvv[1]
        Joo_aa = Joo[1]

        Fov_ii, Fov_aa, Fov_ai, Fov_ia, Fvv_ai, _, _, Foo_ia, _, _ = self._unpack_fock_blocks(Fov, Fvv, Foo)
        
        t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)

        #Non DCA terms:
        if (self.add_DCA):
           Imbje, Imbej, Imnij = self.t2_transform_quadratic_inactive(t2)  


        # Factorized PPL terms.  The two output virtual indices are active, so
        # retain the faster dense path whenever its actual footprint fits.
        self._ppl_used_blocking = False
        R2 = self._ppl_contraction(
            Jvv_ai, Jvv_ai, t2, inact_particle, inact_particle
        )
        R2 += self._ppl_contraction(
            Jvv_ai, Jvv_aa, t2, inact_particle, act_particle
        )
        R2 += self._ppl_contraction(
            Jvv_aa, Jvv_ai, t2, act_particle, inact_particle
        )
        #HHL
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_ia, Joo_ia) 
        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(inact_hole, inact_hole,numpy.arange(n_act_hole),numpy.arange(n_act_hole))]

        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(inact_hole, inact_hole, act_particle, act_particle)])

        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_aa, Joo_ia) 
        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(act_hole, inact_hole,numpy.arange(n_act_hole),numpy.arange(n_act_hole))]

        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(act_hole, self.inact_hole, act_particle, act_particle)])

        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_ia, Joo_aa)

        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(inact_hole,act_hole,numpy.arange(n_act_hole),numpy.arange(n_act_hole))]

        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(inact_hole, act_hole, act_particle, act_particle)])

        Wijmn = None


        #Fock matrix contribution:

        Foo_ia_tmp = Foo_ia.copy()
        Fvv_ai_tmp = Fvv_ai.copy()

        #Foo_aa += lib.einsum("me,ie->im", Fov_ai, t1_ai)

        Foo_ia_tmp += lib.einsum("ic,jc->ij", Fov_ii, t1_ai)
        Foo_ia_tmp += lib.einsum("ic,jc->ij", Fov_ia, t1_aa)

        R2_tmp = -lib.einsum("mi, mjab -> ijab", Foo_ia_tmp, t2[numpy.ix_(inact_hole, act_hole, act_particle, act_particle)]) #only one possibility m has to be inactive.

        #Fvv -= lib.einsum("me, ma -> ae", Fov, t1)

        #Fvv_aa -= lib.einsum("me, ma -> ae", Fov_ia, t1_ia)

        Fvv_ai_tmp -= lib.einsum("lb,la->ab", Fov_ii, t1_ia)
        Fvv_ai_tmp -= lib.einsum("lb,la->ab", Fov_ai, t1_aa)

        R2_tmp += lib.einsum("bc,ijac->ijab", Fvv_ai_tmp, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)]) # only one possibility e has to be inactive.


        #N3V3 terms:

        W_mbje = lib.einsum("Lmj, Lbe -> mbje", Joo_ia, Jvv_ai) 

        if (self.add_DCA):
            W_mbje  -= Imbje[numpy.ix_(inact_hole,numpy.arange(n_act_particle),numpy.arange(n_act_hole),inact_particle)]


        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(act_hole, inact_hole, act_particle, inact_particle)])

        W_mbje = lib.einsum("Lmj,Lbe->mbje", Joo_aa, Jvv_ai)

        if (self.add_DCA):
            W_mbje  -= Imbje[numpy.ix_(act_hole,numpy.arange(n_act_particle),numpy.arange(n_act_hole),inact_particle)]

        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])

        W_mbje = lib.einsum("Lmj,Lbe->mbje",Joo_ia,Jvv_aa)

        if (self.add_DCA):
            W_mbje  -= Imbje[numpy.ix_(inact_hole,numpy.arange(n_act_particle),numpy.arange(n_act_hole),act_particle)]

        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

####
        if (self.add_DCA):
            R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej[numpy.ix_(inact_hole,numpy.arange(n_act_particle), inact_particle,numpy.arange(n_act_hole))],
                                                       t2[numpy.ix_(act_hole, inact_hole, act_particle, inact_particle)])

            R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej[numpy.ix_(act_hole,numpy.arange(n_act_particle), inact_particle,numpy.arange(n_act_hole))],
                                                       t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])

            R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej[numpy.ix_(inact_hole,numpy.arange(n_act_particle), act_particle,numpy.arange(n_act_hole))],
                                                       t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

##########

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo_ia, Jvv_ai)

        if (self.add_DCA):
            W_jema -= 0.5*Imbje[numpy.ix_(inact_hole, numpy.arange(n_act_particle), numpy.arange(n_act_hole), inact_particle)]

        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, inact_hole, inact_particle, act_particle)])

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo_ia, Jvv_aa)      
        if (self.add_DCA):
            W_jema -= 0.5*Imbje[numpy.ix_(inact_hole, numpy.arange(n_act_particle), numpy.arange(n_act_hole), act_particle)]

        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo_aa, Jvv_ai)
        if (self.add_DCA):
            W_jema -= 0.5*Imbje[numpy.ix_(act_hole, numpy.arange(n_act_particle), numpy.arange(n_act_hole), inact_particle)]

        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)])

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))

        return R2

    def _df_imnij(self, lov_n, lov_m, t2_block, allow_dense=True):
        dtype = numpy.result_type(lov_n, lov_m, t2_block)
        result = numpy.zeros(
            (lov_n.shape[1], lov_m.shape[1], t2_block.shape[0], t2_block.shape[1]),
            dtype=dtype,
        )
        if result.size == 0 or t2_block.shape[2] == 0 or t2_block.shape[3] == 0:
            return result

        dense_elements = (
            lov_n.shape[1]
            * lov_m.shape[1]
            * lov_n.shape[2]
            * lov_m.shape[2]
        )
        dense_bytes = numpy.dtype(dtype).itemsize * dense_elements + result.nbytes
        if allow_dense and dense_bytes <= self._available_memory_bytes(fraction=0.8):
            Vnmef = lib.einsum("Lne,Lmf->nmef", lov_n, lov_m)
            return lib.einsum("mnef,ijef->mnij", Vnmef, t2_block)

        elements_per_aux = (
            lov_n.shape[1]
            * t2_block.shape[0]
            * t2_block.shape[1]
            * t2_block.shape[3]
        )
        block_size = self._aux_block_size(elements_per_aux, dtype)
        for p0, p1 in lib.prange(0, self.naux, block_size):
            tmp = lib.einsum(
                "Lne,ijef->Lnijf", lov_n[p0:p1], t2_block
            )
            result += lib.einsum("Lmf,Lnijf->nmij", lov_m[p0:p1], tmp)
        return result

    def _df_mbej(
        self, lov_n, lov_m, t2_block, exchange_output=False, allow_dense=True
    ):
        if exchange_output:
            nj, nn, nf, nb = t2_block.shape
        else:
            nj, nn, nb, nf = t2_block.shape
        dtype = numpy.result_type(lov_n, lov_m, t2_block)
        output_shape = (
            (lov_m.shape[1], nb, nj, lov_n.shape[2])
            if exchange_output
            else (lov_m.shape[1], nb, lov_n.shape[2], nj)
        )
        result = numpy.zeros(output_shape, dtype=dtype)
        if result.size == 0 or nn == 0 or nf == 0:
            return result

        dense_elements = (
            lov_n.shape[1]
            * lov_m.shape[1]
            * lov_n.shape[2]
            * lov_m.shape[2]
        )
        dense_bytes = numpy.dtype(dtype).itemsize * dense_elements + result.nbytes
        if allow_dense and dense_bytes <= self._available_memory_bytes(fraction=0.8):
            Vnmef = lib.einsum("Lne,Lmf->nmef", lov_n, lov_m)
            if exchange_output:
                return lib.einsum("nmef,jnfb->mbje", Vnmef, t2_block)
            return lib.einsum("nmef,jnbf->mbej", Vnmef, t2_block)

        elements_per_aux = lov_m.shape[1] * nj * nn * nb
        block_size = self._aux_block_size(elements_per_aux, dtype)
        for p0, p1 in lib.prange(0, self.naux, block_size):
            if exchange_output:
                tmp = lib.einsum(
                    "Lmf,jnfb->Lmjnb", lov_m[p0:p1], t2_block
                )
                result += lib.einsum(
                    "Lne,Lmjnb->mbje", lov_n[p0:p1], tmp
                )
            else:
                tmp = lib.einsum(
                    "Lmf,jnbf->Lmjnb", lov_m[p0:p1], t2_block
                )
                result += lib.einsum(
                    "Lne,Lmjnb->mbej", lov_n[p0:p1], tmp
                )
        return result

    def t2_transform_quadratic_active_only(self, t2):
        t2_aaaa = t2[
            numpy.ix_(
                self.act_hole,
                self.act_hole,
                self.act_particle,
                self.act_particle,
            )
        ]
        Imnij = self._df_imnij(self.Lov_aa, self.Lov_aa, t2_aaaa)
        Imbej = self._df_mbej(self.Lov_aa, self.Lov_aa, t2_aaaa)
        Imbje = self._df_mbej(
            self.Lov_aa, self.Lov_aa, t2_aaaa, exchange_output=True
        )
        return Imbje, Imbej, Imnij

    def t2_transform_quadratic(self, t2):
        """Return the all-active output generated by non-all-active contractions."""
        Imnij = self._df_imnij(
            self.Lov_ai,
            self.Lov_aa,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    self.inact_particle,
                    self.act_particle,
                )
            ],
        )
        Imnij += self._df_imnij(
            self.Lov_aa,
            self.Lov_ai,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    self.act_particle,
                    self.inact_particle,
                )
            ],
        )
        Imnij += self._df_imnij(
            self.Lov_ai,
            self.Lov_ai,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    self.inact_particle,
                    self.inact_particle,
                )
            ],
        )

        Imbej = self._df_mbej(
            self.Lov_ia,
            self.Lov_ai,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.inact_hole,
                    self.act_particle,
                    self.inact_particle,
                )
            ],
        )
        Imbje = self._df_mbej(
            self.Lov_ia,
            self.Lov_ai,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.inact_hole,
                    self.inact_particle,
                    self.act_particle,
                )
            ],
            exchange_output=True,
        )
        t2_aiaa = t2[
            numpy.ix_(
                self.act_hole,
                self.inact_hole,
                self.act_particle,
                self.act_particle,
            )
        ]
        Imbej += self._df_mbej(self.Lov_ia, self.Lov_aa, t2_aiaa)
        Imbje += self._df_mbej(
            self.Lov_ia, self.Lov_aa, t2_aiaa, exchange_output=True
        )
        Imbej += self._df_mbej(
            self.Lov_aa,
            self.Lov_ai,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    self.act_particle,
                    self.inact_particle,
                )
            ],
        )
        Imbje += self._df_mbej(
            self.Lov_aa,
            self.Lov_ai,
            t2[
                numpy.ix_(
                    self.act_hole,
                    self.act_hole,
                    self.inact_particle,
                    self.act_particle,
                )
            ],
            exchange_output=True,
        )
        return Imbje, Imbej, Imnij

    def t2_transform_quadratic_inactive(self, t2):
        """Build DCA intermediates without forming an O-by-O-by-V-by-V tensor."""
        all_occ = numpy.arange(self.nocc)
        all_vir = numpy.arange(self.nvir)

        dtype = numpy.result_type(self._eris.Lov, t2)
        dense_elements = self.nocc * self.nocc * self.nvir * self.nvir
        dense_bytes = numpy.dtype(dtype).itemsize * dense_elements
        if dense_bytes <= self._available_memory_bytes(fraction=0.8):
            Vnmef = lib.einsum("Lne,Lmf->nmef", self._eris.Lov, self._eris.Lov)
            Imnij = lib.einsum(
                "mnef,ijef->mnij",
                Vnmef,
                t2[numpy.ix_(self.act_hole, self.act_hole, all_vir, all_vir)],
            )
            Imbej = lib.einsum(
                "nmef,jnbf->mbej",
                Vnmef,
                t2[
                    numpy.ix_(
                        self.act_hole, all_occ, self.act_particle, all_vir
                    )
                ],
            )
            Imbje = lib.einsum(
                "nmef,jnfb->mbje",
                Vnmef,
                t2[
                    numpy.ix_(
                        self.act_hole, all_occ, all_vir, self.act_particle
                    )
                ],
            )
            return Imbje, Imbej, Imnij

        t2_hhvv = t2[numpy.ix_(self.act_hole, self.act_hole, all_vir, all_vir)]
        Imnij = self._df_imnij(
            self._eris.Lov, self._eris.Lov, t2_hhvv, allow_dense=False
        )

        t2_hoav = t2[
            numpy.ix_(self.act_hole, all_occ, self.act_particle, all_vir)
        ]
        Imbej = self._df_mbej(
            self._eris.Lov, self._eris.Lov, t2_hoav, allow_dense=False
        )

        t2_hova = t2[
            numpy.ix_(self.act_hole, all_occ, all_vir, self.act_particle)
        ]
        Imbje = self._df_mbej(
            self._eris.Lov,
            self._eris.Lov,
            t2_hova,
            exchange_output=True,
            allow_dense=False,
        )
        return Imbje, Imbej, Imnij


    @dataclass
    class _IMDS:
       R1: numpy.ndarray
       R2: numpy.ndarray
       Joo: numpy.ndarray
       Jvv: numpy.ndarray
       Jvo: numpy.ndarray
       Foo_t1: numpy.ndarray
       Foo_t2: numpy.ndarray
       Fvv_t1: numpy.ndarray
       Fvv_t2: numpy.ndarray
       Fov: numpy.ndarray
       Imbje_active: numpy.ndarray
       Imbej_active: numpy.ndarray
       Imnij_active: numpy.ndarray

    def kernel(self, t1, t2):
       (M0_base, Moo_base, Mvo_base), (M0_full, Moo_full, Mvo_full) = (
           self._build_M_t1_intermediates_pair(t1)
       )
       Mvo_t2 = self._build_Mvo_t2_blocks(t2)

       joo_ai, joo_ia, jvv_ai = self._build_t1_transform_J_blocks_rest(
           t1, Moo_full, Mvo_full
       )
       
       foo_ia, fvv_ai, fov_ii, fov_ia, fov_ai = self._build_t1_transform_fock_blocks_rest(
           t1, M0_full, Moo_full, Mvo_full
       )

       joo_aa, jvv_aa, jvo_aa = self._build_t1_transform_J_blocks_aa(
           t1, Moo_full, Mvo_full, Mvo_t2, joo_ia, True
       )

       joo_aa_only, jvv_aa_only, jvo_aa_only = self._build_t1_transform_J_blocks_aa(
           t1, Moo_base, Mvo_base, Mvo_t2, joo_ia, False
       )

       foo_aa_t1, foo_aa_t2, fvv_aa_t1, fvv_aa_t2, fov_aa = self._build_t1_transform_fock_blocks_aa(
           t1, M0_base, Moo_base, Mvo_base, fov_ai, fov_ia
       )

       Foo = [foo_ia, foo_aa_t1, foo_aa_t2]
       Fvv = [fvv_ai, fvv_aa_t1, fvv_aa_t2]
       Fov = [fov_ii, fov_aa, fov_ai, fov_ia]

       Joo = [joo_ai, joo_aa, joo_ia]
       Jvv = [jvv_ai, jvv_aa]
       Jvo = [jvo_aa]

       Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mvo_t2, t2)

       #Joo, Jvv, Jvo, Foo, Fvv, Fov = self.t1_transform(t1, M0, Moo, Mvo, Mvo_t2)
       #Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mvo_t2, t2)
       R1 = self.R1_residue_active(
           t1, t2, Fov, Fvv, Foo, M0_base, Mvo_base, Mvo_t2
       )
       Imbje_active, Imbej_active,Imnij_active = self.t2_transform_quadratic(t2)

       R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo)
    #  R2 = self.R2_residue_active_debug(t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo)
       imds = self._IMDS(R1 = R1, R2 = R2,
                        Joo = joo_aa_only, Jvv = jvv_aa_only, Jvo = jvo_aa_only,
                        Foo_t1 = Foo[1], Foo_t2 = Foo[2], Fvv_t1 = Fvv[1], Fvv_t2 = Fvv[2],Fov = Fov[1],
                        Imbje_active = Imbje_active, Imbej_active = Imbej_active, Imnij_active = Imnij_active)       

       return imds
