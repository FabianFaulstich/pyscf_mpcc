from pyscf import df
from pyscf import lib
import numpy
from dataclasses import dataclass

class screened:
    def __init__(self, mf, eris, frags, **kwargs):
        self.mf = mf

        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._eris = eris
        self.frag = frags
        self.add_DCA = kwargs.get('DCA', True)
        self._set_integral_blocks()
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
        return self.frag[0]
            
    @property
    def act_particle(self):
        return self.frag[1]
        
    @property
    def inact_hole(self):
        return numpy.setdiff1d(numpy.arange(self.nocc), self.act_hole)
        
    @property
    def inact_particle(self):
        return numpy.setdiff1d(numpy.arange(self.nvir), self.act_particle)

    @property
    def aux_idx(self):
        return numpy.arange(self.naux)

    def _antisymmetrize_t2(self, t2):
        return 2.0 * t2 - t2.transpose(0, 1, 3, 2)

    

    def _build_M_t1_intermediates(self, t1, include_active_terms):
        t1_blocks = dict(zip(("ii", "ia", "ai", "aa"), self._set_t1_blocks(t1)))

        def block_tensor(prefix, key):
            return getattr(self, f"{prefix}_{key}")

        def contract_sum(specs, prefix, einsum_expr):
            result = None
            for tensor_key, t1_key in specs:
                tensor  = block_tensor(prefix, tensor_key)
                t1_block = t1_blocks[t1_key]
                if tensor.size == 0 or t1_block.size == 0:
                    continue            # skip zero-dimensional blocks
                term = lib.einsum(einsum_expr, tensor, t1_block)
                result = term if result is None else result + term
            if result is None:
                # All specs were zero-sized; call einsum on the first spec so that
                # numpy infers the correct output shape (result will be all-zeros).
                tensor   = block_tensor(prefix, specs[0][0])
                t1_block = t1_blocks[specs[0][1]]
                result   = lib.einsum(einsum_expr, tensor, t1_block)
            return result

        m0_specs = [("ii", "ii"), ("ia", "ia"), ("ai", "ai")]
        if include_active_terms:
            m0_specs.append(("aa", "aa"))
        M0 = contract_sum(m0_specs, "Lov", "Lkc,kc->L") * 2.0

        moo_specs = {
            "ii": [("ii", "ii"), ("ia", "ia")],
            "ia": [("ii", "ai")],
            "ai": [("ai", "ii"), ("aa", "ia")],
            "aa": [("ai", "ai")],
        }
        if include_active_terms:
            moo_specs["aa"].append(("aa", "aa"))
            moo_specs["ia"].append(("ia", "aa"))

        mvo_specs = {
            "ii": [("ii", "ii"), ("ia", "ia")],
            "ia": [("ii", "ai")],
            "ai": [("ai", "ii"), ("aa", "ia")],
            "aa": [("ai", "ai")],
        }
        if include_active_terms:
            mvo_specs["ia"].append(("ia", "aa"))
            mvo_specs["aa"].append(("aa", "aa"))

        Moo_ii = contract_sum(moo_specs["ii"], "Lov", "Lia,ja->Lij")
        Moo_ia = contract_sum(moo_specs["ia"], "Lov", "Lia,ja->Lij")
        Moo_ai = contract_sum(moo_specs["ai"], "Lov", "Lia,ja->Lij")
        Moo_aa = contract_sum(moo_specs["aa"], "Lov", "Lia,ja->Lij")

        Mvo_ii = contract_sum(mvo_specs["ii"], "Lvv", "Lac,ic->Lai")
        Mvo_ia = contract_sum(mvo_specs["ia"], "Lvv", "Lac,ic->Lai")
        Mvo_ai = contract_sum(mvo_specs["ai"], "Lvv", "Lac,ic->Lai")
        Mvo_aa = contract_sum(mvo_specs["aa"], "Lvv", "Lac,ic->Lai")

        Moo = [Moo_ii, Moo_ia, Moo_ai, Moo_aa]
        Mvo = [Mvo_ii, Mvo_ia, Mvo_ai, Mvo_aa]
        return M0, Moo, Mvo

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
        
        self.Loo_ii = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, inact_hole)].copy()
        self.Loo_ia = self._eris.Loo[numpy.ix_(naux_idx, inact_hole, act_hole)].copy()
        self.Loo_ai = self._eris.Loo[numpy.ix_(naux_idx, act_hole, inact_hole)].copy()
        self.Loo_aa = self._eris.Loo[numpy.ix_(naux_idx, act_hole, act_hole)].copy()

        self.Lvv_ii = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, inact_particle)].copy()
        self.Lvv_ia = self._eris.Lvv[numpy.ix_(naux_idx, inact_particle, act_particle)].copy()
        self.Lvv_ai = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, inact_particle)].copy()
        self.Lvv_aa = self._eris.Lvv[numpy.ix_(naux_idx, act_particle, act_particle)].copy()

        self.Lov_ii = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, inact_particle)].copy()
        self.Lov_ia = self._eris.Lov[numpy.ix_(naux_idx, inact_hole, act_particle)].copy()
        self.Lov_ai = self._eris.Lov[numpy.ix_(naux_idx, act_hole, inact_particle)].copy()
        self.Lov_aa = self._eris.Lov[numpy.ix_(naux_idx, act_hole, act_particle)].copy()


        self.Lvo_aa = self._eris.Lvo[numpy.ix_(naux_idx, act_particle, act_hole)].copy()



    def _set_t1_blocks(self, t1):

        t1_ii = t1[numpy.ix_(self.inact_hole, self.inact_particle)]
        t1_ia = t1[numpy.ix_(self.inact_hole, self.act_particle)]
        t1_ai = t1[numpy.ix_(self.act_hole, self.inact_particle)]
        t1_aa = t1[numpy.ix_(self.act_hole, self.act_particle)]

        return t1_ii, t1_ia, t1_ai, t1_aa

    def _set_t2_antisym_blocks(self, t2):
        """
        Extract all antisymmetrized t2 blocks in a structured way.
        
        Returns all 16 combinations of t2_antisym[h1_type, h2_type, p1_type, p2_type]
        where types are {i=inact, a=act} for both holes (h1, h2) and particles (p1, p2).
        
        Returns
        -------
        dict
            Dictionary with keys like 'iiii', 'iiia', 'iiai', ..., 'aaaa' where:
            - First two characters denote hole indices types (from {i, a})
            - Last two characters denote particle indices types (from {i, a})
        """
        t2_antisym = self._antisymmetrize_t2(t2)
        
        blocks = {}
        
        # Generate all 16 combinations of hole and particle types
        hole_types = {'i': self.inact_hole, 'a': self.act_hole}
        particle_types = {'i': self.inact_particle, 'a': self.act_particle}
        
        for h1_type, h1_idx in hole_types.items():
            for h2_type, h2_idx in hole_types.items():
                for p1_type, p1_idx in particle_types.items():
                    for p2_type, p2_idx in particle_types.items():
                        key = f"{h1_type}{h2_type}{p1_type}{p2_type}"
                        blocks[key] = t2_antisym[numpy.ix_(h1_idx, h2_idx, p1_idx, p2_idx)].copy()
        
        return blocks

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

    def _build_t1_transform_J_blocks_rest(self, t1, Moo, Mvo, Mvo_t2):

        t1_blocks = dict(zip(("ii", "ia", "ai", "aa"), self._set_t1_blocks(t1)))
        moo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Moo))
        mvo_blocks = dict(zip(("ii", "ia", "ai", "aa"), Mvo))
        mvo_t2_blocks = {
            "aa": Mvo_t2[numpy.ix_(self.aux_idx, self.act_particle, self.act_hole)],
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
            "aa": Mvo_t2[numpy.ix_(self.aux_idx, self.act_particle, self.act_hole)],
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

    def create_M_intermediates(self, t1, t2, include_active_terms):
        """
        Construct M intermediates (M0, Moo, Mvo, Mvo_t2) for CC equations.
        
        Uses elaborate block-based approach with specific Lov blocks and t2 block iteration.
        This is the original implementation for comparison/debugging.
        """
        M0, Moo, Mvo = self._build_M_t1_intermediates(t1, include_active_terms)
        # === Mvo_t2: Elaborate block-based version ===
        # Extract all 16 t2_antisym blocks (exclude 'aaaa')
        t2_blocks = self._set_t2_antisym_blocks(t2)
        
        # Initialize Mvo_t2 to full shape [naux, nvir, nocc] to accommodate all block sizes
        Mvo_t2_full = numpy.zeros((self.naux, self.nvir, self.nocc))
        
        # For each Lov_xy block, contract with all compatible t2 blocks
        # Place each contribution in the correct indices based on h1 and p1 types
        lov_blocks = {'ii': self.Lov_ii, 'ia': self.Lov_ia, 'ai': self.Lov_ai, 'aa': self.Lov_aa}
        
        for xy, lov_xy in lov_blocks.items():
            x, y = xy[0], xy[1]  # hole_type, particle_type of Lov
            x_holes = self.inact_hole if x == 'i' else self.act_hole
            y_particles = self.inact_particle if y == 'i' else self.act_particle
            
            if x_holes .size == 0 or y_particles.size == 0:
                continue  # Skip if no indices in this block

            # Iterate through all h1, p1 combinations (except h1='a', p1='a')
            for h1 in ['i', 'a']:
                for p1 in ['i', 'a']:
                    # Skip 'aaaa' block if h1 and p1 are 'a' and x and y are also 'a':
                    if h1 == 'a' and p1 == 'a' and x == 'a' and y == 'a':
                        continue
                    
                    h1_holes = self.inact_hole if h1 == 'i' else self.act_hole
                    p1_particles = self.inact_particle if p1 == 'i' else self.act_particle
                    
                    if h1_holes.size == 0 or p1_particles.size == 0:
                        continue  # Skip if no indices in this block

                    key = f"{h1}{x}{p1}{y}"
                    if key in t2_blocks:
                        # Contraction: [L, h, c] x [i, h, a, c] -> [L, a, i]
                        # Result shape: [naux, n_p1_particles, n_h1_holes]
                        term = lib.einsum("Lkc, ikac -> Lai", lov_xy, t2_blocks[key])
                        # Place in correct indices of full array
                        Mvo_t2_full[numpy.ix_(self.aux_idx, p1_particles, h1_holes)] += term

        return M0, Moo, Mvo, Mvo_t2_full


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
        Foo_aa = lib.einsum("Lie,Lej->ij", self.Lov_ai, Mvo_t2[numpy.ix_(self.aux_idx, inact_particle, act_hole)])
        Foo_aa += lib.einsum("Lie,Lej->ij", self.Lov_aa, Mvo_t2[numpy.ix_(self.aux_idx, act_particle, act_hole)])
#       Foo_aa += lib.einsum("Lie,Lej->ij", self.Lov_aa, Mvo_t2_additional) #additional contribution from active t2 amplitudes

        Foo_aa_t1 += Foo_aa
        Foo_aa_t2 += Foo_aa

        Foo_ia += lib.einsum("Lie,Lej->ij", self.Lov_ii, Mvo_t2[numpy.ix_(self.aux_idx, inact_particle, act_hole)])
        Foo_ia += lib.einsum("Lie,Lej->ij", self.Lov_ia, Mvo_t2[numpy.ix_(self.aux_idx, act_particle, act_hole)])
        Foo_ia += lib.einsum("Lie,Lej->ij", self.Lov_ia, Mvo_t2_additional) #additional contribution from active t2 amplitudes

  
        Fvv_aa  = -lib.einsum("Lmb,Lam->ab",self.Lov_ia,Mvo_t2[numpy.ix_(self.aux_idx, act_particle, inact_hole)])
        Fvv_aa -= lib.einsum("Lmb,Lam->ab",self.Lov_aa,Mvo_t2[numpy.ix_(self.aux_idx, act_particle, act_hole)])
#        Fvv_aa -= lib.einsum("Lmb,Lam->ab",self.Lov_aa,Mvo_t2_additional) #additional contribution from active t2 amplitudes

        Fvv_aa_t1 += Fvv_aa
        Fvv_aa_t2 += Fvv_aa

        Fvv_ai -= lib.einsum("Lmb,Lam->ab",self.Lov_ii,Mvo_t2[numpy.ix_(self.aux_idx, act_particle, inact_hole)])
        Fvv_ai -= lib.einsum("Lmb,Lam->ab",self.Lov_ai,Mvo_t2[numpy.ix_(self.aux_idx, act_particle, act_hole)])
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

         #construct antisymmetrized t2:
         t2_antisym = self._antisymmetrize_t2(t2)

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
         #R1 -= lib.einsum("me, imae -> ia", Fov_ii, t2_antisym) #many terms
     
         R1 -= lib.einsum("jb, ijab -> ia", Fov_ii,
                          t2_antisym[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.inact_particle)])
     
         R1 += lib.einsum("jb, ijab -> ia", Fov_ai, 
                          t2_antisym[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.inact_particle)])
     
         R1 += lib.einsum("jb, ijab -> ia", Fov_ia, 
                          t2_antisym[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])
     
       
         R1 += lib.einsum("Lia, L -> ia", self.Lov_aa, M0)
         R1 -= lib.einsum("Lji,Laj->ia", self.Loo_ia, Mvo_ai)
         R1 -= lib.einsum("Lji,Laj->ia", self.Loo_aa, Mvo_aa)
         
         R1 -= lib.einsum("Lji, Laj -> ia", self.Loo_ia, Mvo_t2[numpy.ix_(self.aux_idx, act_particle, inact_hole)])
         R1 -= lib.einsum("Lji, Laj -> ia", self.Loo_aa, Mvo_t2[numpy.ix_(self.aux_idx, self.act_particle, self.act_hole)])

         R1 += lib.einsum("Lae, Lei -> ia", self.Lvv_ai, Mvo_t2[numpy.ix_(self.aux_idx, self.inact_particle, self.act_hole)])
         R1 += lib.einsum("Lae, Lei -> ia", self.Lvv_aa, Mvo_t2[numpy.ix_(self.aux_idx, self.act_particle, self.act_hole)])

        # Foo_ai -= lib.einsum("me,ie->im", Fov_ii, t1_ai)*0.5
        # Fvv_ai += lib.einsum("me, ma -> ae", Fov_ii, t1_ia)*0.5

         return R1

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


        #factorized part of the residue:
        #PPL 
        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_ai)
        R2  = lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, inact_particle, inact_particle)])

        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_aa)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)])

        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_aa, Jvv_ai)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])

        Wabef = None
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

    #copy R2_residue_active function and create a new function that prints each intermediate and the final R2 for debugging purposes. You can name it R2_residue_active_debug.
    def R2_residue_active_debug(self, t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo):
        
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
        Fov_ii, Fov_aa, Fov_ai, Fov_ia, Fvv_ai, _, Fvv_aa_t2, Foo_ia, _, Foo_aa_t2 = self._unpack_fock_blocks(Fov, Fvv, Foo)
        t1_ii, t1_ia, t1_ai, t1_aa = self._set_t1_blocks(t1)
        #Non DCA terms:
        Imbje_active = None
        Imbej_active = None
        Imnij_active = None
        if (self.add_DCA):
           Imbje, Imbej, Imnij = self.t2_transform_quadratic_inactive(t2)  
           Imbje_active, Imbej_active, Imnij_active = self.t2_transform_quadratic_active_only(t2) 
           #get the same contributions from self.t2_transform_quadratic and add to the Imnij_active etc contributions
           
           Imbje_active_new, Imbej_active_new, Imnij_active_new = self.t2_transform_quadratic(t2) 

           Imnij_active += Imnij_active_new
           Imbje_active += Imbje_active_new
           Imbej_active += Imbej_active_new
        else:
           Imbje_active = numpy.zeros((len(self.act_hole), len(self.act_particle), len(self.inact_particle), len(self.act_hole)))
           Imbej_active = numpy.zeros((len(self.act_hole), len(self.act_particle), len(self.inact_particle), len(self.act_hole)))
           Imnij_active = numpy.zeros((len(self.act_hole), len(self.act_hole), len(self.act_hole), len(self.act_hole)))

              
        #factorized part of the residue:
       
        #factorized part of the residue:
#        R2 = lib.einsum("Lai, Lbj -> ijab", Jvo_aa, Jvo_aa)

#        print("Initial norm of Jvo_aa:", numpy.linalg.norm(Jvo_aa))
#        print("Initial norm of R2 from Jvo_aa, Jvo_aa:", numpy.linalg.norm(R2))   

        #PPL
        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_ai)
        R2  = lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, inact_particle, inact_particle)])

        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_ai, Jvv_aa)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)])
      
        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_aa, Jvv_ai)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])  
        
        print("norm of R2 after PPL (inactive):", numpy.linalg.norm(R2)) 

        # add Jvv_aa, Jvv_aa contribution to PPL:
        Wabef = lib.einsum("Lae, Lbf -> abef", Jvv_aa, Jvv_aa)
        R2 += lib.einsum("abef, ijef -> ijab", Wabef, t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)])
    
        print("norm of R2 after PPL (Jvv_aa, Jvv_aa):", numpy.linalg.norm(R2))
     
        #HHL
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_ia, Joo_ia) 
        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(inact_hole, inact_hole,numpy.arange(n_act_hole),numpy.arange(n_act_hole))]
        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(inact_hole, inact_hole, act_particle, act_particle)])
        print("norm of R2 after HHL (Joo_ia, Joo_ia):", numpy.linalg.norm(R2))
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_aa, Joo_ia) 
        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(act_hole, inact_hole,numpy.arange(n_act_hole),numpy.arange(n_act_hole))]
        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(act_hole, self.inact_hole, act_particle, act_particle)])

        print("norm of R2 after HHL (Joo_aa, Joo_ia):", numpy.linalg.norm(R2))
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_ia, Joo_aa)
        if (self.add_DCA):
            Wijmn += Imnij[numpy.ix_(inact_hole,act_hole,numpy.arange(n_act_hole),numpy.arange(n_act_hole))]
        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(inact_hole, act_hole, act_particle, act_particle)])
        print("norm of R2 after HHL (Joo_ia, Joo_aa):", numpy.linalg.norm(R2))
        
         #HHL
        Wijmn = lib.einsum("Lmi, Lnj -> mnij", Joo_aa, Joo_aa) + Imnij_active
        R2 += lib.einsum("mnij, mnab -> ijab", Wijmn, t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)]) 

        print("norm of R2 after adding active-only Imnij to HHL:", numpy.linalg.norm(R2))
        #Fock matrix contribution:
        Foo_ia_tmp = Foo_ia.copy()
        Fvv_ai_tmp = Fvv_ai.copy()
        #Foo_aa += lib.einsum("me,ie->im", Fov_ai, t1_ai)
        Foo_ia_tmp += lib.einsum("ic,jc->ij", Fov_ii, t1_ai)
        Foo_ia_tmp += lib.einsum("ic,jc->ij", Fov_ia, t1_aa)
        Foo_aa_t2 += lib.einsum("ic,jc->ij", Fov_aa, t1_aa) #only aa contribution.
        R2_tmp = -lib.einsum("mi, mjab -> ijab", Foo_ia_tmp, t2[numpy.ix_(inact_hole, act_hole, act_particle, act_particle)]) 
        #Foo_aa contribution
        R2_tmp -= lib.einsum("mi, mjab -> ijab", Foo_aa_t2, t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)]) #only one possibility m has to be inactive.
        print("norm of R2_tmp after Foo contribution:", numpy.linalg.norm(R2_tmp))
        #Fvv -= lib.einsum("me, ma -> ae", Fov, t1)
        #Fvv_aa -= lib.einsum("me, ma -> ae", Fov_ia, t1_ia)
        Fvv_ai_tmp -= lib.einsum("lb,la->ab", Fov_ii, t1_ia)
        Fvv_ai_tmp -= lib.einsum("lb,la->ab", Fov_ai, t1_aa)

        Fvv_aa_t2 -= lib.einsum("lb,la->ab", Fov_aa, t1_aa) #only aa contribution
        R2_tmp += lib.einsum("bc,ijac->ijab", Fvv_ai_tmp, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)]) # only one possibility e has to be inactive.
      
        R2_tmp += lib.einsum("bc,ijac->ijab", Fvv_aa_t2, t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)]) # only one possibility e has to be inactive.
      
        print("norm of R2_tmp after Fvv contribution:", numpy.linalg.norm(R2_tmp))

        #=======everything works upto here=========

        #N3V3 terms:
        W_mbje = lib.einsum("Lmj, Lbe -> mbje", Joo_ia, Jvv_ai) 
        if (self.add_DCA):
            W_mbje  -= Imbje[numpy.ix_(inact_hole,numpy.arange(n_act_particle),numpy.arange(n_act_hole),inact_particle)]
        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(act_hole, inact_hole, act_particle, inact_particle)])
        
        print("norm of R2_tmp after W_mbje contributions: step 1", numpy.linalg.norm(R2_tmp))        


        W_mbje = lib.einsum("Lmj,Lbe->mbje", Joo_aa, Jvv_ai)
        if (self.add_DCA):
            W_mbje  -= Imbje[numpy.ix_(act_hole,numpy.arange(n_act_particle),numpy.arange(n_act_hole),inact_particle)]
        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])
        
        print("norm of R2_tmp after W_mbje contributions: step 2", numpy.linalg.norm(R2_tmp))


        W_mbje = lib.einsum("Lmj,Lbe->mbje",Joo_ia,Jvv_aa)
        if (self.add_DCA):  
            W_mbje  -= Imbje[numpy.ix_(inact_hole,numpy.arange(n_act_particle),numpy.arange(n_act_hole),act_particle)]
        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])
        
        print("norm of R2_tmp after W_mbje contributions: step 3", numpy.linalg.norm(R2_tmp))
        #add the active contributions from Imbje to the N3V3 terms:

        W_mbje = lib.einsum("Lmj, Lbe -> mbje", Joo_aa, Jvv_aa) - Imbje_active 
        R2_tmp -= lib.einsum("mbje, imae -> ijab", W_mbje, t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]) # em should be ii, ia, ai types

        print("norm of R2_tmp after W_mbje contributions:", numpy.linalg.norm(R2_tmp))

    #======W_jema contributions:========

        W_jema = lib.einsum("Lmj, Lae -> maje", Joo_ia, Jvv_ai)
        
        if (self.add_DCA):
            W_jema -= 0.5*Imbje[numpy.ix_(inact_hole, numpy.arange(n_act_particle), numpy.arange(n_act_hole), inact_particle)]
        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, inact_hole, inact_particle, act_particle)])
    
    
        W_jema = lib.einsum("Lmj, Lae -> maje", Joo_ia, Jvv_aa)      
        if  (self.add_DCA):
            W_jema -= 0.5*Imbje[numpy.ix_(inact_hole, numpy.arange(n_act_particle), numpy.arange(n_act_hole), act_particle)]
        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)])

        
        W_jema = lib.einsum("Lmj, Lae -> maje", Joo_aa, Jvv_ai)
        if (self.add_DCA):
            W_jema -= 0.5*Imbje[numpy.ix_(act_hole, numpy.arange(n_act_particle), numpy.arange(n_act_hole), inact_particle)]
        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, act_hole, inact_particle, act_particle)])

        if Imbje_active is not None:
            W_jema = lib.einsum("Lmj, Lae -> maje", Joo_aa, Jvv_aa) - 0.5*Imbje_active
        else:
            W_jema = lib.einsum("Lmj, Lae -> maje", Joo_aa, Jvv_aa)
        R2_tmp -= lib.einsum("maje, imeb -> ijab", W_jema, t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)]) # em should be ii, ia, ai types

        print("norm of R2_tmp after W_jema contribution:", numpy.linalg.norm(R2_tmp))

        if (self.add_DCA):  
           R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej[numpy.ix_(inact_hole,numpy.arange(n_act_particle), inact_particle,numpy.arange(n_act_hole))], t2[numpy.ix_(act_hole, inact_hole, act_particle, inact_particle)])
           R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej[numpy.ix_(act_hole,numpy.arange(n_act_particle), inact_particle,numpy.arange(n_act_hole))], t2[numpy.ix_(act_hole, act_hole, act_particle, inact_particle)])
           R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej[numpy.ix_(inact_hole,numpy.arange(n_act_particle), act_particle,numpy.arange(n_act_hole))], t2[numpy.ix_(act_hole, inact_hole, act_particle, act_particle)]) 

        R2_tmp -= lib.einsum("mbej, imae -> ijab", Imbej_active, t2[numpy.ix_(act_hole, act_hole, act_particle, act_particle)]) 

        
        print("norm of R2_tmp after W_mbej contribution:", numpy.linalg.norm(R2_tmp))

        #symmetrize R2_tmp:
        R2 += (R2_tmp + R2_tmp.transpose(1, 0, 3, 2))
        print("Final norm of R2 after symmetrization:", numpy.linalg.norm(R2))
        return R2     

    def t2_transform_quadratic(self,t2):


       #Extract the active part from the follwing terms:

        # We have already built the DCA terms by assembling the factorized terms.  
        # these terms shall be ignored with the DCA approximation.

        #I^mn_ij

        Vnmef = lib.einsum("Lne, Lmf -> nmef", self.Lov_ai, self.Lov_aa)
        Imnij_active = lib.einsum("mnef, ijef->mnij", Vnmef, t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle)])

        Vnmef = lib.einsum("Lne, Lmf -> nmef", self.Lov_aa, self.Lov_ai)
        Imnij_active +=lib.einsum("mnef, ijef -> mnij", Vnmef, t2[numpy.ix_(self.act_hole, self.act_hole,self.act_particle,self.inact_particle)])

        Vnmef = lib.einsum("Lne,Lmf->nmef", self.Lov_ai, self.Lov_ai)
        Imnij_active += lib.einsum("mnef,ijef->mnij",Vnmef, t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.inact_particle)])

        #I^je_bm
        #Vnemf = lib.einsum("Lne, Lmf -> nemf", self.Lov, self.Lov)
        #Ijebm = lib.einsum("nemf, jnbf -> jebm", Vnemf, t2) #nf should be ii, ia, ai types

        Vnmef = lib.einsum("Lne, Lmf -> nmef", self.Lov_ia, self.Lov_ai)

        Imbej_active = lib.einsum("nmef, jnbf -> mbej", Vnmef, t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.inact_particle)])
        Imbje_active = lib.einsum("nmef, jnfb -> mbje", Vnmef, t2[numpy.ix_(self.act_hole, self.inact_hole, self.inact_particle, self.act_particle)])

        Vnmef = lib.einsum("Lne,Lmf->nmef", self.Lov_ia, self.Lov_aa)
        Imbej_active += lib.einsum("nmef,jnbf->mbej", Vnmef, t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])
        Imbje_active += lib.einsum("nmef,jnfb->mbje", Vnmef, t2[numpy.ix_(self.act_hole, self.inact_hole, self.act_particle, self.act_particle)])
        
        Vnmef = lib.einsum("Lne, Lmf -> nmef", self.Lov_aa, self.Lov_ai)
        Imbej_active += lib.einsum("nmef,jnbf->mbej", Vnmef, t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.inact_particle)])
        Imbje_active += lib.einsum("nmef,jnfb->mbje", Vnmef, t2[numpy.ix_(self.act_hole, self.act_hole, self.inact_particle, self.act_particle)])

        return Imbje_active,Imbej_active,Imnij_active

    def t2_transform_quadratic_active_only(self,t2):
        #I_mn^ij 
        Vnemf = lib.einsum("Lne, Lmf -> nmef", self.Lov_aa, self.Lov_aa)
        Imnij = lib.einsum("mnef, ijef -> mnij", Vnemf, t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]) #ef should be ii, ia, ai type
        #I^je_bm
        Imbej = lib.einsum("nmef, jnbf -> mbej", Vnemf, t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]) #nf should be ii, ia, ai type
        #I^je_mb
        Imbje = lib.einsum("nmef, jnfb -> mbje", Vnemf, t2[numpy.ix_(self.act_hole, self.act_hole, self.act_particle, self.act_particle)]) # #nf should be ii, ia, ai ty
        return Imbje, Imbej, Imnij
    

    def t2_transform_quadratic_inactive(self, t2):
        """
        Compute quadratic T2 intermediate contributions for inactive indices.
        
        Extracts the active part from the following terms:
        - I_mn^ij: Occupied-occupied intermediate
        - I^je_bm: Virtual-occupied intermediate (variant 1)
        - I^je_mb: Virtual-occupied intermediate (variant 2)
        
        These DCA terms are built by assembling the factorized terms and should be
        ignored with the DCA approximation.
        
        Parameters
        ----------
        t2 : ndarray
            T2 amplitudes
        
        Returns
        -------
        tuple
            (Imbje, Imbej, Imnij) intermediate tensors
        """
        # Compute array V[n, m, e, f] = L[n, e] * L[m, f]
        Vnmef = lib.einsum("Lne, Lmf -> nmef", self._eris.Lov, self._eris.Lov)
        
        # Prepare index ranges
        all_occ_range = numpy.arange(self.nocc)
        all_vir_range = numpy.arange(self.nvir)
        
        # === I_mn^ij: Occupied-occupied intermediate ===
        # Contract over (e=inactive, f=active), (e=active, f=inactive), (e=inactive, f=inactive)
        Imnij = None
        for e_type, f_type in [('i', 'a'), ('a', 'i'), ('i', 'i'), ('a', 'a')]:
            e_idx = self.inact_particle if e_type == 'i' else self.act_particle
            f_idx = self.inact_particle if f_type == 'i' else self.act_particle
            
            vnmef_slice = Vnmef[numpy.ix_(all_occ_range, all_occ_range, e_idx, f_idx)]
            t2_slice = t2[numpy.ix_(self.act_hole, self.act_hole, e_idx, f_idx)]
            
            term = lib.einsum("mnef, ijef->mnij", vnmef_slice, t2_slice)
            Imnij = term if Imnij is None else Imnij + term
        
        # === I^je_bm: Virtual-occupied intermediate (m-index fixed to inactive) ===
        # Index pattern: Vnmef[inact_hole, all_occ, all_vir, varying]
        # Contraction: nmef x jnbf -> mbej
        Imbej = None
        for n_type, f_type in [('i', 'i'), ('i', 'a'), ('a', 'i'), ('a', 'a')]:
            n_idx = self.inact_hole if n_type == 'i' else self.act_hole
            f_idx = self.inact_particle if f_type == 'i' else self.act_particle
            
            vnmef_slice = Vnmef[numpy.ix_(n_idx, all_occ_range, all_vir_range, f_idx)]
            t2_slice = t2[numpy.ix_(self.act_hole, n_idx, self.act_particle, f_idx)]
            
            term = lib.einsum("nmef, jnbf->mbej", vnmef_slice, t2_slice)
            Imbej = term if Imbej is None else Imbej + term
        
        # === I^je_mb: Virtual-occupied intermediate (m-index fixed to inactive) ===
        # Index pattern: Vnmef[varying, all_occ, all_vir, varying]
        # Contraction: nmef x jnfb -> mbje
        Imbje = None
        for n_type, f_type in [('i', 'i'), ('i', 'a'), ('a', 'i'), ('a', 'a')]:
            n_idx = self.inact_hole if n_type == 'i' else self.act_hole
            f_idx = self.inact_particle if f_type == 'i' else self.act_particle
            
            vnmef_slice = Vnmef[numpy.ix_(n_idx, all_occ_range, all_vir_range, f_idx)]
            t2_slice = t2[numpy.ix_(self.act_hole, n_idx, f_idx, self.act_particle)]
            
            term = lib.einsum("nmef, jnfb->mbje", vnmef_slice, t2_slice)
            Imbje = term if Imbje is None else Imbje + term
        
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
      
       M0, Moo, Mvo, Mvo_t2 = self.create_M_intermediates(t1, t2, True)

       joo_ai, joo_ia, jvv_ai = self._build_t1_transform_J_blocks_rest(t1, Moo, Mvo, Mvo_t2)
       
       foo_ia, fvv_ai, fov_ii, fov_ia, fov_ai = self._build_t1_transform_fock_blocks_rest(t1, M0, Moo, Mvo)

       joo_aa, jvv_aa, jvo_aa = self._build_t1_transform_J_blocks_aa(t1, Moo, Mvo, Mvo_t2, joo_ia, True)

       M0, Moo, Mvo, Mvo_t2 = self.create_M_intermediates(t1, t2, False)

       joo_aa_only, jvv_aa_only, jvo_aa_only = self._build_t1_transform_J_blocks_aa(t1, Moo, Mvo, Mvo_t2, joo_ia, False)

       foo_aa_t1, foo_aa_t2, fvv_aa_t1, fvv_aa_t2, fov_aa = self._build_t1_transform_fock_blocks_aa(t1, M0, Moo, Mvo, fov_ai, fov_ia)

       Foo = [foo_ia, foo_aa_t1, foo_aa_t2]
       Fvv = [fvv_ai, fvv_aa_t1, fvv_aa_t2]
       Fov = [fov_ii, fov_aa, fov_ai, fov_ia]

       Joo = [joo_ai, joo_aa, joo_ia]
       Jvv = [jvv_ai, jvv_aa]
       Jvo = [jvo_aa]

       Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mvo_t2, t2)

       #Joo, Jvv, Jvo, Foo, Fvv, Fov = self.t1_transform(t1, M0, Moo, Mvo, Mvo_t2)
       #Foo, Fvv = self.add_t2_to_fock(Fvv, Foo, Mvo_t2, t2)
       R1 = self.R1_residue_active(t1, t2, Fov, Fvv, Foo, M0, Mvo, Mvo_t2)
       Imbje_active, Imbej_active,Imnij_active = self.t2_transform_quadratic(t2)

       R2 = self.R2_residue_active(t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo)
    #  R2 = self.R2_residue_active_debug(t1, t2, Joo, Jvv, Jvo, Fov, Fvv, Foo)
       imds = self._IMDS(R1 = R1, R2 = R2,
                        Joo = joo_aa_only, Jvv = jvv_aa_only, Jvo = jvo_aa_only,
                        Foo_t1 = Foo[1], Foo_t2 = Foo[2], Fvv_t1 = Fvv[1], Fvv_t2 = Fvv[2],Fov = Fov[1],
                        Imbje_active = Imbje_active, Imbej_active = Imbej_active, Imnij_active = Imnij_active)       

       return imds
