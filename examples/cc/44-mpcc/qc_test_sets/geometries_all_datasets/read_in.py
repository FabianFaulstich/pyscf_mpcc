import os
from pyscf import lib, gto, scf, mp, cc
from pyscf.cc.uccsd import amplitudes_from_rccsd
import numpy as np
from pyscf.mcscf import avas


def read_in_Z_mat(Mol_specs):

    Mol = []
    distances = {}

    read_in_mol = True
    for line in Mol_specs:
        if line == '\n':
            read_in_mol = False
            continue
        
        if read_in_mol:
            Mol.append(line)
        else:
            # Read in Z mat distances
            dist = line.split('=')
            distances[dist[0].replace(' ','')] = float(dist[1])
   
    # replace value in Mol
    for i, elem in enumerate(Mol):
        # elevery line ends with an '\n' so we remove this
        atoms = elem[:-1].split(' ')

        # removing emtpy strings
        atoms = [i for i in atoms if i]
        for k,atom in enumerate(atoms):                
            for key in list(distances.keys()):
                try:
                    if atom[0] == '-':
                        if key == atom[1:]:
                            atoms[k] = str(-distances[key])
                    else:
                        if key == atom:
                            atoms[k] = str(distances[key])
                except:
                    print('HERE')
                    breakpoint()

        Mol[i] = ' '.join(atoms)  

    #if len(Mol) == 1:
    #    # Single atom case
    #    return Mol[0] + " 0 0 0"
    #else:
    return '\n '.join(Mol)

def get_atoms(atom):

    atom_list = atom.split('\n')
    try:
        atom_list.remove('')
    except:
        atom_list = atom_list

    atom_out= []
    
    for elem in atom_list:
        elem_atom = elem.split(' ')
        try:
            elem_atom.remove('')
        except:
            elem_atom = elem_atom

        try:
            atom_out.append(elem_atom[0])
        except:
            print('Could not append')
            breakpoint()
    
    return atom_out

def compute_indiv_atoms(atom_list, basis):

    energies = {}
    for atom in atom_list:

        mol = gto.Mole()
        mol.basis = basis
        mol.atom = atom
        try:
            mol.spin = 0
            mol.build()
        except:
            mol.spin = 1
            mol.build()

        mf = scf.UHF(mol)
        mf.kernel()
        energies[atom] = mf.e_tot
        
    return energies

def get_ao_labels(atoms_list):

    ao_labels = []
    atoms_unique = list(set(atoms_list))
    for atom in atoms_unique:
        if atom == 'H':
            continue

        ao_labels.append(atom + ' 2s')
        ao_labels.append(atom + ' 2p')
        ao_labels.append(atom + ' 3s')
        ao_labels.append(atom + ' 3p')

    return ao_labels

def get_localized_orbs(
    mol,
    mf,
    ao_labels,
    minao="sto-3g",
    openshell_option=4,
    molden_bool=False,
    name=None,
):

    avas_obj = avas.AVAS(mf, ao_labels, minao=minao, openshell_option=openshell_option)
    avas_obj.with_iao = True
    _, _, mocas = avas_obj.kernel()

    if isinstance(mf, scf.uhf.UHF):
        act_hole = (
            np.where(avas_obj.occ_weights[0] > avas_obj.threshold)[0],
            np.where(avas_obj.occ_weights[1] > avas_obj.threshold)[0]
        )
        act_part = (
            np.where(avas_obj.vir_weights[0] > avas_obj.threshold)[0] + mf.mol.nelec[0],
            np.where(avas_obj.vir_weights[1] > avas_obj.threshold)[0] + mf.mol.nelec[1]
        )
    elif isinstance(mf, scf.rohf.ROHF):
        # NOTE check the output format. Do we want a tuple here?
        act_hole = (
            np.where(avas_obj.occ_weights > avas_obj.threshold)[0],
            np.where(avas_obj.occ_weights > avas_obj.threshold)[0]
        )

        act_part = (
            np.where(avas_obj.vir_weights > avas_obj.threshold)[0] + mf.mol.nelec[0],
            np.where(avas_obj.vir_weights > avas_obj.threshold)[0] + mf.mol.nelec[1]
        )
    c_lo = mocas

    if molden_bool:
        write_molden(mol, mf, name)

    return c_lo, act_hole, act_part

def oo_mp2(mf, c_lo, t1, t2):

    if isinstance(mf, scf.rohf.ROHF):
        # Running regular MP2 and CCSD
        mymp = mp.MP2(mf).run(verbose=0)
        mymp.async_io = True

        # Running localized MP2
        eris = mp.mp2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.mp2._iterative_kernel(mymp, eris, t1=t1, t2=t2, verbose=0)

    elif isinstance(mf, scf.uhf.UHF):
        _, sz = mf.spin_square()
        print("spin multiplicity", sz)
        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf).run()

        # Running localized MP2
        eris = mp.ump2._make_eris(mymp, mo_coeff=c_lo)
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, t1=t1, t2=t2, verbose=0)

    return mp_lo

def fragmented_mpcc_unrestricted(frags, mycc, mp_t2, mp_t1, idx_s, idx_d):

    _, t2ab, _ = mp_t2
    nocca, noccb, _, _ = t2ab.shape

    t1a, t1b = mp_t1
    t2 = np.copy(mp_t2)
    t1 = t1a, t1b

    for frag in frags:
        act_hole = [frag[0][0], frag[0][1]]
        act_particle = [frag[1][0] - nocca, frag[1][1] - noccb]
        res = mycc.kernel(act_hole, act_particle, idx_s, idx_d, t1=t1, t2=t2, oo_mp2 = False)
        print("Exact Exact localized CCSD in MPCCSD (4,2) :", res[0])

        t2 = mycc.t2
        t1 = mycc.t1

    return res[0], t1, t2

def run_mpcc(mol, atoms_list, minao, openshell_option):

    mf = scf.UHF(mol)
    mf.kernel()

    # (4,2)
    # this shold be just one entry
    # NOTE These are the active-inactive indices that are kept fix in MPCC
    idx_s = [[0, 1, 2], [0, 1, 2]]
    idx_d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]

    # NOTE These are the active-inactive indices that are kept fix in the OO-MP2 orbital relaxation
    idx_s_oomp2 = [
            np.delete(np.arange(4), np.array(idx_s[0])), 
            np.delete(np.arange(4), np.array(idx_s[1]))
             ]
    idx_d_oomp2 = [
            np.delete(np.arange(16), np.array(idx_d[0])),
            np.delete(np.arange(16), np.array(idx_d[1])),
            np.delete(np.arange(16), np.array(idx_d[2]))
            ]

    # dm0 = mf.make_rdm1()

    # NOTE In case we switch to ROHF later
    mf = scf.addons.convert_to_uhf(mf)
    ao_labels = get_ao_labels(atoms_list)

    c_lo, act_hole, act_part = get_localized_orbs(mol, mf, ao_labels, minao=minao, openshell_option = openshell_option)
    frag = [[act_hole, act_part]]

    print(f"AO labels:       : {ao_labels}")
    print(f"Active hole      : {act_hole}")
    print(f"No of act. hole  : ({len(act_hole[0])},{len(act_hole[1])})")
    print(f"Active particel  : {act_part}")
    print(f"No of act. part. : ({len(act_part[0])},{len(act_part[1])})")

    e_mp_cc_prev = -np.inf
    e_diff = np.inf
    tol = 1e-5
    count = 0
    count_tol = 100

    mycc = cc.umpccsd.CCSD(mf, mo_coeff = c_lo)
 
    mycc.verbose = 4
    mycc.max_cycle = 50
    mycc.diis_space = 8
    mycc.level_shift = 0.5

    e_mp_cc_its = []

    # DIIS
    adiis = lib.diis.DIIS(mycc)

    while e_diff > tol and count < count_tol:

        if count > 0:

            print("=== Starting OOMP2 iterations ======")     

            mycc.max_cycle = 50
            # mp_lo = oo_mp2(mf, c_lo, mp_cc_t1, mp_cc_t2, spin_restricted)
            nocca, noccb, _, _ = mycc.t2[1].shape
            mycc.kernel(act_hole = [frag[0][0][0], frag[0][0][1]], 
                        act_particle = [frag[0][1][0] - nocca, frag[0][1][1] - noccb], 
                        idx_s = idx_s_oomp2, 
                        idx_d = idx_d_oomp2, 
                      #  t1=mycc.t1, 
                      #  t2=mycc.t2, 
                        t1=mp_cc_t1, 
                        t2=mp_cc_t2, 
                        oo_mp2 = True)

            mp_t2 = mycc.t2
            mp_t1 = mycc.t1
        else:
            mp_lo = oo_mp2(mf, c_lo, None, None)
            mp_t2 = mp_lo[2]
            mp_t1 = mp_lo[3]

        print("=== Starting UMPCCSD ======")     
        mycc.max_cycle = 50

        e_mp_cc, mp_cc_t1, mp_cc_t2 = fragmented_mpcc_unrestricted(
            frag, mycc, mp_t2, mp_t1, idx_s, idx_d
        )


        t1shape = [x.shape for x in mp_cc_t1]
        mp_cc_t1 = np.hstack([x.ravel() for x in mp_cc_t1])
        mp_cc_t1 = adiis.update(mp_cc_t1)
        mp_cc_t1 = lib.split_reshape(mp_cc_t1, t1shape)

        e_diff = np.abs(e_mp_cc - e_mp_cc_prev)
        e_mp_cc_its.append(e_mp_cc)
        e_mp_cc_prev = e_mp_cc
        count += 1

        print(f'\n Iterate progress : e-diff = {e_diff} (tol = {tol}) \n')

    print(f"SCF UMPCC stopped after {count} iterations, iterate difference: {e_diff}")
    return e_mp_cc


if __name__ == '__main__':

    # assign directory
    directory = 'w4-11'
    minao = 'dz'
    openshell_option = 4

    for filename in os.scandir(directory):
        if filename.is_file():

            file1 = open(filename.path, 'r')
            Lines = file1.readlines()
           
            conf = Lines[1].split(' ')
            spin = int(conf[1])-1
            charge = float(conf[0])
            Mol_specs = Lines[2:-1]

            # Checking Z matrix or Cartesian
            cart_format = len(Mol_specs[0].split(' ')) > 1

            if cart_format:
                atoms = ' '.join(Mol_specs)

            else:
            
                atoms = read_in_Z_mat(Mol_specs)

            if 'X' in atoms or 'x' in atoms:
                # Skipping Mols with X or x in them
                continue

            mol = gto.Mole()
            # mol.basis = 'sto3g'
            mol.basis = 'augccpvdz'
            mol.atom = atoms
            mol.spin = spin 
            mol.charge = charge
            mol.build()
            
            atoms_list = get_atoms(mol.atom)
            atoms_energies = compute_indiv_atoms(atoms_list, mol.basis)

            # if len(atoms_list)==1:
            #     continue

            # Running mpcc for full molecule
            e_mpcc_mol = run_mpcc(mol, atoms_list, minao, openshell_option)

            # Running mpcc for individual atoms:
            e_mpcc_atom = []
            for atom in atoms_list:

                if atom == 'H':
                    continue

                mol = gto.Mole()
                # mol.basis = 'sto3g'
                mol.basis = 'augccpvdz'
                mol.atom = atom
                try:
                    mol.spin = 0
                    mol.build()
                except:
                    mol.spin = 1
                    mol.build()

                e = run_mpcc(mol, [atom], minao, openshell_option)
                e_mpcc_atom.append(e)

    















