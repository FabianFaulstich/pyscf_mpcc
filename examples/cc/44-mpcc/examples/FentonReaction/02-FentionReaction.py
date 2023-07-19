import numpy as np 
from pyscf import gto, scf, mcscf, mp, cc
import matplotlib.pyplot as plt
from pyscf.mcscf import avas

def load_geometry(IRC):

    name = f'geometries/Fenton_{IRC}'
    file = open(name, 'r')
    Lines = file.readlines()

    atoms = []
    for line in Lines:
        line_str = line.split()
        element = line_str[0]
        geometry = [float(line_str[1]), float(line_str[2]), float(line_str[3])]
        atoms.append([element, geometry])

    return np.array(atoms)

def get_reference_rohf_energies():
    

    name = f'ROHF_energies_Knizia.txt'
    file = open(name, 'r')
    Lines = file.readlines()

    energies = []
    for line in Lines:
        energies.append(float(line))

    return energies

def get_reference_dmrgci_energies():
    

    name = f'DMRGCI_energies_Knizia.txt'
    file = open(name, 'r')
    Lines = file.readlines()

    energies = []
    for line in Lines:
        energies.append(float(line))

    return energies




if __name__ == "__main__":

    IRCs = np.load('geometries/IRCs.npy')
    energies_rohf_ref = get_reference_rohf_energies()
    energies_dmrgci_ref = get_reference_dmrgci_energies()

    plot_knizia = False
    if plot_knizia:
        # making Fig 4 in Knizia reference
        # NOTE something is funn with the units, but the shape looks ok after shifting the curves
        plt.plot(energies_rohf_ref- np.min(energies_rohf_ref), 'b-x', label = 'ROHF')
        plt.plot(energies_dmrgci_ref- np.min(energies_dmrgci_ref), 'g-o', label = 'DMRG-CI')
        plt.legend()
        plt.show()

    energies = np.array([])
    for k, IRC in enumerate(IRCs):
        atoms = load_geometry(IRC)
 
        mol           = gto.M()
        mol.basis     = 'ccpvtz'
        mol.atom      = atoms
        mol.unit      = 'Angstrom'
        #mol.unit      = 'Bohr'
        mol.build()
       
        mf = mol.ROHF()
        mf.max_cycle = 100
        mf.verbose = 4
        
        try:
            dm = np.load(f'dms/rdm1_{mol.unit}_{IRC}.npy')
            print('Loaded 1-RDM')
            mf.kernel(dm0=dm)
        except:
            try:
                dm = np.load(f'dms/rdm1_{mol.unit}_{IRCs[k-1]}.npy')
                print('Running full ROHF (with initial guess from previous IRC) ... ')
                mf.kernel(dm0 = dm)

            except:
                print('Running full ROHF (no initial guess) ... ')
                mf.kernel()

            # storing 1-RDM
            dm = mf.make_rdm1()
            np.save('rdm1_{mol.unit}.npy', dm)



        np.append(energies, mf.e_tot)

        # O 2s + 2p electrons and Fe 3d
        ao_labels = ['O 2s', 'O 2p', 'Fe 3d']
        avas_obj = avas.AVAS(mf, ao_labels, openshell_option=3)
        avas_obj.with_iao = True

        ncas, nelecas, mocas = avas_obj.kernel()

        # Running mcscf
        mycas = mcscf.CASSCF(mf, ncas, nelecas)
        mycas.verbose = 4
        mycas.kernel()
        # mycas.kernel(mocas)

        mf1 = scf.addons.convert_to_uhf(mf)

        # Running regular MP2 and CCSD
        mymp = mp.UMP2(mf1).run()
        mycc = cc.UCCSD(mf1).run()

        # Running localized MP2
        eris = mp.ump2._make_eris(mymp, mo_coeff=(mocas,mocas))
        mp_lo = mp.ump2._iterative_kernel(mymp, eris, verbose=0)


    plt.plot(energies_rohf_ref- np.min(energies_rohf_ref), 'b-', label = 'ROHF (Knizia)')
    plt.plot(energies - np.min(energies), 'rx', label = 'PySCF')
    plt.legend()
    plt.plot()



