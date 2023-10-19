import os
from pyscf import gto,scf
from pyscf.cc.uccsd import amplitudes_from_rccsd

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


if __name__ == '__main__':

    # assign directory
    directory = 'w4-11'

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
            mol.basis = 'sto3g'
            mol.atom = atoms
            mol.spin = spin 
            mol.charge = charge

            mol.build()
            rhf = scf.RHF(mol)
            rhf.kernel()

