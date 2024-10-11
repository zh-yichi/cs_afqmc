import numpy as np

def autofrag(mol, H2heavy=True):
    r'''
    Args:
        mol (gto.Mole):
            PySCF Mole object.
        H2heavy (bool):
            If True, hydrogen atoms are bound with the nearest heavy atom to
            make a fragment. Otherwise, every atom, including hydrogen, makes a
            fragment.
    '''
    if H2heavy:
        get_dist = lambda x,y: ((x[:,None,:]-y)**2.).sum(axis=-1)

        if hasattr(mol, 'lattice_vectors'):  # mol is actually a Cell object
            alat = mol.lattice_vectors()
        else:
            alat = None
        cs = mol.atom_charges()
        rs = mol.atom_coords()
        idx_H = np.where(cs == 1)[0]
        idx_X = np.where(cs > 1)[0]
        if idx_X.size > 0:
            if alat is None:
                d2 = get_dist(rs[idx_H], rs[idx_X])
                H2Xmap = np.argmin(d2, axis=1)
            else:
                d2 = []
                for jx in [-1,0,1]:
                    for jy in [-1,0,1]:
                        for jz in [-1,0,1]:
                            a = np.dot(np.array([jx,jy,jz]), alat)
                            d2.append( get_dist(rs[idx_H], rs[idx_X]+a) )
                d2 = np.hstack(d2)
                H2Xmap = np.argmin(d2, axis=1) % len(idx_X)
            frag_atmlist = [None] * len(idx_X)
            for i,iX in enumerate(idx_X):
                iHs = np.where(H2Xmap==i)[0]
                l = np.asarray(np.concatenate([[iX], idx_H[iHs]]),
                               dtype=int).tolist()
                frag_atmlist[i] = l
        else:   # all-H system
            print('warning: no heavy atom detected in the system; every '
                  'hydrogen atom is treated as a single fragment.')
            frag_atmlist = [[i] for i in idx_H]
    else:
        frag_atmlist = [[i] for i in np.where(mol.atom_charges() > 0)[0]]

    return frag_atmlist

def xyz2atom(fxyz):
    fdata = open(fxyz,'r').read().rstrip('\n').split('\n')
    return '\n'.join(fdata[2:])

def guess_frozen(mol):
    frozen_charge_map = np.array([
        0,
        0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        5, 5, 5, 5, 5, 5, 5, 5,
    ])
    return frozen_charge_map[mol.atom_charges()].sum()


if __name__ == '__main__':
    from pyscf import gto
    atom = '''
 O   -1.485163346097   -0.114724564047    0.000000000000
 H   -1.868415346097    0.762298435953    0.000000000000
 H   -0.533833346097    0.040507435953    0.000000000000
 O    1.416468653903    0.111264435953    0.000000000000
 H    1.746241653903   -0.373945564047   -0.758561000000
 H    1.746241653903   -0.373945564047    0.758561000000
'''
    basis = 'cc-pvdz'
    mol = gto.M(atom=atom, basis=basis)

    frag_atmlist1 = autofrag(mol, H2heavy=True)
    frag_atmlist2 = autofrag(mol, H2heavy=False)
    print(frag_atmlist1)
    print(frag_atmlist2)
