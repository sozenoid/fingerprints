import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as descriptors
from rdkit.Chem import AllChem, Draw
import math
import numpy as np
from SmallestEnclosingCircle_CASTING import returnCircleAsTuple


def get_fingerprint(SMILES=None, RADIUS=(None, None), E_BIND=None):
    """
    PRE: Takes in a MOLECULE as a SMILES
    POST: Prints its finger prints as two list, the first contains the names, the second contains the fingerprints
    """

    def get_atoms_coords(RDKIT_BLOCK):
        """Takes as input an RDKIT BLOCK and returns a list of atoms with a numpy array containing the coordinates"""
        RDKIT_BLOCK = RDKIT_BLOCK.split('\n')
        atm_number = int(RDKIT_BLOCK[3][:3])
        RDKIT_BLOCK = [x.split() for x in RDKIT_BLOCK]
        atm_list = []
        coords_array = np.zeros([atm_number, 3], dtype=float)
        for i, line in enumerate(RDKIT_BLOCK[4:4 + atm_number]):
            coords_atm = line
            atm_list.append(coords_atm[3])
            coords_array[i, :] = coords_atm[:3]
        return atm_list, coords_array

    def get_atom_types(mol):
        """
        PRE: Takes in the mol
        POST: Returns a dictionary with the atom types and numbers
        """
        atom_types = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_types:
                atom_types[symbol] += 1
            else:
                atom_types[symbol] = 1
        return atom_types

    def AreRingFused(mol):
        """
        PRE  : Takes in a mol rdkit
        POST : Returns the max number of fused rings. That is the maximum number of rings any atom belongs to
        """
        rings = Chem.GetSymmSSSR(mol)
        ring_dic = {}
        for ring in rings:
            for atom in list(ring):
                if atom in ring_dic:
                    ring_dic[atom] += 1
                else:
                    ring_dic[atom] = 1
        if ring_dic.values() == []:
            return 0
        else:
            return max(ring_dic.values())

    def getVolume(mol, atom_types):
        """
        PRE: Takes in a mol with HYDROGENS ADDED
        POST: Returns its volume computed as a linear combination of the contribution of the vdW volumes
        """
        index_of_vols = {'H': 7.24, 'C': 20.58, 'N': 15.60, 'O': 14.71, 'F': 13.31, 'Cl': 22.45, 'Br': 26.52,
                         'I': 32.52, 'P': 24.43, 'S': 24.43, 'As': 26.52, 'B': 40.48, 'Si': 38.79, 'Se': 28.73,
                         'Te': 36.62}
        gross_volume = 0
        for sym in atom_types:
            gross_volume += atom_types[sym] * index_of_vols[sym]
        bonds = mol.GetNumBonds()
        rings = Chem.GetSymmSSSR(mol)
        # print 'aromatic ring count is ',descriptors.CalcNumAromaticRings(mol)
        # print 'aliphatic ring count is ',descriptors.CalcNumAliphaticRings(mol)
        ra = 0
        largest_ra = 0
        rna = 0
        largest_rna = 0
        for ringId in range(len(rings)):
            if isRingAromatic(mol, tuple(rings[ringId])):
                ra += 1
                if largest_ra < len(rings[ringId]):
                    largest_ra = len(rings[ringId])
            else:
                rna += 1
                if largest_rna < len(rings[ringId]):
                    largest_rna = len(rings[ringId])
        volume = gross_volume - 5.92 * bonds - 14.7 * ra - 3.8 * rna

        return volume, ra, rna, largest_ra, largest_rna

    def isRingAromatic(mol, ring):
        """
        PRE: Takes in a mol and a ring given as a tuple of atom id
        POST: Returns TRUE is all the atoms inside the ring are aromatic and FALSE otherwise
        """
        aromatic = True
        for ids in ring:
            if mol.GetAtomWithIdx(ids).GetIsAromatic():
                # print ids
                pass
            else:
                aromatic = False
                break
        return aromatic

    mol = SMILES
    features = [
        'atomNbr',
        'Volume',
        'NAtom',
        'OAtom',
        'SAtom',
        'PAtom',
        'ClAtom',
        'BrAtom',
        'FAtom',
        'IAtom',
        'AromaticRingNumber',
        'LargestAromaticRingAtomNbr',
        'NonAromaticRingNumber',
        'LargestNonAromaticRingAtomNbr',
        'MaxNbrFusedRings',
        'SurfaceArea',
        'Charge',
        'MinRadiusOfCylinder',
        'RadiusOfCylinderBestConf',
        'NitroNbr',
        'AlcoholNbr',
        'KetoneNbr',
        'NitrileNbr',
        'ThiolNbr',
        'Phenol_likeNbr',
        'EsterNbr',
        'SulfideNbr',
        'CarboxilicAcidNbr',
        'EtherNbr',
        'AmideNbr',
        'AnilineNbr',
        'PrimaryAmineNbr',
        'SecondaryAmineNbr',
        'EBind',
        'RotableBondNum',
        'HBondDonor',
        'HBondAcceptor',
        'MolLogP',
        'MolMR'
    ]
    for i in range(6):
        features.append('Chi{}v'.format(i + 1))
        features.append('Chi{}n'.format(i + 1))
        if i < 3:
            features.append('Kappa{}'.format(i + 1))

    feature_dic = dict.fromkeys(features)
    if mol == None:
        return sorted(feature_dic.keys())

    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)

    feature_dic['RotableBondNum'] = descriptors.CalcNumRotatableBonds(mol)

    for i in range(6):
        feature_dic['Chi{}v'.format(i + 1)] = descriptors.CalcChiNv(mol, i + 1)
        feature_dic['Chi{}n'.format(i + 1)] = descriptors.CalcChiNn(mol, i + 1)

    feature_dic['Kappa1'] = descriptors.CalcKappa1(mol)
    feature_dic['Kappa2'] = descriptors.CalcKappa2(mol)
    feature_dic['Kappa3'] = descriptors.CalcKappa3(mol)

    feature_dic['HBondAcceptor'] = descriptors.CalcNumHBA(mol)
    feature_dic['HBondDonor'] = descriptors.CalcNumHBD(mol)

    CrippenDescriptors = descriptors.CalcCrippenDescriptors(mol)
    feature_dic['MolLogP'] = CrippenDescriptors[0]
    feature_dic['MolMR'] = CrippenDescriptors[1]

    atom_types = get_atom_types(mol)
    for feat, symbol in zip(['NAtom', 'OAtom', 'SAtom', 'PAtom', 'ClAtom', 'BrAtom', 'FAtom', 'IAtom'],
                            ['N', 'O', 'S', 'P', 'Cl', 'Br', 'F', 'I']):
        if symbol in atom_types:
            feature_dic[feat] = atom_types[symbol]
        else:
            feature_dic[feat] = 0

    feature_dic['atomNbr'] = mol.GetNumHeavyAtoms()
    feature_dic['Volume'], feature_dic['AromaticRingNumber'], feature_dic['NonAromaticRingNumber'], feature_dic[
        'LargestAromaticRingAtomNbr'], feature_dic['LargestNonAromaticRingAtomNbr'] = getVolume(mol, atom_types)
    feature_dic['MaxNbrFusedRings'] = AreRingFused(mol)
    feature_dic['SurfaceArea'] = descriptors.CalcTPSA(mol)
    feature_dic['Charge'] = Chem.GetFormalCharge(mol)

    funct_dic = {
        '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]': 'NitroNbr',
        '[#6][OX2H]': 'AlcoholNbr',
        '[NX1]#[CX2]': 'NitrileNbr',
        '[#6][CX3](=O)[#6]': 'KetoneNbr',
        '[#16X2H]': 'ThiolNbr',
        "[OX2H][cX3][c]": 'Phenol_likeNbr',
        '[#6][CX3](=O)[OX2H0][#6]': 'EsterNbr',
        '[#16X2H0]': 'SulfideNbr',
        '[CX3](=O)[OX2H1]': 'CarboxilicAcidNbr',
        '[OD2]([#6])[#6]': 'EtherNbr',
        # '[NX3][CX3](=[OX1])[#6]':'AmideNbr',
        '[#7X3][#6X3](=[OX1])[#6]': 'AmideNbr',
        '[NX3][cc]': 'AnilineNbr',
        '[NX3H2;!$(NC=O)]': 'PrimaryAmineNbr',
        '[NX3H1;!$(NC=O)]': 'SecondaryAmineNbr'}

    for funct in funct_dic:
        patt = Chem.MolFromSmarts(funct)
        feature_dic[funct_dic[funct]] = len(mol.GetSubstructMatches(patt))

    names, coords = get_atoms_coords(Chem.MolToMolBlock(mol))
    # feature_dic['MinRadiusOfCylinder'] = returnCircleAsTuple(coords[:,1:])[2]
    feature_dic['MinRadiusOfCylinder'] = RADIUS[0]
    feature_dic['RadiusOfCylinderBestConf'] = RADIUS[1]
    feature_dic['EBind'] = E_BIND

    values = []
    for key in sorted(feature_dic.keys()):
        values.append(feature_dic[key])
    # print key, feature_dic[key]
    return values


def get_fingerprints_for_SDF(SDFfile):
    """
    PRE  : Takes in a SDFfile
    POST : Produces a file SDFfile_FINGERPRINTS with the fingerprints as supplied by get_fingerprint()
    """
    supp = Chem.SDMolSupplier(SDFfile, removeHs=False)
    legends = []
    with open(SDFfile + '_FINGERPRINTS', 'wb') as w:
        w.write('MOLECULE_NAME,' + ','.join(get_fingerprint()) + '\n')
        for mol in supp:
            fingerprints = get_fingerprint(mol)
            w.write(mol.GetProp('_Name') + ',' + ','.join(['{0:4.4f}'.format(x) for x in fingerprints]) + '\n')
            AllChem.Compute2DCoords(mol)
            legends.append(mol.GetProp('_Name'))
        img = Chem.Draw._MolsToGridImage(supp, legends=legends, molsPerRow=10, subImgSize=(400, 400))
        img.save('img.png')


def get_fingerprint_for_SUM(SUMfile):
    """
    PRE  : Takes in a SUM file
    POST : Runs the finger prints on it by extracting the radius of best E, smiles and binding energy
    """
    with open(SUMfile, 'rb') as r:
        with open(SUMfile + '_FINGERPRINTS', 'wb') as w:
            w.write('PUBCHEM_NUMBER,' + 'SMILES,' + ','.join(get_fingerprint()) + '\n')
            for i, line in enumerate(r):
                #### Parses the line
                parts = line.split()
                # print zip(range(len(parts)),  parts)
                smiles = parts[1]
                pubchem_number = parts[0]
                radius_E = "na"#parts[4]
                radius_BEST = "na"#parts[8]
                E_BIND = parts[2]
                #### Computes fingerprints
                fingerprints = get_fingerprint(SMILES=smiles, RADIUS=(radius_BEST, radius_E), E_BIND=E_BIND)
                #### Writes to file
                if i % 1000 == 0: print i, zip(get_fingerprint(), fingerprints, range(len(fingerprints)))
                w.write(pubchem_number + ',' + smiles + ',' + ','.join(
                    ['{0:4.4f}'.format(float(x)) for x in fingerprints]) + '\n')
                #### Saves an image
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)
                Chem.Draw.MolToFile(mol, SUMfile + '_{}.png'.format(pubchem_number), size=(400, 400))


if __name__ == "__main__":
    # get_fingerprint('c1ccccc1C(=O)O')
    # get_fingerprint('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
    # get_fingerprints_for_SDF('/home/macenrola/Thesis/GENERATE_FINGERPRINTS/xzamc-min_20_only_mmff94_OUT.sdf_GUESTS.sdf')
    # get_fingerprint_for_SUM(
        # '/home/macenrola/Thesis/PHARMACY_BAD_TASTING/bad_taste_medecine_no_salts/bad_taste_medecine_no_salts.can_SUM')
# for els in get_fingerprint():
# 	print els
# get_fingerprint('C1=C[NH+]=CN1')
get_fingerprint_for_SUM("/home/macenrola/Documents/docked_for_data_analysis/fingerprints/500k_docked_pubsmienergy_restart")