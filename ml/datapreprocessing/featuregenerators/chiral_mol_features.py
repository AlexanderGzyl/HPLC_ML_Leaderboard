from rdkit import Chem
from mapchiral.mapchiral import encode, jaccard_similarity

molecule_1 = Chem.MolFromSmiles('C[C@H](C[S](=O)(=O)c1ccc(C)cc1)c2ccc(cc2)C(F)(F)F')
molecule_2 = Chem.MolFromSmiles('C[C@@H](C[S](=O)(=O)c1ccc(C)cc1)c2ccc(cc2)C(F)(F)F')

fingerprint_1 = encode(molecule_1, max_radius=2, n_permutations=2048, mapping=False)
fingerprint_2 = encode(molecule_2, max_radius=2, n_permutations=2048, mapping=False)

similarity = jaccard_similarity(fingerprint_1, fingerprint_2)

print(similarity)
