from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import pandas as pd
import numpy as np
from typing import List, Tuple, Callable, Union
# testing
smiles = [
    "F[C@H](Cl)Br",  # one enantiomer
    "F[C@@H](Cl)Br"  # mirror image
]


def compute_basic_rdkit_descriptor_df(
    smiles_list: List[str],
    imputer: Union[float, str, Callable[[pd.DataFrame], pd.DataFrame]] = 0.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute RDKit descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        A list of SMILES strings to featurize.

    imputer : float, 'keep', or callable, optional
        - If float, replaces all NaN/inf values with that number (default is 0.0).
        - If 'keep', retains NaN values in the output DataFrame.
        - If callable, should accept and return a DataFrame with imputed values.

    Returns
    -------
    Tuple[pandas.DataFrame, List[str]]
        - DataFrame with descriptor values and SMILES column
        - List of invalid SMILES strings
    """
    all_data = []
    valid_smiles = []
    invalid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append(smi)
            continue

        try:
            desc_dict = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
            all_data.append(desc_dict)
            valid_smiles.append(smi)

        except Exception as e:
            print(f"Error processing {smi}: {e}")
            invalid_smiles.append(smi)

    if not all_data:
        return pd.DataFrame(columns=["SMILES"]), invalid_smiles

    df = pd.DataFrame(all_data)
    df.insert(0, "SMILES", valid_smiles)

    # Apply imputation
    if isinstance(imputer, float) or isinstance(imputer, int):
        df.iloc[:, 1:] = df.iloc[:, 1:].fillna(imputer)
    elif callable(imputer):
        df.iloc[:, 1:] = imputer(df.iloc[:, 1:])
    elif imputer == "keep":
        pass  # leave NaNs as-is
    else:
        raise ValueError("Invalid value for 'imputer'. Use float, 'keep', or a callable.")

    return df, invalid_smiles


def compute_3D_rdkit_descriptor_df(smiles_list: List[str]) -> pd.DataFrame:
    """
    Compute RDKit 3D molecular descriptors for a list of SMILES strings.

    This includes descriptors such as:
        - Plane of Best Fit (PBF)
        - Labute ASA
        - AUTOCORR3D (12 bins)
        - Principal Moments of Inertia (PMI1–3)
        - Shape descriptors (Inertial Shape Factor, Spherocity, Eccentricity, Asphericity)
        - Normalized Principal Ratios (NPR1, NPR2)
        - Radius of Gyration
        - Molecular Density
        - Approximate Molecular Volume

    Parameters
    ----------
    smiles_list : List[str]
        A list of SMILES strings to compute 3D descriptors for.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a molecule and each column is a 3D descriptor.
        Molecules that fail to embed will have NaN values.
    """
    feature_data = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        if success != 0:
            # Embedding failed; fill with NaNs
            feature_data.append({
                "SMILES": smi,
                **{f"AUTOCORR3D_bin_{i}": np.nan for i in range(12)},
                "PBF": np.nan,
                "LabuteASA": np.nan,
                "InertialShapeFactor": np.nan,
                "SpherocityIndex": np.nan,
                "Eccentricity": np.nan,
                "Asphericity": np.nan,
                "PMI1": np.nan,
                "PMI2": np.nan,
                "PMI3": np.nan,
                "NPR1": np.nan,
                "NPR2": np.nan,
                "RadiusOfGyration": np.nan,
                "Density": np.nan,
                "MolVolume": np.nan,
            })
            continue

        try:
            desc = {
                "SMILES": smi,
                "PBF": rdMolDescriptors.CalcPBF(mol),
                "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
                "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol),
                "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol),
                "Eccentricity": rdMolDescriptors.CalcEccentricity(mol),
                "Asphericity": rdMolDescriptors.CalcAsphericity(mol),
                "PMI1": rdMolDescriptors.CalcPMI1(mol),
                "PMI2": rdMolDescriptors.CalcPMI2(mol),
                "PMI3": rdMolDescriptors.CalcPMI3(mol),
                "NPR1": rdMolDescriptors.CalcNPR1(mol),
                "NPR2": rdMolDescriptors.CalcNPR2(mol),
                "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration(mol),
            }
            volume = AllChem.ComputeMolVolume(mol)
            exact_mw = rdMolDescriptors.CalcExactMolWt(mol)
            desc["MolVolume"] = volume
            desc["ExactMolWt"] = exact_mw
            desc["Density"] = exact_mw / volume if volume != 0 else np.nan
            autocorr3d = rdMolDescriptors.CalcAUTOCORR3D(mol)
            for i, val in enumerate(autocorr3d):
                desc[f"AUTOCORR3D_bin_{i}"] = val

            feature_data.append(desc)
            valid_smiles.append(smi)

        except Exception as e:
            print(f"Error processing {smi}: {e}")
            continue

    return pd.DataFrame(feature_data)


def compute_flattened_coulomb_matrix_rdkit_descriptor_df(
    smiles_list: List[str],
    num_atoms: int = 20
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute flattened Coulomb matrix vectors from SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to featurize.

    num_atoms : int
        Target number of atoms. Matrices are padded/truncated to num_atoms × num_atoms,
        and the upper triangle is flattened.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - DataFrame with flattened Coulomb matrix vectors.
        - List of SMILES that failed processing.
    """
    feature_data = []
    failed_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_smiles.append(smi)
            continue

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            failed_smiles.append(smi)
            continue

        try:
            n_atoms = mol.GetNumAtoms()
            matrix = rdMolDescriptors.CalcCoulombMat(mol, confId=0)

            # Convert flat list to square matrix
            mat = np.array(matrix).reshape((n_atoms, n_atoms))

            # Pad/truncate to target size
            padded = np.zeros((num_atoms, num_atoms))
            n = min(num_atoms, n_atoms)
            padded[:n, :n] = mat[:n, :n]

            # Flatten upper triangle
            iu = np.triu_indices(num_atoms)
            flat = padded[iu]

            # Build row
            row = {"SMILES": smi}
            for i, v in enumerate(flat):
                row[f"Coulomb_{i}"] = v
            feature_data.append(row)

        except Exception as e:
            print(f"Error processing {smi}: {e}")
            failed_smiles.append(smi)

    df = pd.DataFrame(feature_data)
    return df, failed_smiles



def compute_eem_charges(
    smiles_list: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute EEM partial atomic charges for a list of SMILES strings.

    This function embeds each molecule in 3D using ETKDG, then calculates
    electronegativity equalization method (EEM) charges per atom. The output
    is a DataFrame where each row corresponds to a molecule, and columns are
    named EEM_0, EEM_1, ..., up to the max number of atoms encountered.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to compute EEM charges for.

    Returns
    -------
    Tuple[pandas.DataFrame, List[str]]
        - DataFrame with columns: SMILES, EEM_0, EEM_1, ..., EEM_N
        - List of invalid or failed SMILES strings
    """
    charge_data = []
    failed = []
    max_atoms = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(smi)
            continue

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            failed.append(smi)
            continue

        try:
            charges = rdMolDescriptors.CalcEEMcharges(mol, confId=0)
            charges = list(charges)
            max_atoms = max(max_atoms, len(charges))
            row = {"SMILES": smi}
            for i, c in enumerate(charges):
                row[f"EEM_{i}"] = c
            charge_data.append(row)
        except Exception as e:
            print(f"Error processing {smi}: {e}")
            failed.append(smi)

    # Pad missing charge columns with NaN
    df = pd.DataFrame(charge_data)
    for i in range(max_atoms):
        col = f"EEM_{i}"
        if col not in df.columns:
            df[col] = np.nan

    return df, failed


