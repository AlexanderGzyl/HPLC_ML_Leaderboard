from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem,rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdFingerprintGenerator
import matplotlib.pyplot as plt
import pandas as pd


def grand_tanimoto_similarity_mean(smiles_series,smiles_series_to_compare):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fp_list = [mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_series]
    fp_list_compare = [mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_series_to_compare]
    
    
    # Calculate Tanimoto similarities
    similarity_matrix = []
    for i in range(len(fp_list)):
        row = []
        for j in range(len(fp_list_compare)):
            similarity = TanimotoSimilarity(fp_list[i], fp_list_compare[j])
            row.append(similarity)
        similarity_matrix.append(sum(row)/len(row))

    return sum(similarity_matrix)/len(similarity_matrix)


def tanimoto_similarity_mean(smiles_series,smiles_series_to_compare):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fp_list = [mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_series]
    fp_list_compare = [mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_series_to_compare]
    
    
    # Calculate Tanimoto similarities
    similarity_list = []
    for i in range(len(fp_list)):
        row = []
        for j in range(len(fp_list_compare)):
            similarity = TanimotoSimilarity(fp_list[i], fp_list_compare[j])
            row.append(similarity)
        similarity_list.append(sum(row)/len(row))

    return similarity_list

from rdkit import Chem
from collections import Counter

from rdkit import Chem
from collections import Counter

def atom_counts(smiles):
    """
    Analyzes a molecule from its SMILES string to count atom types.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        dict or None: A dictionary of atom counts if successful, otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # Return None for invalid SMILES

    atom_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    return dict(atom_counts)

def plot_atom_counts_from_smiles_list(smiles_list):
    """
    Analyzes a list of SMILES strings and plots the cumulative atom counts.

    Args:
        smiles_list (list): A list of SMILES strings.
    """
    total_atom_counts = Counter()
    invalid_smiles_count = 0

    for smiles in smiles_list:
        counts = atom_counts(smiles)
        if counts is not None:
            total_atom_counts.update(counts)
        else:
            print(f"Warning: Invalid SMILES string skipped: {smiles}")
            invalid_smiles_count += 1

    if not total_atom_counts:
        print("No valid molecules processed to plot atom counts.")
        return

    # Prepare data for plotting
    atom_types = list(total_atom_counts.keys())
    counts = list(total_atom_counts.values())

    # Sort by count for better visualization
    sorted_ = sorted(zip(atom_types, counts), key=lambda x: x[1], reverse=True)
    sorted_atom_types, sorted_counts = zip(*sorted_)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_atom_types, sorted_counts, color='skyblue')
    for i in range(len(sorted_atom_types)):
        plt.text(i, sorted_counts[i], sorted_counts[i], ha='center')
    plt.xlabel('Atom Type')
    plt.ylabel('Cumulative Count Across Molecules')
    plt.title('Cumulative Atom Counts for a List of Molecules')
    plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

    if invalid_smiles_count > 0:
        print(f"\nNote: Encountered {invalid_smiles_count} invalid SMILES string(s) which were skipped.")

def plot_num_atoms_per_molecule(smiles_list):
    """
    Calculates the number of atoms for each valid molecule in a list of SMILES
    and plots a histogram of the distribution.

    Args:
        smiles_list (list): A list of SMILES strings.
    """
    num_atoms_list = []
    invalid_smiles_count = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            num_atoms_list.append(num_atoms)
        else:
            print(f"Warning: Invalid SMILES string skipped: {smiles}")
            invalid_smiles_count += 1

    if not num_atoms_list:
        print("No valid molecules processed to plot the number of atoms.")
        return

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(num_atoms_list, bins=50, color='lightgreen', edgecolor='black') # You can adjust the number of bins
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Atoms per Molecule')
    plt.grid(axis='y', alpha=0.75)

    # Add some descriptive statistics to the plot (optional)
    # plt.axvline(sum(num_atoms_list)/len(num_atoms_list), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {sum(num_atoms_list)/len(num_atoms_list):.2f}')
    # plt.legend()

    plt.tight_layout()
    plt.show()

    if invalid_smiles_count > 0:
        print(f"\nNote: Encountered {invalid_smiles_count} invalid SMILES string(s) which were skipped.")


def plot_stereogenic_centers_per_molecule(smiles_list):
    """
    Calculates the number of potential tetrahedral stereogenic centers
    for each valid molecule in a list of SMILES and plots a histogram
    of the distribution.

    Args:
        smiles_list (list): A list of SMILES strings.
    """
    num_chiral_centers_list = []
    invalid_smiles_count = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Find tetrahedral chiral centers.
            # includeExamples=False means we just get the atom indices.
            # use ChiralityParse=True if you want to consider explicitly defined stereocenters in SMILES like @ or @@
            # For a general "potential centers" plot, includeExamples=False and the default ChiralityParse=False is often suitable
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            num_chiral_centers = len(chiral_centers)
            num_chiral_centers_list.append(num_chiral_centers)
        else:
            print(f"Warning: Invalid SMILES string skipped: {smiles}")
            invalid_smiles_count += 1

    if not num_chiral_centers_list:
        print("No valid molecules processed to plot the number of stereogenic centers.")
        return

    # Count the frequency of each number of chiral centers
    counts = Counter(num_chiral_centers_list)
    chiral_center_counts = list(counts.keys())
    frequencies = list(counts.values())

    # Sort by number of chiral centers for plotting
    sorted_pairs = sorted(zip(chiral_center_counts, frequencies))
    sorted_chiral_center_counts, sorted_frequencies = zip(*sorted_pairs)

    # Create the histogram/bar plot
    plt.figure(figsize=(10, 6))
    # Use a bar plot for discrete integer counts
    plt.bar(sorted_chiral_center_counts, sorted_frequencies, color='salmon', edgecolor='black')

    plt.xlabel('Number of Potential Tetrahedral Stereogenic Centers')
    plt.ylabel('Frequency')
    plt.title('Distribution of Potential Tetrahedral Stereogenic Centers per Molecule')
    plt.xticks(sorted_chiral_center_counts) # Ensure all occurring counts are labeled on x-axis
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()

    if invalid_smiles_count > 0:
        print(f"\nNote: Encountered {invalid_smiles_count} invalid SMILES string(s) which were skipped.")

def plot_rotatable_bonds_per_molecule(smiles_list):
    """
    Calculates the number of rotatable bonds for each valid molecule
    in a list of SMILES and plots a histogram of the distribution.

    Args:
        smiles_list (list): A list of SMILES strings.
    """
    num_rotatable_bonds_list = []
    invalid_smiles_count = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Calculate the number of rotatable bonds
            num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            num_rotatable_bonds_list.append(num_rotatable_bonds)
        else:
            print(f"Warning: Invalid SMILES string skipped: {smiles}")
            invalid_smiles_count += 1

    if not num_rotatable_bonds_list:
        print("No valid molecules processed to plot the number of rotatable bonds.")
        return

    # Count the frequency of each number of rotatable bonds
    counts = Counter(num_rotatable_bonds_list)
    rotatable_bond_counts = list(counts.keys())
    frequencies = list(counts.values())

    # Sort by number of rotatable bonds for plotting
    sorted_pairs = sorted(zip(rotatable_bond_counts, frequencies))
    sorted_rotatable_bond_counts, sorted_frequencies = zip(*sorted_pairs)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_rotatable_bond_counts, sorted_frequencies, color='purple', edgecolor='black')

    plt.xlabel('Number of Rotatable Bonds')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rotatable Bonds per Molecule')
    plt.xticks(sorted_rotatable_bond_counts) # Ensure all occurring counts are labeled on x-axis
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()

    if invalid_smiles_count > 0:
        print(f"\nNote: Encountered {invalid_smiles_count} invalid SMILES string(s) which were skipped.")


def plot_rotatable_bonds_vs_property(df: pd.DataFrame, smiles_col: str = 'smiles', property_col: str = 'RT'):
    """
    Calculates the number of rotatable bonds for molecules in a DataFrame
    and plots it against a specified property column.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES and the property.
        smiles_col (str): The name of the column containing SMILES strings (default: 'smiles').
        property_col (str): The name of the column containing the property values (default: 'RT').
    """
    # --- Input Validation ---
    if smiles_col not in df.columns:
        print(f"Error: DataFrame must contain a '{smiles_col}' column.")
        return
    if property_col not in df.columns:
        print(f"Error: DataFrame must contain a '{property_col}' column.")
        return

    # --- Data Processing ---
    num_rotatable_bonds_list = []
    property_values_list = []
    invalid_smiles_count = 0
    processed_count = 0

    # Iterate through DataFrame rows
    for index, row in df.iterrows():
        smiles = row[smiles_col]
        property_value = row[property_col]

        # Ensure property value is numerical and not NaN
        if pd.isna(property_value) or not isinstance(property_value, (int, float)):
            print(f"Warning: Skipping row {index} due to invalid or missing property value in column '{property_col}': {property_value}")
            continue # Skip this row if property value is invalid

        mol = Chem.MolFromSmiles(str(smiles)) # Ensure smiles is treated as string
        if mol is not None:
            try:
                num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                num_rotatable_bonds_list.append(num_rotatable_bonds)
                property_values_list.append(property_value)
                processed_count += 1
            except Exception as e:
                # Catch potential RDKit errors during descriptor calculation
                print(f"Warning: Could not calculate rotatable bonds for SMILES '{smiles}' (row {index}). Error: {e}")
                invalid_smiles_count += 1
        else:
            print(f"Warning: Invalid SMILES string skipped at row {index}: {smiles}")
            invalid_smiles_count += 1

    # --- Plotting ---
    if not num_rotatable_bonds_list:
        print("No valid data points processed to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Create a scatter plot
    plt.scatter(num_rotatable_bonds_list, property_values_list, alpha=0.5) # alpha for transparency if points overlap

    plt.xlabel('Number of Rotatable Bonds')
    plt.ylabel(property_col) # Use the actual property column name for the y-label
    plt.title(f'Number of Rotatable Bonds vs. {property_col}')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Optional: Add correlation coefficient to the plot (requires numpy)
    # if processed_count > 1:
    #     correlation = np.corrcoef(num_rotatable_bonds_list, property_values_list)[0, 1]
    #     plt.text(0.95, 0.95, f'Correlation: {correlation:.2f}',
    #              verticalalignment='top', horizontalalignment='right',
    #              transform=plt.gca().transAxes,
    #              fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    plt.tight_layout()
    plt.show()

    print(f"\nProcessed {processed_count} valid molecules.")
    if invalid_smiles_count > 0:
        print(f"Skipped {invalid_smiles_count} rows due to invalid SMILES or calculation errors.")




