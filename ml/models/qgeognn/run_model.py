import os
import random
import scipy.sparse as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from torch_geometric.data import DataLoader
import torch_geometric
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import rdMolDescriptors
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
from mordred import Calculator, descriptors, is_missing
from pyDOE import *
from utils import *
from model import *

# Paths
full_path = './ml/models/qgeognn/sourcedata/'

# Parameters
MODEL = 'Train'  # ['Train','Test','Test_other_method','Test_enantiomer','Test_excel']
test_mode = 'fixed'  # fixed or random or enantiomer(extract enantimoers)
transfer_target = 'All_column_test'  # trail name
Use_geometry_enhanced = True  # default:True
Use_column_info = True  # default: True

atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = [
    "bond_dir", "bond_type", "is_in_ring"]

if Use_geometry_enhanced == True:
    bond_float_names = ["bond_length", 'prop']

if Use_geometry_enhanced == False:
    bond_float_names = ['prop']

bond_angle_float_names = ['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']

column_specify = {'ADH':[1,5,0,0],'ODH':[1,5,0,1],'IC':[0,5,1,2],'IA':[0,5,1,3],'OJH':[1,5,0,4],
                'ASH':[1,5,0,5],'IC3':[0,3,1,6],'IE':[0,5,1,7],'ID':[0,5,1,8],'OD3':[1,3,0,9],
                'IB':[0,5,1,10],'AD':[1,10,0,11],'AD3':[1,3,0,12],'IF':[0,5,1,13],'OD':[1,10,0,14],
                'AS':[1,10,0,15],'OJ3':[1,3,0,16],'IG':[0,5,1,17],'AZ':[1,10,0,18],'IAH':[0,5,1,19],
                'OJ':[1,10,0,20],'ICH':[0,5,1,21],'OZ3':[1,3,0,22],'IF3':[0,3,1,23],'IAU':[0,1.6,1,24]}

column_smile = ['O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(C2=CC=C(C)C=C2)=O)[C@@H](OC(C3=CC=C(C)C=C3)=O)[C@@H]1OC)C4=CC=C(C)C=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(N[C@@H](C)C2=CC=CC=C2)=O)[C@@H](OC(N[C@@H](C)C3=CC=CC=C3)=O)[C@H]1OC)N[C@@H](C)C4=CC=CC=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC=CC(Cl)=C3)=O)[C@H]1OC)NC4=CC=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(N[C@@H](C)C2=CC=CC=C2)=O)[C@@H](OC(N[C@@H](C)C3=CC=CC=C3)=O)[C@H]1OC)N[C@@H](C)C4=CC=CC=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(C2=CC=C(C)C=C2)=O)[C@@H](OC(C3=CC=C(C)C=C3)=O)[C@@H]1OC)C4=CC=C(C)C=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(Cl)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(C2=CC=C(C)C=C2)=O)[C@@H](OC(C3=CC=C(C)C=C3)=O)[C@@H]1OC)C4=CC=C(C)C=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(Cl)=CC(Cl)=C2)=O)[C@@H](OC(NC3=CC(Cl)=CC(Cl)=C3)=O)[C@@H]1OC)NC4=CC(Cl)=CC(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC=C(C)C(Cl)=C2)=O)[C@@H](OC(NC3=CC=C(C)C(Cl)=C3)=O)[C@H]1OC)NC4=CC=C(C)C(Cl)=C4',
'O=C(OC[C@@H](O1)[C@@H](OC)[C@H](OC(NC2=CC(C)=CC(C)=C2)=O)[C@@H](OC(NC3=CC(C)=CC(C)=C3)=O)[C@H]1OC)NC4=CC(C)=CC(C)=C4']

column_name = ['ADH','ODH','IC','IA','OJH','ASH','IC3','IE','ID','OD3', 'IB','AD','AD3',
            'IF','OD','AS','OJ3','IG','AZ','IAH','OJ','ICH','OZ3','IF3','IAU']
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)


if Use_column_info:
    bond_id_names.extend(['coated', 'immobilized'])
    bond_float_names.extend(['diameter'])
    if Use_geometry_enhanced:
        bond_angle_float_names.extend(['column_TPSA', 'column_TPSA', 'column_TPSA', 'column_MDEC', 'column_MATS'])
    else:
        bond_float_names.extend(['column_TPSA', 'column_TPSA', 'column_TPSA', 'column_MDEC', 'column_MATS'])
    full_bond_feature_dims.extend([2, 2])

calc = Calculator(descriptors, ignore_3D=False)

def mord(mol, nBits=1826, errors_as_zeros=True):
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    
    except:
        return np.NaN if not errors_as_zeros else np.zeros(
            (nBits,), dtype=np.float32
            )


def load_3D_mol():
    dir = 'mol_save/'
    for root, dirs, files in os.walk(dir):
        file_names = files
    file_names.sort(key=lambda x: int(x[x.find('_') + 5:x.find(".")]))
    mol_save = []
    for file_name in file_names:
        mol_save.append(Chem.MolFromMolFile(dir + file_name))
    return mol_save


def parse_args():
    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_root', type=str, default="dataset",
                        help='dataset root')
    args = parser.parse_args()

    return args


def calc_dragon_type_desc(mol):
    compound_mol = mol
    compound_MolWt = Descriptors.ExactMolWt(compound_mol)
    compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
    compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
    compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
    compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
    compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
    return (
        rdMolDescriptors.CalcAUTOCORR3D(mol) +
        rdMolDescriptors.CalcMORSE(mol) +
        rdMolDescriptors.CalcRDF(mol) +
        rdMolDescriptors.CalcWHIM(mol) +
        ([
            compound_MolWt, compound_TPSA,
            compound_nRotB, compound_HBD,
            compound_HBA, compound_LogP
            ])
        )


def preparation(args):
    save_dir = os.path.join('saves', args.task_name)
    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)


def q_loss(q, y_true, y_pred):
    e = (y_true-y_pred)
    return torch.mean(torch.maximum(q*e, (q-1)*e))


def eval(model, device, loader_atom_bond, loader_bond_angle):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_10 = []
    y_pred_90 = []

    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond, batch_bond_angle)[0]

            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:,1].detach().cpu())
            y_pred_10.append(pred[:,0].detach().cpu())
            y_pred_90.append(pred[:,2].detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_pred_10 = torch.cat(y_pred_10, dim=0)
    y_pred_90 = torch.cat(y_pred_90, dim=0)
    # plt.plot(y_pred.cpu().data.numpy(),c='blue')
    # plt.plot(y_pred_10.cpu().data.numpy(),c='yellow')
    # plt.plot(y_pred_90.cpu().data.numpy(),c='black')
    # plt.plot(y_true.cpu().data.numpy(),c='red')
    #plt.show()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return torch.mean((y_true - y_pred) ** 2).data.numpy()


def train(
    model,
    device,
    loader_atom_bond,
    loader_bond_angle,
    optimizer,
    criterion_fn
    ):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
        batch_atom_bond = batch[0]
        batch_bond_angle = batch[1]
        batch_atom_bond = batch_atom_bond.to(device)
        batch_bond_angle = batch_bond_angle.to(device)
        pred = model(batch_atom_bond, batch_bond_angle)[0]  #.view(-1, )
        true = batch_atom_bond.y
        optimizer.zero_grad()
        loss = (
            q_loss(0.1, true,pred[:, 0]) +
            torch.mean((true-pred[:, 1])**2) +
            q_loss(0.9, true, pred[:, 2]) +
            torch.mean(torch.relu(pred[:, 0]-pred[:, 1])) +
            torch.mean(torch.relu(pred[:, 1]-pred[:, 2])) +
            torch.mean(torch.relu(2-pred))
            )
        
        #loss = criterion_fn(pred, batch_atom_bond.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def test(
    model,
    device,
    loader_atom_bond,
    loader_bond_angle
    ):
    model.eval()
    y_pred = []
    y_true = []
    y_pred_10 = []
    y_pred_90 = []
    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond, batch_bond_angle)[0]
            y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1,))
            y_pred.append(pred[:, 1].detach().cpu())
            y_pred_10.append(pred[:, 0].detach().cpu())
            y_pred_90.append(pred[:, 2].detach().cpu())
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_10 = torch.cat(y_pred_10, dim=0)
        y_pred_90 = torch.cat(y_pred_90, dim=0)
        # plt.figure(10)
        # box=np.array([y_pred_10.cpu().data.numpy(),y_pred.cpu().data.numpy(),y_pred_90.cpu().data.numpy()])
        # plt.boxplot(box)
        # plt.scatter(np.arange(0,y_true.shape[0],1),y_true.cpu().data.numpy(), c='red')

        # plt.plot(y_pred.cpu().data.numpy(), c='blue')
        # plt.plot(y_pred_10.cpu().data.numpy(), c='yellow')
        # plt.plot(y_pred_90.cpu().data.numpy(), c='black')
        # plt.plot(y_true.cpu().data.numpy(), c='red')
        #plt.show()

    R_square = 1 - (
        ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean())**2).sum()
        )
    test_mae = torch.mean((y_true - y_pred) ** 2)
    print(R_square)
    return y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90


def get_feature(model, device, loader):
    model.eval()
    y_pred = []
    y_true = []
    GNN_feature = []
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            pred, h_graph = model(batch)
            y_pred.append(pred.detach().cpu())
            y_true.append(batch.y.detach().cpu().reshape(pred.shape[0], pred.shape[1]))
            GNN_feature.append(h_graph.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0).cpu().data.numpy()
    y_true = torch.cat(y_true, dim=0).cpu().data.numpy()
    GNN_feature = torch.cat(GNN_feature, dim=0).cpu().data.numpy()
    T_1_pred = y_pred[:, 0]
    T_1_true = y_true[:, 0]
    return y_pred, y_true, GNN_feature


def cal_prob(prediction):
    '''
    calculate the separation probability Sp
    '''
    #input  prediction=[pred_1,pred_2]
    #output: Sp
    a = prediction[0][0]
    b = prediction[1][0]
    if a[2] < b[0]:
        return 1
    elif a[0] > b[2]:
        return 1
    else:
        length = min(a[2], b[2]) - max(a[0], b[0])
        all = max(a[2], b[2]) - min(a[0], b[0])
        return 1-length/(all)


def Convert_adjacency_matrix(
    edge_index,
    plot_name,
    plot=False
    ):
    size = edge_index.shape[1]
    atom_num = torch.max(edge_index)+1
    adj = torch.zeros(size, size)
    edge_w = torch.ones(size)
    adj_matrix = sp.coo_matrix(arg1=(edge_w, (edge_index[0, :], edge_index[1, :])), shape=(size, size))
    adj_matrix = adj_matrix.todense()[0:atom_num, 0:atom_num]

    plt.figure(10, figsize=(2, 2), dpi=300)
    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, atom_num, 1))
    ax.set_yticks(np.arange(0, atom_num, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, atom_num, 1))
    ax.set_yticklabels(np.arange(0, atom_num, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, atom_num-1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, atom_num-1, 1), minor=True)

    # Gridlines based on minor ticks
    if plot == True:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        plt.imshow(adj_matrix, cmap='Purples', vmin=-0.3, vmax=3)
        plt.tight_layout()
        plt.savefig(
            f'fig_save/{plot_name}.tiff',
            bbox_inches='tight',
            pad_inches=0
            )
        plt.savefig(
            f'fig_save/{plot_name}.pdf',
            bbox_inches='tight',
            pad_inches=0
            )
        plt.show()
    
    return adj_matrix


def Construct_dataset(
    dataset,
    data_index,
    T1,
    speed,
    eluent,
    column,
    column_name,
    column_specify,
    atom_id_names,
    bond_id_names,
    Use_geometry_enhanced: bool = True,
    Use_column_info: bool = True
    ):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []

    full_path = './ml/models/qgeognn/sourcedata/'
    column_descriptor = np.load(
        os.path.join(full_path, 'column_descriptor.npy'), allow_pickle=True
        )
    all_descriptor = np.load(
        os.path.join(full_path, 'descriptor_all_column.npy')
        )

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        col_info = column_name[i]
        col_specify = column_specify[col_info]
        col_des = np.array(column_descriptor[col_specify[3]])
        atom_feature = []
        bond_feature = []

        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names[0:3]:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([float(T1[i]) * float(speed[i])])
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int = torch.from_numpy(np.array(data_index[i])).to(torch.int64)

        prop = torch.ones([bond_feature.shape[0]])*eluent[i]
        coated = torch.ones([bond_feature.shape[0]]) * col_specify[0]
        diameter = torch.ones([bond_feature.shape[0]]) * col_specify[1]
        immobilized = torch.ones([bond_feature.shape[0]]) * col_specify[2]

        TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820]/100
        RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
        RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
        MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
        MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

        if Use_geometry_enhanced:
            col_TPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[820] / 100
            col_RASA = torch.ones([bond_angle_feature.shape[0]]) * col_des[821]
            col_RPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[822]
            col_MDEC = torch.ones([bond_angle_feature.shape[0]]) * col_des[1568]
            col_MATS = torch.ones([bond_angle_feature.shape[0]]) * col_des[457]
        else:
            col_TPSA = torch.ones([bond_feature.shape[0]]) * col_des[820] / 100
            col_RASA = torch.ones([bond_feature.shape[0]]) * col_des[821]
            col_RPSA = torch.ones([bond_feature.shape[0]]) * col_des[822]
            col_MDEC = torch.ones([bond_feature.shape[0]]) * col_des[1568]
            col_MATS = torch.ones([bond_feature.shape[0]]) * col_des[457]
        if Use_column_info:
            bond_feature = torch.cat([bond_feature, coated.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, immobilized.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)

        if Use_column_info:
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)

        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
        bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

        if Use_column_info == True:
            if Use_geometry_enhanced == True:
                bond_angle_feature = torch.cat([bond_angle_feature, col_TPSA.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_RASA.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_RPSA.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_MDEC.reshape(-1, 1)], dim=1)
                bond_angle_feature = torch.cat([bond_angle_feature, col_MATS.reshape(-1, 1)], dim=1)
            else:
                bond_feature = torch.cat([bond_feature, col_TPSA.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_RASA.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_RPSA.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_MDEC.reshape(-1, 1)], dim=1)
                bond_feature = torch.cat([bond_feature, col_MATS.reshape(-1, 1)], dim=1)

        if y[0] > 60:
            big_index.append(i)
            continue

        data_atom_bond = Data(
            atom_feature, edge_index, bond_feature, y,data_index=data_index_int
            )
        data_bond_angle = Data(
            edge_index=bond_index, edge_attr=bond_angle_feature
            )
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    return (graph_atom_bond, graph_bond_angle, big_index)


#-------------Construct data----------------
'''
The Graph construction is prepared and saved beforehand to accelerate the process by the code:
for smile in smiles:
    mol = obtain_3D_mol(smile, 'trail')
    mol = Chem.MolFromMolFile(f"trail.mol")
    all_descriptor.append(mord(mol))
    dataset.append(mol_to_geognn_graph_data_MMFF3d(mol))
'''


def main():
    import os
    full_path = './ml/models/qgeognn/sourcedata/'
    HPLC_all = pd.read_csv(os.path.join(full_path, 'All_column_charity.csv'))

    bad_all_index = np.load(os.path.join(full_path, 'bad_all_column.npy')) #Some compounds that cannot get 3D conformer by RDKit
    HPLC_all = HPLC_all.drop(bad_all_index)
    all_smile_all = HPLC_all['SMILES'].values
    T1_all = HPLC_all['RT'].values
    Speed_all = HPLC_all['Speed'].values
    Prop_all = HPLC_all['i-PrOH_proportion'].values
    dataset_all = np.load(os.path.join(full_path, 'dataset_charity_all_column.npy'), allow_pickle=True).tolist()
    index_all = HPLC_all['Unnamed: 0'].values
    Column_info = HPLC_all['Column'].values
    Column_charity_index = HPLC_all['index'].values

    #===========NN setting===============
    train_ratio = 0.90
    validate_ratio = 0.05
    test_ratio = 0.05
    args = parse_args()
    preparation(args)
    nn_params = {
        'bond_float_names': bond_float_names,
        'bond_angle_float_names': bond_angle_float_names,
        'bond_id_names': bond_id_names,
        'full_atom_feature_dims': full_atom_feature_dims,
        'full_bond_feature_dims': full_bond_feature_dims,
        'num_tasks': 3,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827,
        'Use_geometry_enhanced': Use_geometry_enhanced
    }

    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
    # writer = SummaryWriter(log_dir=args.save_dir)
    not_improved = 0
    best_valid_mae = 9999
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)


    if MODEL in ['Train', 'Test', 'Test_other_method']:
        dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = Construct_dataset(
            dataset_all,
            index_all,
            T1_all,
            Speed_all,
            Prop_all,
            transfer_target,
            Column_info,
            column_specify=column_specify,
            atom_id_names=atom_id_names,
            bond_id_names=bond_id_names,
            Use_geometry_enhanced=Use_geometry_enhanced,
            Use_column_info=Use_column_info
            )
        
        print("data set constructed!")
        total_num = len(dataset_graph_atom_bond)
        print(f"length of total num {total_num}")
        # automatic dataloading and splitting

        if test_mode == 'enantiomer':
            '''
            Randomly select enantiomers
            '''
            index_all = []
            fix_index = []
            for i in range(len(dataset_graph_atom_bond)):
                index_all.append(int(dataset_graph_atom_bond[i].data_index.data.numpy()))

            charity_all_index = np.unique(np.array(Column_charity_index)).tolist()
            HPLC_all_save = pd.read_csv(os.path.join(full_path, 'All_column_charity.csv'))
            Column_charity_index_save = HPLC_all_save['index'].values
            random.seed(1101)
            select_num = random.sample(charity_all_index, 500)
            #print(select_num[0:10])

            index_loc = []
            for i in select_num:
                loc = np.where(np.array(Column_charity_index_save) == i)[0]
                index_loc.extend(loc)
            #print(index_loc[0:10])

            #(HPLC_all_save.iloc[index_loc]).to_excel('dataset/test_compound_charity.xlsx')

            for i in index_loc:
                if len(np.where(np.array(index_all) == i)[0]) > 0:
                    fix_index.append(np.where(np.array(index_all) == i)[0][0])
            #print(fix_index[0:10])

            print(len(fix_index))
            data_array = np.arange(0, total_num, 1)
            data_array = np.delete(data_array, fix_index)


        else:
            data_array = np.arange(0, total_num, 1)

        #given random seed
        np.random.seed(388)
        torch.random.manual_seed(525)
        rngenerator = np.random.default_rng(seed=42)
        rngenerator.shuffle(data_array)
        # np.random.shuffle(data_array)

        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        train_column_atom_bond = []
        valid_column_atom_bond = []
        test_column_atom_bond = []
        train_column_bond_angle = []
        valid_column_bond_angle = []
        test_column_bond_angle = []
        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)

        if test_mode == 'enantiomer':
            train_num = int(total_num * train_ratio)
            val_num = int(total_num * validate_ratio)

            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            test_index = fix_index

        else:
            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            if test_mode == 'fixed':
                test_index = data_array[total_num-test_num:]
            if test_mode=='random':
                test_index = data_array[train_num + val_num:train_num + val_num + test_num]


        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])

        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])

        for i in train_index:
            
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])
    # save indexes into a file in order to see what was used
        
        filename = 'model_data_split_repeat.npz'
        np.savez(
            filename,
            train=train_index,
            valid=valid_index,
            test=test_index
            )
        print('data saved')

        print('========Data  preprared!=============')
        print(test_data_atom_bond[0].y,test_data_atom_bond[477].data_index)

        train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        if MODEL == 'Train':
            print('=========Start training!=================\n')
            
            if not Use_geometry_enhanced:
                file_name = 'saves/model_{transfer_target}_GNN'
                try:
                    os.makedirs(f'saves/model_{transfer_target}_GNN')
                except OSError:
                    pass

            elif not Use_column_info:
                file_name = f'saves/model_{transfer_target}_no_column_info'
                try:
                    os.makedirs(f'saves/model_{transfer_target}_no_column_info')
                except OSError:
                    pass
            else:
                file_name = f'saves/model_{transfer_target}'
                try:
                    os.makedirs(f'saves/model_{transfer_target}')
                except OSError:
                    pass

            if test_mode == 'enantiomer':
                file_name = f'saves/model_{transfer_target}_enantiomer'
                try:
                    os.makedirs(f'saves/model_{transfer_target}_enantiomer')
                except OSError:
                    pass
    # original epoch range 1500 changed to 5
            for epoch in tqdm(range(5)):
                train_mae = train(
                    model,
                    device,
                    train_loader_atom_bond,
                    train_loader_bond_angle,
                    optimizer,
                    criterion_fn
                    )

                if (epoch + 1) % 100 == 0:
                    valid_mae = eval(
                        model,
                        device,
                        valid_loader_atom_bond,
                        valid_loader_bond_angle
                        )
                    print(train_mae, valid_mae)
                    torch.save(model.state_dict(), file_name+f'/model_save_{epoch + 1}.pth')

        if MODEL == 'Test':
            print('==================Start testing==============')
            import os
            current_directory = os.getcwd()
            print(current_directory)
            if Use_geometry_enhanced:
                if Use_column_info:
                    print(model)
                    print(model._load_from_state_dict)
                    print(model.load_state_dict)
                    model.load_state_dict(
                        torch.load(r'models\geognn\model_save_800.pth',map_location=torch.device('cpu')))
                else:
                    model.load_state_dict(
                        torch.load(f'saves/model_All_column_no_column_info/model_save_800.pth'))
            else:
                model.load_state_dict(
                    torch.load(f'saves/model_All_column_GNN/model_save_800.pth'))

            y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90 = test(
                model,
                device,
                test_loader_atom_bond,
                test_loader_bond_angle
                )
            y_pred=y_pred.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            y_pred_10=y_pred_10.cpu().data.numpy()
            y_pred_90=y_pred_90.cpu().data.numpy()
            print('relative_error',np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)))
            print('MAE',np.mean(np.abs(y_true - y_pred) / y_true))
            print('RMSE',np.sqrt(np.mean((y_true - y_pred) ** 2)))
            print(torch.mean((y_true - y_pred) ** 2).data.numpy())
            R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
            print(R_square)
            plt.figure(1,figsize=(3,3),dpi=300)
            plt.style.use('ggplot')
            plt.scatter(y_true, y_pred, c='#8983BF',s=15,alpha=0.4)
            plt.plot(np.arange(0, 60), np.arange(0, 60),linewidth=1.5,linestyle='--',color='black')
            plt.yticks(fontproperties='Arial', size=8)
            plt.xticks(np.arange(0,66,10),fontproperties='Arial', size=8)
            # plt.xlabel('Observed data', fontproperties='Arial', size=8)
            # plt.ylabel('Predicted data', fontproperties='Arial', size=8)
            if Use_geometry_enhanced==True:
                if Use_column_info==True:
                    plt.savefig(f'fig_save/predict_{transfer_target}.tiff', bbox_inches='tight', dpi=300)
                    plt.savefig(f'fig_save/predict_{transfer_target}.pdf', bbox_inches='tight', dpi=300)
                    plt.savefig(f'fig_save/predict_{transfer_target}.jpg', bbox_inches='tight', dpi=300)
                else:
                    plt.savefig(f'fig_save/predict_{transfer_target}_no_column_info.tiff', bbox_inches='tight', dpi=300)
                    plt.savefig(f'fig_save/predict_{transfer_target}_no_column_info.pdf', bbox_inches='tight', dpi=300)
                    plt.savefig(f'fig_save/predict_{transfer_target}_no_column_info.jpg', bbox_inches='tight', dpi=300)
            else:
                plt.savefig(f'fig_save/predict_{transfer_target}_GNN.tiff', bbox_inches='tight', dpi=300)
                plt.savefig(f'fig_save/predict_{transfer_target}_GNN.pdf', bbox_inches='tight', dpi=300)
                plt.savefig(f'fig_save/predict_{transfer_target}_GNN.jpg', bbox_inches='tight', dpi=300)
            plt.show()


if __name__ == '__main__':
    main()
    # if MODEL =='Test_other_method' :
    #     dataset = dataset_all
    #     all_descriptor = np.load('sourcedata/descriptor_all_column.npy')
    #     column_descriptor = np.load('sourcedata/column_descriptor.npy', allow_pickle=True)
    #     dataset_Finger = []
    #     from rdkit.Chem import MACCSkeys
    #     index=0
    #     coated=[]
    #     diameter=[]
    #     immobilized=[]
    #     col_TPSA=[]
    #     col_RASA = []
    #     col_RPSA = []
    #     col_MDEC = []
    #     col_MATS = []
    #     for smile in tqdm(all_smile_all):
    #         col_info = Column_info[index]
    #         col_specify = column_specify[col_info]
    #         col_des = np.array(column_descriptor[col_specify[3]])
    #         mol = Chem.MolFromSmiles(smile)
    #         Finger = MACCSkeys.GenMACCSKeys(mol)
    #         Figerprint = np.array([x for x in Finger])
    #         dataset_Finger.append(Figerprint)
    #         coated.append(col_specify[0])
    #         diameter.append(col_specify[1])
    #         immobilized.append(col_specify[2])
    #         col_TPSA.append(col_des[820] / 100)
    #         col_RASA.append(col_des[821])
    #         col_RPSA.append(col_des[822])
    #         col_MDEC.append(col_des[1568])
    #         col_MATS.append(col_des[457])
    #         index+=1

    #     dataset_Finger = np.array(dataset_Finger)
    #     dataset = np.hstack((dataset_Finger, Prop_all.reshape(Prop_all.shape[0], 1)))
    #     for col in [820,821,822,1568,457]:
    #         dataset = np.hstack((dataset, all_descriptor[:,col].reshape(-1,1)))
    #     dataset = np.hstack((dataset, np.array(coated).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(diameter).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(immobilized).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(col_TPSA).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(col_RASA).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(col_RPSA).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(col_MDEC).reshape(-1, 1)))
    #     dataset = np.hstack((dataset, np.array(col_MATS).reshape(-1, 1)))
    #     y = []
    #     for i in range(dataset.shape[0]):
    #         y.append([float(T1_all[i]) * float(Speed_all[i])])
    #     y = np.array(y)


    #     large_index=np.where((y>60))[0]
    #     y=np.delete(y,large_index).reshape(-1,1)
    #     dataset = np.delete(dataset, large_index,axis=0)




    #     from torch.autograd import Variable
    #     from xgboost.sklearn import XGBRegressor
    #     import torch.nn.functional as F
    #     import pandas as pd
    #     import lightgbm as lgb
    #     from rdkit.Chem import Descriptors
    #     from tqdm import tqdm

    #     X_train = dataset[data_array[0:train_num]]
    #     y_train = y[data_array[0:train_num]]
    #     X_valid = dataset[data_array[train_num :train_num + val_num]]
    #     y_valid = y[data_array[train_num :train_num + val_num]]
    #     if test_mode=='random':
    #         X_test = dataset[data_array[train_num + val_num:train_num + test_num + val_num]]
    #         y_test = y[data_array[train_num + val_num:train_num + test_num + val_num]]
    #     if test_mode=='fixed':
    #         X_test = dataset[data_array[total_num - test_num:]]
    #         y_test = y[data_array[total_num - test_num:]]
    #     if test_mode=='enantiomer':
    #         X_test = dataset[test_index]
    #         y_test = y[test_index]


    #     dataset_actual=X_test[[414,415,476,477],:]
    #     y_actual = y_test[[414,415,476,477], :]
    #     print(y_actual)

    #     model_lgb = lgb.LGBMRegressor(objective='regression', max_depth=5,
    #                                   num_leaves=25,
    #                                   learning_rate=0.007, n_estimators=1000)

    #     model_xgb = XGBRegressor(seed=525,
    #                              n_estimators=200,
    #                              max_depth=3,
    #                              eval_metric='rmse',
    #                              learning_rate=0.1,
    #                              min_child_weight=1,
    #                              subsample=1,
    #                              colsample_bytree=1,
    #                              colsample_bylevel=1,
    #                              gamma=0)

    #     model_xgb.fit(X_train, y_train.reshape(y_train.shape[0]))
    #     model_lgb.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
    #                   eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
    #                             (X_valid, list(y_valid.reshape(y_valid.shape[0], )))],
    #                   eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=200,
    #                   verbose=False)
    #     from sklearn.ensemble import RandomForestRegressor


    #     class ANN(nn.Module):
    #         '''
    #         Construct artificial neural network
    #         '''

    #         def __init__(self, in_neuron, hidden_neuron, out_neuron):
    #             super(ANN, self).__init__()
    #             self.input_layer = nn.Linear(in_neuron, hidden_neuron)
    #             self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
    #             self.output_layer = nn.Linear(hidden_neuron, out_neuron)

    #         def forward(self, x):
    #             x = self.input_layer(x)
    #             x = F.leaky_relu(x)
    #             x = self.hidden_layer(x)
    #             x = F.leaky_relu(x)
    #             x = self.hidden_layer(x)
    #             x = F.leaky_relu(x)
    #             x = self.hidden_layer(x)
    #             x = F.leaky_relu(x)
    #             x = self.output_layer(x)
    #             return x


    #     y_pred_xgb = model_xgb.predict(X_test).reshape(y_test.shape[0], 1)
    #     y_pred_lgb = model_lgb.predict(X_test).reshape(y_test.shape[0], 1)
    #     print('single predict xgb:',model_xgb.predict(dataset_actual).reshape(-1,1))
    #     print('single predict lgb:',model_lgb.predict(dataset_actual).reshape(-1, 1))
    #     delete_index = np.where(y_test > 70)[0]
    #     y_test = np.delete(y_test, delete_index)
    #     y_pred_xgb = np.delete(y_pred_xgb, delete_index)
    #     y_pred_lgb = np.delete(y_pred_lgb, delete_index)



    #     plt.figure(3,figsize=(3,3),dpi=300)
    #     plt.style.use('ggplot')
    #     plt.scatter(y_test, y_pred_xgb, c='#8983BF',s=15,alpha=0.4)
    #     plt.plot(np.arange(0, 60), np.arange(0, 60), linewidth=1.5, linestyle='--', color='black')
    #     plt.yticks(fontproperties='Arial', size=8)
    #     plt.xticks(np.arange(0,66,10),fontproperties='Arial', size=8)
    #     # plt.xlabel('Observed data', fontproperties='Arial', size=8)
    #     # plt.ylabel('Predicted data', fontproperties='Arial', size=8)
    #     # plt.savefig(f'fig_save/compare_xgb.tiff', bbox_inches='tight', dpi=300)
    #     # plt.savefig(f'fig_save/compare_xgb.pdf', bbox_inches='tight', dpi=300)
    #     # plt.savefig(f'fig_save/compare_xgb.jpg', bbox_inches='tight', dpi=300)
    #     plt.show()

    #     print('xgb relative error:',np.sqrt(np.sum((y_test - y_pred_xgb) ** 2) / np.sum(y_test ** 2)))
    #     print('xgb RMSE:',np.sqrt(np.mean((y_test - y_pred_xgb) ** 2)))
    #     R_square = 1 - (((y_test - y_pred_xgb) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    #     print('xgb R2:',R_square)

    #     plt.figure(4,figsize=(3,3),dpi=300)
    #     plt.style.use('ggplot')
    #     plt.scatter(y_test, y_pred_lgb, c='#8983BF',s=15,alpha=0.4)
    #     plt.plot(np.arange(0, 60), np.arange(0, 60), linewidth=1.5, linestyle='--', color='black')
    #     plt.yticks(fontproperties='Arial', size=8)
    #     plt.xticks(np.arange(0,66,10),fontproperties='Arial', size=8)
    #     # plt.xlabel('Observed data', fontproperties='Arial', size=8)
    #     # plt.ylabel('Predicted data', fontproperties='Arial', size=8)
    #     # plt.savefig(f'fig_save/compare_lgb.tiff', bbox_inches='tight', dpi=300)
    #     # plt.savefig(f'fig_save/compare_lgb.pdf', bbox_inches='tight', dpi=300)
    #     # plt.savefig(f'fig_save/compare_lgb.jpg', bbox_inches='tight', dpi=300)
    #     plt.show()
    #     print('lgb relative error:',np.sqrt(np.sum((y_test - y_pred_lgb) ** 2) / np.sum(y_test ** 2)))
    #     print('lgb RMSE:', np.sqrt(np.mean((y_test - y_pred_lgb) ** 2)))
    #     R_square = 1 - (((y_test - y_pred_lgb) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    #     print('xgb R2:',R_square)

    #     X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(device), requires_grad=True)
    #     y_train = Variable(torch.from_numpy(y_train.astype(np.float32)).to(device))
    #     X_valid = Variable(torch.from_numpy(X_valid.astype(np.float32)).to(device), requires_grad=True)
    #     y_valid = Variable(torch.from_numpy(y_valid.astype(np.float32)).to(device))
    #     X_test = Variable(torch.from_numpy(X_test.astype(np.float32)).to(device), requires_grad=True)
    #     y_test = Variable(torch.from_numpy(y_test.astype(np.float32)).to(device), requires_grad=True)
    #     dataset_actual = Variable(torch.from_numpy( dataset_actual .astype(np.float32)).to(device), requires_grad=True)


    #     Net = ANN(X_train.shape[1], 50, 1).to(device)
    #     optimizer = torch.optim.Adam(Net.parameters())
    #     MSELoss = torch.nn.MSELoss()
    #     for epoch in range(10000):
    #         optimizer.zero_grad()
    #         prediction = Net(X_train)
    #         prediction_validate = Net(X_valid)
    #         loss = MSELoss(y_train, prediction)
    #         loss_validate = MSELoss(y_valid, prediction_validate)
    #         loss.backward()
    #         optimizer.step()
    #         if (epoch + 1) % 100 == 0:
    #             print(epoch + 1, loss.item(), loss_validate.item())

    #     print('single predict NN:',Net(dataset_actual).cpu().data.numpy())
    #     y_pred = Net(X_test).cpu().data.numpy()
    #     y_test = y_test.cpu().data.numpy()
    #     y_pred = np.delete(y_pred, delete_index)
    #     plt.figure(5,figsize=(3,3),dpi=300)
    #     plt.style.use('ggplot')
    #     plt.scatter(y_test, y_pred, c='#8983BF',s=15,alpha=0.4)
    #     plt.plot(np.arange(0, 60), np.arange(0, 60), linewidth=1.5, linestyle='--', color='black')
    #     plt.yticks(fontproperties='Arial', size=8)
    #     plt.xticks(np.arange(0,66,10),fontproperties='Arial', size=8)
    #     # plt.xlabel('Observed data', fontproperties='Arial', size=8)
    #     # plt.ylabel('Predicted data', fontproperties='Arial', size=8)
    #     # plt.savefig(f'fig_save/compare_NN.tiff', bbox_inches='tight', dpi=300)
    #     # plt.savefig(f'fig_save/compare_NN.pdf', bbox_inches='tight', dpi=300)
    #     # plt.savefig(f'fig_save/compare_NN.jpg', bbox_inches='tight', dpi=300)
    #     plt.show()
    #     print('NN relative error:',np.sqrt(np.sum((y_test - y_pred) ** 2) / np.sum(y_test ** 2)))
    #     print('NN RMSE:', np.sqrt(np.mean((y_test - y_pred) ** 2)))
    #     R_square = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    #     print('NN R_square:',R_square)
    #     plt.show()

# if MODEL=='Test_excel':
#     '''
#     Input an excel and get prediction for all compounds in the excel
#     setting：
#     Use_geometry_enhanced=True
#     Use_column_info=True 
#     '''

#     out_sample=pd.read_excel('dataset/test_compound_enantiomer.xlsx')
#     smiles=out_sample['smiles'].values
#     eluent=out_sample['prop'].values
#     #label=out_sample['label'].values
#     true_value=out_sample['RT'].values
#     speed=out_sample['speed'].values
#     column=out_sample['column'].values
#     column_descriptor = np.load('sourcedata/column_descriptor.npy', allow_pickle=True)

#     dataset=[]
#     all_descriptor=[]
#     graph_atom_bond=[]
#     graph_bond_angle=[]
#     y_pred = []

#     model.load_state_dict(
#         torch.load(f'saves/model_All_column_enantiomer/model_save_1500.pth'))
#     model.eval()

#     for i in tqdm(range(len(smiles))):
#         smile=smiles[i]
#         mol = obtain_3D_mol(smile, 'trail')
#         mol = Chem.MolFromMolFile(f"trail.mol")
#         all_descriptor.append(mord(mol))
#         dataset.append(mol_to_geognn_graph_data_MMFF3d(mol))
#     SE=0
#     for i in range(0, len(dataset)):
#         data = dataset[i]
#         col_specify = column_specify[column[i]]
#         col_des = np.array(column_descriptor[col_specify[3]])
#         atom_feature = []
#         bond_feature = []
#         for name in atom_id_names:
#             atom_feature.append(data[name])
#         for name in bond_id_names[0:3]:
#             bond_feature.append(data[name])
#         atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
#         bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
#         bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
#         bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
#         y = torch.Tensor([float(speed[i])*float(true_value[i])])
#         edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
#         bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)

#         prop = torch.ones([bond_feature.shape[0]]) * eluent[i]
#         coated = torch.ones([bond_feature.shape[0]]) * col_specify[0]
#         diameter = torch.ones([bond_feature.shape[0]]) * col_specify[1]
#         immobilized = torch.ones([bond_feature.shape[0]]) * col_specify[2]

#         TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][820] / 100
#         RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][821]
#         RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][822]
#         MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][1568]
#         MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][457]

#         col_TPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[820] / 100
#         col_RASA = torch.ones([bond_angle_feature.shape[0]]) * col_des[821]
#         col_RPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[822]
#         col_MDEC = torch.ones([bond_angle_feature.shape[0]]) * col_des[1568]
#         col_MATS = torch.ones([bond_angle_feature.shape[0]]) * col_des[457]

#         bond_feature = torch.cat([bond_feature, coated.reshape(-1, 1)], dim=1)
#         bond_feature = torch.cat([bond_feature, immobilized.reshape(-1, 1)], dim=1)
#         bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
#         bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)
#         bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)

#         bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, col_TPSA.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, col_RASA.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, col_RPSA.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, col_MDEC.reshape(-1, 1)], dim=1)
#         bond_angle_feature = torch.cat([bond_angle_feature, col_MATS.reshape(-1, 1)], dim=1)

#         data_atom_bond = Data(atom_feature, edge_index, bond_feature, y)
#         data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)

#         pred, h_graph = model(data_atom_bond.to(device), data_bond_angle.to(device))
#         y_pred.append(pred.detach().cpu().data.numpy() / speed[i])
#         SE+=((true_value[i]-pred[:,1].detach().cpu().data.numpy() / speed[i])*speed[i])**2


#     MSE=SE/(len(dataset))
#     print(MSE)
#     print(len(y_pred))

#     np.save('result_save/pred_out_of_sample.npy', y_pred, allow_pickle=True)

# if MODEL == 'Test_enantiomer':
#     '''
#     Given two compounds and predict the RT in different condition
#     '''
#     transfer_target='All_column'
#     predict_column_all=['ADH']
#     smiles = ['N[C@@H](C)C(O)=O', 'N[C@H](C)C(O)=O']
#     speed = [1.0, 1.0]
#     eluent_all=[[0.005,0.005]]
#     true=[18.584,20.022]
#     draw_picture = False
#     plot_box=False
#     col_iter=0
#     column_descriptor = np.load('sourcedata/column_descriptor.npy', allow_pickle=True)
#     plt.figure(10, figsize=(2, 2), dpi=300)
#     for predict_column in predict_column_all:
#         col_specify=column_specify[predict_column]
#         col_des = np.array(column_descriptor[col_specify[3]])
#         y_pred=[]
#         for eluent in eluent_all:
#             mols=[]
#             all_descriptor=[]
#             dataset=[]
#             for smile in smiles:
#                 mol = Chem.MolFromSmiles(smile)
#                 mols.append(mol)
#             from rdkit.Chem import Draw
#             if draw_picture==True:
#                 index=0
#                 for mol in mols:
#                     smiles_pic = Draw.MolToImage(mol, size=(200, 100), kekulize=True)
#                     plt.imshow(smiles_pic)
#                     plt.axis('off')
#                     plt.savefig(f'fig_save/molecular_{index}.tiff')
#                     index+=1
#                     plt.show()
#                     plt.clf()

#             for smile in smiles:
#                 mol = obtain_3D_mol(smile, 'trail')
#                 mol = Chem.MolFromMolFile(f"trail.mol")
#                 all_descriptor.append(mord(mol))
#                 dataset.append(mol_to_geognn_graph_data_MMFF3d(mol))

#             for i in range(0, len(dataset)):
#                 data = dataset[i]
#                 atom_feature = []
#                 bond_feature = []
#                 for name in atom_id_names:
#                     atom_feature.append(data[name])
#                 for name in bond_id_names[0:3]:
#                     bond_feature.append(data[name])
#                 atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
#                 bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
#                 bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
#                 bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
#                 y = torch.Tensor([float(speed[i])])
#                 edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
#                 bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)

#                 prop = torch.ones([bond_feature.shape[0]]) * eluent[i]
#                 coated = torch.ones([bond_feature.shape[0]]) * col_specify[0]
#                 diameter = torch.ones([bond_feature.shape[0]]) * col_specify[1]
#                 immobilized = torch.ones([bond_feature.shape[0]]) * col_specify[2]

#                 TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][820] / 100
#                 RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][821]
#                 RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][822]
#                 MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][1568]
#                 MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i][457]

#                 col_TPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[820] / 100
#                 col_RASA = torch.ones([bond_angle_feature.shape[0]]) * col_des[821]
#                 col_RPSA = torch.ones([bond_angle_feature.shape[0]]) * col_des[822]
#                 col_MDEC = torch.ones([bond_angle_feature.shape[0]]) * col_des[1568]
#                 col_MATS = torch.ones([bond_angle_feature.shape[0]]) * col_des[457]

#                 bond_feature = torch.cat([bond_feature, coated.reshape(-1, 1)], dim=1)
#                 bond_feature = torch.cat([bond_feature, immobilized.reshape(-1, 1)], dim=1)
#                 bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
#                 bond_feature = torch.cat([bond_feature, prop.reshape(-1, 1)], dim=1)
#                 bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)

#                 bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, col_TPSA.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, col_RASA.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, col_RPSA.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, col_MDEC.reshape(-1, 1)], dim=1)
#                 bond_angle_feature = torch.cat([bond_angle_feature, col_MATS.reshape(-1, 1)], dim=1)


#                 print(bond_angle_feature[0])
#                 data_atom_bond = Data(atom_feature, edge_index, bond_feature, y)
#                 data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)
#                 model.load_state_dict(
#                     torch.load(f'saves/model_All_column/model_save_1500.pth'))
#                 model.eval()

#                 pred, h_graph = model(data_atom_bond.to(device),data_bond_angle.to(device))
#                 y_pred.append(pred.detach().cpu().data.numpy()/speed[i])
#             #print(pred.detach().cpu().data.numpy()/speed[i])
#             print(y_pred)
#             print(cal_prob(y_pred))


#             if plot_box==True:
#                 plt.style.use('ggplot')
#                 plt.boxplot(np.squeeze(np.array(y_pred)).T,medianprops={'color': '#838AB4'})
#                 plt.scatter([1,2],true,marker='x')
#                 plt.yticks(fontproperties='Arial', size=8)
#                 plt.xticks([1,2],['Enantiomer 1','Enantiomer 2'],fontproperties='Arial', size=8)
#                 #plt.ylabel('Predicted RT', fontproperties='Arial', size=8)
#                 plt.title(predict_column,fontproperties='Arial', size=8)
#                 col_iter+=1

#                 plt.savefig(f'fig_save/box_{transfer_target}_{predict_column}_1.tiff', bbox_inches='tight', dpi=300)
#                 plt.savefig(f'fig_save/box_{transfer_target}_{predict_column}_1.pdf', bbox_inches='tight', dpi=300)
#                 plt.savefig(f'fig_save/box_{transfer_target}_{predict_column}_1.jpg', bbox_inches='tight', dpi=300)
#                 plt.show()