
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data import collate_pool, CIFData
from model import CrystalGraphConvNet


def load_model(model_path, device):
    """Load the pretrained model with an EA/IP dual-output architecture."""
    checkpoint = torch.load(model_path, map_location=device)
    args_saved = checkpoint.get('args', {})
    state_dict = checkpoint['state_dict']

    # Retrieve model hyperparameters
    atom_fea_len = args_saved.get('atom_fea_len', 64)
    n_conv = args_saved.get('n_conv', 3)
    h_fea_len = args_saved.get('h_fea_len', 128)
    n_h = args_saved.get('n_h', 1)

    # Infer input feature dimensions from the checkpoint
    orig_atom_fea_len = state_dict['embedding.weight'].shape[1]
    fc_full_in = state_dict['convs.0.fc_full.weight'].shape[1]
    nbr_fea_len = fc_full_in - 2 * atom_fea_len

    # Initialize the CGCNN model
    model = CrystalGraphConvNet(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=atom_fea_len, n_conv=n_conv,
        h_fea_len=h_fea_len, n_h=n_h, classification=False
    )

    # Configure the EA/IP dual-output layer and load pretrained weights
    num_ftrs = model.fc_out.in_features
    model.fc_out = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Retrieve target normalization parameters
    norm_dict = checkpoint.get('normalizer', None)
    return model, norm_dict, device


def predict(model, norm_dict, loader, device):
    all_preds, all_ids = [], []
    mean = torch.tensor(norm_dict['mean']).to(device)
    std = torch.tensor(norm_dict['std']).to(device)

    with torch.no_grad():
        for input_data, _, batch_cif_ids in loader:
            input_var = (
                input_data[0].to(device),
                input_data[1].to(device),
                input_data[2].to(device),
                [i.to(device) for i in input_data[3]]
            )

            # Predict EA/IP values and reverse target normalization
            pred_norm = model(*input_var)
            pred = pred_norm * std + mean
            pred = pred.cpu().numpy()

            for i, cif_id in enumerate(batch_cif_ids):
                all_ids.append(cif_id)
                all_preds.append(pred[i])

    return all_ids, np.array(all_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('data_path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_dict, device = load_model(args.model_path, device)

    # Load the dataset for batch prediction
    dataset = CIFData(args.data_path)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_pool, num_workers=0)
    ids, preds = predict(model, norm_dict, loader, device)

    # Save EA, IP, and predicted energy gap
    res_df = pd.DataFrame({
        'material_id': ids,
        'pred_EA': preds[:, 0],
        'pred_IP': preds[:, 1],
        'pred_gap': preds[:, 0] - preds[:, 1]
    })
    res_df.to_csv(os.path.join(args.data_path, 'batch_predictions.csv'), index=False)
