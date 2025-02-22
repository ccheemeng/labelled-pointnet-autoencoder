import chamferdist
import faiss
import torch
from tqdm import tqdm

from datasets import LabelledPointCloudDataset
from models import LabelledPointNet, Decoder, LabelledPointNetAE

import argparse
import json
import logging
import os
from pathlib import Path

def main(args):
    output_dir = os.path.join("runs", args.name)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(output_dir, f"{args.name}.hyperparameters"), 'w') as fp:
        json.dump(vars(args), fp)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(output_dir, f"{args.name}.log"),
                        level=logging.NOTSET)

    dataset = LabelledPointCloudDataset(args.dir, args.num_classes,
                                        args.radius, args.max_points)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    n = dataset.getN()
    model = LabelledPointNetAE(n, args.num_classes).to(args.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    logging.info("Training...")
    model.train()
    i = 1
    for epoch in range(args.num_epochs):
        total_loss = 0
        logging.info(f"========Epoch {i}========")
        for batch in tqdm(loader):
            optimiser.zero_grad()
            x = batch.to(args.device)
            out = model(x)
            loss = combined_loss(x, out)
            logging.debug(f"Loss: {loss}")
            total_loss += loss.item()
            loss.backward()
            optimiser.step()
        logging.info(f"Epoch {i} complete.\tAverage loss: {total_loss / len(loader)}")
        if i % args.save_freq == 0:        
            torch.save(model.state_dict(), os.path.join(output_dir, f"{i}-model-state-dict.pt"))
            torch.save(optimiser.state_dict(), os.path.join(output_dir, f"{i}-optimiser-state-dict.pt"))
        i += 1
        curr_loss = total_loss
    logging.info("Training complete.")

    torch.save(model.state_dict(), os.path.join(output_dir, "model-state-dict.pt"))
    torch.save(optimiser.state_dict(), os.path.join(output_dir, "optimiser-state-dict.pt"))

def combined_loss(x, out, cd_weight=0.5, nll_weight=0.5):
    pointsx = x[:, :3, :]
    pointsout = out[:, :3, :]
    labelsx = x[:, 3:, :]
    labelsout = out[:, 3:, :]

    nll = nll_loss(pointsx, pointsout, labelsx, labelsout)
    logging.debug(f"NLLLoss: {nll.detach().cpu().item()}")

    chamferDist = chamferdist.ChamferDistance(batch_reduction="sum")
    cd = chamferDist(pointsx, pointsout)
    logging.debug(f"Chamfer distance: {cd.detach().cpu().item()}")

    return cd_weight * cd + nll_weight * nll

def nll_loss(pointsx, pointsout, labelsx, labelsout):
    device = pointsx.device
    batch_size = pointsx.size()[0]
    loss = torch.tensor(0.0, device=device)
    nllLoss = torch.nn.NLLLoss(reduction="sum")

    for i in range(batch_size):
        pointsout_i = pointsout[i].T.detach().cpu().numpy()
        pointsx_i = pointsx[i].T.detach().cpu().numpy()
        labelsx_i = labelsx[i].T
        labelsout_i = labelsout[i].T

        index = faiss.IndexFlatL2(3)
        index.add(pointsx_i.astype("float32"))
        
        _, nearest_idx = index.search(pointsout_i.astype("float32"), 1)
        nearest_idx = torch.tensor(nearest_idx.squeeze(), device=device)
        nearest_labels = torch.gather(labelsx_i, dim=0, index=nearest_idx.unsqueeze(1).expand(-1, labelsx.size(1)))
        labels = torch.argmax(nearest_labels, dim=1)

        loss_i = nllLoss(labelsout_i, labels)
        loss += loss_i
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str
    )
    parser.add_argument(
        "-c", "--num-classes",
        type=int
    )
    parser.add_argument(
        "-r", "--radius",
        type=float,
        default=100.0,
        required=False
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "--cd-weight",
        type=float,
        default=0.5,
        required=False
    )
    parser.add_argument(
        "--nll-weight",
        type=float,
        default=0.5,
        required=False
    )
    parser.add_argument(
        "-d", "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        required=False
    )
    parser.add_argument(
        "-w", "--num-workers",
        default=1,
        type=int,
        required=False
    )
    parser.add_argument(
        "-l", "--lr",
        default=1E-4,
        type=float,
        required=False
    )
    parser.add_argument(
        "-b", "--batch-size",
        default=50,
        type=int,
        required=False
    )
    parser.add_argument(
        "-e", "--num_epochs",
        default=500,
        type=int,
        required=False
    )
    parser.add_argument(
        "--name",
        default="train",
        type=str,
        required=False
    )
    args = parser.parse_args()
    main(args)