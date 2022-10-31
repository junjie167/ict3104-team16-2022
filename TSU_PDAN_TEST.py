import argparse
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-JSON', type=str, default='')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-batch_size', type=str, default='1')
parser.add_argument('-num_classes', type=str, default='51')
args = parser.parse_args()

sys.argv = sys.argv[0:1] + ["-batch_size", args.batch_size]
from TSU_PDAN import HOI_PDAN
from HOI.smarthome_i3d_per_video import TSU as Dataset
from HOI.smarthome_i3d_per_video import TSU_collate_fn as collate_fn

batch_size = int(args.batch_size)
classes = int(args.num_classes)

if len(args.JSON) > 0:
    dataset = Dataset(args.JSON, 'training', args.rgb_root, batch_size, classes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                pin_memory=True, collate_fn=collate_fn)
    dataloader.root = args.rgb_root

    val_dataset = Dataset(args.JSON, 'testing', args.rgb_root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4,
                                                    pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = args.rgb_root

    modelrunner = HOI_PDAN()
    modelrunner.PDAN_training_parameters()

else:
    raise argparse.ArgumentError("Missing json file. Specify with -JSON")

