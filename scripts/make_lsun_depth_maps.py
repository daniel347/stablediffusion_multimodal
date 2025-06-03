from PIL import Image
import ml_depth_pro.src.depth_pro as depth_pro
import torch
import torchvision.utils as tvu
from ldm.data.dataset_NYU_Depth import NYUDepth
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import tqdm
import time
import multiprocessing as mp

device = torch.device("cuda:1")

# Load model and preprocessing transform
model, _ = depth_pro.create_model_and_transforms(device=device)
model.eval()

transform = v2.Compose([
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            v2.ConvertImageDtype(torch.float)])

# create RGB dataset
tx = {"colour": transform}

dataset = NYUDepth("/home/ds3119/datasets/lsun",
                   transform=tx,
                   split="all",
                   modality=["colour"],
                   classes=["bedroom"],
                   modality_exts={"colour": "jpg"},
                   return_row=True)

def save_batch_outputs(outs, paths):
    for out, path in zip(outs, paths):
        cv2.imwrite(str(path), out.detach().cpu().numpy().astype(np.uint16))

max_save_delay = 25 # batches

batch_count = 0

save_queue = mp.Queue(maxsize=50)


def save_images():
    while True:
        if not save_queue.empty():
            depth_norm, out_paths = save_queue.get()
            if depth_norm is None and out_paths is None:
                break
            cv2.imwrite(str(out_paths), depth_norm)
            print(f"Saved depth map to {out_paths}")
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    print("Starting depth estimation...")
    save_proc = mp.Process(target=save_images, args=())
    save_proc.start()

    for img, row in tqdm.tqdm(dataset):
        img = img[0].to(device)
        prediction = model.infer(img, f_px=None)
        depth = prediction["depth"]  # Depth in [m]
        depth_norm = (torch.clip(depth, 0, 10) * 65535.0 / 10.0).detach().cpu().numpy().astype(np.uint16)
        out_paths = str(row.colour_path).replace("colour", "depth")
        out_paths = out_paths.replace(".jpg", ".png")

        save_queue.put((depth_norm, out_paths))

        while save_queue.full():
            print("Waiting for save queue to clear...")
            time.sleep(1)

        print(out_paths)
        cv2.imwrite(str(out_paths), depth_norm)
        batch_count += 1

        if batch_count > 100:
            print("Processed 100 batches, stopping.")
            save_queue.put((None, None))
            save_proc.join()
            break

    print("Depth estimation completed.")

