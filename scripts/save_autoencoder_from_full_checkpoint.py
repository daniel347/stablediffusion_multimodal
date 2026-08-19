import torch

ckpt_path = "../checkpoints/sd-v1-4.ckpt"
ckpt = torch.load(ckpt_path)["state_dict"]

def extract_autoencoder(state_dict):
    """
    This function removes the "module." prefix from the state_dict keys.
    This is commonly needed after wrapping a model with DataParallel or DistributedDataParallel.
    """
    new_state_dict = {}

    start_key = "first_stage_model."

    for key, value in state_dict.items():
        if key.startswith(start_key):
            print(f"saving: {key}")
            new_key = key[len(start_key):]  # Remove the "module." prefix (7 characters)
            new_state_dict[new_key] = value

    return new_state_dict

autoencoder_state_dict = extract_autoencoder(ckpt)

autoencoder_path = "../checkpoints/sd-v1-4-autoencoder.ckpt"
torch.save(autoencoder_state_dict, autoencoder_path)