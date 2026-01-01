import torch

from ldm.modules.midas.api import load_midas_transform


class AddMiDaS(object):
    def __init__(self, model_type):
        super().__init__()
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample['jpg'])
        x = self.transform({"image": x})["image"]
        sample['midas_in'] = x
        return sample



def depth_display(self, img, opencv=False):
    print("Depth img min: ", img.min().item(), "max:", img.max().item())
    if opencv:
        return (torch.clamp(img, 0, 1) * 65535).to(torch.uint16)[0].numpy()
    else:
        return img.repeat(3, 1, 1)

def colour_display(self, img, opencv=False):
    print("Colour img min: ", img.min().item(), "max:", img.max().item())
    if opencv:
        return (torch.clamp(img, 0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()[:, :, ::-1]
    else:
        return img

def semseg_display(self, mask, opencv=False):
    if len(mask.shape) == 2:
        # We have a mask of class indices
        mask = self.index_semseg_to_one_hot(mask, num_classes=len(self.semseg_labels))

    mask_rgb = torch.zeros(*(3, *mask.shape[1:]), dtype=torch.float32)
    if self.bisenet_compat:
        # We force the mask to be single class, based on class priority
        mask_inds = self.mask_to_single_labels(mask)

        semseg_display_colours_t = torch.stack([col for col in self.semseg_display_colours.values()] + [torch.zeros(3)])
        mask_rgb = semseg_display_colours_t[mask_inds.to(int)]
        mask_rgb =  mask_rgb.permute((2, 0, 1))
    else:
        # Norm so that multiple classes mix colours - We clamp to allow background to be 0 everywhere.
        mask_norm = mask/torch.clamp(mask.sum(dim=0, keepdim=True), 1.0)
        mask_norm = torch.nan_to_num(mask_norm)

        for i in range(mask.shape[0]):
            mask_rgb += self.semseg_display_colours[i][:, None, None] * mask_norm[i, :, :].unsqueeze(0).expand(3, -1, -1).float()

    i