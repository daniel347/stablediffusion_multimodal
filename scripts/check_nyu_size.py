from torchvision.transforms import v2
from ldm.data.dataset_NYU_Depth import NYUDepth

input_size = 256
datadir = "../mmvaeplus/data/nyu_data"

modalities = ['colour']
kwargs = {'num_workers': 4, 'pin_memory': True}
tx = v2.Compose([v2.Resize([256, 256])])

train_dataset = NYUDepth(datadir, transform=tx, train=True, modality=modalities, p_hor_flip=0.5,
                         p_colour_jitter=0.1)
test_dataset = NYUDepth(datadir, transform=tx, train=False, modality=modalities)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")