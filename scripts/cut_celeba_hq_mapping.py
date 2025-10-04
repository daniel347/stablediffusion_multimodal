import os
import pandas as pd

root_data_dir = "../../datasets/CelebAMask-HQ/"

index_mapping = pd.read_csv(os.path.join(root_data_dir, "CelebA-HQ-to-CelebA-mapping.txt"), delim_whitespace=True)
file_present = index_mapping.idx.apply(lambda x: str(x)[:3] == "100")

index_mapping = index_mapping[file_present]
index_mapping = index_mapping.drop("idx", axis=1)

index_mapping.to_csv(os.path.join(root_data_dir, "new_mapping.txt"), sep=" ", index_label="idx")

