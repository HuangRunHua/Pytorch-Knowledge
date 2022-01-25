import pandas as pd
import os

img_labels = pd.read_csv("test.csv", names=['file_name', 'label'])
print(img_labels)

img_path = os.path.join("Data", img_labels.iloc[0, 0])
print(img_path)