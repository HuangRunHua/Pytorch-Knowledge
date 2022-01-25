import pandas as pd

img_labels = pd.read_csv("test.csv", names=['file_name', 'label'])
print(img_labels)