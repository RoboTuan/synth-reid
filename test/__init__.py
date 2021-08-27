import zipfile
import gdown
import os
import shutil

root_test = "./"

url = 'https://drive.google.com/uc?id=1bItIfpmm766Bj3QtdKQ2EKNWVw5nuFzE'
output = 'GTA_synthReid.zip'
dataset = 'GTA_synthReid/'
print("Downloading the dataset")
gdown.download(url, output, quiet=False)

print("Unzipping the dataset")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("./")

print("Removing the zipped dataset")
if os.path.exists(output):
    os.remove(output)
print("Removing the non necessary folders")
if os.path.exists("./__MACOSX"):
    shutil.rmtree("./__MACOSX")

