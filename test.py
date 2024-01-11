import torch

if torch.cuda.is_available():
    print("gpu")
else:
    print("cpu")