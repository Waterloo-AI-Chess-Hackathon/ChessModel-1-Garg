from model import ChessModel
import torch

device = torch.device("mps")
model = ChessModel()

model.to(device)
model.eval()

policy, value = model.forward(torch.randn(1, 119, 8, 8).to(device))



# print(policy.shape)
# print(value.shape)
