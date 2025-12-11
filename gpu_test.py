import torch

print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    x = torch.randn(10000, 10000, device="cuda")
    # 让 GPU 做一个真正的计算
    y = x @ x
    torch.cuda.synchronize()
    print("GPU computation success. Result sum:", y.sum().item())
else:
    print("GPU not available!")
