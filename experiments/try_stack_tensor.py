import time
import torch
data = [torch.rand(22) for _ in range(10240)]
t1 = time.time()
for i in range(300): d = torch.stack(data)

t2 = time.time()
for i in range(300): d = torch.cat(data).view(-1, 22)

t3 = time.time()
print("torch.stack time: {}, torch.cat time: {}".format(t2 - t1, t3 - t2))
