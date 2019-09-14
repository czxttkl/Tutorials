import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Linear(5, 6)
        self.conv2 = nn.Linear(6, 2)

    def forward(self, input: torch.Tensor, mode: str):
        if mode == "a":
            input = F.relu(self.conv1(input))
            input = F.relu(self.conv2(input))
            return input
        else:
            return torch.ones(2)


scripted_module = torch.jit.script(MyModule())
print("scripted_module graph:\n", scripted_module.graph)

test_input = torch.randn(3, 5)
print(scripted_module(test_input, "a"))
print(scripted_module(test_input, "b"))

m = MyModule()
m.eval()
print(m(test_input, "a").requires_grad)
with torch.no_grad():
    print(m(test_input, "a").requires_grad)



