import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


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

m = MyModule()
scripted_module = torch.jit.script(m)
# print("scripted_module graph:\n", scripted_module.graph)

test_input = torch.randn(3, 5)

# error: Type 'Tuple[Tensor, str]' cannot be traced.
# Only Tensors and (possibly nested) Lists, Dicts, and Tuples of Tensors can be traced
# traced_module = torch.jit.trace(m, (test_input, "a"))

print("original with input a:", m(test_input, "a"))
print("scripted_module with input a:", scripted_module(test_input, "a"))
# print("traced_module with input a:", traced_module(test_input, "a"))
print("scripted_module with input b:",scripted_module(test_input, "b"))



# test that only using torch.no_grad() can return output with requires_grad=False
m.eval()
print(m(test_input, "a").requires_grad)
with torch.no_grad():
    print(m(test_input, "a").requires_grad)


class Batch(NamedTuple):
    input: torch.Tensor


class MyModuleUp(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, batch: Batch):
        return self.base_model(batch.input, "a")

class MyModuleUpUp(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input: torch.Tensor):
        return self.base_model(Batch(input=input))


m_up = MyModuleUp(m)
m_up_up = MyModuleUpUp(m_up)
traced_module_up_up = torch.jit.trace(m_up_up, test_input)
scripted_module_up_up = torch.jit.script(m_up_up)
print("original module of ModuleUpUp", m_up_up(test_input))
print("traced module of ModuleUpUp", traced_module_up_up(test_input))
print("scripted module of ModuleUpUp", scripted_module_up_up(test_input))





