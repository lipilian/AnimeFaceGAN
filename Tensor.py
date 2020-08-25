# %% import package for tensor practicing
import torch as t 




# %% test with Tensor 
a = t.Tensor(3,3) # create tensor with size
b = t.Tensor([[1,2,3],[1,2,3]]) # create tensor with list
b.tolist()
b.size()
b.numel()
b.shape
t.arange(1,8,2)
# %%
a = t.arange(0,6)
a.view(2,3)
b = a.view(-1,3)
a.view(2,-1)
b.unsqueeze(1)
c = b.view(1,1,1,2,3)
c.squeeze(0)
# %%
a[0] = 100
b
# %%
t.FloatTensor(2,3)

# %% select index based tensor dimension




