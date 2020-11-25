from visualization import Visualization
from numpy import random


x=random.rand(1000)
vis=Visualization()
vis.plt_test(x)
# plt.hist(x)
# plt.title('bla')

a=x.mean()
print(a)
b=x.std()
print(b)
