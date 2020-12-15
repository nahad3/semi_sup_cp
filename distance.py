import torch.nn as nn
import  torch
from torch.autograd import Variable
eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).

class KLDiv(nn.Module):
    # Calculate KL-Divergence
        
    def forward(self, predict, target):
       assert predict.ndimension() == 2, 'Input dimension must be 2'
       target = target.detach()

       # KL(T||I) = \sum T(logT-logI)
       predict = eps + predict
       target = eps + target


       logI = torch.log(predict)
       logT = torch.log(target)
       logdiff = logT - logI
       TlogTdI = target * (logdiff)
       kld = TlogTdI.sum(1)
     #  criter = nn.MSELoss()
     #  kld = criter(predict,target)

       return kld


'Kl divergence loss'
def kl_loss(p,q):
   kl = KLDiv()

   kl_div = torch.mean(kl(p,q),dim=0)
   kl_div.requires_grad = True
   return kl_div

