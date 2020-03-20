from fastai.vision import *
import model_utils_ext as m_util_x
import pretrainedmodels


def se_resnext50(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=pretrained)
    return model


class SAI(nn.Module):
    ''' SelfAttention with Identity '''
    
    def __init__(self, nf):
        super(SAI, self).__init__()
        
        self.sa = PooledSelfAttention2d(nf)
        self.bn = nn.BatchNorm2d(nf)
        self.do = nn.Dropout(0.4),
        
        
    def forward(self, x):
        ident = x
        out = self.sa(x)
        out = ident + out
        
        out = self.do(self.bn(out)) #
        
        return nn.ReLU(out)

    
class HeadBlock(nn.Module):
    
    def __init__(self, nf, nc):
        super(HeadBlock, self).__init__()
        
        self.se1 = m_util_x.SEBlock(nf//2)
        self.head1 = create_head(nf, nc[0])
        self.se2 = m_util_x.SEBlock(nf//2)
        self.head2 = create_head(nf, nc[1])
        self.se3 = m_util_x.SEBlock(nf//2)
        self.head3 = create_head(nf, nc[2])
        
        
    def forward(self, x):
        x1 = self.se1(x)
        x1 = self.head1(x1)
        
        x2 = self.se2(x)
        x2 = self.head2(x2)
        
        x3 = self.se3(x)
        x3 = self.head3(x3)
        
        return x1, x2, x3
    
class HeadBlock0(nn.Module):
    
    def __init__(self, nf, nc):
        super(HeadBlock0, self).__init__()
        
        self.head1 = create_head(nf, 1024, bn_final = True)
        self.at1 = SelfAttention(nf//4)
        self.rl1 = nn.LeakyReLU(0.1, inplace=True)
        self.lin1 = nn.Linear(1024, nc[0])
        
        self.head2 = create_head(nf, 1024, bn_final = True)
        self.at2 = SelfAttention(nf//4)
        self.rl2 = nn.LeakyReLU(0.1, inplace=True)
        self.lin2 = nn.Linear(1024, nc[1])
        
        self.head3 = create_head(nf, 1024, bn_final = True)
        self.at3 = SelfAttention(nf//4)
        self.rl3 = nn.LeakyReLU(0.1, inplace=True)
        self.lin3 = nn.Linear(1024, nc[2])
        
        
    def forward(self, x):
        
        x1 = self.head1(x)
        x1 = self.at1(x1)
        x1 = self.rl1(x1)
        x1 = self.lin1(x1) 
        
        x2 = self.head2(x)
        x2 = self.at2(x2)
        x2 = self.rl2(x2)
        x2 = self.lin2(x2) 
        
        x3 = self.head3(x)
        x3 = self.at3(x3)
        x3 = self.rl3(x3)
        x3 = self.lin3(x3) 
        
        
        return x1, x2, x3
    
    
    
class HeadBlockEff(nn.Module):
    
    def __init__(self, nf, nc):
        super(HeadBlockEff, self).__init__()
        
        #self.head1 = create_head(nf, 1024, bn_final = True)
        self.at1 = SelfAttention(nf)
        self.rl1 = nn.LeakyReLU(0.1, inplace=True)
        self.lin1 = nn.Linear(nf, nc[0])
        #self.sw1 = MemoryEfficientSwish()
        
        #self.head2 = create_head(nf, 1024, bn_final = True)
        self.at2 = SelfAttention(nf)
        self.rl2 = nn.LeakyReLU(0.1, inplace=True)
        self.lin2 = nn.Linear(nf, nc[1])
        #self.sw2 = MemoryEfficientSwish()
        
        #self.head3 = create_head(nf, 1024, bn_final = True)
        self.at3 = SelfAttention(nf)
        self.rl3 = nn.LeakyReLU(0.1, inplace=True)
        self.lin3 = nn.Linear(nf, nc[2])
        #self.sw3 = MemoryEfficientSwish()
        
        
    def forward(self, x):
        
        #x1 = self.head1(x)
        x1 = self.at1(x)
        x1 = self.rl1(x1)
        x1 = self.lin1(x1)
        #x1 = self.sw1(x1) 
        
        #x2 = self.head2(x)
        x2 = self.at2(x)
        x2 = self.rl2(x2)
        x2 = self.lin2(x2) 
        #x2 = self.sw2(x2) 
        
        #x3 = self.head3(x)
        x3 = self.at3(x)
        x3 = self.rl3(x3)
        x3 = self.lin3(x3) 
        #x3 = self.sw3(x3) 
        
        
        return x1, x2, x3
    
    