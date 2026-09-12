import torch
from torch import nn

class EEGNet(nn.Module):
    def __init__(self,C,T,D,N,F_1,F_2):
        super().__init__()
        self.C=C
        self.T=T
        self.D=D
        self.F_1=F_1
        self.F_2=F_2
        self.N=N
        #EEG blks 
        self.b1=self.blk_1()
        self.b2=self.blk_2()
        self.net=nn.Sequential(
            self.b1,
            self.b2,
            nn.Linear(self.F_2 * ((self.T // 4 - 1) // 8),self.N)
		)
    def blk_1(self):
        return nn.Sequential(
                nn.Conv2d(
                    1,
                    self.F_1,
                    kernel_size=(1,64),
                    padding=(0,31)
                    ),#F_1@C*T
                nn.BatchNorm2d(self.F_1),
                nn.Conv2d(
                    self.F_1,
                    self.D*self.F_1,
                    kernel_size=(self.C,1),
                    groups=self.F_1
                    ),#(D*F_1)@1*T
                nn.BatchNorm2d(self.D*self.F_1),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1,4)),#(D*F_1)@1*(T//4)
                nn.Dropout(p=0.25)#or p=0.5
                )

    def blk_2(self):
        return nn.Sequential(
                nn.Conv2d(
                    self.D*self.F_1,
                    self.D*self.F_1,
                    kernel_size=(1,16),
                    padding=(0,7),
                    groups=self.D*self.F_1,
                    bias=False
                    ),
                nn.Conv2d(
                    self.D*self.F_1,
                    self.F_2,
                    kernel_size=1,
                    bias=True
                    ),#F_2@1*(T//4)
                nn.BatchNorm2d(self.F_2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1,8)),#F_2@1*(T//32)
                nn.Dropout(p=0.25),#or p=0.5
                nn.Flatten()#F_2*(T//32)
                )

    def forward(self,x):
        x=self.net(x)
        return x
