import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, ci, co, k=3, s=1, act=True):
        super().__init__(); self.conv=nn.Conv2d(ci,co,k,s,k//2,bias=False); self.bn=nn.BatchNorm2d(co); self.act=nn.SiLU() if act else nn.Identity()
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class DSBlock(nn.Module):
    def __init__(self, ci, co, s=1):
        super().__init__(); self.dw=ConvBNAct(ci,ci,3,s); self.pw=ConvBNAct(ci,co,1,1); self.skip=ConvBNAct(ci,co,1,s,False) if s!=1 or ci!=co else nn.Identity()
    def forward(self,x): return F.silu(self.pw(self.dw(x))+self.skip(x))

class GlobalPixelAttention(nn.Module):
    def __init__(self,c=96,d=24):
        super().__init__(); self.q=nn.Conv1d(c,d,1,bias=False); self.k=nn.Conv1d(c,d,1,bias=False); self.v=nn.Conv1d(c,d,1,bias=False); self.proj=nn.Conv1d(d,c,1,bias=False); self.scale=d**-0.5
    def forward(self,x):
        b,c,h,w=x.shape; z=x.flatten(2); q=self.q(z).transpose(1,2); k=self.k(z); v=self.v(z).transpose(1,2); a=torch.softmax(torch.bmm(q,k)*self.scale,dim=-1); y=self.proj(torch.bmm(a,v).transpose(1,2)); return x+y.reshape(b,c,h,w)

class TinyGlobalDet(nn.Module):
    def __init__(self,nc=7,slots=2):
        super().__init__(); self.nc=nc; self.slots=slots
        self.stem=ConvBNAct(3,16,3,2); self.b1=nn.Sequential(DSBlock(16,24,2),DSBlock(24,24)); self.b2=nn.Sequential(DSBlock(24,32,2),DSBlock(32,32)); self.b3=nn.Sequential(DSBlock(32,48,2),DSBlock(48,48)); self.b4=nn.Sequential(DSBlock(48,56),DSBlock(56,56)); self.b5=nn.Sequential(DSBlock(56,56)); self.attn=GlobalPixelAttention(56,16); self.lat=ConvBNAct(56,32,1,1); self.skip=ConvBNAct(32,32,1,1); self.head=nn.Conv2d(32,slots*(5+nc),1)
    def forward(self,x):
        x=self.stem(x); x=self.b1(x); x=self.b2(x); skip=x; x=self.b3(x); x=self.b4(x); x=self.b5(x); x=self.attn(x); x=F.interpolate(self.lat(x),size=skip.shape[-2:],mode='bilinear',align_corners=False)+self.skip(skip); return self.head(F.silu(x)).reshape(x.shape[0],self.slots,5+self.nc,30,40)
