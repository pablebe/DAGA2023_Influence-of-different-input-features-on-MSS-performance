
import torch 
from openunmix_slicq.nsgt_mod  import NSGT
from openunmix_slicq.nsgt_mod  import LogScale, VQLogScale, VQLogScaleMod
import matplotlib.pyplot as plt

class CQT():
    def __init__(self, fmin, fmax, fbins, fs=44100, audio_len=44100, device="cpu", split_0_nyq=False):
#        fmax=fs/2
        Ls=audio_len
        scl = LogScale(fmin, fmax, fbins)
        self.cq = NSGT(scl, fs, Ls,
                 real=True,
                 matrixform=True, 
                 multichannel=False,
                 device=device
                 )
        #self.cq=self.cq.to(device)
        self.split_0_nyq=split_0_nyq

    def fwd(self,x):
        c= self.cq.forward(x)
        c=c.squeeze(0)
        c= torch.view_as_real(c)
        c=c.permute(0,3,2,1)
        if self.split_0_nyq:
            c0=c[:,:,:,0]
            cnyq=c[:,:,:,-1]
            cfreqs=c[:,:,:,1:-1]
            return cfreqs, c0, cnyq
        else:
            return c

    def bwd(self,cfreqs,c0=None, cnyq=None):

        if self.split_0_nyq:
            c=torch.cat((c0.unsqueeze(3),cfreqs,cnyq.unsqueeze(3)),dim=3)
        else:
            c=cfreqs
        
        c=c.permute(0,3,2,1)
        c=c.unsqueeze(0)
        c=c.contiguous()
        c=torch.view_as_complex(c)
        xrec= self.cq.backward(c)
        return xrec


class VCQT():
    def __init__(self,fmin, fmax, fbins, Qvar, fs=44100, audio_len=44100, multichannel=False, device="cpu", split_0_nyq=False):
#        fmax=fs/2
        Ls=audio_len
        scl = LogScale(fmin=fmin,fmax=fmax,bnds=fbins)
#        scl_0 = VQLogScale(fmin=fmin,fmax=fmax,bnds=fbins,gamma=0)

        # f,q = scl()
        # fig = plt.figure()
        # plt.plot(f,q)
        # f,q = scl_0()
        # fig2 = plt.figure()
        # plt.plot(f,q)
        
        self.cq = NSGT(scl, fs, Ls, Qvar,
                 real=True,
                 matrixform=True, 
                 multichannel=multichannel,
                 device=device
                 )
        #self.cq=self.cq.to(device)
        self.split_0_nyq=split_0_nyq

    def fwd(self,x):
        # x ... input tensor of size [batch_size x n_channels x n_time_steps]
        # c ... ouput tensor of size [batch_size x n_channels x n_freqs x n_time_frames x 2(complex)]
        n_samples, n_channels, n_time_steps = x.size()
        x = x.reshape(1,n_samples*n_channels,n_time_steps)
        c = self.cq.forward(x)
        c = c.squeeze(0)
        c = torch.view_as_real(c).reshape(n_samples,n_channels,c.size(-2),c.size(-1),2)
#        c=c.permute(0,3,2,1)
        if self.split_0_nyq:
            c0=c[...,0,:,:]
            cnyq=c[...,-1,:,:]
            cfreqs=c[...,1:-1,:,:]
            return cfreqs, c0, cnyq
        else:
            return c#.permute(0,3,2,1)

    def bwd(self,cfreqs,c0=None, cnyq=None):
        # c ... input tensor of size [batch_size x n_channels x n_freqs x n_time_frames x 2(complex)]
        # xrec ... output tensor of size [batch_size x n_channels x n_time_steps]
        if self.split_0_nyq:
            c=torch.cat((c0.unsqueeze(2),cfreqs,cnyq.unsqueeze(2)),dim=2)
#            c=c.permute(0,3,2,1)
        else:
            c=cfreqs
        
        c=c.unsqueeze(0)
        c=c.contiguous()
        c=torch.view_as_complex(c)
        xrec= self.cq.backward(c)
        return xrec