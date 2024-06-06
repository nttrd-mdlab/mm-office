import os
import random
import pandas as pd
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
import glob


class LoadDataset(Dataset):
    """
    children class of torch.util.data.Dataset
    This class make sample for torch.util.data.DataLoader
    """
    def __init__(self, cfg, train):
        self.train = train
        sp_param = cfg['sp_cfg']
        dnn_param = cfg['dnn_cfg']
        self.use_mic = dnn_param["use_mic"]
        self.use_cam = dnn_param["use_cam"]
        
        
        if self.train:
            self.afiledir = cfg['atraindir']
            self.vfiledir = cfg['vtraindir']
        else:            
            self.afiledir = cfg['atestdir']
            self.vfiledir = cfg['vtestdir']        

        self.datatype = cfg['dataType']
        
        self.afiles = sorted(glob.glob(self.afiledir+'/*_id00*'))
        self.vfiles = sorted(glob.glob(self.vfiledir+'/*_id00*'))
        self.shift = sp_param['shift_len']
        self.fs = sp_param['fs']
        
        self.input_len = sp_param['input_len']
        self.freq = int(sp_param['fft_len']/2)+1
        
        self.recinfo = pd.read_csv(cfg['rec_info'], header=0, names=['recid','sceneid','patternid'])
        self.eventlabels = pd.read_csv(cfg['event_info'])
        self.div = 0

        self.t_reso = sp_param['t_reso']
        self.n_class = dnn_param['n_class']
                
    def __len__(self):
        return len(self.afiles)

    def __getitem__(self, idx):
        data = self._getdata(idx)
        return data
    
    def _getdata(self, idx):
        id1 = idx
        afn1 = self.afiles[id1]
        length1 = 0.99*os.path.getsize(afn1)*8/16/1/48000
        offrand = random.random()
        #offset1 = (length1-self.input_len)*offrand
        offset1 = max(0.0,(length1-self.input_len)*offrand)
        
        self.div = int(self.afiles[id1][-5])
        
        if offset1<0.0:
            print(offset1)
            print(length1)
            print(self.input_len)
        ax1 = self._getaudiodata(id1,length1, offset1)
        vx1,offset = self._getvideodata(id1,length1, offset1)
        
        #should be refact
        if(vx1==None):
            vx1 = ax1
        if(ax1==None):
            ax1 = vx1
        
        elabel1 = self._geteventlabel(id1)

        if self.train:           
            sample = {'audio1': ax1,
                        'video1': vx1,
                        'elabel1': elabel1,
                        'offset': offset,
                        'fn1': afn1}
        else:
            tlabel1 = self._gettimelabel(id1,offset1)
            sample = {'audio1': ax1,
                        'video1': vx1,
                        'elabel1': elabel1,
                        'offset': offset,
                        'fn1': afn1,
                        'tlabel1': tlabel1}
                
        return sample

    def _getaudiodata(self, idx,length, offset):
        afn_former = self.afiles[idx][:-21]#id前
        afn_latter = self.afiles[idx][-19:]#id後
        axs = []
        for i in range(8):
            if i in self.use_mic:
                afn = afn_former + str(i).zfill(2) + afn_latter
                if(length<self.input_len):
                    ax, fs0 = torchaudio.load(afn, num_frames=int(48000.0*length),offset=0)
                    z = ax[:,: int(48000 * self.input_len) - int(48000 * length)] * 0
                    ax = torch.cat([ax,z],dim=-1)
                else:
                    ax, fs0 = torchaudio.load(afn, num_frames=int(48000.0*self.input_len),
                                            offset=int(48000.0*offset))
                ax = self._preprocess_audio(ax, self.input_len)
                axs.append(ax)
        if len(axs)>0:
            ax = torch.cat(axs,dim=0)#8xT
            return ax
        else:
            return None
    
    def _getvideodata(self, idx, length, offset):
        vfn1 = self.vfiles[idx][:-21]
        vfn2 = self.vfiles[idx][-19:]
        vxs = []
        for i in range(4):
            if i in self.use_cam:
                vfn = vfn1 + str(i).zfill(2) + vfn2
                if length<self.input_len:
                    vx, _, info = torchvision.io.read_video(vfn, start_pts=0,
                                                            end_pts=length, pts_unit='sec')                
                    #full size@len=25.6 torch.Size([513, 180, 320, 3]),20.039fps
                    z = vx[:int((self.input_len - length)*20.039)]*0
                    vx = torch.cat([vx,z],dim=0)
                else:
                    vx, _, info = torchvision.io.read_video(vfn, start_pts=offset,
                                                            end_pts=offset+self.input_len, pts_unit='sec')   
                vx = self._preprocess_video(vx, self.t_reso, 224)        
                vxs.append(vx.unsqueeze(0))

        if len(vxs)>0:
            vx = torch.cat(vxs)
            return vx,offset
        else:
            return None,0.0

    def _geteventlabel(self, idx):
        afn = self.afiles[idx]
        recid = self._getrecid(afn)
        thisinfo = self.recinfo[self.recinfo['recid']==recid]
        sceneid = thisinfo['sceneid'].values[0]
        patternid = thisinfo['patternid'].values[0]
        thisevent = self.eventlabels[(self.eventlabels['sceneid']==sceneid)&
                                    (self.eventlabels['patternid']==patternid)&
                                    (self.eventlabels['division']==self.div)].values[0][3:]
        return thisevent 

    def _gettimelabel(self, idx, offset):
        afn = self.afiles[idx]
        recid = self._getrecid(afn)
        df = pd.read_csv('../dataset/label/timeanotation3/recid'+str(recid).zfill(3)+'.csv')
        tlabel = torch.zeros((self.t_reso,13))#self.n_class))
        for t in range(self.t_reso):
            time = (t + 0.5) * (self.input_len/self.t_reso) + offset
            for i in range(len(df)):
                start = df['starttime'][i]
                end = df['endtime'][i]
                act = df['action'][i]# -1 with bg
                if time>=start and time<=end and act>=0:
                    tlabel[t][act] = 1                        
        return tlabel

    def _preprocess_audio(self, x, L):
        x = torchaudio.transforms.Resample(48000, self.fs)(x)
        if x.shape[1] < self.fs*L:
            res = int(self.fs*L - x.shape[1])
            z = torch.zeros([x.shape[0], res], dtype=torch.float)
            x = torch.cat([x, z], dim=1)
        return x

    def _preprocess_video(self, x, frames, size):
        # sampling:
        px = torch.zeros([3, frames, size, size], dtype=torch.float)
        ratio = int(x.shape[0]/frames)  # shold be larger than 1
        if ratio != 0:
            k = 0
            for i in range(0, x.shape[0]-ratio, ratio):
                if k > frames-1:
                    break
                offset = random.randrange(ratio-1)
                sample = x[i+offset]  # H,W,C
                sample = torch.transpose(sample, 0, 2)  # C,W,H
                sample = torch.transpose(sample, 1, 2)  # C,H,W
                sample = torchvision.transforms.ToPILImage()(sample)
                sample = torchvision.transforms.Resize((size, size))(sample)
                sample = torchvision.transforms.ToTensor()(sample)
                sample = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(sample)
                px[:, k, :, :] = sample  # C,T,H,W
                k += 1
        else:
            print('error')
            print(x.shape)
        return px

    def _getrecid(self, fn):
        return int(fn[-9:-6])

class TestLoadDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        sp_param = cfg['sp_cfg']
        dnn_param = cfg['dnn_cfg']
        
        self.use_mic = dnn_param["use_mic"]
        self.use_cam = dnn_param["use_cam"]

        self.afiledir = cfg['atestdir']
        self.vfiledir = cfg['vtestdir']        
        self.datatype = cfg['dataType']
        
        self.testlist = pd.read_csv(cfg['testlist_path'])
        self.recinfo = pd.read_csv(cfg['rec_info'], header=0, names=['recid','sceneid','patternid'])
        self.eventlabels = pd.read_csv(cfg['event_info'])

        #recid, N, flag, offset        
        #self.afiles = sorted(glob.glob(self.afiledir+'/*_id00*'))
        #self.vfiles = sorted(glob.glob(self.vfiledir+'/*_id00*'))
        self.shift = sp_param['shift_len']
        self.fs = sp_param['fs']
        
        self.input_len = sp_param['input_len']
        self.freq = int(sp_param['fft_len']/2)+1

        self.t_reso = sp_param['t_reso']
        self.n_class = dnn_param['n_class']
                
    def __len__(self):
        return len(self.testlist)

    def __getitem__(self, idx):
        data = self._getdata(idx)
        return data
    
    def _getdata(self, idx):
        recid1 = int(self.testlist["recid"][idx])
        afn1 = glob.glob(self.afiledir+'/*_id00*recid' + str(recid1).zfill(3) + '*')[0]
        vfn1 = glob.glob(self.vfiledir+'/*_id00*recid' + str(recid1).zfill(3) + '*')[0]
        length1 = 0.99*os.path.getsize(afn1)*8/16/1/48000
        filesplit = int(self.testlist["filesplit"][idx])        
        offset1 = 0.5 * self.input_len * filesplit  - self.testlist["offset"][idx] #0.5: half overlap evaluation

        if offset1 < 0.5* (length1 - self.input_len):            
            self.div = 0
        else:
            self.div = 1

        ax1 = self._getaudiodata(afn1,offset1)
        vx1 = self._getvideodata(vfn1,offset1)
        #should be refact
        if(vx1==None):
            vx1 = ax1
        if(ax1==None):
            ax1 = vx1
        
        elabel1 = self._geteventlabel(afn1)
        tlabel1 = self._gettimelabel(afn1,offset1)

        sample = {'audio1': ax1,
                    'video1': vx1,
                    'elabel1': elabel1,
                    'tlabel1': tlabel1,
                    'afile1': afn1,
                    'vfile1': vfn1
                    }

        return sample

    def _getaudiodata(self, afn, offset):  
        afn_former = afn[:-19]#id前
        afn_latter = afn[-17:]#id後
        axs = []
        for i in range(8):
            if i in self.use_mic:
                afn = afn_former + str(i).zfill(2) + afn_latter
                try:
                    ax, fs0 = torchaudio.load(afn, num_frames=int(48000.0*self.input_len),
                                            offset=int(48000.0*offset))
                except:
                    print(afn)
                    print(offset)
                ax = self._preprocess_audio(ax, self.input_len)
                axs.append(ax)
        if len(axs)>0:
            ax = torch.cat(axs,dim=0)#8xT
            return ax    
        else:
            return None
    
    def _getvideodata(self, vfn, offset):
        vfn1 = vfn[:-19]
        vfn2 = vfn[-17:]
        vxs = []
        for i in range(4):
            if i in self.use_cam:
                vfn = vfn1 + str(i).zfill(2) + vfn2
                vx, _, info = torchvision.io.read_video(vfn, start_pts=offset,
                                                        end_pts=offset+self.input_len, pts_unit='sec')
                vx = self._preprocess_video(vx, self.t_reso, 224)        
                vxs.append(vx.unsqueeze(0))

        if len(vxs)>0:
            vx = torch.cat(vxs)
            return vx
        else:
            return None

    def _geteventlabel(self, afn):
        recid = self._getrecid(afn)
        thisinfo = self.recinfo[self.recinfo['recid']==recid]
        sceneid = thisinfo['sceneid'].values[0]
        patternid = thisinfo['patternid'].values[0]
        thisevent = self.eventlabels[(self.eventlabels['sceneid']==sceneid)&
                                    (self.eventlabels['patternid']==patternid)&
                                    (self.eventlabels['division']==self.div)].values[0][3:]
        return thisevent

    def _gettimelabel(self, afn, offset):
        recid = self._getrecid(afn)
        df = pd.read_csv(self.cfg['testlabeldir'] + 'recid'+str(recid).zfill(3)+'.csv')
        tlabel = torch.zeros((self.t_reso,13))#self.n_class))
        for t in range(self.t_reso):
            time = (t + 0.5) * (self.input_len/self.t_reso) + offset#0.5?
            for i in range(len(df)):
                start = df['starttime'][i]
                end = df['endtime'][i]
                act = df['eventclass'][i]# -1 with bg
                if time>=start and time<=end and act>=0:
                    tlabel[t][act] = 1                    
        return tlabel
    
    def _preprocess_audio(self, x, L):
        x = torchaudio.transforms.Resample(48000, self.fs)(x)
        if x.shape[1] < self.fs*L:
            res = int(self.fs*L - x.shape[1])
            z = torch.zeros([x.shape[0], res], dtype=torch.float)
            x = torch.cat([x, z], dim=1)
        return x

    def _preprocess_video(self, x, frames, size):
        # sampling:
        px = torch.zeros([3, frames, size, size], dtype=torch.float)
        ratio = int(x.shape[0]/frames)  # shold be larger than 1
        if ratio != 0:
            k = 0
            for i in range(0, x.shape[0]-ratio, ratio):
                if k > frames-1:
                    break
                offset = random.randrange(ratio-1)
                sample = x[i+offset]  # H,W,C
                sample = torch.transpose(sample, 0, 2)  # C,W,H
                sample = torch.transpose(sample, 1, 2)  # C,H,W
                sample = torchvision.transforms.ToPILImage()(sample)
                sample = torchvision.transforms.Resize((size, size))(sample)
                sample = torchvision.transforms.ToTensor()(sample)
                sample = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(sample)
                px[:, k, :, :] = sample  # C,T,H,W
                k += 1
        else:
            print('error')
            print(x.shape)
        return px
    
    def _getrecid(self, fn):
        return int(fn[-7:-4])     
