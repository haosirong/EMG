import os
import sys
import time
import pywt
import datetime
import numpy as np
import scipy.signal
import scipy.io as scio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchsummary import summary
from tensorboardX import SummaryWriter



# Hyper parameters
num_epochs = 50
num_classes = 27
batch_size = 256
learning_rate = 0.001
channels_num = 168

WINDOW_LEN = 256
OVERLAP_LEN = 128
LENGTH = 1000
SEGMENT_NUM = (LENGTH - (WINDOW_LEN - OVERLAP_LEN)) / OVERLAP_LEN  #18
wavelet = pywt.Wavelet('db5')
low_filter = torch.Tensor(wavelet.dec_lo[::-1])
high_filter = torch.Tensor(wavelet.dec_hi[::-1])
low_filters = low_filter.repeat(128,1).reshape(128,1,10).cuda()
high_filters = high_filter.repeat(128,1).reshape(128,1,10).cuda()


def window_slice(data,window_len,overlap_len):
    aquisition = torch.Tensor(data) #shape(channel,wave_length)
    total_length = aquisition.size(1)
    segments = aquisition[:,0:window_len]

    segments = segments.reshape(1,-1,window_len)

    #segment_inc = torch.Tensor(channels_num,window_len)
    #segment_inc1 = torch.Tensor(1,channels_num,window_len)
    #print(segments.size())
    time = overlap_len #incremental window
    while time + window_len <= total_length:
        segment_inc = aquisition[:,time:time + window_len]
        segment_inc1 = segment_inc.reshape(1,-1,window_len)
        segments = torch.cat((segments,segment_inc1),0)
        time = time + overlap_len
    '''
    print('test')
    print(segments.size())
    print(segments)
    print(channel_list)
    print(torch.mean(abs(aquisition),1))
    print(torch.mean(abs(segments),1))
    '''
    return segments

#window_slice(data['data'],WINDOW_LEN,OVERLAP_LEN)

def load_dataset(sub,ses,ges,tri):
    for i in ges:
        for j in ses:
            for k in sub:
                mat_path = 'subject{0}/session{1}/gest{2}.mat'.format(k,j,i)
                data = scio.loadmat(mat_path)
                for l in tri:
                    trial = data['gestures'][tri.index(l),0]
                    trial = np.delete(trial,np.s_[7:192:8],0)
                    if tri.index(l) == 0:
                        trial_total = window_slice(trial,WINDOW_LEN,OVERLAP_LEN)
                        print('subject{0}/session{1}/gest{2}.mat'.format(k,j,i))
                        print(trial_total[0,0,0])
                    else:
                        trial_single = window_slice(trial,WINDOW_LEN,OVERLAP_LEN)
                        trial_total = torch.cat((trial_total,trial_single),0)
                if sub.index(k) == 0:
                    sub_total = trial_total
                else:
                    sub_total = torch.cat((sub_total,trial_total),0)
            if ses.index(j) == 0:
                ses_total = sub_total
            else:
                ses_total = torch.cat((ses_total,sub_total),0) 
        if ges.index(i) == 0:
            ges_total = ses_total
        else:
            ges_total = torch.cat((ges_total,ses_total),0) 
    return ges_total


class EMGData(Dataset): #继承Dataset
    def __init__(self,sub,ses,ges,trial): #__init__是初始化该类的一些基础参数
        #self.root_dir = root_dir   #文件目录
        #self.transform = transform #变换
        #self.images = os.listdir(self.root_dir)#目录里的所有文件
        self.dataset = load_dataset(sub,ses,ges,trial) #shape ---> [total_number,channel,wave_length]
        #self.dataset = torch.norm(torch.rfft(self.dataset,1),p=2,dim=4)

        print('----- > emg tensor size')
        print(self.dataset.size())
        self.len = self.dataset.size(0)
        self.number = self.len // num_classes #each gesture number
        self.gesture_type = num_classes
        
        self.features_rms = torch.sqrt(torch.mean(self.dataset*self.dataset,2)).reshape(self.len,24,7)
        #f_max = torch.max(features_rms).item()
        #f_min = torch.min(features_rms).item()
        #f_mean = torch.mean(features_rms).item()
        #f_std = torch.std(features_rms).item()

        y1 = self.dataset[:,:,0:WINDOW_LEN - 1]
        y2 = self.dataset[:,:,1:WINDOW_LEN]
        self.features_wl = torch.sum(abs(y1 - y2),2).reshape(self.len,24,7)
        #f_max = torch.max(features_wl).item()
        #f_min = torch.min(features_wl).item()
        #f_mean = torch.mean(features_wl).item()
        #f_std = torch.std(features_wl).item()
        x1 = torch.sign(self.dataset[:,:,0:WINDOW_LEN - 1])
        x2 = torch.sign(self.dataset[:,:,1:WINDOW_LEN])
        zc = torch.eq(x1,x2)
        zc_sum = WINDOW_LEN - torch.sum(zc,2)
        self.features_zc = zc_sum.reshape(self.len,24,7)
        self.features_zc = self.features_zc.type(torch.FloatTensor)
        
        x = torch.norm(torch.rfft(self.dataset,1),p=2,dim=3)
        y = torch.linspace(0,128,129).reshape(129,1)
        self.features_sn = torch.matmul(x*x,y).reshape(self.len,24,7)

    def __len__(self):#返回整个数据集的大小
        #return len(self.images)
        return self.len
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        
        label = index // self.number
        #print('{0}:{1}'.format(label,num))
        data = self.dataset[index] 
        feature_rms = self.features_rms[index] 
        feature_wl = self.features_wl[index] 
        feature_zc = self.features_zc[index] 
        feature_sn = self.features_sn[index] 
        #sample = {'data111':data,'label222':label}
        return data,torch.cat((feature_rms,feature_wl,feature_zc,feature_sn),dim=0),label
        #return data,feature_rms,label

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16,init_weights=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1   = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2   = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_planes // ratio)
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        
        #auxiliary convnet
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(4, 256, 3, padding=1, bias=False,groups = 4)
        self.conv2 = nn.Conv2d(256, 256, 1, bias=False)#,groups = 2)
        self.fc = nn.Linear(128*256,128,bias=False)
        '''
        self.fc1   = nn.Conv1d(16, 8, 1, bias=False)
        self.fc1_   = nn.Conv1d(8, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(4)
        self.fc2   = nn.Conv1d(8, 16, 1, bias=False)
        self.fc2_   = nn.Conv1d(4, 8, 1, bias=False)
        '''
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        if init_weights:
            self._initialize_weights()

    def forward(self, x, layer = 0):
        #print(x.size())
        #print(self.avg_pool(x).size())
        #avg_pool = self.avg_pool(torch.abs(x))
        #avg_pool = torch.mean(torch.abs(x),2).reshape(x.size(0),x.size(1),1)
        #avg_pool = torch.sum(x*x, dim=1).reshape(x.size(0),x.size(2),1) # weighting in frequency domain, need a transpose
        #max_pool = self.max_pool(torch.abs(x))
        #rms = torch.sqrt(torch.mean(x*x,2)).reshape(x.size(0),1,16,8)
        
        #y1 = x[:,:,0:WINDOW_LEN - 1]
        #y2 = x[:,:,1:WINDOW_LEN]
        #wl = torch.sum(abs(y1 - y2),2).reshape(x.size(0),1,16,8)
        #print('---->\n',rms[0])
        #print(wl[0])
        #x = torch.cat((wl,rms),dim=1)
            
        '''
        x1 = torch.sign(x[:,:,0:WINDOW_LEN - 1])
        x2 = torch.sign(x[:,:,1:WINDOW_LEN])
        zc = torch.eq(x1,x2)
        zc_sum = WINDOW_LEN - torch.sum(zc,2)
        zc_sum = zc_sum.reshape(x.size(0),1,16,8)
        zc_sum = zc_sum.type(torch.cuda.FloatTensor)
        #x = torch.cat((rms,zc_sum),dim=1)
        '''


        '''
        subband1_energy = torch.randn(x.size(0),x.size(1),1)
        subband2_energy = torch.randn(x.size(0),x.size(1),1)
        subband3_energy = torch.randn(x.size(0),x.size(1),1)
        subband4_energy = torch.randn(x.size(0),x.size(1),1)
        for i in range(x.size(0)):
            for j in range(128):
                wp = pywt.WaveletPacket(x[i,j].cpu().detach().numpy(), wavelet='sym5',mode='symmetric',maxlevel=2)
                subband1_energy[i,j,0] = torch.Tensor(np.array(np.linalg.norm(wp['aa'].data,ord=None)))
                subband2_energy[i,j,0] = torch.Tensor(np.array(np.linalg.norm(wp['ad'].data,ord=None)))
                subband3_energy[i,j,0] = torch.Tensor(np.array(np.linalg.norm(wp['da'].data,ord=None)))
                subband4_energy[i,j,0] = torch.Tensor(np.array(np.linalg.norm(wp['dd'].data,ord=None)))
            #print(i,":",subband1_energy[i].reshape(16,8))
        '''
        '''
        x_a = F.conv1d(x,low_filters.cuda(),padding = 9,groups = 128)
        x_aa = F.conv1d(x_a,low_filters.cuda(),padding = 9,groups = 128)
        x_ad = F.conv1d(x_a,low_filters.cuda(),padding = 9,groups = 128)
        x_aa_rms = torch.sqrt(torch.mean(x_aa*x_aa,2)).reshape(x.size(0),1,16,8)
        x_ad_rms = torch.sqrt(torch.mean(x_ad*x_ad,2)).reshape(x.size(0),1,16,8)

        x_d = F.conv1d(x,high_filters.cuda(),padding = 9,groups = 128)
        x_da = F.conv1d(x_d,high_filters.cuda(),padding = 9,groups = 128)
        x_dd = F.conv1d(x_d,high_filters.cuda(),padding = 9,groups = 128)
        x_da_rms = torch.sqrt(torch.mean(x_da*x_da,2)).reshape(x.size(0),1,16,8)
        x_dd_rms = torch.sqrt(torch.mean(x_dd*x_dd,2)).reshape(x.size(0),1,16,8)
        x = torch.cat((x_aa_rms,x_ad_rms,x_da_rms,x_dd_rms),dim=1)
        '''

        '''
        x_att = x.cpu().detach().numpy()
        zc_sum = np.zeros((x.size(0),128,1))
        zc = np.zeros((x.size(0),128,199))
        #x_arr = x.detach.numpy()
        for i in range(x.size(0)):
            for j in range(128):
                for k in range(199):
                    zc[i][j][k] = -np.sign(x_att[i][j][k])*np.sign(x_att[i][j][k+1])
                    print(i,j,k)
                zc_sum[i][j][0] = np.where(zc[i][j][k] > 0)[0].shape[0]
        print(zc_sum[0].reshape(16,8))
        '''
        #avg_out = self.fc2(self.relu1(self.bn1(self.fc1(x)))) 
        #avg_out = self.fc2(self.relu1(self.bn1(self.fc1(sqrt_avg_pool)))) 
        #avg_out2 = self.fc2(self.relu1(self.bn1(self.fc1(subband2_energy)))) 
        #avg_out3 = self.fc2(self.relu1(self.bn1(self.fc1(subband3_energy)))) 
        #avg_out4 = self.fc2(self.relu1(self.bn1(self.fc1(subband4_energy)))) 
        #avg_out = avg_out.transpose(2,1)
        #max_out = self.fc2(self.relu1(self.bn1(self.fc1(max_pool))))
        #sqrt_out = self.fc2(self.relu1(self.bn1(self.fc1(sqrt_avg_pool))))
        #out = self.fc2(self.relu1(self.bn1(self.fc1(zc_sum))))
        #out = self.fc2(self.relu1(self.bn1(self.fc1(zc_sum))))
        
        #print("-------------------------")
        #print(avg_pool[-1].reshape(16,8))
        #print(avg_out[-1].reshape(16,8))
        #print(self.sigmoid(avg_out[-1].reshape(16,8)))
        #print(max_pool[-1].reshape(16,8))
        #print(max_out[-1].reshape(16,8))
        #print(self.sigmoid(max_out[-1].reshape(16,8)))
        #print(sqrt_avg_pool[-1].reshape(16,8))
        #print(sqrt_out[-1].reshape(16,8))
        #print(zc_sum[-1].reshape(16,8))
        #print(self.sigmoid(zc_out[-1].reshape(16,8)))
        avg_out = self.conv1(x)
        avg_out = self.conv2(avg_out)
        #avg_out = self.relu1(avg_out)
        avg_out = self.fc(avg_out.reshape(x.size(0),-1))
        #out = avg_out + max_out
        return self.sigmoid(avg_out).reshape(x.size(0),128,1)
        #return self.softmax(out)
        '''
        if layer == 1:
            avg_pool1 = torch.mean(avg_pool.reshape(-1,16,8),2)   # shape [batch , 16]
            avg_pool1 = avg_pool1.reshape(x.size(0),16,1)
            avg_out1 = self.fc2(self.relu1(self.bn1(self.fc1(avg_pool1)))) # mean is not a salient feature

            max_pool1 = torch.mean(max_pool.reshape(-1,16,8),2)   # shape [batch , 16]
            max_pool1 = max_pool1.reshape(x.size(0),16,1)
            max_out1 = self.fc2(self.relu1(self.bn1(self.fc1(max_pool1))))
            out1 = avg_out1 + max_out1
            out1 = torch.stack((out1,out1,out1,out1,out1,out1,out1,out1),dim=2).reshape(x.size(0),128,1)
            return self.sigmoid(out1)

        if layer == 2: 
            avg_pool2 = torch.mean(avg_pool.reshape(-1,16,8).transpose(2,1),2)   # shape [batch , 8]
            avg_pool2 = avg_pool2.reshape(x.size(0),8,1)
            avg_out2 = self.fc2_(self.relu1(self.bn2(self.fc1_(avg_pool2)))) # mean is not a salient feature

            max_pool2 = torch.mean(max_pool.reshape(-1,16,8).transpose(2,1),2)   # shape [batch , 8]
            max_pool2 = max_pool2.reshape(x.size(0),8,1)
            max_out2 = self.fc2_(self.relu1(self.bn2(self.fc1_(max_pool2))))
            out2 = avg_out2 + max_out2
            out2 = torch.stack((out2,out2,out2,out2,out2,out2,out2,out2,out2,out2,out2,out2,out2,out2,out2,out2),dim=2).transpose(2,1).reshape(x.size(0),128,1)
            return self.sigmoid(out2)
            '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal(param)

# Convolutional neural network part of VGGNet
class FIXED_TICNN_AT(nn.Module):
    def __init__(self,used_gpu, nclass, dropout, output_weight=False, init_weights=True):
        super(FIXED_TICNN_AT, self).__init__()
        self.hidden_dim = 32
        #self.features = features
        #print(self.features)
        self.used_gpu = used_gpu
        self.nclass = nclass
        self.output_weight = output_weight
        self.relu = nn.ReLU(inplace = True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim = -1)

        #   'N': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
        #  rewrite cnn in init in order to add se module per layer 
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        #self.layer_16_64 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=3, padding=1),nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        #self.layer_32_64 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=1),nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        #self.layer_128_64 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=3, padding=1),nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        #self.layer_256_64 = nn.Sequential(nn.Conv1d(256, 64, kernel_size=3, padding=1),nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        #self.layer_64_128 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1),nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        #self.layer_128_256 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, padding=1),nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        #self.layer_256_256 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=1),nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        #self.layer_256_512 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=3, padding=1),nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        #self.layer_512_512 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=3, padding=1),nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.bn_128_0 = nn.BatchNorm1d(128)

        self.layer_128_128 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn_128 = nn.BatchNorm1d(128)

        self.layer_168_256_1 = nn.Conv1d(168, 256, kernel_size=3, padding=1)
        self.bn_256_1 = nn.BatchNorm1d(256)
        self.layer_256_256_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_2 = nn.BatchNorm1d(256)
        self.layer_256_256_3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_3 = nn.BatchNorm1d(256)
        self.layer_256_256_4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_4 = nn.BatchNorm1d(256)
        self.layer_256_512_5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn_512_5 = nn.BatchNorm1d(512)
        self.layer_512_512_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn_512_6 = nn.BatchNorm1d(512)
        
        self.downsample_1 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=3),nn.BatchNorm1d(256))
        self.downsample_2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, stride=3),nn.BatchNorm1d(256))
        self.downsample_3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, stride=3),nn.BatchNorm1d(256))

        self.group_cnn_64 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 42) # aggragate spatial channel
        self.group_cnn_8 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 8) # aggragate spatial channel
        self.group_cnn_4 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 6) # aggragate spatial channel
        self.group_cnn_2 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 3) # aggragate spatial channel
        self.group_cnn_16 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 12) # aggragate spatial channel
        self.group_cnn_32 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 84) # aggragate spatial channel
        self.group_cnn_128 = nn.Conv1d(channels_num, 168, kernel_size=1, groups = 168) # aggragate spatial channel
        self.group_cnn_256 = nn.Conv1d(256, 256, kernel_size=1, groups = 256) # aggragate spatial channel
        self.group_cnn_512 = nn.Conv1d(512, 512, kernel_size=1, groups = 512) # aggragate spatial channel
        self.ca_input = ChannelAttention(168)	
        self.ca_64 = ChannelAttention(64)	
        self.ca_128 = ChannelAttention(128)	
        self.ca_256_1 = ChannelAttention(256)	
        self.ca_256_2 = ChannelAttention(256)	
        self.ca_256_3 = ChannelAttention(256)	
        self.ca_512 = ChannelAttention(512)	

        self.rnn = nn.LSTM(512, self.hidden_dim, 2, batch_first=True, dropout=dropout)
        print(self.rnn)
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, nclass))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * nclass, nclass))
        self.classifier2 = nn.Sequential(nn.Linear(512, nclass))
        #self.classifier2 = nn.Sequential(nn.Linear(WINDOW_LEN // 3**4 * 512, 1024),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(1024,256),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(256,nclass))
        print(self.classifier2)
        if init_weights:
            self._initialize_weights()


    #def forward(self, x, length):
    def forward(self, x, y):

        # Covolutions
        '''
        if x.size(0) != batch_size:
            weight = self.ca_input(x,layer = 1)[0].view(16,8)
            picture = torch.mean(abs(x[0]),1).view(16,8)
            scale = weight[0][0] / picture[0][0]
            propotion = weight/picture/scale
            print(weight)
            print(picture)
            print(torch.mean(picture,1))
            #print(propotion)
            print('max:{},min{}'.format(propotion.max().item(),propotion.min().item()))
        '''


        if sys.argv[1] == '1':
            #x = self.ca_input(y.reshape(y.size(0),1,16,8)) * x #N*C*1   *   N*C*L
            #x = torch.norm(torch.rfft(x,1),p=2,dim=3)
            #y = torch.linspace(0,100,101).reshape(101,1)
            #z = torch.matmul(x2*x2,y).reshape(x.size(0),1,16,8)
            #x = self.ca_input(z) * x #N*C*1   *   N*C*L
            #z = torch.mean(x*x,2).reshape(x.size(0),x.size(1),1)
            #x = self.ca_input(y[:,0,:,:].reshape(x.size(0),1,16,8)) * x #N*C*1   *   N*C*L

            #sqrt_avg_pool = torch.sqrt(torch.mean(x*x,2)).reshape(x.size(0),x.size(1),1)
            x = self.ca_input(y) * x #N*C*1   *   N*C*L
            #x = self.ca_input(x)*x #N*C*1   *   N*C*L
            
            '''
            x_a = F.conv1d(x,low_filters.cuda(),padding = 9,groups = 128)
            x_aa = F.conv1d(x_a,low_filters.cuda(),padding = 9,groups = 128)
            x_ad = F.conv1d(x_a,low_filters.cuda(),padding = 9,groups = 128)
            x_aa = self.ca_input(x_aa) * x_aa #N*C*1   *   N*C*L
            x_ad = self.ca_input(x_ad) * x_ad #N*C*1   *   N*C*L

            x_d = F.conv1d(x,high_filters.cuda(),padding = 9,groups = 128)
            x_da = F.conv1d(x_d,high_filters.cuda(),padding = 9,groups = 128)
            x_dd = F.conv1d(x_d,high_filters.cuda(),padding = 9,groups = 128)
            x_da = self.ca_input(x_da) * x_da #N*C*1   *   N*C*L
            x_dd = self.ca_input(x_dd) * x_dd #N*C*1   *   N*C*L

            x = x_aa + x_ad + x_da + x_dd
            '''

        if sys.argv[1] == '2':
            # method 1
            x = self.group_cnn_128(x)

            # method 2
            #x = x.reshape(x.size(0),16,8,WINDOW_LEN).transpose(2,1).reshape(x.size(0),128,WINDOW_LEN)
            #x = self.group_cnn_16(x)
            
            # method 3
            '''
            x3 = None 
            for i in range(8):   #8*16 channel  2*2 cluster
                for j in range(4):
                    if i == 0 and j == 0:
                        x3 = x[:,[0,1,8,9],:]
                    else:
                        x3 = torch.cat([x3,x[:,[16*i+2*j,16*i+2*j+1,16*i+2*j+8,16*i+2*j+9],:]],dim=1)
            x = self.group_cnn_32(x3)

            x = self.bn_128(x)
            x = self.relu(x)
            '''

        #if sys.argv[1] == '3':
            #x = self.ca_input(x)*x #N*C*1   *   N*C*L
            #x = self.group_cnn_16(x)
            #x = self.bn_128_0(x)
            #x = self.relu(x)

        #x = self.features(x)
        #   'N': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
        #  rewrite cnn in init in order to add se module per layer 

        #out = x
        #out = self.downsample_1(out)

        x = self.layer_168_256_1(x)
        x = self.bn_256_1(x)
        x = self.relu(x)
        x = self.pooling(x)
        #x = self.ca_256_1(x)*x 
        #x = out + x

        '''
        x_t = x.transpose(2,1) 
        c_att = torch.matmul(x,x_t)
        if x.size(0) != batch_size:
            print(x[0],x.size())
            print(x_t[0],x_t.size())
            print(c_att[0])
        c_att = c_att - torch.min(c_att, -1, keepdim=True)[0].expand_as(c_att)

        c_att = self.softmax(c_att)
        x = torch.matmul(c_att,x)
        if x.size(0) != batch_size:
            print(c_att[0])
        '''

        #out = x
        #out = self.downsample_2(out)
        x = self.layer_256_256_2(x)
        x = self.bn_256_2(x)
        x = self.relu(x)
        x = self.pooling(x)
        #x = self.ca_256_2(x)*x 
        #x = out + x

        #out = x
        #out = self.downsample_3(out)
        x = self.layer_256_256_3(x)
        x = self.bn_256_3(x)
        x = self.relu(x)
        x = self.layer_256_256_4(x)
        x = self.bn_256_4(x)
        x = self.relu(x)
        x = self.pooling(x)
        #x = self.ca_256_3(x)*x 
        #x = out + x

        x = self.layer_256_512_5(x)
        x = self.bn_512_5(x)
        x = self.relu(x)
        x = self.layer_512_512_6(x)
        x = self.bn_512_6(x)
        x = self.relu(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal(param)


# Generate network layers in accordance to configuration(config)
# Used for the construction of VGGNet
def make_layers(cfg, batch_norm=False):
    layers = []
    '''
    in_channels = channels_num #input channel,12 leads
    conv1d = nn.Conv1d(in_channels, 256, kernel_size=1)
    layers += [conv1d]
    '''
    in_channels = 128
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=3, stride=3)]
            #layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# Configuration of different VGG Networks
cfg = {
    'N': [256, 'M', 256, 'M', 256, 256, 'M', 512, 512],
    #'N': [256, 'M', 256, 'M', 384, 384, 'M', 512, 512, 'M'],
    #'N': [256, 'M', 256, 'M', 512, 512, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# Instantiation function of VGG-16 Network with Batch Normalization
def fixedticnn16at_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    #model = FIXED_TICNN_AT(make_layers(cfg['N'], batch_norm=True), **kwargs)
    model = FIXED_TICNN_AT(**kwargs)
    return model


if sys.argv[2] == '0':
    time = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    writer = SummaryWriter(logdir='runs/'+time)
    print('----> train data begin')
    data_train = EMGData(
               #[1],#subject
               [1,2,3,4,5],#subject
               #[1],#session
               [1,2,3,4,5],#session
               [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],#gesture  
               #[0,1],#gesture  
               #[1,3,5,7,9]                         #trial
               [1]                         #trial
               )#初始化类，设置数据集所在路径以及变换

    dataloader_train = DataLoader(data_train,batch_size=batch_size,shuffle=True)#使用DataLoader加载数据
    print('----> train data ready:len',len(dataloader_train))

    print('----> validation data begin')
    data_val = EMGData(
               #[1],#subject
               [1,2,3,4,5],#subject
               #[1],#session
               [1,2,3,4,5],#session
               [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],#gesture  
               [2,4,6,8,10],  #trial
               #[6,7,8,9,10]                         #trial
               )#初始化类，设置数据集所在路径以及变换

    dataloader_val = DataLoader(data_val,batch_size=batch_size,shuffle=False)#使用DataLoader加载数据
    print('----> validation data ready:len',len(dataloader_val))
else:
    print('----> test data begin')
    data_test = EMGData(
               #[1],#subject
               [1,2,3,4,5],#subject
               [2],#session
               #[1,2,3,4,5],#session
               [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],#gesture  
               #[0,1],
               [4,6,8,10]                         #trial
               )#初始化类，设置数据集所在路径以及变换

    dataloader_test = DataLoader(data_test,batch_size=batch_size,shuffle=False)#使用DataLoader加载数据
    print('----> test data ready:len',len(dataloader_test))

if torch.cuda.is_available():
#if 0:
    device = 'cuda:0'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = fixedticnn16at_bn(used_gpu=[0],nclass=num_classes,dropout=0.5)
    model = torch.nn.parallel.DataParallel(model, device_ids=[0])

else:
    print('use cpu')
    device = 'cpu'
    model = fixedticnn16at_bn(used_gpu=None,nclass=num_classes,dropout=0.5)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#,weight_decay=1e-5)

#print(summary(model,[(128,200),(2,16,8)],batch_size = 10,device = 'cpu'))

if sys.argv[2] == '0':
    # Train the model
    total_step = len(dataloader_train)
    loss_list = []
    print('Training...')
    for epoch in range(num_epochs):
        model.train()
        loss_acc = 0
        for i, (images, features,labels) in enumerate(dataloader_train):
            #print('---->' + images + ':' + labels)
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            if i == 0:
                print(labels)
            # Forward pass
            outputs = model(images,features)
            loss = criterion(outputs, labels)
            loss_acc += loss.item() 
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Train      Epoch [{}/{}], Step [{}/{}], Loss: {:.7f},Time:{}' .format(epoch+1, num_epochs, i+1, total_step, loss_acc,datetime.datetime.now()))
        writer.add_scalar('train', loss_acc, global_step=epoch)
        loss_list.append(loss.item())

        model.eval()
        with torch.no_grad():
            loss_acc = 0
            correct = 0
            total = 0
            correct_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            for i, (images, features,labels) in enumerate(dataloader_val):
                #print('---->' + images + ':' + labels)
                images = images.to(device)
                features = features.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images,features)
                loss = criterion(outputs, labels)
                loss_acc += loss.item() 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_label = (predicted == labels)
                for j in range(num_classes):
                    correct_ele = (predicted == j)
                    correct_list[j] = correct_list[j] + (correct_label & correct_ele).sum().item()#(predicted == i).sum().item()

            writer.add_scalar('validation', loss_acc, global_step=epoch)
            print ('Validation Epoch [{}/{}], Step [{}/{}], Loss: {:.7f},Time:{}' .format(epoch+1, num_epochs, i+1, total_step, loss_acc,datetime.datetime.now()))
            print(correct)
            print(correct_list)

        


        if epoch == 20:
            for p in optimizer.param_groups:
                print('1111')
                p['lr'] *= 0.1

        if epoch == 40:
            for p in optimizer.param_groups:
                print('1111')
                p['lr'] *= 0.1
        if epoch == 60:
            for p in optimizer.param_groups:
                print('1111')
                p['lr'] *= 0.1

        '''
        plt.plot(range(num_epochs),loss_list,color = 'r')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        '''
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
    writer.close()

else:
    #Load model
    model.load_state_dict(torch.load('model.ckpt'))

    acc_list = []
    print('Testing...')
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for name,param in model.named_parameters():
            print(name,param.size())
            #if name == "module.group_cnn_128.weight":
            #    print(param.data.reshape(num_classes,y))
                
        correct = 0
        total = 0
        correct_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        confusion_mat = np.zeros((num_classes,num_classes))
        for images, features,labels in dataloader_test:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(images,features)
            #print('---->{}'.format(outputs.data[0]))
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            for i in range(labels.size(0)):
                confusion_mat[labels[i]][predicted[i]] = confusion_mat[labels[i]][predicted[i]] + 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_label = (predicted == labels)
            for i in range(num_classes):
                correct_ele = (predicted == i)
                correct_list[i] = correct_list[i] + (correct_label & correct_ele).sum().item()#(predicted == i).sum().item()
                #for j in range(8):
                #    correct_list[i] = correct_list[i] + (correct_label & correct_ele).sum().item()#(predicted == i).sum().item()
        #if epoch % 5 == 0:
        #    acc = 100 * correct / total 
        #    acc_list.append(acc)
        print('Test Accuracy of the model {}: {} %'.format(sys.argv[1],100 * correct / total))
        print(correct_list)
        for i in range(num_classes):
            print('Label {} Accuracy:{} %'.format(i,100*correct_list[i]/(total/num_classes)))
        print(confusion_mat)
    '''
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(num_epochs),loss_list , label = 'Loss')
    ax2.plot(range(0,num_epochs,5), acc_list , color = 'r' , label = 'Accuracy')
    #plt.plot(range(num_epochs),loss_list,color = 'r')
    ax1.set_xlabel('Number of Iteration')
    ax1.set_ylabel('Train Loss')
    ax2.set_ylabel('Test Accuracy %')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2 , labels1+labels2, loc='upper right')
    plt.show()
    '''
