from __future__ import division
import time
import math
import random
 
import numpy as np
from scipy.ndimage.measurements import find_objects,label
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class ObjectDetector:
    def __init__(self, verbose=2, batch_size=2, nb_epochs=10):
        # experimentaly, a batch_size higher than 2 yielded worse results. 
        # try at your own risk.
        self.net = Net()
        self.net.cuda()
        self.verbose = verbose
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.sp = Smart_parser()

    def fit(self, X, y):
        X = X.reshape(-1,224,224)
        mean = np.mean(X, axis=(1,2))
        std = np.std(X, axis=(1,2))

        batch_size = self.batch_size
        nb_epochs = self.nb_epochs
        lr = 1e-1 # TODO : could be improved with grid-search
        net = self.net
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) # TODO : could be improved with grid-search
        criterion = nn.BCELoss().cuda()

        for epoch in range(nb_epochs):
            t0 = time.time()
            net.train() # train mode
            train_loss = []
            train_acc = []
            
            for i in range(int(np.floor(X.shape[0]/batch_size))):
                idx = np.arange(X.shape[0])[i*batch_size:min((i+1)*batch_size,X.shape[0])] # batch
                X_batch = np.array([((X[i]-mean[i])/std[i]) for i in idx]).reshape(-1,1,224,224) # preprocess
                X_batch = _make_variable(X_batch)
                y_batch = _make_variable(to_image(y[idx]).reshape(-1,1,224,224))

                optimizer.zero_grad() # zero-out the gradients because they accumulate by default
                y_pred = net(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward() # compute gradients
                optimizer.step() # update params

                # Loss and accuracy
                train_acc.extend(self._get_acc(y_pred, y_batch))
                train_loss.append(loss.data[0])
                
                del X_batch
                del y_batch
                del y_pred

                if (self.verbose >= 2):
                    progress = i*batch_size/X.shape[0]
                    print('\r-Train F1 : ' + str(np.mean(np.array(train_acc)[:,0])) + '|' +
                        str(np.mean(np.array(train_acc)[-100:,0])) + '|' + 
                        str(np.mean(np.array(train_acc)[-10:,0]))  + 
                        ' \t precision : ' + str(np.mean(np.array(train_acc)[:,1])) + '|' + 
                        str(np.mean(np.array(train_acc)[-100:,1])) + '|' + 
                        str(np.mean(np.array(train_acc)[-10:,1])) + 
                        ' \t recall : ' + str(np.mean(np.array(train_acc)[:,2])) + '|' + 
                        str(np.mean(np.array(train_acc)[-100:,2])) + '|' + 
                        str(np.mean(np.array(train_acc)[-10:,2])) + 
                        ' \t loss : ' + str(loss.data[0]) + ' \t'
                        + bar_graph(progress) ,end='')

            if (self.verbose >= 2):
                print()

            delta_t = time.time() - t0
            if(self.verbose >= 1):
                print('Finished epoch ', (epoch + 1))
                print('Time spent : ', delta_t)
                print('Train F1 : ', np.mean(train_acc))


        # Fitting the Smart_parser
        X_parser = None
        y_parser = None
        print('generating train_set for smart parsing...')
        for i in range(int(np.floor(X.shape[0]/self.batch_size))):
            idx = np.arange(X.shape[0])[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0])] #batch
            X_batch = np.array([((X[i]-mean[i])/std[i]) for i in idx]).reshape(-1,224,224,1) # preprocess
            X_batch = _make_variable(X_batch)
                
            y_proba = self.net(X_batch).cpu()
            y_proba = y_proba.float().data.numpy().reshape(-1,224,224)

            # pipelined to save RAM
            X_sub, y_sub = self.sp._conv_parser(y_proba, y) 

            if (X_parser is None or y_parser is None):
                X_parser = X_sub
                y_parser = y_sub
            else:
                X_parser = np.concatenate((X_parser,X_sub))
                y_parser = np.concatenate((y_parser,y_sub))

            del X_batch
            del y_proba


        print("fitting the smart parsing...")
        self.sp.fit(X_parser, y_parser)
        print("done.")

        return self
    
    
    def _get_raw(self, X_input): # return the mask for debug
        X_input = X_input.astype(np.float)
        self.net.eval()
    
        X_batch = ((X_input-np.mean(X_input))/np.std(X_input)).reshape(-1,1,224,224) # preprocess
        X_batch = _make_variable(X_batch)        
        y_proba = self.net(X_batch)
        y_proba = y_proba.cpu().float().data.numpy().reshape(-1,224,224)
        del X_batch

        return y_proba

    def predict(self, X, threashold=0.25):
        # threashold here makes sure that the hole circle is capture, including edges
        # TODO : threashold could be improved with grid-search
        X = X.reshape(-1,224,224)
        mean = np.mean(X, axis=(1,2))
        std = np.std(X, axis=(1,2))
        self.net.eval()
        
        output = np.empty(X.shape[0], dtype=object)
        for i in range(int(np.floor(X.shape[0]/self.batch_size))):
            idx = np.arange(X.shape[0])[i*self.batch_size:min((i+1)*self.batch_size,X.shape[0])] #batch
            X_batch = np.array([((X[i]-mean[i])/std[i]) for i in idx]).reshape(-1,1,224,224) # preprocess
            X_batch = _make_variable(X_batch)
                
            y_proba = self.net(X_batch).cpu()
            y_proba = y_proba.float().data.numpy().reshape(-1,224,224)
            
            output[idx] = self.sp.predict(y_proba) # convert to the desired output
            del X_batch
            del y_proba

        return output

    def _get_acc(self, y_pred, y_true):
        pred = (y_pred[:,0] > 0.5).float().cpu().data.numpy().reshape(-1,224*224)
        true = y_true[:,0].cpu().data.numpy().reshape(-1,224*224)
        out = []
        for i in range(len(y_true)):
            if (np.sum(pred[i]) != 0 and np.sum(true[i]) != 0):
                tn, fp, fn, tp = confusion_matrix(true[i],pred[i]).ravel()
                F1 = (tp*2)/(fn+fp+(tp*2))
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                out.append((F1,precision,recall))
            elif(np.sum(pred[i]) == 0 and np.sum(true[i]) == 0):
                out.append((1,1,1))
            else:
                out.append((0,0,0))
        return out
    
def _make_variable(X):
    return Variable(torch.from_numpy(X).float()).cuda()

def bar_graph(n, nb_chars=30):
    n = round(n*nb_chars)
    if n >= nb_chars:
        return "[" + ('=' * nb_chars) + "]"
    else:
        return "[" + ('=' * max(n,0)) + '>' + ('-' * max(nb_chars-n-1,0)) + "]"

def _edge_block(in_chan,mid,out_chan):
    return nn.Sequential(
        nn.Conv2d(in_chan, mid, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(mid),
        nn.Conv2d(mid, out_chan, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_chan)
    )
    
def _block(in_chan,layers,out_chan):
    return nn.Sequential(
        nn.Conv2d(in_chan, layers[0], 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(layers[0]),
        nn.Conv2d(layers[0], layers[1], 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(layers[1]),
        nn.Conv2d(layers[1], out_chan, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_chan),
    )
    
class Net(nn.Module): # based on the SegNet architecture with some of SSD's "feed-forward" design
    
    def __init__(self): 
        # was trained on gtx960 with 2G of VRAM. As such, feel free to increase the number of neurons per layer.
        super(Net, self).__init__()
        
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.d1 = _edge_block(1,8,16)
        self.d2 = _edge_block(16,24,32)
        self.d3 = _block(32,[48,64],80)
        self.d4 = _block(80,[128,128],256)
        self.d5 = _block(256,[512,512],512)
        
        self.u5 = _block(512,[512,512],256)
        self.u4 = _block(256,[128,128],80)
        self.u3 = _block(80,[64,48],32)
        self.u2 = _edge_block(32,24,16)
        self.u1 = _edge_block(16,8,1)
        
        self.fuse_scale = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5,1,1,padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 , indices_1 = self.pool(self.d1(x))
        x2 , indices_2 = self.pool(self.d2(x1))
        x3 , indices_3 = self.pool(self.d3(x2))
        x4 , indices_4 = self.pool(self.d4(x3))
        x5 , indices_5 = self.pool(self.d5(x4))
        
        x5 = self.u5(self.unpool(x5,indices_5))
        x5 = self.u4(self.unpool(x5,indices_4))
        x5 = self.u3(self.unpool(x5,indices_3))
        x5 = self.u2(self.unpool(x5,indices_2))
        x5 = self.u1(self.unpool(x5,indices_1))

        x4 = self.u4(self.unpool(x4,indices_4))
        x4 = self.u3(self.unpool(x4,indices_3))
        x4 = self.u2(self.unpool(x4,indices_2))
        x4 = self.u1(self.unpool(x4,indices_1))

        x3 = self.u3(self.unpool(x3,indices_3))
        x3 = self.u2(self.unpool(x3,indices_2))
        x3 = self.u1(self.unpool(x3,indices_1))

        x2 = self.u2(self.unpool(x2,indices_2))
        x2 = self.u1(self.unpool(x2,indices_1))

        x1 = self.u1(self.unpool(x1,indices_1))

        x = torch.cat((x1,x2,x3,x4,x5), dim=1) # SSD-like output

        return self.fuse_scale(x)

def to_image(y, scale=1): # turn (x,y,r) to a 2d mask
    xx, yy = np.mgrid[:(224*scale), :(224*scale)]
    out = np.zeros((len(y),(224*scale),(224*scale)),dtype=np.int)
    for i in range(len(y)):
        for j in range(len(y[i])):
            out[i,((xx - (y[i][j][0]*scale)) ** 2 + (yy - (y[i][j][1]*scale)) ** 2) <= ((y[i][j][2]*scale) ** 2)] = 1
            
    return out



###############################################################
############## the parser from mask to pred ###################
###############################################################

from sklearn.ensemble import RandomForestRegressor
from scipy.misc import imresize
from scipy.ndimage.measurements import center_of_mass
#import numpy as np

class Smart_parser:

    def __init__(self):
        self.parser_r = RandomForestRegressor(n_estimators=25, n_jobs=4)

    def fit(self, X_parser, y_parser):
        if X_parser.shape[0] == 0:
            print("Error, Smart_parser cannot fit on 0 samples!")
            return self

        filter_idx = np.not_equal(y_parser, None)

        self.parser_r.fit(X_parser[filter_idx][:,1:], [sub[2] for sub in y_parser[filter_idx]])
        return self
        
    def predict(self, X, return_proba=True):
        X_parser, loc = self._conv_parser(X, return_loc=True)

        if X_parser.shape[0] == 0:
            return [[] for x in X]

        img_id = X_parser[:,0]
        r_pred = self.parser_r.predict(X_parser[:,1:])

        y_out = np.empty(X.shape[0], dtype=object)
        for i in range(X.shape[0]):
            y_sub = []
            idx = (img_id == i)
            for k in range(idx.sum()):
                if r_pred[idx][k] > 1:
                    if return_proba:
                        y_sub.append((np.mean(X_parser[idx][k][3:]),
                                      loc[idx][k][0], 
                                      loc[idx][k][1], 
                                      r_pred[idx][k]))
                    else:
                        y_sub.append((loc[idx][k][0], 
                                      loc[idx][k][1], 
                                      r_pred[idx][k]))

            y_out[i] = y_sub

        return y_out

    def _get_dist(self, true, pred):
        return ((np.round((pred[0].stop+pred[0].start)/2-0.5) - true[0]) + 
            (np.round((pred[1].stop+pred[1].start)/2-0.5) - true[1]),0)

    def _conv_parser(self, pred_raw, y=None, return_loc=False, size=(12,12), frame_bias=10):
        X_parser = np.empty((0,1+2+size[0]*size[1]))
        if return_loc:
            loc = np.empty((0,2))
        if y is not None:
            y_parser = np.empty(0, dtype=object)

        for i in range(pred_raw.shape[0]):

            objs = find_objects(label(pred_raw[i]>0.5)[0])
            X_sub = np.empty((len(objs),1+2+size[0]*size[1]))
            X_sub[:,0] = i
            if return_loc:
                loc_sub = np.empty((len(objs),2))

            if y is not None:
                y_sub = np.empty(len(objs), dtype=object)
                y_sub[:] = None

                seen_objs = []
                if len(objs) > 0 :
                    for l in range(len(y[i])):
                        min_pred = (self._get_dist(y[i][l],objs[0]),0)
                        for k in range(1,len(objs)):
                            if k not in seen_objs:
                                diff = self._get_dist(y[i][l],objs[k])
                                if(diff < min_pred[0]):
                                    min_pred = (diff,k)
                        y_sub[min_pred[1]] = y[i][l]
                        seen_objs.append(min_pred[1])

            for k in range(len(objs)):
                coor_x_low = max(objs[k][0].start - frame_bias,0)
                coor_x_high = min(objs[k][0].stop + frame_bias,224)
                coor_y_low = max(objs[k][1].start - frame_bias,0)
                coor_y_high = min(objs[k][1].stop + frame_bias,224)

                X_sub[k,3:] = imresize(pred_raw[i][coor_x_low:coor_x_high,coor_y_low:coor_y_high],size).reshape(-1)

                center = center_of_mass(pred_raw[i][coor_x_low:coor_x_high,coor_y_low:coor_y_high])
                if return_loc:
                    loc_sub[k,0] = np.round(coor_x_low+center[0])
                    loc_sub[k,1] = np.round(coor_y_low+center[1])
                X_sub[:,1] = center[0]
                X_sub[:,2] = center[1]

            X_parser = np.concatenate((X_parser, X_sub))
            if return_loc:
                loc = np.concatenate((loc, loc_sub))
            if y is not None:
                y_parser = np.concatenate((y_parser, y_sub))

        if y is not None:
            if return_loc:
                return (X_parser, y_parser, loc)
            else:
                return (X_parser, y_parser)
        else:
            if return_loc:
                return (X_parser, loc)
            else:
                return X_parser
