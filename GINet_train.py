import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore")
from datasets.preprocess import process_scan
from utils.Gsupport import listdir_nohidden,normalize_radius,convert2GI,model_load,DiceLoss,sub_sampling_blockwise
from model.GINet import GINet_

import argparse


parser = argparse.ArgumentParser(description="Hyperparameters for the model")

parser.add_argument("--GPU_id", type=str, default="-1", help="ID for GPUs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--n_pc", type=int, default=4096, help="Number of point cloud")
parser.add_argument("--T_id", type=int, default=0, help="ID for T")
parser.add_argument("--seed", type=int, default=999, help="baseline test")
parser.add_argument("--batch", type=int, default=1, help="baseline test")
parser.add_argument("--samples_per_block", type=int, default=1, help="baseline test")
parser.add_argument("--con", type=bool, default=False, help="name add")
parser.add_argument("--t_patch", type=int, default=64, help="name add")
parser.add_argument("--start_epoch", type=int, default=0, help="name add")
parser.add_argument("--para", type=bool, default=False, help="name add")
parser.add_argument("--set_sdf", type=bool, default=False, help="name add")
parser.add_argument("--model", type=int, default=0, help="name add")
parser.add_argument("--position", type=int, default=1, help="name add")
parser.add_argument("--position_list", type=str, default="0,1", help="name add")
parser.add_argument("--id", type=int, default=0, help="name add")
parser.add_argument("--simple", type=bool, default=False, help="name add")

args = parser.parse_args()
seed = args.seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)
position_list = args.position_list 
position_list = [int(x) for x in position_list.split(',')]
position = args.position
para = args.para

set_sdf = args.set_sdf
id_ = args.id
start_epoch = args.start_epoch
n_epochs = args.n_epochs
n_pc = args.n_pc
T_id = args.T_id
lr = args.lr
GPU_id = args.GPU_id
alpha0 = args.alpha
batch_size = args.batch
model_index = args.model
con = args.con
LR = lr
simple = args.simple
target_patch = args.t_patch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_pc = args.n_pc
n_patch = int(n_pc**0.5)  

indices, points = sub_sampling_blockwise(
    n_regions_phi=n_patch,
    n_regions_theta=n_patch,
    sample_regions_phi=target_patch,
    sample_regions_theta=target_patch,
    samples_per_block=args.samples_per_block
)


final_save_name = 'T0_Files/SG.pth'
print('model name:',model_index, final_save_name)


beta1 = 0.5
beta2 = 0.999

if GPU_id !='-1':
    print('using GPU: {}'.format(GPU_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id

    device_ids = list(np.arange(len(GPU_id)//2+1))
    device_ids = [int(device_ids[i]) for i in device_ids]


# data preprocessing
tumor_path_all ='tumor_data'
tumor_patients = np.load('data_utils/tumor_patients.npy')
patients=np.load('data_utils/H_patients.npy')
# for center crop 
center_h359 = np.load('data_utils/center359.npy')

GI_H = np.load('data_utils/cGI_H_{}rpt_preC_noc.npy'.format(n_pc))
g_tem = np.load('data_utils/cGI_tem_{}rpt.npy'.format(n_pc))
center_g =np.load('data_utils/gcenter_tem_{}rpt.npy'.format(n_pc)) 


g_tem = np.expand_dims(g_tem, axis=0)
g_tem_nor = (g_tem - center_g[np.newaxis,np.newaxis,:]) #  (1,n_pc,3)
GI_H_nor = GI_H - center_g[np.newaxis,np.newaxis,:]
GI_tem_in  = convert2GI(g_tem_nor,n_patch)
GI_H_in  =  convert2GI(GI_H_nor,n_patch)
GI_tem_in[:,:,:3] = GI_tem_in[:,:,:3] + center_g[np.newaxis,np.newaxis,:]
GI_H_in[:,:,:3] = GI_H_in[:,:,:3] + center_g[np.newaxis,np.newaxis,:]
normal_min_r = np.min(GI_H_in [:, :, 3])
normal_max_r = np.max(GI_H_in [:, :, 3])
GI_tem_in  = normalize_radius(GI_H_in,GI_tem_in ,True)
GI_H_in= normalize_radius(GI_H_in)


r_mean = GI_H_in[:,indices,3].mean(axis=0)
r_std = GI_H_in[:,indices,3].std(axis=0)
center_H_pre = [center_h359[i] for i in range(len(center_h359))]

GI_H = [GI_H_in[i] for i in range(GI_H_in.shape[0])]
images_path = 'H_data' + '/' + 'Original'
labels_path =  'H_data' + '/' + 'STAPLE'

skull_brain_path = []
brain_masks = []

for p_fold in listdir_nohidden(images_path):
    patient_id = p_fold[:-7]
    if len(patient_id)>2 and 'nii.gz' in p_fold:
        BrainSkull_path = images_path + '/' + patient_id + '.nii.gz'
        Brain_mask = labels_path + '/' + patient_id + '_staple.nii.gz'
        skull_brain_path.append(BrainSkull_path) 
        brain_masks.append(Brain_mask) 

                
skull_brain_path_order = []
for j in range(len(patients)):
    for i in range(len(skull_brain_path)):

        if patients[j] in skull_brain_path[i].split('/')[-1].split('.')[0]:
            skull_brain_path_order.append(skull_brain_path[i])
brain_masks_order = []
for j in range(len(patients)):
    for i in range(len(brain_masks)):

        if  patients[j] in brain_masks[i].split('/')[-1].split('.')[0]:   
     
            brain_masks_order.append(brain_masks[i])     
            
skull_brain_path = skull_brain_path_order
brain_masks = brain_masks_order

class myDataset(Dataset):
    def __init__(self, skull_brain_path, brain_mask, g_map ,N_H = None, T_all =False, center_pre=None):
        self.x_path = skull_brain_path
        self.gt_path = brain_mask
        self.g_map = g_map
        self.center_pre = center_pre
        self.T_all = T_all
        self.N_H = N_H
    def __len__(self):
        return len(self.gt_path)
    def __getitem__(self, i):

        shapes_ =   (192,192,192)
        reshapes_ = (1,192,192,192)     

        x = process_scan(self.x_path[i],norm_method = 'zs',output_shape=shapes_,center=self.center_pre[i])
        y = process_scan(self.gt_path[i],mask=True,output_shape=shapes_,center=self.center_pre[i])
     
        x = x.reshape(reshapes_)   
        y = y .reshape(reshapes_)
       
        GI_all = self.g_map[i][indices,:]
      
        data = (torch.tensor(x).contiguous(),torch.tensor(y).contiguous(), torch.tensor(GI_all).contiguous())
        return data


n_H_Train = int(len(skull_brain_path)*0.7)
n_H_Test = len(skull_brain_path[int(len(skull_brain_path)*0.7):])

print('n_H_Train: {}, n_H_Test: {}'.format(n_H_Train,n_H_Test))

unseen_id = [1,2,3,4,5]
     
        
skull_train_final  = skull_brain_path[:int(len(skull_brain_path)*0.7)] 
skull_test_final  =  skull_brain_path[int(len(skull_brain_path)*0.7):] 

brain_train_final  = brain_masks[:int(len(skull_brain_path)*0.7)] 
brain_test_final  =  brain_masks[int(len(skull_brain_path)*0.7):] 

GI_train_final  = GI_H[:int(len(skull_brain_path)*0.7)] 
GI_test_final  =  GI_H[int(len(skull_brain_path)*0.7):] 



center_train_final_pre  = center_H_pre[:int(len(skull_brain_path)*0.7)] 
center_test_final_pre  =  center_H_pre[int(len(skull_brain_path)*0.7):]


train_dataset = myDataset(skull_train_final,brain_train_final,GI_train_final,n_H_Train,center_pre=center_train_final_pre )
train_loader = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = myDataset(skull_test_final,brain_test_final,GI_test_final,n_H_Test,center_pre=center_test_final_pre  )
test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)


print('nums in training | test | unseen |  are:',train_loader.__len__()*batch_size, test_loader.__len__()*batch_size)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

coors_tem  = torch.tensor(GI_tem_in).contiguous().type(Tensor)
coors_tem = torch.transpose(coors_tem , 1, 2)[:,:,indices]


model = GINet_(method ='gate_v2',set_sdf=set_sdf,n_pc=int(args.t_patch**2))

lr_r = lr
lr_d = lr

if model_index in [1,2,3,4,5]:
    params_r = (
        [p for p in model.convd_list.parameters()] +
        [p for p in model.convu_list.parameters()] +
        [p for p in model.pc_evolve.parameters()]  # Assuming pc_evolve affects g_in generation
    )

    # Parameters for optimizer_d (refined layers)
    params_d = (
        [p for p in model.convd_list_refined.parameters()] +
        [p for p in model.convu_list_refined.parameters()] +
        [p for p in model.refine_list.parameters()]
    )

    # Initialize separate optimizers
    optimizer_r = AdamW(params_r, lr=lr_r, betas=(beta1, beta2), weight_decay=0.00005)
    optimizer_d = AdamW(params_d, lr=lr_d, betas=(beta1, beta2), weight_decay=0.00005)
else:
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2), weight_decay=0.00005)


from utils.Gatten_support import model_load
if con==True:
    model_load(model,final_save_name)
    lr = lr*0.1
    LR=lr
if eval==True:
    model_load(model,final_save_name)
if para and device_ids:
    print('ids',device_ids)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
if cuda:
    model.cuda()
    r_mean =  torch.tensor(r_mean).contiguous().type(Tensor)
    r_std =  torch.tensor(r_std).contiguous().type(Tensor)

def custom_loss(r, r_gt, r_mean=r_mean, r_std=r_std):
    # Ensure that r_mean and r_std are not zero to avoid division by zero
    epsilon = 1e-8
    weights = r_mean * (1 + r_std) + epsilon
    loss = ((r - r_gt)**2) / weights
    return torch.sqrt(torch.mean(loss)) 

label_pooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
dice =  DiceLoss()
def custom_loss_dice(y_pre, ys_pre, y_gt, r=0.8):
    # Compute the coefficients for weighted loss
    a = (1 - r) / (1 - r ** 6)
    coeff = [a * r ** i for i in range(6)]

    ys_gt_list = [y_gt]
    for i in range(5):
        ys_gt = label_pooling(ys_gt_list[-1])  # Use the last item in the list for pooling
        ys_gt_list.append(ys_gt)

    # Calculate loss for the original scale
    loss = coeff[0] * dice(y_pre, y_gt)
    # Calculate loss for downscaled predictions
    for i, (pre, gt) in enumerate(zip(ys_pre, ys_gt_list[::-1])):
        
        loss = loss + coeff[5 - i] * dice(pre, gt)
    return loss


loss_MSE = custom_loss
loss_Dice =custom_loss_dice

def adjust_learning_rate(optimizer, LR, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = LR * ((1-epoch/n_epochs ).__pow__(0.9))
        param_group['lr'] = lr
        print('...working on epoch: {}/{} with learning rate: {:.7f}'.format(epoch,
              n_epochs , param_group['lr']))
    # alpha = alpha0* ((1-epoch/n_epochs ).__pow__(0.9))




eval_Dice = DiceLoss()
loss_eval_list = []


loss_R_list = []
loss_CD_list =[]
loss_Dice_list = []


stop_flag = 0
eval_best_c = float('inf')
for epoch in range(start_epoch, n_epochs):
    torch.cuda.empty_cache()
    adjust_learning_rate(optimizer_r, LR, epoch)  # adjust lr
    adjust_learning_rate(optimizer_d, LR, epoch)  # adjust lr


    for i, (skull_brain,brain_gt,GI_all) in enumerate(train_loader):
        B = skull_brain.shape[0]
        coors_tem_in = coors_tem.repeat(B,1,1)
        coors_tem_in[:, :3, :]  = Tensor(coors_tem_in[:, :3, :] / 191.0).requires_grad_(False)
        center =  Tensor(center_g).unsqueeze(0).repeat(B, 1)
        img = skull_brain.type(Tensor)

        label_r = GI_all[:,:,3].type(Tensor)
        # label_coor = (GI_all[:,:,:3]/191.0).type(Tensor)
        label_img = brain_gt.type(Tensor)

        r_out, y_refined, ys_pre = model(img, coors_tem_in, center)

        loss_r = loss_MSE(r_out.view(B,-1),label_r.view(B,-1))
        if epoch>20:
            loss_d =  loss_Dice(y_refined,ys_pre,label_img)
            loss_r.backward()
            optimizer_r.step()
            optimizer_r.zero_grad()

            loss_d.backward()
            optimizer_d.step()
            optimizer_d.zero_grad()
        else:
            loss_r.backward()
            optimizer_r.step()
            optimizer_r.zero_grad()

        loss_R_list.append(loss_r.detach().cpu().item())
        loss_Dice_list.append(loss_d.detach().cpu().item() if epoch > 50 else 0)

        if i %40 ==0:
            if epoch>20:
                print('loss_r {:.5f}; loss_dice {:.5f}'.format(loss_r.item(),loss_d.item()))
            else:
                print('loss_r {:.5f}'.format(loss_r.item()))
    if epoch % 10 == 0:
        with torch.no_grad():
            eval_l = []
            loss_r_list = []
            loss_dice_list = []
            loss_dice_list_eval = []
            for i, (skull_brain, brain_gt, GI_all) in enumerate(test_loader):
                B = skull_brain.shape[0]
                coors_tem_in = coors_tem.repeat(B, 1, 1)
                coors_tem_in[:, :3, :]  = Tensor(coors_tem_in[:, :3, :] / 191.0).requires_grad_(False)
                center =  Tensor(center_g).unsqueeze(0).repeat(B, 1)
                img = skull_brain.type(Tensor)

                label_r = GI_all[:, :, 3].type(Tensor)
                # label_coor = (GI_all[:, :, :3] / 191.0).type(Tensor)
                label_img = brain_gt.type(Tensor)

                r_out, y_refined,ys_pre = model(img, coors_tem_in, center)

                loss_r = loss_MSE(r_out.view(B, -1), label_r.view(B, -1))
                if epoch > 20:
                    loss_d = loss_Dice(y_refined,ys_pre,label_img)
                    eval_d = eval_Dice(y_refined,label_img)
                eval_l.append( loss_r.detach().cpu().item() + loss_d.detach().cpu().item() if epoch > 20 else loss_r.detach().cpu().item())
                loss_r_list.append(loss_r.detach().cpu().item())
                loss_dice_list.append(loss_d.detach().cpu().item()  if epoch > 20 else 0)
                loss_dice_list_eval.append(eval_d.detach().cpu().item()  if epoch > 20 else 0)


            eval_c_H = np.mean(eval_l)
            avg_loss_r_H = np.mean(loss_r_list)
            avg_loss_dice_H = np.mean(loss_dice_list)
            avg_loss_dice_H_eval = np.mean(loss_dice_list_eval)

            eval_c =  eval_c_H

            loss_eval_list.append(eval_c)
            print('model name:',model_index, final_save_name)
            print(f"Epoch {epoch},  Healthy Loss: {eval_c_H}, H Loss R: {avg_loss_r_H}, H Loss CD: 0, H Loss Dice: {avg_loss_dice_H}")

            print('evaluation', eval_c)
            print(f"Real Dice  H :  {avg_loss_dice_H_eval}")

            if (eval_c > eval_best_c) and (epoch>20):
                stop_flag += 1
            elif (eval_c <= eval_best_c) and (epoch>20):
                print('saving current model')
                eval_best_c = eval_c  # Update the best evaluation loss
                stop_flag = 0  # Reset stop flag since we improved
                torch.save(model.state_dict(), final_save_name)
            if epoch==20:
                torch.save(model.state_dict(), final_save_name.replace('SG','pre_SG'))
    if stop_flag > 6:  # If no improvement for more than one checkpoint
        print("Stopping early due to no improvement.")
        break

