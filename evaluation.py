import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from datasets.preprocess import process_scan
from utils.Gsupport import listdir_nohidden,normalize_radius,convert2GI,model_load,sub_sampling_blockwise
from datasets.load_data import load_H_extra
from utils.metrics import HausdorffDistance_incase,HausdorffDistance,Loss_all_batch,HausdorffDistance95,Loss_all_batch_each,DiceLoss_batch_each,cal_hd_hd95
from utils.Gsupport import listdir_nohidden,DiceLoss_batch
from model.GINet import GINet_
import json

import argparse

parser = argparse.ArgumentParser(description="Hyperparameters for the model")
parser.add_argument("--GPU_id", type=str, default="-1", help="ID for GPUs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--n_pc", type=int, default=4096, help="Number of point cloud")
parser.add_argument("--T_id", type=int, default=0, help="ID for T")
parser.add_argument("--seed", type=int, default=999, help="baseline test")
parser.add_argument("--batch", type=int, default=1, help="baseline test")
parser.add_argument("--con", type=bool, default=False, help="name add")
parser.add_argument("--start_epoch", type=int, default=0, help="name add")
parser.add_argument("--para", type=bool, default=False, help="name add")
parser.add_argument("--model", type=int, default=0, help="name add")
parser.add_argument("--position", type=int, default=1, help="name add")
parser.add_argument("--position_list", type=str, default="0,1", help="name add")
parser.add_argument("--id", type=int, default=0, help="name add")
parser.add_argument("--eval", type=bool, default=False, help="name add")
parser.add_argument("--H95", type=bool, default=False, help="name add")
parser.add_argument("--model_pth", type=str, default='T0_Files/SG_v5_T32.pth', help="name add")
parser.add_argument("--samples_per_block", type=int, default=1, help="baseline test")
parser.add_argument('--t_patch', type=int, default=0, help='Target patch index or identifier')


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

eval_all_metrics= args.eval
print('eval_all_metrics',eval_all_metrics ) 

H95 = args.H95
id_ = args.id
start_epoch = args.start_epoch
n_epochs = args.n_epochs

T_id = args.T_id
lr = args.lr
GPU_id = args.GPU_id
alpha0 = args.alpha
batch_size = args.batch
model_index = args.model
con = args.con
LR = lr
target_patch = args.t_patch

final_save_name = args.model_pth
print('model name:',model_index, final_save_name)
n_pc = args.n_pc
n_patch = int(n_pc**0.5)  

indices, points = sub_sampling_blockwise(
    n_regions_phi=n_patch,
    n_regions_theta=n_patch,
    sample_regions_phi=target_patch,
    sample_regions_theta=target_patch,
    samples_per_block=args.samples_per_block
)

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
center_t125 = np.load('data_utils/centerT125.npy')

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
center_T_pre = [center_t125[i] for i in range(len(center_t125))]

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

tumor_skull_path = []
tumor_brain_masks = []

for data_folder in listdir_nohidden(tumor_path_all):
    folder_path = tumor_path_all + '/' + data_folder
    
    files =[]
    for p_fold in listdir_nohidden(folder_path):
        files.append(p_fold)    
    tumor_skull_path.append(folder_path + '/' + 'T1C.nii.gz')
    if 'T1C_brain_mask_mc.nii.gz' in files:
        tumor_brain_masks.append(folder_path + '/' + 'T1C_brain_mask_mc.nii.gz')
    elif 'T1C_brain_mask_m.nii.gz' in files:
        tumor_brain_masks.append(folder_path + '/' + 'T1C_brain_mask_m.nii.gz')
    else:
        print('error  ',folder_path)
    
tumor_skull_path_order = []
tumor_brain_masks_order = []

for j in range(len(tumor_patients)):
    for i in range(len(tumor_skull_path)):
        if tumor_patients[j] == tumor_skull_path[i].split('/')[-2]:
            tumor_skull_path_order.append(tumor_skull_path[i])
    for i in range(len(tumor_brain_masks)):
        if tumor_patients[j] == tumor_brain_masks[i].split('/')[-2]:
            tumor_brain_masks_order.append(tumor_brain_masks[i])
tumor_skull_path = tumor_skull_path_order
tumor_brain_masks = tumor_brain_masks_order  
print("Num of Healthy: {}, Tumor: {}" .format(len(skull_brain_path),len(tumor_skull_path)))  

class myDataset(Dataset):
    def __init__(self, skull_brain_path, brain_mask, g_map=None,N_H = None, T_all =False, center_pre=None):
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
        if self.g_map!=None:
            GI_all = self.g_map[i][indices,:]
        else:
            GI_all =  None

        data = (torch.tensor(x).contiguous(),torch.tensor(y).contiguous(), torch.tensor(GI_all).contiguous())
        return data

skull_T_train = []
skull_T_test = []
skull_T_unseen = []

mask_T_train =[]
mask_T_test =[]
mask_T_unseen =[]

GI_T_train = []
GI_T_test = []

center_T_train_pre = []
center_T_test_pre =[]
center_T_unseen_pre =[]

n_H_Train = int(len(skull_brain_path)*0.7)
n_H_Test = len(skull_brain_path[int(len(skull_brain_path)*0.7):])

print('n_H_Train: {}, n_H_Test: {}'.format(n_H_Train,n_H_Test))

unseen_id = [1,2,3,4,5]

if T_id <= 0:
    pass
else:
    for i in range(T_id):

        skull_T_train += tumor_skull_path[25*(i):25*(i+1)][:17]
        mask_T_train += tumor_brain_masks[25*(i):25*(i+1)][:17]
        center_T_train_pre +=center_T_pre[25*(i):25*(i+1)][:17]

        #######   #######   #######   #######  #######  #######  
        skull_T_test += tumor_skull_path[25*(i):25*(i+1)][17:]
        mask_T_test += tumor_brain_masks[25*(i):25*(i+1)][17:]
        center_T_test_pre +=center_T_pre[25*(i):25*(i+1)][17:]
        
        unseen_id.remove(i+1)
        
for j in unseen_id:
    i = j-1
    skull_T_unseen += tumor_skull_path[25*(i):25*(i+1)]
    mask_T_unseen += tumor_brain_masks[25*(i):25*(i+1)]
    center_T_unseen_pre +=center_T_pre[25*(i):25*(i+1)]       
        
skull_train_final  = skull_brain_path[:int(len(skull_brain_path)*0.7)] + skull_T_train
skull_test_final  =  skull_brain_path[int(len(skull_brain_path)*0.7):] + skull_T_test

brain_train_final  = brain_masks[:int(len(skull_brain_path)*0.7)] + mask_T_train
brain_test_final  =  brain_masks[int(len(skull_brain_path)*0.7):] + mask_T_test

GI_train_final  = GI_H[:int(len(skull_brain_path)*0.7)] + GI_T_train
GI_test_final  =  GI_H[int(len(skull_brain_path)*0.7):] + GI_T_test

center_train_final_pre  = center_H_pre[:int(len(skull_brain_path)*0.7)] +  center_T_train_pre
center_test_final_pre  =  center_H_pre[int(len(skull_brain_path)*0.7):] +  center_T_test_pre


train_dataset = myDataset(skull_train_final,brain_train_final,g_map = GI_train_final,N_H=n_H_Train,center_pre=center_train_final_pre )
train_loader = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True)

test_dataset = myDataset(skull_test_final,brain_test_final,g_map=None,N_H=n_H_Test,center_pre=center_test_final_pre  )
test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

unseen_dataset = myDataset(skull_T_unseen,mask_T_unseen, g_map =None, T_all =True,center_pre=center_T_unseen_pre)
unseen_loader = DataLoader(unseen_dataset,  batch_size=batch_size, shuffle=False)

print('nums in training | test | unseen |  are:',train_loader.__len__()*batch_size, test_loader.__len__()*batch_size, unseen_loader.__len__()*batch_size)

from datasets.load_data import load_H_extra

skull_paths_LBPA40, brain_mask_paths_LBPA40, skull_paths_NFBS, brain_mask_paths_NFBS =load_H_extra('Health_extra')
center_LBPA_pre = np.load('data_utils/center_LBPA_pre.npy')
center_NFBS_pre = np.load('data_utils/center_NFBS_pre.npy')
center_NFBS_pre = [center_NFBS_pre[i] for i in range(len(center_NFBS_pre))]
center_LBPA_pre = [center_LBPA_pre[i] for i in range(len(center_LBPA_pre))]


NFBS_dataset = myDataset(skull_paths_NFBS, brain_mask_paths_NFBS, g_map =None,center_pre=center_NFBS_pre)
NFBS_loader = DataLoader(NFBS_dataset,  batch_size=batch_size, shuffle=False)

LBPA_dataset = myDataset(skull_paths_LBPA40, brain_mask_paths_LBPA40,  g_map =None,center_pre=center_LBPA_pre)
LBPA_loader = DataLoader(LBPA_dataset,  batch_size=batch_size, shuffle=False)
print('model name:',final_save_name)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

coors_tem  = torch.tensor(GI_tem_in).contiguous().type(Tensor)
coors_tem = torch.transpose(coors_tem , 1, 2)[:,:,indices]


print('from eval module')


Gmodel = GINet_(method ='gate_v2',set_sdf=False,n_pc=int(args.t_patch**2))

model_load(Gmodel,final_save_name)

if para and device_ids:
    print('ids',device_ids)
    Gmodel = torch.nn.DataParallel(Gmodel, device_ids=device_ids)
if cuda:
    Gmodel.cuda()
    r_mean =  torch.tensor(r_mean).contiguous().type(Tensor)
    r_std =  torch.tensor(r_std).contiguous().type(Tensor)


if H95:
    compute_hausdorff_distance = HausdorffDistance95()
else:
    compute_hausdorff_distance = HausdorffDistance()

chd_incase = HausdorffDistance_incase()



def criterion(pred, gt, eval_all_metrics=False):
    metrics = {}
    if eval_all_metrics:

        precision, recall, FPR, FNR = Loss_all_batch()(pred, gt)

        try:
            HD = compute_hausdorff_distance.compute(pred, gt)
        except:
            
            HD = chd_incase.compute(pred, gt)
            print('error HD', HD)

        metrics.update({
            'precision': precision.item(),
            'recall': recall.item(),
            'FPR': FPR.item(),
            'FNR': FNR.item(),
            'HD': HD.mean().item()

        })


    dice = DiceLoss_batch()(pred, gt)
    metrics['dice'] = dice.item()

    return metrics



def criterion_each(pred, gt, eval_all_metrics=False):
    metrics = {}
    if eval_all_metrics:
        # Assume Loss_all_batch_each is modified according to your previous description
       

        precision, recall, FPR, FNR = Loss_all_batch_each()(pred, gt)

        HD = cal_hd_hd95(pred,gt)
        # Store tensors directly in the dictionary
        metrics.update({
            'precision': precision,
            'recall': recall,
            'FPR': FPR,
            'FNR': FNR,
            'HD': HD
        })
    
    # Compute Dice for each batch item
    dice = DiceLoss_batch_each()(pred, gt)
    metrics['dice'] = dice

    return metrics

def evaluate_model(model, data_loader, eval_all_metrics=False, base=False):
    results = {
        'dice': [],
        'precision': [],
        'recall': [],
        'FPR': [],
        'FNR': [],
        'HD': []
    }
    with torch.no_grad():
        for skull_brain, brain_gt, _ in data_loader:
            B = skull_brain.shape[0]
            coors_tem_in = coors_tem.repeat(B, 1, 1)
            coors_tem_in[:, :3, :]  = Tensor(coors_tem_in[:, :3, :] / 191.0).requires_grad_(False)
            label_img = brain_gt.type(Tensor)
            img = skull_brain.type(Tensor)
            center =  Tensor(center_g).unsqueeze(0).repeat(B, 1)
            _, y_pred, _ = model(img, coors_tem_in, center)


            metrics = criterion_each(y_pred, label_img, eval_all_metrics=eval_all_metrics)
            for k, v in metrics.items():
                if torch.is_tensor(v):
               
                    results[k].extend(v.tolist())  # Convert tensor to list and extend the corresponding results list
                elif isinstance(v, list):
           
                    results[k].extend(v) 
                else:
                    results[k].append(v)  # Handling non-tensor values (if any)

    # Filter out empty lists and convert lists to tensors if desired
    return {k: torch.tensor(v) for k, v in results.items() if v}


def repeat_evaluation(model, data_loaders, model_name, is_base=False, num_repeats=5, eval_all_metrics=False):

    results = {name: {} for name in data_loaders}
    total_times = {}
    for name, loader in data_loaders.items():
        start_time = time.time()
        for repeat in tqdm(range(num_repeats), desc=f"Progress for {model_name} - {name}", unit="repeat"):
            metrics = evaluate_model(model, loader, base=is_base, eval_all_metrics=eval_all_metrics)
            for k, v in metrics.items():
                if k not in results[name]:
                    results[name][k] = []
                
                results[name][k].extend(v.tolist())
                
        total_times[name] = time.time() - start_time
    return results, total_times

def print_evaluation_results(results, times):
    for model_name, datasets in results.items():
        print(f"Results for {model_name}:")
        for dataset_name, metrics in datasets.items():
            print(f'Results for {dataset_name}:')
            for metric, values in metrics.items():
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()
                    print('transfered into cpu')
                mean, std = np.mean(values), np.std(values)
                print(f'  {metric}: Mean = {mean:.7f}, Std = {std:.7f}')
            print(f'  Total time for {model_name} on {dataset_name}: {times[model_name][dataset_name]:.2f} seconds')


models = {'model': (Gmodel, False)}
data_loaders = {'CC359': test_loader, 'GBM125': unseen_loader, 'NFBS125': NFBS_loader,'LPBA40': LBPA_loader}

all_results = {}
all_times = {}
for model_name, (model, is_base) in models.items():
    model.cuda()  # Adjust according to the device available
    results, times = repeat_evaluation(model, data_loaders, model_name, is_base, eval_all_metrics=eval_all_metrics)
    all_results[model_name] = results

    all_times[model_name] = times


with open('all_results.json', 'w') as f:
    json.dump(all_results, f)

print_evaluation_results(all_results, all_times)