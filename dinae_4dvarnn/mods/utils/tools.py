from dinae_4dvarnn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to create recursive paths
def mk_dir_recursive(dir_path):
    if os.path.isdir(dir_path):
        return
    h, t = os.path.split(dir_path)  # head/tail
    if not os.path.isdir(h):
        mk_dir_recursive(h)

    new_path = join_paths(h, t)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

# convert from masked_array to array replacing masks by 0.
def compress_masked_array(vals, axis=-1, fill=0.0):
    cnt = vals.mask.sum(axis=axis)
    shp = vals.shape
    num = shp[axis]
    mask = (num - cnt[..., np.newaxis]) > np.arange(num)
    n = fill * np.ones(shp)
    n[mask] = vals.compressed()
    return n

def Gradient(img, order):
    """ calculate x, y gradient and magnitude """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = sobelx/8.0
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = sobely/8.0
    sobel_norm = np.sqrt(sobelx*sobelx+sobely*sobely)
    if (order==0):
        return sobelx
    elif (order==1):
        return sobely
    else:
        return sobel_norm

# dim is the number of images in the time series
def gradient_imageTS(x):
    # Gradx
    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    kernel_weights=np.zeros((x.shape[1],x.shape[1],3,3))
    for i in range(x.shape[1]):
        kernel_weights[i,i,:,:]=a
    conv1=torch.nn.Conv2d(x.shape[1], x.shape[1], kernel_size=3, stride=1, padding=int(3/2), bias=False)
    conv1.weight=torch.nn.Parameter(torch.from_numpy(kernel_weights).float().to(device))
    G_x=conv1(x).data.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])

    # Grady
    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    kernel_weights=np.zeros((x.shape[1],x.shape[1],3,3))
    for i in range(x.shape[1]):
        kernel_weights[i,i,:,:]=b
    conv2=torch.nn.Conv2d(x.shape[1], x.shape[1], kernel_size=3, stride=1, padding=int(3/2), bias=False)
    conv2.weight=torch.nn.Parameter(torch.from_numpy(kernel_weights).float().to(device))
    G_y=conv2(x).data.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])

    # Grad
    G=torch.sqrt(torch.pow(G_x,2) + torch.pow(G_y,2))
    return G

# compute the along-track SSH gradient
def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        # returns False if target is greater than any value
        return -999
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def along_track_gradient_loss(obs,mask,itrp,time,id_sat):
    batch_size=obs.shape[0]
    for ibatch in range(batch_size):
        obs_    = obs[ibatch]
        mask_   = mask[ibatch]
        itrp_   = itrp[ibatch]
        time_   = time[ibatch]
        id_sat_ = id_sat[ibatch]
        # Get observation mask Omega
        mask1 = np.where(mask_.flatten()!=0.)[0]
        # Split Dataset by satellite
        list_sat = np.unique(id_sat_)[1:]
        res = 0.
        for i in range(len(list_sat)):
            mask2 = np.where(id_sat_.flatten()==list_sat[i])[0]
            mask12  = np.intersect1d(mask1,mask2)
            time_mask = time_.flatten()[mask12]
            gnn_time_mask = np.array([ find_nearest_above(time_mask,time_mask[index]) \
                       for index in range(len(time_mask))])
            # remove False index
            id_rm = np.where(gnn_time_mask==-999)[0]
            obs_mask = obs_.flatten()[mask12]
            itrp_mask = itrp_.flatten()[mask12]
            alt_grad_obs = np.delete(obs_mask,id_rm)-\
                        obs_mask[np.delete(gnn_time_mask,id_rm)]
            alt_grad_itrp = np.delete(itrp_mask,id_rm)-\
                        itrp_mask[np.delete(gnn_time_mask,id_rm)]
            res_tmp = torch.mean((alt_grad_obs-alt_grad_itrp)**2)
            if not torch.isnan(res_tmp):
                res+=res+res_tmp
    return res

def thresholding(x,thr):
    greater = K.greater_equal(x,thr) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater


