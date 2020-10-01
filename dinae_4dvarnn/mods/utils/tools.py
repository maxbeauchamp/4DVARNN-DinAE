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

def thresholding(x,thr):
    greater = K.greater_equal(x,thr) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater


