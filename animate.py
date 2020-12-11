from netCDF4 import Dataset
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def gradient_norm(a):
    return np.array(np.linalg.norm(np.gradient(a),axis=0))
def Animate_SSH(file1):
    nc_data_test = Dataset(file1, "r")
    gt_test = np.array(nc_data_test['gt_test'])
    x_test_missing = np.array(nc_data_test['x_test_missing'])
    mask_test = np.array(nc_data_test['mask_test'])
    x_test_pred = np.array(nc_data_test['x_test_pred'])
    rec_AE_Tt = np.array(nc_data_test['rec_AE_Tt'])
    x_test_OI = np.array(nc_data_test['x_test_OI'])

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.title.set_text('GT')
    ax2.title.set_text('Obs')
    ax3.title.set_text('Pred')
    ax4.title.set_text('OI')
    plt.set_cmap('ocean')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ims = []
    for ind in range(len(gt_test)):
        a1=ax1.imshow(gt_test[ind])
        a2=ax2.imshow(np.where(mask_test[ind,:,:].squeeze()==0, np.nan, gt_test[ind,:,:].squeeze()))
        a3=ax3.imshow(x_test_pred[ind])
        a4=ax4.imshow(x_test_OI[ind])
        ims.append([a1,a2,a3,a4])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    ani.save('Animation_SSH.gif', writer='pillow', fps=10)
    plt.show()
    return 1

def Animate_SSHGrad(file1):
    nc_data_test = Dataset(file1, "r")

    gt_grad_test = gradient_norm(np.array(nc_data_test['gt_test']))
    gt_test = np.array(nc_data_test['gt_test'])
    mask_test = np.array(nc_data_test['mask_test'])
    x_grad_test_missing = gradient_norm(np.array(nc_data_test['x_test_missing']))
    x_grad_test_pred = gradient_norm(np.array(nc_data_test['x_test_pred']))
    rec_AE_grad_Tt = gradient_norm(np.array(nc_data_test['rec_AE_Tt']))
    x_grad_test_OI = gradient_norm(np.array(nc_data_test['x_test_OI']))

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.title.set_text('GT')
    ax2.title.set_text('Obs')
    ax3.title.set_text('Pred')
    ax4.title.set_text('OI')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ims = []
    for ind in range(len(gt_grad_test)):
        a1=ax1.imshow(gt_grad_test[ind])
        a2=ax2.imshow(gradient_norm(np.where(mask_test[ind,:,:].squeeze()==0, np.nan, gt_test[ind,:,:].squeeze())))
        a3=ax3.imshow(x_grad_test_pred[ind])
        a4=ax4.imshow(x_grad_test_OI[ind])
        ims.append([a1,a2,a3,a4])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    ani.save('Animation_SSHGrad.gif', writer='pillow', fps=10)
    plt.show()
    return 1


Animate_SSHGrad("/home/administrateur/Desktop/test1.nc")