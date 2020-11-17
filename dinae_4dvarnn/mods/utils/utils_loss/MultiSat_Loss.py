from dinae_4dvarnn import *

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff.cpu(), 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        # returns False if target is greater than any value
        return -999
    masked_diff = np.ma.masked_array(diff.cpu(), mask)
    return masked_diff.argmin()

class MultiSat_Loss(torch.nn.Module):
    def __init__(self, N_sat):
        super(MultiSat_Loss, self).__init__()
        self.Nsat    = N_sat
        self._weights = torch.nn.Parameter(torch.ones(N_sat,device=device),requires_grad=True)
        self._regul   = torch.nn.Parameter(torch.ones(2,device=device),requires_grad=True)
        self.loss    = torch.nn.L1Loss().to(device)
        self.activation = torch.nn.Softmax(dim=0)

    @property
    def norm_weights(self):
        return self.activation(self._weights)

    @property
    def norm_regul(self):
        return self.activation(self._regul)

    def forward(self, obs, mask, itrp, time, IDsat, list_sat):

        # Get observation mask Omega
        mask_all      = torch.where(torch.flatten(mask)!=0.)[0].long()
        obs_      = torch.flatten(obs)[mask_all.long()]
        itrp_     = torch.flatten(itrp)[mask_all.long()]
        time_     = torch.flatten(time)[mask_all.long()]
        IDsat_    = torch.flatten(time)[mask_all.long()]
        # Split Dataset by satellite
        list_sat = torch.unique(list_sat,sorted=True)[1:]
        res_R = torch.zeros(1).to(device)
        res_G = torch.zeros(1).to(device)
        for i in range(len(list_sat)):
            mask         = torch.where(IDsat_==list_sat[i])[0].long()
            ratio = len(mask)/len(mask_all)
            # Keep only Observations of sat i
            obs_mask      = obs_[mask.long()]
            itrp_mask     = itrp_[mask.long()]
            time_mask     = time_[mask.long()]
            gnn_time_mask = torch.from_numpy(np.array([find_nearest_above(time_mask,time_mask[index]) \
                       for index in range(len(time_mask))])).long()
            # remove False index
            index = torch.where(gnn_time_mask!=-999.)[0]
            index = index.long()
            alt_grad_obs = obs_mask[index]-obs_mask[gnn_time_mask[index]]
            alt_grad_itrp = itrp_mask[index]-itrp_mask[gnn_time_mask[index]]
            # add weighted along-track gradient loss for satellite N.i
            index = list_sat[i]-1
            index = index.long()
            res_tmp = self.loss(alt_grad_obs,alt_grad_itrp)
            if res_tmp.numel()!=0:
                if torch.isnan(res_tmp):
                    res_tmp = torch.zeros(1)
            else:
                res_tmp = torch.zeros(1)
            res_tmp = res_tmp.to(device)
            # res_G = res_G + torch.mul(res_tmp,self.norm_weights[index])
            res_G = res_G + res_tmp
            # add weighted recontruction loss for satellite N.i
            index = list_sat[i]-1
            index = index.long()
            res_tmp = self.loss(obs_mask,itrp_mask)
            if res_tmp.numel()!=0:
                if torch.isnan(res_tmp):
                    res_tmp = torch.zeros(1)
            else:
                res_tmp = torch.zeros(1)
            res_tmp = res_tmp.to(device)
            #res_R = res_R + torch.mul(res_tmp,self.norm_weights[index])
            res_R = res_R + res_tmp
        #print(self.norm_weights)
        #print(self.norm_regul)
        loss = self.norm_regul[0]*res_R + self.norm_regul[1]*res_G
        loss = res_R
        return loss


