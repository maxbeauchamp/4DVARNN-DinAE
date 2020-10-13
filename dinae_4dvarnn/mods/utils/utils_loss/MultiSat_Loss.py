
def torch_delete(tensor,index):


class MultiSat_Loss(torch.nn.Module):
    def __init__(self, N_sat):
        super(MultiSatAndTime_Loss, self).__init__()
        self.lambda = torch.nn.Parameter(torch.ones(N_sat))
        self.loss = torch.nn.L1Loss()

    def forward(self, obs, mask, itrp, time, IDsat):

	# Get observation mask Omega
        mask1 = torch.where(torch.flatten(mask)!=0.)
        # Split Dataset by satellite
        list_sat = torch.unique(IDsat,sorted=True)[1:]
        res = []
        for i in range(len(list_sat)):
            mask2 = torch.where(torch.flatten(IDsat)==list_sat[i])[0]
            mask12  = np.intersect1d(mask1,mask2)
            time_mask = torch.flatten(time)[mask12]
            gnn_time_mask = np.array([ find_nearest_above(time_mask,time_mask[index]) \
                       for index in range(len(time_mask))])
            # remove False index
            obs_mask = torch.flatten(obs)[mask12]
            itrp_mask = torch.flatten(itrp)[mask12]
            alt_grad_obs = torch.empty()
            alt_grad_obs = torch.empty()
            for j in range(len(gnn_time_mask)):
                if gnn_time_mask[j]!=-999:

            alt_grad_obs = obs_mask-obs_mask[gnn_time_mask]
            alt_grad_itrp = np.delete(itrp_mask,id_rm)-\
                        itrp_mask[np.delete(gnn_time_mask,id_rm)]
            res.append(self.loss(alt_grad_obs-alt_grad_itrp))
	res = np.nanmean(res/self.lambda)
        return res
