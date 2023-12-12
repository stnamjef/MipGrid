import torch


class Loss:
    def __init__(
        self,
        model,
        lr_decay_factor,
        l1_weight_init,     # an initial weighting factor for the L1 regularization term
        l1_weight_rest,     # a final initial weighting factor for L1 reg. after updating the alpha mask
        ortho_weight,       # a weighting factor for the orthogonal regularization term
        tv_weight_den,      # a weighting factor for the total variance term w.r.t density grid
        tv_weight_app,      # a weighting factor for the total variance term w.r.t appearance grid
        end_lossmult = None
    ):
        self.model = model
        self.lr_decay_factor = lr_decay_factor
        self.l1_weight_init = l1_weight_init
        self.l1_weight_rest = l1_weight_rest
        self.ortho_weight = ortho_weight
        self.tv_weight_den = tv_weight_den
        self.tv_weight_app = tv_weight_app
        self.end_lossmult = end_lossmult

        self.l1_weight = l1_weight_init

    def _vector_diffs(self, channels, vecs):
        total = 0
        for ch, vec in zip(channels, vecs):
            vec = torch.squeeze(vec)
            dot = torch.matmul(vec, vec.T)
            off_diag = dot.view(-1)[1:].view(ch-1, ch+1)[:, :-1]
            total = total + torch.mean(torch.abs(off_diag))
        return total

    def _L1(self, mats, vecs):
        total = 0
        for mat, vec in zip(mats, vecs):
            total += torch.mean(torch.abs(mat)) + torch.mean(torch.abs(vec))
        return total
    
    def _TV(self, mats):
        def tensor_size(x):
            ch, m_size0, m_size1 = x.shape[1:]
            return ch * m_size0 * m_size1
        
        def total_variance(x):
            batch, _, h, w = x.shape
            count_h = tensor_size(x[:,:,1:,:])
            count_w = tensor_size(x[:,:,:,1:])
            h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h-1,:]), 2).sum()
            w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w-1]), 2).sum()
            return 2 * (h_tv/count_h + w_tv/count_w) / batch
        
        total = 0
        for mat in mats:
            total = total + total_variance(mat) * 1e-2
        
        return total
    
    def reset_l1_weight(self):
        self.l1_weight = self.l1_weight_rest
        print(f"  reset L1 reg. weight from {self.l1_weight_init} to {self.l1_weight_rest}")

    def accumulate_gradients(self, i, rgbs_pred, rgbs_gt, lossmults):
        # compute L2 loss
        if lossmults is not None and i <= self.end_lossmult:
            loss = torch.mean(lossmults * (rgbs_pred - rgbs_gt)**2)
        else:
            loss = torch.mean((rgbs_pred - rgbs_gt)**2)

        # orthogonal reg
        if self.ortho_weight > 0:
            ortho_reg = self._vector_diffs(self.model.den_channels, self.model.den_vecs) + \
                        self._vector_diffs(self.model.app_channels, self.model.app_vecs)
            loss = loss + self.ortho_weight * ortho_reg

        # L1 reg
        if self.l1_weight > 0:
            l1_reg = self._L1(self.model.den_mats, self.model.den_vecs)
            loss = loss + self.l1_weight * l1_reg

        # TV reg (density)
        if self.tv_weight_den > 0:
            self.tv_weight_den *= self.lr_decay_factor
            tv_reg_den = self._TV(self.model.den_mats)
            loss = loss + self.tv_weight_den * tv_reg_den
        
        # TV reg (appearance)
        if self.tv_weight_app > 0:
            self.tv_weight_app *= self.lr_decay_factor
            tv_reg_app = self._TV(self.model.app_mats)
            loss = loss + self.tv_weight_app * tv_reg_app

        loss_value = loss.detach().item()

        # backprop
        loss.backward()

        return loss_value
        
