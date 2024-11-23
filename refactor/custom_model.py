import torch
import lightning as L
from models.unet_adaptive_bins import UnetAdaptiveBins
from loss import SILogLoss, BinsChamferLoss

class LitModel(L.LightningModule):
    def __init__(self, n_bins, min_val, max_val, norm):
        super().__init__()
        self.net = UnetAdaptiveBins.build(n_bins=n_bins, min_val=min_val, max_val=max_val, norm=norm)
        self.criterion_ueff = SILogLoss()
        self.criterion_bins = BinsChamferLoss()
        self.chamfer_weight = 0.0
    
    # Uncomment to find unused params        
    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(f"Unused: {name}")
    
    def training_step(self, batch, batch_idx):
        img = batch['image']
        depth = batch['depth']
        # depth_mask = batch['depth_mask']
        intrinsics = batch['intrinsics']
        sam_feats = batch['sam_feats']
        depth_mask = torch.logical_and(depth > self.net.min_val, depth < self.net.max_val)
    
        # # Long range augmentation
        # if random.random() < 0.2:
        #     img, depth, intrinsics = augment_long_range_tensors(img, depth, intrinsics, alpha=1.333)
        #     depth_mask = torch.logical_and(depth > args.min_depth, depth < args.max_depth)
        
        # TODO: Check if needed        
        # img = img.to(device)
        # depth = depth.to(device)
        # depth_mask = depth_mask.to(device)
        # intrinsics = intrinsics.to(device)
        
        bin_edges, pred = self.net(img, intrinsics, sam_feats)
        l_dense = self.criterion_ueff(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)

        if self.chamfer_weight > 0:
            l_chamfer = self.criterion_bins(bin_edges, depth)
        else:
            l_chamfer = torch.Tensor([0]).to(img.device)

        loss = l_dense + self.chamfer_weight * l_chamfer
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=0.1, lr=1e-4)
        return optimizer