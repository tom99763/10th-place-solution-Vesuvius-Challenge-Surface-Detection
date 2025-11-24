import torch
import torch.nn.functional as F

def gradient_loss(disp):
    dz = torch.abs(disp[:,:,1:,:,:] - disp[:,:,:-1,:,:])
    dy = torch.abs(disp[:,:,:,1:,:] - disp[:,:,:,:-1,:])
    dx = torch.abs(disp[:,:,:,:,1:] - disp[:,:,:,:,:-1])
    return (dz.mean() + dy.mean() + dx.mean())

def dice_coef(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(1,2,3,4))
    sums = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
    dice = (2.0 * intersection + eps) / (sums + eps)
    return dice.mean()

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_coef(pred, target)
    return bce + (1.0 - dice)

def jacobian_determinant(displacement):
    disp = displacement
    def diff_axis(t, axis):
        if axis == 0: return t[:,:,1:,:,:] - t[:,:,:-1,:,:]
        if axis == 1: return t[:,:,:,1:,:] - t[:,:,:,:-1,:]
        if axis == 2: return t[:,:,:,:,1:] - t[:,:,:,:,:-1]
    dz = diff_axis(disp, 0)
    dy = diff_axis(disp, 1)
    dx = diff_axis(disp, 2)
    dz = dz[:,:,:,:-1,:-1]
    dy = dy[:,:,:-1,:,:-1]
    dx = dx[:,:,:-1,:-1,:]
    d_dispx_dx = dx[:,0,:,:,:]; d_dispx_dy = dy[:,0,:,:,:]; d_dispx_dz = dz[:,0,:,:,:]
    d_dispy_dx = dx[:,1,:,:,:]; d_dispy_dy = dy[:,1,:,:,:]; d_dispy_dz = dz[:,1,:,:,:]
    d_dispz_dx = dx[:,2,:,:,:]; d_dispz_dy = dy[:,2,:,:,:]; d_dispz_dz = dz[:,2,:,:,:]
    J11 = 1.0 + d_dispx_dx; J12 = d_dispx_dy; J13 = d_dispx_dz
    J21 = d_dispy_dx; J22 = 1.0 + d_dispy_dy; J23 = d_dispy_dz
    J31 = d_dispz_dx; J32 = d_dispz_dy; J33 = 1.0 + d_dispz_dz
    det = (J11*(J22*J33 - J23*J32)
           - J12*(J21*J33 - J23*J31)
           + J13*(J21*J32 - J22*J31))
    return det

def jacobian_negative_penalty(displacement):
    det = jacobian_determinant(displacement)
    neg = F.relu(-det)
    return neg.mean()