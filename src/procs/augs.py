import torch
import torch.nn.functional as F


def random_thicken_mask_3d(
    mask,              # (B, 1, D, H, W)
    t_min=1,
    t_max=3,
    prob=0.5,
):
    if torch.rand(1, device=mask.device) > prob:
        return mask

    B, C, D, H, W = mask.shape
    t = torch.randint(t_min, t_max + 1, (1,), device=mask.device).item()

    out = mask.clone()

    # randomly choose axes (prefer sheet-safe behavior)
    axes = []
    if torch.rand(1) < 0.7:
        axes.append(2)  # depth
    if torch.rand(1) < 0.7:
        axes.append(3)  # height
    if torch.rand(1) < 0.7:
        axes.append(4)  # width

    if len(axes) == 0:
        axes = [2]  # fallback

    for axis in axes:
        for d in range(1, t + 1):
            out = torch.maximum(out, torch.roll(mask, d, dims=axis))
            out = torch.maximum(out, torch.roll(mask, -d, dims=axis))

    return out


def random_thicken_mask_3d_stochastic(
    mask,
    t_min=1,
    t_max=3,
    prob=0.5,
    drop_prob=0.3,
):
    if torch.rand(1, device=mask.device) > prob:
        return mask

    B, _, _, _, _ = mask.shape
    t = torch.randint(t_min, t_max + 1, (1,), device=mask.device).item()

    out = mask.clone()
    axes = [2, 3, 4]

    for axis in axes:
        for d in range(1, t + 1):
            rolled = torch.roll(mask, d, dims=axis)
            keep = (torch.rand_like(rolled) > drop_prob).float()
            out = torch.maximum(out, rolled * keep)

            rolled = torch.roll(mask, -d, dims=axis)
            keep = (torch.rand_like(rolled) > drop_prob).float()
            out = torch.maximum(out, rolled * keep)

    return out


def random_thicken_mask_3d_field(
    mask,
    t_min=1,
    t_max=3,
    prob=0.5,
    field_sigma=7,
):
    if torch.rand(1, device=mask.device) > prob:
        return mask

    t = torch.randint(t_min, t_max + 1, (1,), device=mask.device).item()

    # smooth random field
    field = torch.randn_like(mask)
    field = F.avg_pool3d(
        field, kernel_size=field_sigma, stride=1, padding=field_sigma // 2
    )
    field = (field > 0).float()

    out = mask.clone()
    for axis in [2, 3, 4]:
        for d in range(1, t + 1):
            rolled = torch.roll(mask, d, dims=axis)
            out = torch.maximum(out, rolled * field)

            rolled = torch.roll(mask, -d, dims=axis)
            out = torch.maximum(out, rolled * field)

    return out


def thicken_then_erode(mask, t=2):
    out = mask.clone()
    for _ in range(t):
        out = torch.maximum(
            out,
            torch.roll(out, 1, dims=2)
        )
    # erosion
    eroded = torch.minimum(
        out,
        torch.roll(out, 1, dims=2)
    )
    return eroded


def random_thicken_mask_sheetwise(
        mask,
        t_min=1,
        t_max=3,
        prob=0.5,
):
    """
    Randomly thickens mask along the depth (sheet-wise), preserving contiguous sheets.

    Args:
        mask: (B, 1, D, H, W) binary mask
        t_min, t_max: min/max thickness to apply
        prob: probability of applying augmentation
    Returns:
        Augmented mask
    """
    if torch.rand(1, device=mask.device) > prob:
        return mask

    B, _, D, H, W = mask.shape
    out = mask.clone()

    for b in range(B):
        m = mask[b, 0]  # (D, H, W)
        depth_presence = (m.sum(dim=(1, 2)) > 0)
        idx = torch.where(depth_presence)[0]  # indices of non-empty slices

        if len(idx) == 0:
            continue  # skip empty masks

        # find breaks in contiguous slices
        breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1

        # compute sizes of each contiguous sheet
        sheet_sizes = torch.diff(torch.cat([
            torch.tensor([0], device=idx.device),
            breaks,
            torch.tensor([len(idx)], device=idx.device)
        ]))

        sheets = torch.split(idx, sheet_sizes.tolist())

        for depths in sheets:
            t = torch.randint(t_min, t_max + 1, (1,), device=mask.device).item()
            slab = m[depths]

            # thicken both directions along depth
            for d in range(1, t + 1):
                # extend upwards
                if depths[0] - d >= 0:
                    out[b, 0, depths[0] - d] = torch.maximum(
                        out[b, 0, depths[0] - d], slab[0]
                    )
                # extend downwards
                if depths[-1] + d < D:
                    out[b, 0, depths[-1] + d] = torch.maximum(
                        out[b, 0, depths[-1] + d], slab[-1]
                    )

    return out

def apply_random_thickness_augmentation(mask_oof):
    mask_oof = random_thicken_mask_sheetwise(mask_oof, t_min=1, t_max=2, prob=0.5)
    mask_oof = random_thicken_mask_3d_stochastic(mask_oof, t_min=1, t_max=2, prob=0.4, drop_prob=0.2)
    mask_oof = random_thicken_mask_3d_field(mask_oof, t_min=1, t_max=2, prob=0.2, field_sigma=7)
    return mask_oof