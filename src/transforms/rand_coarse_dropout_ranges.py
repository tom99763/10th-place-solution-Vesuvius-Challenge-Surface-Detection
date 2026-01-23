import numpy as np
from monai.transforms import MapTransform, RandCoarseDropoutd

class RandCoarseDropoutdWithRanges(MapTransform):
    def __init__(
        self,
        keys,
        prob=1.0,
        holes_range=(8, 12),
        spatial_size_range=(10, 30),
        fill_value=0.0,
    ):
        super().__init__(keys)
        self.prob = prob
        self.holes_range = holes_range
        self.spatial_size_range = spatial_size_range
        self.fill_value = fill_value

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        num_holes = np.random.randint(
            self.holes_range[0], self.holes_range[1] + 1
        )
        hole_d = np.random.randint(
            self.spatial_size_range[0], self.spatial_size_range[1] + 1
        )
        hole_h = np.random.randint(
            self.spatial_size_range[0], self.spatial_size_range[1] + 1
        )
        hole_w = np.random.randint(
            self.spatial_size_range[0], self.spatial_size_range[1] + 1
        )

        transform = RandCoarseDropoutd(
            keys=self.keys,
            prob=1.0,
            holes=num_holes,
            spatial_size=(hole_d, hole_h, hole_w),
            fill_value=self.fill_value,
        )
        return transform(data)