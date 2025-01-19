import torch
import h5py
import copy
import numpy as np
from scOT.problems.Mybase import BaseTimeDataset, BaseDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS

class CompressibleBase(BaseTimeDataset):
    def __init__(self, file_path, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 200

        self.N_max = 3
        self.N_val = 1
        self.N_test = 1
        self.resolution = 128 
        self.tracer = tracer

        data_path = self.data_path + file_path
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")
      

        self.constants = copy.deepcopy(CONSTANTS)

        self.input_dim = 1
        self.label_description = (
            "[rho]"
        )

        self.pixel_mask = (
            torch.tensor([False])
        )
        
        
        # self.input_dim = 4 if not tracer else 5
        # self.label_description = (
        #     "[rho],[u,v],[p]" if not tracer else "[rho],[u,v],[p],[tracer]"
        # )

        # self.pixel_mask = (
        #     torch.tensor([False, False, False, False])
        #     if not tracer
        #     else torch.tensor([False, False, False, False, False])
        # )

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        # inputs = (
        #     torch.from_numpy(self.reader["data"][i + self.start, t1, 0:4])
        #     .type(torch.float32)
        #     .reshape(4, self.resolution, self.resolution)
        # )
        # label = (
        #     torch.from_numpy(self.reader["data"][i + self.start, t2, 0:4])
        #     .type(torch.float32)
        #     .reshape(4, self.resolution, self.resolution)
        # )
        
        inputs = (
            torch.from_numpy(self.reader["data"][i + self.start, t1, 0:1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["data"][i + self.start, t2, 0:1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        
        comparison = inputs == label
        true_count = torch.sum(comparison)
        print("it is ", true_count)
        import matplotlib.pyplot as plt
        plt.imshow(label[0,:,:], cmap='viridis')
        plt.colorbar(label='Value')
        plt.title('2D Tensor Visualization')
        plt.savefig("./labels.png")

        # if self.tracer:
        #     input_tracer = (
        #         torch.from_numpy(self.reader["data"][i + self.start, t1, 4:5])
        #         .type(torch.float32)
        #         .reshape(1, self.resolution, self.resolution)
        #     )
        #     output_tracer = (
        #         torch.from_numpy(self.reader["data"][i + self.start, t2, 4:5])
        #         .type(torch.float32)
        #         .reshape(1, self.resolution, self.resolution)
        #     )
        #     inputs = torch.cat([inputs, input_tracer], dim=0)
        #     label = torch.cat([label, output_tracer], dim=0)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }
 
class BubbleC(CompressibleBase):
    def __init__(self, *args, tracer=False, **kwargs):
        file_path = "/JXFdensity_n_128.nc"
        super().__init__(file_path, *args, tracer=tracer, **kwargs)
