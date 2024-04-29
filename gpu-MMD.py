import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import attr
import os
from materials import mdict

mask_path=None

def apply_mask(x,path):
    return x

def to_HU(mu,E0):
    data = []
    f = open('/data_new3/username/gpu-MMD/input/vmi_data.txt', 'r')
    for line in f.readlines()[1:]:
        data.append(np.array(line.split(), dtype=np.float64))
    f.close()
    data = np.array(data).T
    E, mac1_E, mac2_E, mac_water_E = data
    uw = np.interp(E0, E, mac_water_E)
    data1 = []
    f1 = open('/data_new3/username/gpu-MMD/input/air.dat', 'r')
    for line in f1.readlines()[0:]:
        data1.append(np.array(line.split(), dtype=np.float64))
    f1.close()
    data1=np.array(data1).T
    E,e1,mua=data1
    ua = np.interp(E0,E,mua)
    return 1000*(mu-uw)/(uw-ua)

def to_mu(HU, E0):
    data = []
    f = open('/data_new3/username/gpu-MMD/input/vmi_data.txt', 'r')
    for line in f.readlines()[1:]:
        data.append(np.array(line.split(), dtype=np.float64))
    f.close()
    data = np.array(data).T
    E, mac1_E, mac2_E, mac_water_E = data
    uw = np.interp(E0, E, mac_water_E)
    data1 = []
    f1 = open('/data_new3/username/gpu-MMD/input/air.dat', 'r')
    for line in f1.readlines()[0:]:
        data1.append(np.array(line.split(), dtype=np.float64))
    f1.close()
    data1=np.array(data1).T
    E,e1,mua=data1
    ua = np.interp(E0,E,mua)
    return (HU/1000)*(uw-ua)+uw

def pixel2HU(img,E):
    img = img.astype(np.float32)
    return img - 1024

class tri(object):
    points=attr.ib(default=None)
    simplices=attr.ib(default=None)


class MMD(object):
    def __init__(self, tri, E1, E2, mask_path,mu1,mu2,mdict):
        self.tri = tri
        self.E1 = E1
        self.E2 = E2
        self.mu1 = torch.tensor(mu1, dtype=torch.float32, device="cuda")
        self.mu2 = torch.tensor(mu2, dtype=torch.float32, device="cuda")
        self.mask_path = mask_path
        self.matnames = ['water', 'air', 'bone', 'omni300_in_blood_1000', 'fat']
        self.material_color_dict = {
            'water': 'Blues', 'air': 'Greys', 'bone': 'Purples', 
            'omni300_in_blood_1000': 'BuGn', 'fat': 'OrRd'
        }
        self.mask_color_dict = {'0':'inferno', '1':'inferno', '2':'inferno'}
        self.mats = []
        for m in self.matnames:
            self.mats.append(mdict[m])
        self.E_vec = [self.E1, self.E2]
        points = []
        for mat in self.mats:
            mat.init_atten_coeffs(self.E_vec)
            points.append(mat.mu)
        points = np.array(points)
        self.points = torch.tensor(points, dtype=torch.float32, device="cuda")
        simplices = torch.tensor([[0, 2, 3], [1,3,4], [0,3,4]], dtype=torch.long, device="cuda")
        self.tri.points = self.points
        self.tri.simplices = simplices
        self.M_dict = {name: torch.zeros([512, 512], dtype=torch.float32, device="cuda") for name in self.matnames}
        self.tri_type = {name: torch.zeros([512, 512], dtype=torch.float32, device="cuda") for name in self.mask_color_dict.keys()}

    def min_distance_to_triangle(self, mu1, mu2, triples):

        triples = triples.unsqueeze(0).unsqueeze(0).expand(512, 512, 2, 3)  # [512, 512, 2, 3]

        mus = torch.stack((mu1, mu2), dim=-1).unsqueeze(-1)  # [512, 512, 2, 1]

        point_distances = torch.zeros([512, 512, 3], dtype=torch.float32, device="cuda")
        for i in range(3):
            point_distances[:, :, i] = torch.norm(mus[..., :, 0] - triples[..., :, i], dim=2).squeeze(-1)


        edge_distances = torch.zeros_like(point_distances).unsqueeze(-1)  # [512, 512, 3, 1]
        for i in range(3):
            line_start = triples[..., :, i].unsqueeze(-1)
            line_end = triples[..., :, (i + 1) % 3].unsqueeze(-1)
            edge_distances[:, :, i,:] = self.point_to_line_distance(mus, line_start, line_end)

        edge_distances = edge_distances.squeeze(-1)  # [512, 512, 3]


        all_distances = torch.cat((point_distances, edge_distances), dim=2)
        min_distances = torch.min(all_distances, dim=2).values

        return min_distances

    def point_to_line_distance(self,points, line_start, line_end):


        line_vec = line_end - line_start
        point_vec = points - line_start

        line_vec_mag_squared = torch.sum(line_vec * line_vec, dim=2, keepdim=True)
        proj_length = torch.sum(point_vec * line_vec, dim=2, keepdim=True) / line_vec_mag_squared
        proj_length = torch.clamp(proj_length, 0, 1)

        proj_point = line_start + proj_length * line_vec

        dist = torch.norm(points - proj_point, dim=2)

        return dist

    def is_inside(self,mu1, mu2, triple, epsilon=1e-12):

        triple = triple.unsqueeze(0).unsqueeze(0).expand(512, 512, 2, 3)

        points = torch.stack((mu1, mu2), dim=2).unsqueeze(-1)  # [512, 512, 2, 1]

        total_area = self.tri_area(triple[..., :, 0], triple[..., :, 1], triple[..., :, 2])

        area1 = self.tri_area(points[..., :, 0], triple[..., :, 1], triple[..., :, 2])
        area2 = self.tri_area(points[..., :, 0], triple[..., :, 2], triple[..., :, 0])
        area3 = self.tri_area(points[..., :, 0], triple[..., :, 0], triple[..., :, 1])

        area_sum = area1 + area2 + area3

        is_inside_mask = torch.abs(total_area - area_sum) < epsilon

        return is_inside_mask

    def tri_area(self,p1, p2, p3):
        cross_product = p1[:,:, 0] * (p2[:,:, 1] - p3[:,:, 1]) + p2[:,:, 0] * (p3[:,:, 1] - p1[:,:, 1]) + p3[:,:, 0] * (p1[:,:, 1] - p2[:,:, 1])
        return torch.abs(cross_product) / 2

    def md3_parallel(self,mu1, mu2, triple):
        x = torch.ones((512, 512, 3), device="cuda", dtype=torch.float32)
        x[:, :, :2] = torch.stack((mu1, mu2), dim=2)
        M = torch.ones((512, 512, 3, 3), device="cuda", dtype=torch.float32)
        M[:, :, :2, :] = triple.unsqueeze(0).unsqueeze(0).expand(512, 512, 2, 3)
        M_inv = torch.linalg.inv(M)
        alphas = torch.matmul(M_inv, x.unsqueeze(-1)).squeeze(-1)
        return alphas
     
    def compute_material_decomposition(self):
        min_distances_total = torch.zeros([512,512,3], dtype=torch.float32, device="cuda")
        inside_total = torch.zeros([512,512], dtype=torch.bool, device="cuda")
        flaged = torch.zeros([512,512], dtype=torch.bool, device="cuda")
        for tri_indices,i in zip(self.tri.simplices,range(3)):
            triples = self.tri.points[tri_indices].T
            min_distances = self.min_distance_to_triangle(self.mu1, self.mu2, triples)
            min_distances_total[:,:,i] = min_distances
            inside_mask = self.is_inside(self.mu1, self.mu2, triples)
            inside_total = inside_total | inside_mask
        minest = torch.min(min_distances_total, dim=2).values
        for tri_indices,i in zip(self.tri.simplices,range(3)):
            tri_name = []
            for idx, name in enumerate(self.matnames):
                if idx in tri_indices:
                    tri_name.append(name)
            triples = self.tri.points[tri_indices].T
            min_distances = self.min_distance_to_triangle(self.mu1, self.mu2, triples)
            inside_mask = self.is_inside(self.mu1, self.mu2, triples)
            alphas = self.md3_parallel(self.mu1, self.mu2, triples)
            flag = (inside_mask |( (min_distances == minest) & (inside_total == False))) & (flaged == False)
            flaged = flaged | flag
            flag = flag.to(torch.float32)
            self.tri_type[i] = flag
            for i, name in enumerate(tri_name):
                self.M_dict[name] += flag[:,:] * alphas[:,:,i]

def main():
    start_time = time.time()
    idx = 0

    matnames = ['water',  'air', 'bone', 'omni300_in_blood_1000','fat']

    E_vec = [62, 93]
    image_folder = '/data_new3/username/DualEnergyCTSynthesis/dataset/valid'
    save_path = '/data_new3/username/decomposition_dataset/valid'

    files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]

    tris = tri()

    for file in files:
        print(f'Processing {file}')
        idx += 1
        file_dir = os.path.join(image_folder, file)
        data = np.load(file_dir, allow_pickle=True).item()
        image = data['data']
        label = data['label']
        patient_id = data['patient_id']
        image_id = data['image_id']
        mu1 = apply_mask(to_mu(pixel2HU(image.reshape([512, 512]), E_vec[0]), E_vec[0]), None)
        mu2 = apply_mask(to_mu(pixel2HU(label.reshape([512, 512]), E_vec[1]), E_vec[1]), None)

        mmd = MMD(tris, E_vec[0], E_vec[1], None, mu1, mu2, mdict)
        mmd.compute_material_decomposition()

        result_data = {'data': mu1, 'label': mu2, 'patient_id': patient_id, 'image_id': image_id}
        for name in matnames:
            result_data[name] = mmd.M_dict[name]

        decomposed_file_name = f'decomposed_{file}'
        np.save(os.path.join(save_path, decomposed_file_name), result_data)
        print(f'Saved to {decomposed_file_name}')
    print(f'Finished in {time.time() - start_time:.2f} seconds for {idx} image pairs, average {1000*(time.time() - start_time)/idx:.2f} ms per image pair')
main()


'''mu1 = apply_mask(to_mu(pixel2HU(np.load('/data_new3/username/gpu-MMD/img1.npy').reshape([512,512]),E_vec[0]), E_vec[0]), mask_path)
    mu2 = apply_mask(to_mu(pixel2HU(np.load('/data_new3/username/gpu-MMD/img2.npy').reshape([512,512]),E_vec[1]), E_vec[1]), mask_path)
    tris = tri()

    mmd = MMD(tris, E_vec[0], E_vec[1], mask_path, mu1, mu2, mdict)
    mmd.compute_material_decomposition()'''
'''M_dict = mmd.M_dict
    tritype= mmd.tri_type
    scale = 2.5
    fig, ax = plt.subplots(1, 8, figsize=(scale*8, scale))
    for idx, name in enumerate(matnames):
        vmax=min(np.max(M_dict[name].cpu().numpy()),1)
        vmin=max(np.min(M_dict[name].cpu().numpy()),0)
        if name=='omni300_in_blood_1000':
            vmax=vmax*0.5
        print(f'{name} vmax:{vmax} vmin:{vmin}')
        ax[idx].imshow(M_dict[name].cpu().numpy(), cmap=mmd.material_color_dict[name],vmin=vmin,vmax=vmax)
        ax[idx].set_title(name)
        ax[idx].axis('off')
    for i, name in enumerate(mmd.mask_color_dict.keys()):
        ax[i+5].imshow(tritype[i].cpu().numpy(), cmap=mmd.mask_color_dict[name])
        ax[i+5].set_title(name)
        ax[i+5].axis('off')
    plt.savefig(f'/data_new3/username/gpu-MMD/output/output{time.time()}.png')'''