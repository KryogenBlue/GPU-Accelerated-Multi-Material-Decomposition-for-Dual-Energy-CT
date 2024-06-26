## some materials to import for our mat decomp
## we use the material composition to calc mass-atten coeffs
## 
## this is set up to convert the NIST lists of 
## compound material compositions by weight
## into matcomp strings for xcompy 
##
## ex matcomp_string:
## 'H(6.3984)C(27.8)N(2.7)O(41.0016)Mg(0.2)P(7.0)S(0.2)Ca(14.7)' # ICRU bone

from periodictable import elements
import xcompy as xc
import numpy as np

def convert_nist_str(d):
    '''
    easy conversion from NIST list of materials to xcompy format
    just remember to add commas after each line when pasting
    '''
    
    s = ''
    for Z in d.keys():
        s+= f'{elements[Z]}({100*d[Z]:.4f})'
        
    return s

class Material:
    def __init__(self, name, density, matcomp_dict):
        self.name = name
        self.density = density
        self.matcomp_dict = matcomp_dict
        self.matcomp = convert_nist_str(matcomp_dict)

    def set_density(self, p):
        self.density = p

    def init_atten_coeffs(self, E):
        mass_attens = xc.mixatten(self.matcomp, np.array(E).astype(np.float64))
        linear_attens = self.density * mass_attens
        self.E = E
        self.mu_rho = mass_attens
        self.mu = linear_attens
        return mass_attens, linear_attens

    def add_water(self):
        mu_water = xc.mixatten('H(11.1898)O(88.8102)', self.E)
        self.mu += mu_water # mess up mu_rho?
        
def dilute_contrast(mat_0, mat_contrast, mg_ml):
    '''
    Create new diluted contrast material in a medium
    '''
    g_ml = mg_ml/1000.0

    # total mass per cm3, or total density
    p_total = g_ml + mat_0.density

    # percentages by weight
    p_0 = mat_0.density/p_total
    p_contrast = g_ml/p_total

    # create new dictionary
    new_matcomp_dict = {}
    for Z in mat_0.matcomp_dict:
        new_matcomp_dict[Z] = p_0*mat_0.matcomp_dict[Z]
    for Z in mat_contrast.matcomp_dict:
        new_matcomp_dict[Z] = p_contrast*mat_contrast.matcomp_dict[Z]
    
    # new material
    new_name = f'{mat_contrast.name}_in_{mat_0.name}_{int(mg_ml)}'
    mat_diluted = Material(new_name, p_total, new_matcomp_dict)
    return mat_diluted


## define three materials
## using ICRU materials
## https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

all_mats = []
mdict = {}
for name, p, d in [
   ['fat', 
    0.950,
    {1: 0.114000,
     6: 0.598000,
     7: 0.007000,
     8: 0.278000,
     11: 0.001000,
     16: 0.001000,
     17: 0.001000}],
   ['bone',
    1.450,
    {1: 0.065473,
     6: 0.536942,
     7: 0.021500,
     8: 0.032084,
     9: 0.167415,
     20: 0.176585}],
   ['blood',
    1.060,
    {1: 0.102000,
     6: 0.110000,
     7: 0.033000,
     8: 0.745000,
     11: 0.001000,
     15: 0.001000,
     16: 0.002000,
     17: 0.003000,
     19: 0.002000,
     26: 0.001000}],
    ['air',
     1.205e-3,
     {6: 0.000124,
     7: 0.755268,
     8: 0.231781,
     18: 0.012827}],
    ['tissue_soft',
     1.060,
     {1: 0.102000,
      6: 0.143000,
      7: 0.034000,
      8: 0.708000,
      11: 0.002000,
      15: 0.003000,
      16: 0.003000,
      17: 0.002000,
      19: 0.003000}],
    ['calcium',
     1.550,
     {20: 1.0}],
    ['iodine',
     4.930,
     {53: 1.0}],
    ['water',
     1.0,
     {1: 0.111898,
      8: 0.888102}],
    ['omni300',
     1.06,
     {1: 0.03191478751101292, 
                         6: 0.2779110718133381, 
                         7: 0.05117301559050043,
                         8: 0.1753598375717305,
                         53: 0.46364128751341815}],
    ['titanium',
     6.45,
     {22: 0.44,
      28: 0.56},],
    ['lightbone',
     1.14,
     {1: 0.065473*0.1+0.114000*0.4+0.102000*0.5,
     6: 0.536942*0.1+0.598000*0.4+0.110000*0.5,
     7: 0.021500*0.1+ 0.007000*0.4+0.033000*0.5,
     8: 0.032084*0.1+0.278000*0.4+0.745000*0.5,
     11: 0.001000*0.4+0.001000*0.5,
     15: 0.001000*0.5,
     9: 0.167415,
     16: 0.001000*0.4+0.002000*0.5,
     17: 0.001000*0.4+ 0.003000*0.5,
     19: 0.002000*0.5,
     26: 0.001000*0.5,
     20: 0.176585}]
    ]:
    mat = Material(name, p, d)
    all_mats.append(mat)
    mdict[name] = mat

all_mats.append(dilute_contrast(mdict['blood'], mdict['iodine'], mg_ml=100.0))
all_mats.append(dilute_contrast(mdict['water'], mdict['calcium'], mg_ml=600.0))
all_mats.append(dilute_contrast(mdict['blood'], mdict['omni300'], mg_ml=1000.0))

for mat in all_mats:
    if mat not in mdict:
        mdict[mat.name] = mat
