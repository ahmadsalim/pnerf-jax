
"""
pNeRF algorithm for parallelized conversion from internal to Cartesian coordinates.
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

# Imports
import jax.numpy as np
import numpy as onp
from jax.ops import index_update, index
from jax.lax import fori_loop, dynamic_update_slice
import collections

# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = np.array([145.801, 152.326, 132.868], dtype='float32')
BOND_ANGLES  = np.array([  2.124,   1.941,   2.028], dtype='float32')

def normalize(v, axis=-1, order=2):
    # From .https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    norm = np.atleast_1d(np.linalg.norm(v, order, axis))
    return v / np.expand_dims(norm, axis)


def dihedral_to_point(dihedral, r=BOND_LENGTHS, theta=BOND_ANGLES):
    """ Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points ready for use in
        reconstruction of coordinates. Bond lengths and angles are based on idealized averages.

    Args:
        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    Returns:
                  [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    num_steps  = dihedral.shape[0]
    batch_size = dihedral.shape[1]

    r_cos_theta = r * np.cos(np.pi - theta) # [NUM_DIHEDRALS]
    r_sin_theta = r * np.sin(np.pi - theta) # [NUM_DIHEDRALS]

    pt_x = np.tile(np.reshape(r_cos_theta, [1, 1, -1]), [num_steps, batch_size, 1]) # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_y = np.cos(dihedral) * r_sin_theta  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_z = np.sin(dihedral) * r_sin_theta # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    pt = np.stack([pt_x, pt_y, pt_z]) # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_perm = np.transpose(pt) # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
    pt_final = np.reshape(pt_perm, [num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS]) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]

    return pt_final

def point_to_coordinate(pt, num_fragments=6):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.

        Reconstruction is done in parallel, by independently reconstructing num_fragments fragments and then 
        reconstituting the chain at the end in reverse order. The core reconstruction algorithm is NeRF, based on 
        DOI: 10.1002/jcc.20237 by Parsons et al. 2005. The parallelized version is described in 
        https://www.biorxiv.org/content/early/2018/08/06/385450.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Opts:
        num_fragments: Number of fragments to reconstruct in parallel. If None, the number is chosen adaptively

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """                             

    # compute optimal number of fragments if needed
    s = pt.shape[0] # NUM_STEPS x NUM_DIHEDRALS
    if num_fragments is None:
        num_fragments = np.cast(np.sqrt(np.cast(s, dtype='float32')), dtype='int32')

    # initial three coordinates (specifically chosen to eliminate need for extraneous matmul)
    Triplet = collections.namedtuple('Triplet', 'a, b, c')
    batch_size = pt.shape[1] # BATCH_SIZE
    init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]], dtype='float32')
    init_coords = Triplet(*[np.reshape(np.tile(row[np.newaxis], np.stack([num_fragments * batch_size, 1])), 
                                       [num_fragments, batch_size, NUM_DIMENSIONS]) for row in init_mat])
                  # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
    
    # pad points to yield equal-sized fragments
    r = ((num_fragments - (s % num_fragments)) % num_fragments)          # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
    pt = np.pad(pt, [[0, r], [0, 0], [0, 0]])                            # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    pt = np.reshape(pt, [num_fragments, -1, batch_size, NUM_DIMENSIONS]) # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
    pt = np.transpose(pt, [1, 0, 2, 3])                                  # [FRAG_SIZE, NUM_FRAGS,  BATCH_SIZE, NUM_DIMENSIONS]

    # extension function used for single atom reconstruction and whole fragment alignment
    def extend(tri, pt, multi_m):
        """
        Args:
            tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
            pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
            multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
        """

        bc = normalize(tri.c - tri.b, axis=-1)                                        # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]        
        n = normalize(np.cross(tri.b - tri.a, bc), axis=-1)                            # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
        if multi_m: # multiple fragments, one atom at a time. 
            m = np.transpose(np.stack([bc, np.cross(n, bc), n]), [1, 2, 3, 0])        # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
        else: # single fragment, reconstructed entirely at once.
            s = onp.pad(pt.shape, [[0, 1]], constant_values=3)                                    # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
            m = np.transpose(np.stack([bc, np.cross(n, bc), n]), [1, 2, 0])                     # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
            m = np.reshape(np.tile(m, [s[0], 1, 1]), s)                                    # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
        coord = np.squeeze(np.matmul(m, np.expand_dims(pt, 3)), axis=3) + tri.c  # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
        return coord
    
    # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
    coords = np.zeros_like(pt)  # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
    
    def loop_extend(i, dt): # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        tri, coords = dt
        coord = extend(tri, pt[i], True)
        return (Triplet(tri.b, tri.c, coord), index_update(coords, i, coord))

    tris, coords_pretrans = fori_loop(0, pt.shape[0], loop_extend, (init_coords, coords))
                                  # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS], 
                                  # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
    # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
    coords_pretrans = np.transpose(coords_pretrans, [1, 0, 2, 3]) # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    n = coords_pretrans.shape[0] # NUM_FRAGS
    fs = coords_pretrans.shape[1] # FRAG_SIZE


    res_array = np.zeros((n * coords_pretrans.shape[1], *coords_pretrans.shape[2:]))
    def loop_trans(j, coords):
        i = (n - j) - 1
        transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
        return dynamic_update_slice(transformed_coords, coords_pretrans[i], [fs*i] + [0] * (transformed_coords.ndim - 1))

    res_array = index_update(res_array, index[fs*(n-1):fs*n], coords_pretrans[-1])
    coords_trans = fori_loop(0, n, loop_trans, res_array) # coords_pretrans[-1]) # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

    # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
    coords = np.pad(coords_trans[:s-1], [[1, 0], [0, 0], [0, 0]]) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    return coords

if __name__ == "__main__":
    import time
    import math
    from jax import random
    key = random.PRNGKey(0)
    angles = (-math.pi-math.pi)*random.uniform(key, (1000,10,3), dtype=np.float32)+ math.pi
    start=time.time()
    print(point_to_coordinate(dihedral_to_point(angles,r=BOND_ANGLES,theta=BOND_LENGTHS)))
    stop=time.time()
    print("Computational time: " + str(stop-start))

