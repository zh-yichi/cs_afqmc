from jax import numpy as jnp
from jax import vmap

def walker_norhf_energy(walker,ham_data,trial_mo):
    '''rhf walker's energy with non-orthonal trial'''
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    green_walker = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gia,ak->gik',rot_chol[:,:nocc,nocc:],
                   green_walker.T[nocc:,:nocc],optimize='optimal')
    tf = vmap(jnp.trace)(f)
    tc = vmap(jnp.trace)(rot_chol[:,:nocc,:nocc])
    ene11 = 2*jnp.einsum('ia,ia->',green_walker[:nocc,nocc:],rot_h1[:nocc,nocc:])
    ene12 = 2*jnp.sum(tf*tc) \
            - jnp.einsum('gik,gki->',f,rot_chol[:,:nocc,:nocc])
    ene1 = ene11-2*ene12
    ene2 = 2*jnp.sum(tf*tf) \
           - jnp.einsum('gij,gji->',f,f)
    return ene1+ene2 #,ene1,ene2,

def walker_norhf_orb_en(walker,ham_data,wave_data):
    '''rhf walker's energy with non-orthonal trial'''
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    nocc = rot_h1.shape[0]
    green_walker = (walker.dot(jnp.linalg.inv(
        wave_data['mo_coeff'].T.conj() @ walker))).T
    f = jnp.einsum('gia,ak->gik',rot_chol[:,:nocc,nocc:],
                   green_walker.T[nocc:,:nocc],optimize='optimal')
    tf = vmap(jnp.trace)(f)
    # tc = vmap(jnp.trace)(rot_chol[:,:nocc,:nocc])
    ene11 = 2*jnp.einsum('ia,ka,ik->',green_walker[:nocc,nocc:],rot_h1[:nocc,nocc:],m)
    ene12 = 2*jnp.einsum('gik,ik,gjj->',f,m,rot_chol[:,:nocc,:nocc]) \
            - jnp.einsum('gij,gjk,ik->',f,rot_chol[:,:nocc,:nocc],m)
    ene1 = ene11-2*ene12
    ene2 = 2*jnp.einsum('gik,ik,g->',f,m,tf) \
           - jnp.einsum('gij,gjk,ik->',f,f,m)
    return ene1+ene2