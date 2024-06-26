import jax
from IPython.core.magic import register_cell_magic

#function that saves the code in cell and runs that cell afterwards
@register_cell_magic
def write_and_run(line, cell):
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)
    

#replace small valeus in array with some threshold value
@jax.jit
def replace_small_entries_with_threshold(k, threshold):
    mask = k < threshold
    k = jnp.where(mask, threshold, k)
    return k