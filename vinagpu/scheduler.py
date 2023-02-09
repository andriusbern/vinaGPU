from multiprocessing import Pool, current_process, Queue
from vinagpu import VinaGPU

NUM_GPUS = 4
PROC_PER_GPU = 1

runners = [VinaGPU(gpu_id) for gpu_id in range(NUM_GPUS)]

queue = Queue()

def dock(smiles):
    gpu_id = queue.get()
    try:
        # run processing on GPU <gpu_id>
        ident = current_process().ident
        print('{}: starting process on GPU {}'.format(ident, gpu_id))

        runner = runners[gpu_id]
        scores = runner.dock(
            target_pdb_path=target_path,
            smiles=[smiles],
            output_subfolder=output_subfolder, 
            active_site_coords=active_site,
            verbose=True)
        # ... process filename
        print('{}: finished'.format(ident))
    except Exception as e:
        print(e)
    finally:
        queue.put(gpu_id)

def parallel_dock(target_path, smiles_list,
                  output_subfolder, 
                  active_site,
                  verbose):

    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
    
    for _ in pool.imap_unordered(dock, smiles_list):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    target_path = '/home/andrius/git/vinaGPU/examples/P21918.pdb'
    output_subfolder = 'test'
    active_site = (0, 0, 0)
    smiles = ['CCCCCCCCCCCCCCCCCCCCCCCCC', 'CCO', 'CCOCC', 'CCOCCO', 'CCOCCOCC', 'CCOCCOCCO', 'CCOCCOCCOCC', 'CCOCCOCCOCCO']
    verbose = True
    parallel_dock(target_path, smiles, output_subfolder, active_site, verbose)
