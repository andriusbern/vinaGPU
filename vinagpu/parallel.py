from multiprocessing import Pool, current_process, Queue
from vinagpu import VinaCPU, VinaGPU
import os
import time
from vinagpu.utils import read_log

def docking_job(smiles: list):
    """
    This function is called by each process in the pool.

    Arguments:
        smiles (list)           : list of SMILES
    """
    ident = current_process().ident
    device_id = queue.get()
    if device_id < 0:
        device_id = -device_id - 1
        runners = cpu_runners
        print('{}: starting process on CPU {}'.format(ident, device_id))
    else:
        runners = gpu_runners
        print(f'{ident}: starting process on GPU {device_id}, docking {len(smiles)} ligands.')

    try:
        # Run processing on GPU/CPU 
        docking_kwargs['smiles'] = smiles
        scores = runners[device_id].dock(**docking_kwargs)
        
        print('{}: finished'.format(ident))
    except Exception as e:
        print(e)
        runners[device_id].remove_docker_container() 
    # KeyboardsInterrupt is raised when the process is terminated by the user
    except KeyboardInterrupt:
        print('Process terminated by the user.')
        runners[device_id].remove_docker_container()
    finally:
        queue.put(device_id)


def parallel_dock(target_pdb_path, smiles=[], ligand_pdbqt_paths=[], output_subfolder='', 
                  box_center=(0,0,0), box_size=(20,20,20), search_depth=3,
                  threads=256, threads_per_call=256, clean=True, verbose=True, 
                  visualize_in_pymol=False, write_log=True, 
                  gpu_ids=[0,1,2,3], workers_per_gpu=1,
                  num_cpu_workers=0, threads_per_cpu_worker=1, exhaustiveness=8):
    """
    Dock a list of SMILES using multiple GPUs or CPUs (using Autodock Vina).

    Arguments:
        target_pdb_path (str)                 : path to the target PDB file
        smiles (list)                         : list of SMILES
        ligand_pdbqt_paths (list)             : list of paths to ligand PDBQT files (alternative to SMILES)
        output_subfolder (str)                : path to the output folder
        active_site_coords (tuple)            : coordinates of the active site (x,y,z)
        bbox_size (tuple)                     : size of the bounding box (x,y,z)
        clean (bool)                          : clean the output folder (remove ligand .pdbqt files)
        verbose (bool)                        : print details in the console
        visualize_in_pymol (bool)             : visualize the results in PyMOL
        write_log (bool)                      : write the log file

        # GPU arguments
        gpu_ids (list)                        : list of GPU ids
        workers_per_gpu (int)                 : number of workers per GPU
        search_depth (int)                    : search depth
        threads (int)                         : number of threads to use (look up Vina-GPU documentation)
        threads_per_call (int)                : number of threads per single call (look up Vina-GPU documentation)

        # CPU arguments
        num_cpus (int)                        : number of CPU workers
        threads_per_cpu_worker (int)          : number of threads per CPU worker
        exhaustiveness (int)                  : Vina CPU exhaustiveness
    
    Returns:
        scores (list)                         : list of scores

    """

    ## Declare global variables to be used in the docking_job function
    global docking_kwargs
    global queue
    global gpu_runners
    global cpu_runners
    docking_kwargs = locals()
    queue = Queue()
    gpu_runners = [VinaGPU(devices=[str(gpu_id)]) for gpu_id in gpu_ids]
    cpu_runners = [VinaCPU(cpu=threads_per_cpu_worker, device_id=i) for i in range(num_cpu_workers)]

    # initialize the queue with the GPU ids
    num_gpus = len(gpu_ids)
    num_gpu_workers = workers_per_gpu * num_gpus
    for gpu_ids in range(num_gpus):
        for _ in range(workers_per_gpu):
            queue.put(gpu_ids)

    # initialize the queue with the CPU ids (negative values to distinguish from GPU ids)
    for cpu_id in range(num_cpu_workers):
        queue.put(-cpu_id - 1)
    
    ## Split the list of SMILES into <num_splits> parts
    n_smiles = len(smiles)
    splits = num_gpu_workers
    w = (n_smiles // splits) + 1
    smiles_splits = [smiles[i*w:(i+1)*w] for i in range(splits)]

    t0 = time.time() 
    # Start the worker pool
    pool = Pool(processes=num_gpu_workers + num_cpu_workers)
    for _ in pool.imap_unordered(docking_job, smiles_splits):
        pass
    pool.close()
    pool.join()
    print(f'Docking finished. Time elapsed: {time.time() - t0} seconds.')

    ## Read generated scores from the log file
    log = read_log(os.path.join('output', output_subfolder, 'log.tsv'))
    scores = []
    processed_smiles = [entry[0] for entry in log]
    for ligand in smiles:
        if ligand in processed_smiles:
            idx = processed_smiles.index(ligand)
            best_score = log[idx][2][0]
            scores.append(best_score)
        else:
            scores.append(100.0)

    return scores



