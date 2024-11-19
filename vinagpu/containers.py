import docker
import os
import shutil


class VinaContainer:
    def __init__(self, dir_to_mount):

        self.client = docker.from_env()
        self.dir_to_mount = dir_to_mount
        
        self.container = None

        ## Container paths
        self.container_name = 'vina-cl'
        self.working_dir = '/vina-gpu-dockerized/Vina-GPU-2.1/QuickVina2-GPU-2.1'
        self.docking_dir = self.working_dir + '/docking'
        self.executables = dict(
            vina='QuickVina2-GPU-2-1',
            adfr='/htd/ADFRsuite-1.0/adfr'
        )

        ## Device
        self.device = 'gpu'
        self.device_id = None
        self.docker_kwargs = dict(
            image=self.container_name,
            volumes = [f'{self.dir_to_mount}:{self.docking_dir}'],
            device_requests=[docker.types.DeviceRequest(device_ids=self) ]

        )

    def start(self):
        self.container = self.client.containers.run(
            command='sleep infinity', # Keeps the container running until it is killed
            detach=True,              # Run container in background
            **self.docker_kwargs
        )

    def remove(self):
        self.container.remove(force=True) 
        self.container = None
