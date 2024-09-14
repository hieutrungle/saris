from multiprocessing import Process, Queue
import multiprocessing

multiprocessing.get_context("spawn")
multiprocessing.set_start_method("spawn", force=True)


class Multiprocessor:

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets


####################################################################################################
# Example usage of the code above
####################################################################################################

# @dataclass
# class Args:
#     config_file: str
#     compute_scene_path: str
#     viz_scene_path: str
#     saved_path: str
#     results_path: str
#     use_cmap: bool = False
#     verbose: bool = False
#     seed: int = 0


# args = Args(
#     config_file="/home/hieule/research/saris/configs/sionna_L_hallway_1.yaml",
#     compute_scene_path="/home/hieule/research/saris/local_assets/blender/hallway_L_1/ceiling_idx/hallway.xml",
#     viz_scene_path="/home/hieule/research/saris/local_assets/blender/hallway_L_1/idx/hallway.xml",
#     saved_path="/home/hieule/research/saris/local_assets/images/test",
#     results_path="/home/hieule/research/saris/tmp/test.pkl",
#     use_cmap=False,
#     verbose=False,
#     seed=0,
# )


# def get_paths(args):
#     for i in range(20):
#         config = utils.load_config(args.config_file)

#         if args.verbose:
#             utils.log_args(args)
#             utils.log_config(config)

#         tf.random.set_seed(args.seed)

#         # Prepare folders
#         sig_cmap = signal_cmap.SignalCoverageMap(
#             config, args.compute_scene_path, args.viz_scene_path, args.verbose
#         )

#         if not args.use_cmap:
#             paths = sig_cmap.compute_paths()
#             cir = paths.cir()
#     return cir


# if __name__ == "__main__":

#     start_time = time.time()
#     mp = Multiprocessor()
#     mp.run(get_paths, args)
#     rets = mp.wait()
#     print(f"time: {time.time() - start_time}")
