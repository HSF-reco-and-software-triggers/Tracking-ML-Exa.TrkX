from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..segment_base import SegmentBase
from ..utils.segmentation_utils import label_graph


class TrackMLSegment(SegmentBase):

    """
    The segmentation data module specific to the TrackML pipeline
    """

    def __init__(self, params):
        super().__init__(hparams)

        """Init method for class.
        
        Args:
            params (type): Description of params.
            
        """

    def prepare_data(self):

        all_files = [
            os.path.join(self.hparams["input_dir"], file)
            for file in os.listdir(self.hparams["input_dir"])
        ][: self.n_files]
        all_files = np.array_split(all_files, self.n_tasks)[self.task]

        os.makedirs(self.output_dir, exist_ok=True)
        print("Writing outputs to " + self.output_dir)

        process_func = partial(label_graph, **self.hparams)
        process_map(process_func, all_events, max_workers=self.n_workers)
