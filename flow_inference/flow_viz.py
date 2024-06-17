# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
import os
import argparse

import torch
import numpy as np
import cv2
from skvideo.io import FFmpegWriter
import h5py
from tqdm import tqdm


from metavision_core_ml.video_to_event.simu_events_iterator import SimulatedEventsIterator
from metavision_ml.flow.lightning_model import FlowModel
from metavision_ml.data import CDProcessorIterator, HDF5Iterator
from metavision_ml.flow.viz import get_arrows, get_dense_flow
from metavision_ml.preprocessing.viz import normalize, filter_outliers


@torch.no_grad()
def _proc(preprocessor, flow_dataset, mode="arrows", video_process=None, mask_by_input=True, display=True, step=4):
    """Sub function performing preprocessing influence and visualization. """

    for tensor, flow in tqdm(zip(preprocessor, flow_dataset), total=len(flow_dataset)):

        tensor = tensor.cpu().detach().numpy()[0]
        img = preprocessor.get_vis_func()(tensor)
        mask = (np.sum(np.abs(tensor), axis=0) > 0) if mask_by_input else None
        if step:
            img = get_arrows(torch.from_numpy(flow), base_img=img, step=step, mask=mask)
        else:
            img = get_dense_flow(torch.from_numpy(flow), base_img=img, mask=mask)

        if display:
            current_time = int(preprocessor.step * preprocessor.delta_t)
            img = cv2.putText(img, f"{current_time:d}us", (10, 20),
                              cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(12, 240, 50), thickness=2)
            cv2.imshow(mode, img[..., ::-1])
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
        if video_process is not None:
            video_process.writeFrame(img)


def main(path, flow_path, write_video="", mode="arrows", height=None, width=None,
         max_duration=None, mask_by_input=True, start_ts=0, display=True, step=4):
    assert mode in ("sharp", "arrows")
    flow_file = h5py.File(flow_path, 'r')
    flow_attrs = dict(flow_file['flow'].attrs)
    delta_t = flow_attrs.get("delta_t", 10000)

    # create the preprocessor
    if path.endswith('h5'):
        preprocessor = HDF5Iterator(path, height=height, width=width)
    elif path.lower().endswith(".mp4"):
        simu_iterator = SimulatedEventsIterator(
            path, mode="delta_t", delta_t=delta_t, max_duration=max_duration, relative_timestamps=True, height=height,
            width=width, Cp=0.11, Cn=0.1, refractory_period=1e-3, sigma_threshold=0.0, cutoff_hz=0, leak_rate_hz=0,
            shot_noise_rate_hz=0)
        preprocessor = CDProcessorIterator.from_iterator(simu_iterator, "histo", height=height, width=width)
    else:
        preprocessor = CDProcessorIterator(path, "histo", delta_t=delta_t, max_duration=max_duration,
                                           height=height, width=width, start_ts=start_ts)
    # create the outputs
    if display:
        cv2.namedWindow(mode, cv2.WINDOW_NORMAL)
    process = FFmpegWriter(write_video) if write_video else None

    _proc(preprocessor, flow_file["flow"], mode=mode, video_process=process, mask_by_input=mask_by_input,
          display=display, step=step)

    # clean up
    if write_video:
        process.close()

    if display:
        cv2.destroyWindow(mode)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Visualize the content of a flow file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', help='RAW, HDF5 or DAT filename, leave blank to use a camera.'
                        'Warning if you use an HDF5 file the parameters used for precomputation must match those of'
                        ' the model.')
    parser.add_argument('flow_path', help=' HDF5 file containing a flow dataset corresponding to the file in input')
    parser.add_argument('--start-ts', type=int, default=0,
                        help='timestamp (in microseconds) from which the computation begins. ')
    parser.add_argument('--max-duration', type=int, default=None,
                        help='maximum duration of the inference file in us.')
    parser.add_argument('--step', type=int, default=4,
                        help='pixel interval between two arrows, 0 triggers dense visualization')
    parser.add_argument('--height-width', nargs=2, default=None, type=int,
                        help="if set, downscale the feature tensor to the requested resolution using interpolation"
                        " Possible values are only power of two of the original resolution.")
    parser.add_argument('--show-flow-everywhere', dest="mask_by_input", action="store_false", help="if set, this will "
                        "show flow arrows everywhere and not just when there are input events.")

    parser.add_argument("-w", "--write-video", default='',
                        help='if set, save the visualization in a .mp4 video at the indicated path.')
    parser.add_argument(
        "--no-display", dest="display", action="store_false", help='if set, deactivate the display Window')

    return parser.parse_args(argv) if argv is not None else parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    [height, width] = ARGS.height_width if ARGS.height_width is not None else [None, None]
    main(ARGS.path, ARGS.flow_path, write_video=ARGS.write_video, height=height, width=width,
         max_duration=ARGS.max_duration, mask_by_input=ARGS.mask_by_input, start_ts=ARGS.start_ts,
         display=ARGS.display, step=ARGS.step)
