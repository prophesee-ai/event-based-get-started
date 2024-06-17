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

from metavision_core_ml.video_to_event import SimulatedEventsIterator
from metavision_ml.flow.lightning_model import FlowModel
from metavision_ml.data import CDProcessorIterator, HDF5Iterator
from metavision_ml.flow.viz import get_arrows, get_dense_flow
from metavision_ml.preprocessing.viz import normalize, filter_outliers

from metavision_ml.utils.h5_writer import HDF5Writer


def inference_sharpening(flow_model, tensor):
    """Performs inference of a flow model on an input tensor and builds a sharpening visualization.

    Args:
        flow_model (FlowModel): instance of flow lightnings Module.
        tensor (torch.Tensor): input tensor for the network representing one time bin of the network preprocessing.
    Returns:
        flow (torch.Tensor): output flow shape (1, 2, height, width)
        img (numpy.ndarray): color visualization of shape (height, width, 3) and dtype np.uint8
    """
    # sharpen means to move together the micro tbins of the tensor according to the optical flow.
    flow, sharp_img = flow_model.network.sharpen(tensor)
    channels, height, width = sharp_img.shape[-3:]
    num_micro_tbins = channels // 2
    sharp_img = sharp_img.reshape(num_micro_tbins, 2, height, width)
    sharp_img = sharp_img.mean(0)
    sharp_img = sharp_img.detach().cpu()
    sharp_img = sharp_img[1] - sharp_img[0]
    sharp_img = np.uint8(255 * normalize(filter_outliers(sharp_img, 7)))
    blur_img = tensor.reshape(-1, 2, *tensor.shape[-2:]).mean(0).detach().cpu()
    blur_img = blur_img[1] - blur_img[0]
    blur_img = np.uint8(255 * normalize(filter_outliers(blur_img, 9)))
    blur_img = get_arrows(flow[0], blur_img, step=8, mask=blur_img != 127)
    sharp_img = sharp_img[..., None].repeat(3, 2)
    img = np.concatenate((sharp_img, blur_img), axis=1)

    return flow, img


def inference_arrows(flow_model, tensor, viz_func, mask_by_input=True):
    """Performs inference of a flow model on an input tensor and draws arrows at regular intervals
    where the flow should be.

    Args:
        flow_model (FlowModel): instance of flow lightnings Module.
        tensor (torch.Tensor): input tensor for the network representing one time bin of the network preprocessing.
        mask_by_input (boolean): if True only display flow arrows on pixel with non null input.

    Returns:
        flow (torch.Tensor): output flow shape (1, 2, height, width)
        img (numpy.ndarray): color visualization of shape (height, width, 3) and dtype np.uint8
    """
    flow = flow_model.network(tensor)[-1][0, 0]
    tensor = tensor.detach()
    tensor = tensor[0, 0].cpu().numpy()
    img_gray = cv2.cvtColor(viz_func(tensor), cv2.COLOR_BGR2GRAY)
    img = np.dstack([img_gray, img_gray, img_gray])
    mask = (np.sum(np.abs(tensor), axis=0) > 0) if mask_by_input else None
    img = get_arrows(flow, base_img=img, step=4, mask=mask)
    return flow, img


@torch.no_grad()
def _proc(preprocessor, flow_model, mode="arrows", video_process=None,
          h5writer=None, mask_by_input=True, display=True):
    """Sub function performing preprocessing inference and visualization. """

    for tensor in tqdm(preprocessor):
        tensor = tensor[None]
        if mode == 'sharp':
            flow, img = inference_sharpening(flow_model, tensor)
        else:
            flow, img = inference_arrows(flow_model, tensor, preprocessor.get_vis_func(), mask_by_input=mask_by_input)

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

        if h5writer is not None:
            h5writer.write(flow.cpu().numpy()[None])


def main(path, delta_t, checkpoint_path, write_video="", cuda=False, mode="arrows", height=None, width=None,
         max_duration=None, save="", mask_by_input=True, start_ts=0, display=True):
    assert mode in ("sharp", "arrows")
    # load the network
    flow_model = FlowModel.load_from_checkpoint(checkpoint_path, strict=False)
    print(flow_model.hparams)

    device = torch.device('cuda') if cuda else torch.device('cpu')
    if cuda:
        flow_model.cuda()
    # create the preprocessor
    if path.endswith('h5'):
        preprocessor = HDF5Iterator(path, device=device, height=height, width=width)
        # ensures that HDF5 attributes match the desired input
        preprocessor.checks(flow_model.hparams["preprocess"], delta_t=delta_t)
    elif path.lower().endswith(".mp4"):
        simu_iterator = SimulatedEventsIterator(
            path, mode="delta_t", delta_t=delta_t, max_duration=max_duration, relative_timestamps=True, height=height,
            width=width, Cp=0.11, Cn=0.1, refractory_period=1e-3, sigma_threshold=0.0, cutoff_hz=0, leak_rate_hz=0,
            shot_noise_rate_hz=0)
        preprocessor = CDProcessorIterator.from_iterator(simu_iterator, flow_model.hparams["preprocess"],
                                                         device=device, height=height, width=width)

    else:
        preprocessor = CDProcessorIterator(path, flow_model.hparams["preprocess"], delta_t=delta_t,
                                           max_duration=max_duration, device=device,
                                           height=height, width=width, start_ts=start_ts)
    # create the outputs
    if display:
        cv2.namedWindow(mode, cv2.WINDOW_NORMAL)
    process = FFmpegWriter(write_video) if write_video else None
    if save:
        dirname = os.path.dirname(save)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        shape = (2,) + tuple(preprocessor.array_dim[-2:])
        h5w = HDF5Writer(
            save, "flow", shape, dtype=np.float32,
            attrs={"events_to_tensor": np.string_(flow_model.hparams["preprocess"]),
                   'checkpoint_path': os.path.basename(checkpoint_path),
                   'input_file_name': os.path.basename(path),
                   "delta_t": np.uint32(delta_t),
                   'event_input_height': preprocessor.event_input_height,
                   "event_input_width": preprocessor.event_input_width})
        h5w.dataset_size_increment = 100
    else:
        h5w = None

    flow_model.network.eval()
    flow_model.network.reset()
    _proc(preprocessor, flow_model, mode=mode, video_process=process, h5writer=h5w, mask_by_input=mask_by_input,
          display=display)

    # clean up
    if write_video:
        process.close()
    if save:
        h5w.close()
        # Update hdf5
        flow_h5 = h5py.File(save, "r+")
        T, C, H, W = flow_h5["flow"].shape
        flow_start_ts_np = np.arange(0, T * delta_t, delta_t)
        flow_end_ts_np = np.arange(delta_t, T * delta_t + 1, delta_t)
        flow_h5.create_dataset("flow_start_ts", data=flow_start_ts_np, compression="gzip")
        flow_h5.create_dataset("flow_end_ts", data=flow_end_ts_np, compression="gzip")
        flow_h5.close()

    if display:
        cv2.destroyWindow(mode)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Perform inference with a flow network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('checkpoint', help='path to the checkpoint containing the neural network name.')
    parser.add_argument('path', help='RAW, HDF5 or DAT filename, leave blank to use a camera.'
                        'Warning if you use an HDF5 file the parameters used for precomputation must match those of'
                        ' the model.')
    parser.add_argument('--delta-t', type=int, default=50000,
                        help='duration of timeslice (in us) in which events are accumulated'
                        ' to compute features.')
    parser.add_argument('--start-ts', type=int, default=0,
                        help='timestamp (in microseconds) from which the computation begins. ')
    parser.add_argument('--max-duration', type=int, default=None,
                        help='maximum duration of the inference file in us.')
    parser.add_argument('--mode', default="arrows", choices=('sharp', "arrows"),
                        help='Either show arrows or show the sharpening effect')
    parser.add_argument('--height-width', nargs=2, default=None, type=int,
                        help="if set, downscale the feature tensor to the requested resolution using interpolation"
                        " Possible values are only power of two of the original resolution.")
    parser.add_argument('--show-flow-everywhere', dest="mask_by_input", action="store_false", help="if set, this will "
                        "show flow arrows everywhere and not just when there are input events.")
    parser.add_argument("--cuda", action="store_true", help='run on GPU')
    parser.add_argument("-s", "--save-flow-hdf5", default='',
                        help='if set save the flow in a HDF5 format at the given path')
    parser.add_argument("-w", "--write-video", default='',
                        help='if set, save the visualization in a .mp4 video at the indicated path.')
    parser.add_argument(
        "--no-display", dest="display", action="store_false", help='if set, deactivate the display Window')

    return parser.parse_args(argv) if argv is not None else parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    [height, width] = ARGS.height_width if ARGS.height_width is not None else [None, None]
    main(ARGS.path, ARGS.delta_t, ARGS.checkpoint, write_video=ARGS.write_video, cuda=ARGS.cuda, mode=ARGS.mode,
         height=height, width=width, max_duration=ARGS.max_duration, save=ARGS.save_flow_hdf5,
         mask_by_input=ARGS.mask_by_input, start_ts=ARGS.start_ts, display=ARGS.display)
