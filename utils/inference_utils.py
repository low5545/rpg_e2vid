from .util import robust_min, robust_max
from .path_utils import ensure_dir
from .timers import Timer, CudaTimer
from .loading_utils import get_device
from os.path import join, splitext, relpath
from math import ceil, floor
from torch.nn import ReflectionPad2d
import numpy as np
import torch
import cv2
from collections import deque
import atexit
import scipy.stats as st
import torch.nn.functional as F
from math import sqrt, pi, atan
import json

from external.event_nerf.datasets import CameraPose
from external.event_nerf.trajectories import LinearTrajectory


def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview


def gkern(kernlen=5, nsig=1.0):
    """Returns a 2D Gaussian kernel array."""
    """https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).float()


class EventPreprocessor:
    """
    Utility class to preprocess event tensors.
    Can perform operations such as hot pixel removing, event tensor normalization,
    or flipping the event tensor.
    """

    def __init__(self, options):

        print('== Event preprocessing ==')
        self.no_normalize = options.no_normalize
        if self.no_normalize:
            print('!!Will not normalize event tensors!!')
        else:
            print('Will normalize event tensors.')

        self.hot_pixel_locations = []
        if options.hot_pixels_file:
            try:
                self.hot_pixel_locations = np.loadtxt(options.hot_pixels_file, delimiter=',').astype(np.int)
                print('Will remove {} hot pixels'.format(self.hot_pixel_locations.shape[0]))
            except IOError:
                print('WARNING: could not load hot pixels file: {}'.format(options.hot_pixels_file))

        self.flip = options.flip
        if self.flip:
            print('Will flip event tensors.')

    def __call__(self, events):

        # Remove (i.e. zero out) the hot pixels
        for x, y in self.hot_pixel_locations:
            events[:, :, y, x] = 0

        # Flip tensor vertically and horizontally
        if self.flip:
            events = torch.flip(events, dims=[2, 3])

        # Normalize the event tensor (voxel grid) so that
        # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
        if not self.no_normalize:
            with CudaTimer('Normalization'):
                nonzero_ev = (events != 0)
                num_nonzeros = nonzero_ev.sum()
                if num_nonzeros > 0:
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array
                    mean = events.sum() / num_nonzeros
                    stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                    mask = nonzero_ev.float()
                    events = mask * (events - mean) / stddev

        return events


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self, options):
        self.auto_hdr = options.auto_hdr
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = options.auto_hdr_median_filter_size
        self.Imin = options.Imin
        self.Imax = options.Imax

    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:
            with CudaTimer('Compute Imin/Imax (auto HDR)'):
                Imin = torch.min(img).item()
                Imax = torch.max(img).item()

                # ensure that the range is at least 0.1
                Imin = np.clip(Imin, 0.0, 0.45)
                Imax = np.clip(Imax, 0.55, 1.0)

                # adjust image dynamic range (i.e. its contrast)
                if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                    self.intensity_bounds.popleft()

                self.intensity_bounds.append((Imin, Imax))
                self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
                self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        with CudaTimer('Intensity rescaling'):
            img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
            img.clamp_(0.0, 255.0)
            img = img.byte()  # convert to 8-bit tensor

        return img


class ImageWriter:
    """
    Utility class to write images to disk.
    Also writes the image timestamps into a text file.
    """

    def __init__(self, options):

        self.output_folder = options.output_folder
        self.dataset_name = options.dataset_name
        self.save_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show
        print('== Image Writer ==')
        if self.output_folder:
            ensure_dir(self.output_folder)
            ensure_dir(join(self.output_folder, self.dataset_name))
            print('Will write images to: {}'.format(join(self.output_folder, self.dataset_name)))
            self.transforms_path = join(self.output_folder,
                                        f"transforms_{self.dataset_name}.json")

            if self.save_events:
                self.event_previews_folder = join(self.output_folder, self.dataset_name, 'events')
                ensure_dir(self.event_previews_folder)
                print('Will write event previews to: {}'.format(self.event_previews_folder))

            # initialize transforms
            camera_calibration_path = join(
                options.input_file, "camera_calibration.npz"
            )
            camera_calibration = np.load(camera_calibration_path)
            self.intrinsics = camera_calibration["intrinsics"]
            self.distortion_params = camera_calibration["distortion_params"]
            self.distortion_model = camera_calibration["distortion_model"]
            img_width = int(camera_calibration["img_width"])
            img_height = int(camera_calibration["img_height"])

            if len(self.distortion_params) == 0:
                self.new_intrinsics = self.intrinsics
            elif self.distortion_model == "plumb_bob":
                self.new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(
                    self.intrinsics, self.distortion_params,
                    (img_width, img_height), alpha=0
                )
                assert (roi == (0, 0, img_width - 1, img_height - 1))
            elif self.distortion_model == "equidistant":
                self.new_intrinsics = (
                    cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        self.intrinsics, self.distortion_params,
                        (img_width, img_height), R=np.eye(3, dtype=np.float32),
                        balance=0
                    )
                )
            elif self.distortion_model == "fov":
                raise NotImplementedError       # TODO
            else:
                raise NotImplementedError

            self.transforms = {
                "frames": []
            }
            new_focal_len_x = float(self.new_intrinsics[0, 0])
            new_focal_len_y = float(self.new_intrinsics[1, 1])
            new_principal_pt_x = float(self.new_intrinsics[0, 2])
            new_principal_pt_y = float(self.new_intrinsics[1, 2])
            if (
                (new_focal_len_x == new_focal_len_y)
                and (new_principal_pt_x == (img_width - 1) / 2)
                and (new_principal_pt_y == (img_height - 1) / 2)
            ):
                self.transforms["camera_angle_x"] = (
                    2 * atan((img_width / 2) / new_focal_len_x)
                )
            else:
                self.transforms["intrinsics"] = self.new_intrinsics.tolist()

            # arbitrarily define `self.stepsize`
            events_path = join(options.input_file, "raw_events.npz")
            events = np.load(events_path)

            N = int(img_width * img_height * options.num_events_per_pixel)
            num_events = len(events["polarity"])
            num_views = round(num_events / N)
            self.stepsize = 2 * pi / num_views

            # initialize trajectory
            camera_poses = CameraPose(
                root_directory=options.input_file,
                permutation_seed=None,
                downsampling_factor=options.traj_downsampling_factor,
                upsampling_factor=1,
                upsampling_algo="linear"
            )
            self.trajectory = LinearTrajectory(camera_poses)

            atexit.register(self.__cleanup__)
        else:
            print('Will not write images to disk.')

    def __call__(self, img, event_tensor_id, stamp=None, events=None):
        if not self.output_folder:
            return

        img_index = len(self.transforms["frames"])
        if self.save_events and events is not None:
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            cv2.imwrite(join(self.event_previews_folder,
                             'events_{:d}.png'.format(img_index)), event_preview)

        # undistort image & save to disk
        if len(self.distortion_params) > 0:
            if self.distortion_model == "plumb_bob":
                img = cv2.undistort(
                    img, self.intrinsics, self.distortion_params,
                    newCameraMatrix=self.new_intrinsics
                )
            elif self.distortion_model == "equidistant":
                img = cv2.fisheye.undistortImage(
                    img, self.intrinsics, self.distortion_params,
                    Knew=self.new_intrinsics
                )
            elif self.distortion_model == "fov":
                raise NotImplementedError       # TODO
            else:
                raise NotImplementedError
        cv2.imwrite(join(self.output_folder, self.dataset_name,
                         'r_{:d}.png'.format(img_index)), img)
        if stamp is not None:
            # interpolate the camera pose at the image timestamp
            timestamp_ns = torch.tensor(
                stamp * 1e+9, dtype=torch.get_default_dtype()
            )
            T_wc_position, T_wc_orientation = self.trajectory(timestamp_ns)

            # convert the camera pose from a common to the OpenGL convention
            T_CCOMMON_COPENGL_ORIENTATION = torch.tensor(
                [[1,  0,  0],
                 [0, -1,  0],
                 [0,  0, -1]],
                dtype=T_wc_orientation.dtype
            )
            T_w_ccommon_orientation = T_wc_orientation
            T_w_copengl_orientation = T_w_ccommon_orientation \
                                      @ T_CCOMMON_COPENGL_ORIENTATION
            T_wc_orientation = T_w_copengl_orientation

            # convert the camera pose to a homogenous transformation matrix
            T_wc = torch.eye(4, dtype=T_wc_orientation.dtype)
            T_wc[:3, :3] = T_wc_orientation
            T_wc[:3, 3] = T_wc_position

            # append to the transforms
            self.transforms["frames"].append({
                "file_path": join(".", self.dataset_name,
                                  f"r_{img_index}"),
                "rotation": self.stepsize,
                "transform_matrix": T_wc.tolist()
            })

    def __cleanup__(self):
        if self.output_folder:
            with open(self.transforms_path, "w") as f:
                json.dump(self.transforms, f, indent=4)


class ImageDisplay:
    """
    Utility class to display image reconstructions
    """

    def __init__(self, options):
        self.display = options.display
        self.show_events = options.show_events
        self.color = options.color
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show

        self.window_name = 'Reconstruction'
        if self.show_events:
            self.window_name = 'Events | ' + self.window_name

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.border = options.display_border_crop
        self.wait_time = options.display_wait_time

    def crop_outer_border(self, img, border):
        if self.border == 0:
            return img
        else:
            return img[border:-border, border:-border]

    def __call__(self, img, events=None):

        if not self.display:
            return

        img = self.crop_outer_border(img, self.border)

        if self.show_events:
            assert(events is not None)
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            event_preview = self.crop_outer_border(event_preview, self.border)

        if self.show_events:
            img_is_color = (len(img.shape) == 3)
            preview_is_color = (len(event_preview.shape) == 3)

            if(preview_is_color and not img_is_color):
                img = np.dstack([img] * 3)
            elif(img_is_color and not preview_is_color):
                event_preview = np.dstack([event_preview] * 3)

            img = np.hstack([event_preview, img])

        cv2.imshow(self.window_name, img)
        cv2.waitKey(self.wait_time)


class UnsharpMaskFilter:
    """
    Utility class to perform unsharp mask filtering on reconstructed images.
    """

    def __init__(self, options, device):
        self.unsharp_mask_amount = options.unsharp_mask_amount
        self.unsharp_mask_sigma = options.unsharp_mask_sigma
        self.gaussian_kernel_size = 5
        self.gaussian_kernel = gkern(self.gaussian_kernel_size,
                                     self.unsharp_mask_sigma).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, img):
        if self.unsharp_mask_amount > 0:
            with CudaTimer('Unsharp mask'):
                blurred = F.conv2d(img, self.gaussian_kernel,
                                   padding=self.gaussian_kernel_size // 2)
                img = (1 + self.unsharp_mask_amount) * img - self.unsharp_mask_amount * blurred
        return img


class ImageFilter:
    """
    Utility class to perform some basic filtering on reconstructed images.
    """

    def __init__(self, options):
        self.bilateral_filter_sigma = options.bilateral_filter_sigma

    def __call__(self, img):

        if self.bilateral_filter_sigma:
            with Timer('Bilateral filter (sigma={:.2f})'.format(self.bilateral_filter_sigma)):
                filtered_img = np.zeros_like(img)
                filtered_img = cv2.bilateralFilter(
                    img, 5, 25.0 * self.bilateral_filter_sigma, 25.0 * self.bilateral_filter_sigma)
                img = filtered_img

        return img


def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = np.expand_dims(X[dy, :], axis=0)
    elif dy < 0:
        X[dy:, :] = np.expand_dims(X[dy, :], axis=0)
    if dx > 0:
        X[:, :dx] = np.expand_dims(X[:, dx], axis=1)
    elif dx < 0:
        X[:, dx:] = np.expand_dims(X[:, dx], axis=1)
    return X


def upsample_color_image(grayscale_highres, color_lowres_bgr, colorspace='LAB'):
    """
    Generate a high res color image from a high res grayscale image, and a low res color image,
    using the trick described in:
    http://www.planetary.org/blogs/emily-lakdawalla/2013/04231204-image-processing-colorizing-images.html
    """
    assert(len(grayscale_highres.shape) == 2)
    assert(len(color_lowres_bgr.shape) == 3 and color_lowres_bgr.shape[2] == 3)

    if colorspace == 'LAB':
        # convert color image to LAB space
        lab = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2LAB)
        # replace lightness channel with the highres image
        lab[:, :, 0] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=lab, code=cv2.COLOR_LAB2BGR)
    elif colorspace == 'HSV':
        # convert color image to HSV space
        hsv = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HSV)
        # replace value channel with the highres image
        hsv[:, :, 2] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hsv, code=cv2.COLOR_HSV2BGR)
    elif colorspace == 'HLS':
        # convert color image to HLS space
        hls = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HLS)
        # replace lightness channel with the highres image
        hls[:, :, 1] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hls, code=cv2.COLOR_HLS2BGR)

    return color_highres_bgr


def merge_channels_into_color_image(channels):
    """
    Combine a full resolution grayscale reconstruction and four color channels at half resolution
    into a color image at full resolution.

    :param channels: dictionary containing the four color reconstructions (at quarter resolution),
                     and the full resolution grayscale reconstruction.
    :return a color image at full resolution
    """

    with Timer('Merge color channels'):

        assert('R' in channels)
        assert('G_TR' in channels)
        assert('G_BL' in channels)
        assert('B' in channels)
        assert('grayscale' in channels)

        # upsample each channel independently
        for channel in ['R', 'G_TR', 'G_BL', 'B']:
            channels[channel] = cv2.resize(channels[channel], dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Shift the channels so that they all have the same origin
        channels['B'] = shift_image(channels['B'], dx=1, dy=1)
        channels['G_TR'] = shift_image(channels['G_TR'], dx=1, dy=0)
        channels['G_BL'] = shift_image(channels['G_BL'], dx=0, dy=1)

        # merge the interpolated, full resolution top-right & bottom-left green
        # channel images
        H, W = channels['G_TR'].shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        is_top_right = (x % 2 == 1) & (y % 2 == 0)
        is_bottom_left = (x % 2 == 0) & (y % 2 == 1)
        is_others = ~(is_top_right | is_bottom_left)

        channels['G_MEAN'] = cv2.addWeighted(
            src1=channels['G_TR'], alpha=0.5,
            src2=channels['G_BL'], beta=0.5,
            gamma=0.0, dtype=cv2.CV_8U
        )
        channels['G'] = np.empty_like(channels['G_TR'])
        channels['G'][is_top_right] = channels['G_TR'][is_top_right]
        channels['G'][is_bottom_left] = channels['G_BL'][is_bottom_left]
        channels['G'][is_others] = channels['G_MEAN'][is_others]

        # reconstruct the color image at half the resolution using the reconstructed channels RGBW
        reconstruction_bgr = np.dstack([channels['B'],
                                        channels['G'],
                                        channels['R']])

        reconstruction_grayscale = channels['grayscale']

        # combine the full res grayscale resolution with the low res to get a full res color image
        upsampled_img = upsample_color_image(reconstruction_grayscale, reconstruction_bgr)
        return upsampled_img

    return upsampled_img


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events)
        with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(device)

        with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices]
                                  * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                  + (tis_long[valid_indices] + 1) * width * height,
                                  source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid
