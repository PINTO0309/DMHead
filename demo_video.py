
import cv2
import time
import math
import copy
import argparse
import onnxruntime
import numpy as np
from math import cos, sin
from typing import Tuple, Optional, List


class YOLOv7ONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'yolov7_tiny_head_0.768_post_480x640.onnx',
        class_score_th: Optional[float] = 0.20,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """YOLOv7ONNX
        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for YOLOv7
        class_score_th: Optional[float]
        class_score_th: Optional[float]
            Score threshold. Default: 0.30
        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOv7ONNX
        Parameters
        ----------
        image: np.ndarray
            Entire image
        Returns
        -------
        face_boxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]
        face_scores: np.ndarray
            Predicted face box scores: [facecount, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        scores, boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )

        # PostProcess
        face_boxes, face_scores = self.__postprocess(
            image=temp_image,
            scores=scores,
            boxes=boxes,
        )

        return face_boxes, face_scores


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess
        Parameters
        ----------
        image: np.ndarray
            Entire image
        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)
        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess
        Parameters
        ----------
        image: np.ndarray
            Entire image.
        scores: np.ndarray
            float32[N, 1]
        boxes: np.ndarray
            int64[N, 6]
        Returns
        -------
        faceboxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]
        facescores: np.ndarray
            Predicted face box confs: [facecount, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Head Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0
            classid -> always 0: "Head"
        scores: float32[N,1],
        batchno_classid_y1x1y2x2: int64[N,6],
        """
        scores = scores
        keep_idxs = scores[:, 0] > self.class_score_th
        scores_keep = scores[keep_idxs, :]
        boxes_keep = boxes[keep_idxs, :]
        faceboxes = []
        facescores = []

        if len(boxes_keep) > 0:
            for box, score in zip(boxes_keep, scores_keep):
                x_min = max(int(box[3]), 0)
                y_min = max(int(box[2]), 0)
                x_max = min(int(box[5]), image_width)
                y_max = min(int(box[4]), image_height)

                faceboxes.append(
                    [x_min, y_min, x_max, y_max]
                )
                facescores.append(
                    score
                )

        return np.asarray(faceboxes), np.asarray(facescores)



def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    if math.isnan(yaw) or math.isnan(pitch) or math.isnan(roll):
        return img
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def main(args):
    # YOLOv7_tiny_Head
    yolov7_head = YOLOv7ONNX(
        class_score_th=0.20,
    )

    # DMHead
    model_file_path = ''
    dmhead_input_name = None
    mask_or_nomask = args.mask_or_nomask

    if mask_or_nomask == 'mask':
        model_file_path = 'dmhead_mask_Nx3x224x224.onnx'
    elif mask_or_nomask == 'nomask':
        model_file_path = 'dmhead_nomask_Nx3x224x224.onnx'

    dmhead = onnxruntime.InferenceSession(
        path_or_bytes=model_file_path,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    )
    dmhead_input_name = dmhead.get_inputs()[0].name
    dmhead_H = dmhead.get_inputs()[0].shape[2]
    dmhead_W = dmhead.get_inputs()[0].shape[3]

    cap_width = int(args.height_width.split('x')[1])
    cap_height = int(args.height_width.split('x')[0])
    if args.device.isdecimal():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    WINDOWS_NAME = 'Demo'
    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS_NAME, cap_width, cap_height)

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # ============================================================= YOLOv7_tiny_Head
        heads, head_scores = yolov7_head(frame)

        canvas = copy.deepcopy(frame)
        # ============================================================= DMHead
        croped_resized_frame = None

        if len(heads) > 0:
            dmhead_inputs = []
            dmhead_position = []

            for head in heads:
                x_min = int(head[0])
                y_min = int(head[1])
                x_max = int(head[2])
                y_max = int(head[3])

                # enlarge the bbox to include more background margin
                y_min = max(0, y_min - abs(y_min - y_max) / 10)
                y_max = min(frame.shape[0], y_max + abs(y_min - y_max) / 10)
                x_min = max(0, x_min - abs(x_min - x_max) / 5)
                x_max = min(frame.shape[1], x_max + abs(x_min - x_max) / 5)
                x_max = min(x_max, frame.shape[1])
                croped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                # h,w -> 224,224
                croped_resized_frame = cv2.resize(croped_frame, (dmhead_W, dmhead_H))
                # bgr --> rgb
                rgb = croped_resized_frame[..., ::-1]
                # hwc --> chw
                chw = rgb.transpose(2, 0, 1)
                dmhead_inputs.append(chw)
                dmhead_position.append([x_min,y_min,x_max,y_max])
            # chw --> nchw
            nchw = np.asarray(dmhead_inputs, dtype=np.float32)
            positions = np.asarray(dmhead_position, dtype=np.int32)

            yaw = 0.0
            pitch = 0.0
            roll = 0.0
            # Inference DMHead
            outputs = dmhead.run(
                None,
                input_feed = {dmhead_input_name: nchw}
            )[0]

            for (yaw, roll, pitch), position in zip(outputs, positions):
                yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
                print(f'yaw: {yaw}, pitch: {pitch}, roll: {roll}')

                x_min,y_min,x_max,y_max = position

                # BBox draw
                deg_norm = 1.0 - abs(yaw / 180)
                blue = int(255 * deg_norm)
                cv2.rectangle(
                    canvas,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    color=(blue, 0, 255-blue),
                    thickness=2
                )

                # Draw
                draw_axis(
                    canvas,
                    yaw,
                    pitch,
                    roll,
                    tdx=(x_min+x_max)/2,
                    tdy=(y_min+y_max)/2,
                    size=abs(x_max-x_min)//2
                )
                cv2.putText(
                    canvas,
                    f'yaw: {np.round(yaw)}',
                    (int(x_min), int(y_min)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 255, 0),
                    1
                )
                cv2.putText(
                    canvas,
                    f'pitch: {np.round(pitch)}',
                    (int(x_min), int(y_min) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 255, 0),
                    1
                )
                cv2.putText(
                    canvas,
                    f'roll: {np.round(roll)}',
                    (int(x_min), int(y_min)-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 255, 0),
                    1
                )

        time_txt = f'{(time.time()-start)*1000:.2f} ms (inference+post-process)'
        cv2.putText(
            canvas,
            time_txt,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            time_txt,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        cv2.imshow(WINDOWS_NAME, canvas)
        video_writer.write(canvas)

    cv2.destroyAllWindows()

    if video_writer:
        video_writer.release()

    if cap:
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        '--height_width',
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    parser.add_argument(
        '--mask_or_nomask',
        type=str,
        default='mask',
        choices=[
            'mask',
            'nomask',
        ],
        help='\
            Select either a model that provides high accuracy when wearing \
            a mask or a model that provides high accuracy when not wearing a mask.',
    )
    args = parser.parse_args()
    main(args)
