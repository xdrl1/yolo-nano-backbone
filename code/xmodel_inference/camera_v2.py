import cv2
import time
import numpy as np
import vart
import xir
import torch
import torchvision

def GetDPUIDims(dpu):
    return [a.dims for a in dpu.get_input_tensors()]

def GetDPUODims(dpu):
    return [a.dims for a in dpu.get_output_tensors()]

def GetDPUOBuffer(dpu):
    return [np.zeros(a, dtype='float32') for a in GetDPUODims(dpu)]

def execute_async(dpu, iBuffer, oBuffer):
    jid = dpu.execute_async(iBuffer, oBuffer)
    return dpu.wait(jid)

def ResizeImg(img):
    img = np.pad(img, ((0, 0), (8, 8), (0, 0), (0, 0)), 'constant')
    return img

def RunDPU(dpu, img):
    out = GetDPUOBuffer(dpu)
    execute_async(dpu, [img], out)
    return out

def postprocess_1(outputs):
    outputs = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs]
    hw = [x.shape[-2:] for x in outputs]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    outputs[..., 4:] = outputs[..., 4:].sigmoid()
    return decode_outputs(hw, outputs, dtype=outputs[0].type())

def decode_outputs(hw, outputs, dtype):
    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, [8, 16, 32]):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs

def postprocess_2(prediction, num_classes=1, conf_thre=0.04, nms_thre=0.25, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def inference(dpu):
    gst_pipeline = (
        "rtspsrc location=rtsp://admin:vogelki@192.168.178.28/Preview_01_main latency=0 ! "
        "rtph265depay ! h265parse ! omxh265dec ! "
        "videoconvert ! appsink"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    img_counter = 0
    color = (0, 0, 255)
    thickness = 2

    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return

    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (3840, 2176))
            img = img[np.newaxis, :, :, :].astype(np.float32)
            
            out = RunDPU(dpu, img)
            out = postprocess_2(postprocess_1(out))

            if out[0] is not None:
                for pred in out:
                    for bbox in pred:
                        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                        start_point = (int(x1/2), int(y1/2))
                        end_point = (int(x2/2), int(y2/2))
                        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            
            cv2.imshow('Frame with Bounding Box', frame)

            frame_end_time = time.time()
            capture_duration = frame_end_time - frame_start_time

            if capture_duration > 0:
                achieved_fps = 1.0 / capture_duration
                print(f"Frame {img_counter}: Achieved FPS: {achieved_fps:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()

g = xir.Graph.deserialize('./no_qat/compiled_bird.xmodel')
sg = g.get_root_subgraph().toposort_child_subgraph()
dpuYoloN = vart.Runner.create_runner(sg[1], "run")

inference(dpuYoloN)
