import cv2
import time
import os
import vart
import xir
import numpy
import torch
import torchvision

def GetDPUIDims(dpu):
    return [a.dims for a in dpu.get_input_tensors()]


def GetDPUODims(dpu):
    return [a.dims for a in dpu.get_output_tensors()]


def GetDPUOBuffer(dpu):
    return [numpy.zeros(a, dtype='float32') for a in GetDPUODims(dpu)]


def execute_async(dpu, iBuffer, oBuffer):
    jid = dpu.execute_async(iBuffer, oBuffer)
    return dpu.wait(jid)


def ResizeImg(img):
    # (1, 2160, 4096, 3) to (1, 2176, 3840, 3)
    img = numpy.pad(img, ((0, 0), (8, 8), (0, 0), (0, 0)), 'constant')
    return img


def RunDPU(dpu, img):
    out = GetDPUOBuffer(dpu)
    execute_async(dpu, [img], out)
    return out

def postprocess_1(outputs):
        outputs = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs]
        hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        # outputs = torch.cat([x.view(x.size(0), x.size(1), -1) for x in outputs], dim=2).permute(0, 2, 1)
        outputs[..., 4:] = outputs[..., 4:].sigmoid()
        return decode_outputs(hw, outputs, dtype=outputs[0].type())

def decode_outputs(hw, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw,[8, 16, 32]):
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

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
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
    # Initialize the webcam
    cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
   

    img_counter = 0
    color = (0, 0, 255)
    thickness = 2

    while True:
        frame_start_time = time.time()  # Start time for frame capture
        ret, frame = cap.read()
        if ret:
            #print(frame.shape)
            img = cv2.resize(frame,(3840,2176))
            #print(img.shape) #(2176, 3840, 3)
            img = img[numpy.newaxis, :, :, :].astype(numpy.float32)
            #print(img.shape, img.dtype) #(1, 2176, 3840, 3)
            
            out = RunDPU(dpuYoloN, img)
            out = postprocess_2(postprocess_1(out))
            # print("oit imgpre",img_pre.shape)
            # print("\nout 1 :", out[0].shape)
            # print("\nout 2 :", out[1].shape)
            # print("\nout 3 :", out[2].shape)
            # #print("\nout length is:", len(out))
            #prediction = postprocess_2(torch.from_numpy(out))
            #print(len(out), out[0])

            
            
            if out[0] is not None:
                for pred in out:
                    #print(pred)
                    # Iterate over each prediction tensor
                    for bbox in pred:
                        #print(bbox)
                        # Extract and convert the bounding box coordinates
                        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                        #print(x1,y1,x2,y2)
                        # calculate coordinates
                        start_point = (int(x1/2),int(y1/2))
                        end_point = (int(x2/2),int(y2/2))
                        #print(start_point,end_point)
                        #print(frame.shape)
                        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            cv2.imshow('Frame with Bounding Box', frame)


            # Calculate the time taken to capture the frame
            frame_end_time = time.time()
            capture_duration = frame_end_time - frame_start_time

            # Calculate and print the achieved FPS
            if capture_duration > 0:
                achieved_fps = 1.0 / capture_duration
                print(f"Frame {img_counter}: Achieved FPS: {achieved_fps:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            img_counter += 1

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()



g = xir.Graph.deserialize('./no_qat/compiled_bird.xmodel')
#subgraph_YOLOX__YOLOX_YOLOPAFPN_backbone__BaseConv_lateral_conv0__Conv2d_conv__ret_217 is the yolo model
sg = g.get_root_subgraph().toposort_child_subgraph()
dpuYoloN = vart.Runner.create_runner(sg[1], "run")

inference(dpuYoloN)
