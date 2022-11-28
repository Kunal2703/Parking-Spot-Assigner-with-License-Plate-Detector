from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.plots import plot_one_box
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
import re
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, ones, float32, maximum, minimum, zeros, uint8, absolute, copyto
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import mysql.connector


mydb = mysql.connector.connect(host = "localhost", user = "root", passwd = "1234", database="minorproject")     #establishing connection with mysql
mycursor = mydb.cursor()                                                                                        #creating cursor object

query="DELETE FROM entry"                                                                                       #removing previous records
mycursor.execute(query)
mydb.commit()

entry=dict()                                                                                                    
dept1=[1,2,3,4,5,6,7,8,9,10]                                                                                    #parking slots for dept
dept2=[11,12,13,14,15,16,17,18,19,20]
dept3=[21,22,23,24,25,26,27,28,29,30]
dept4=[31,32,33,34,35,36,37,38,39,40]


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = maximum(sharpened, zeros(sharpened.shape))
    sharpened = minimum(sharpened, 255 * ones(sharpened.shape))
    sharpened = sharpened.round().astype(uint8)
    if threshold > 0:
        low_contrast_mask = absolute(image - blurred) < threshold
        copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def search(numplate):
    global entry
    if numplate not in entry.keys():
        tup1=(numplate,)
        query1="SELECT dept_no FROM empdetails where numplate = %s"
        mycursor.execute(query1, tup1)                                                                         #searching numplate in database
        res=mycursor.fetchall()
        if res:
            dept=res[0]
            dept=dept[0]
            if dept==1:
                slot=dept1.pop(0)                                                                              #assigning slot
                entry[numplate]=[slot,dept]
            elif dept==2:
                slot=dept2.pop(0)
                entry[numplate]=[slot,dept]
            elif dept==3:
                slot=dept3.pop(0)
                entry[numplate]=[slot,dept]
            elif dept==4:
                slot=dept4.pop(0)
                entry[numplate]=[slot,dept]
        
            query2="INSERT INTO entry (Numberplate, Slot, Dept_no) VALUES(%s, %s, %s)"                          #inserting detail in table
            tup2=(numplate,slot,dept)
            mycursor.execute(query2, tup2)
            mydb.commit()
            print("slot assigned: ",slot)
        
        else:
            print("not found")
    
    else:
        return

def imgtotext(cropped_img):
    
    img = cropped_img  # loading image
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to grayscale
    # noise reduction using bilateral filter
    denoised_img = cv2.bilateralFilter(gray_img, 11, 17, 17)
    thresh1 = cv2.adaptiveThreshold(
        denoised_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
    # text = pytesseract.image_to_string(thresh1, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11')
    text = pytesseract.image_to_string(
        denoised_img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3 --user-patterns C:/Users/prana/Desktop/Minor Project/ANPR/ANPRyolov7/yolov7-anpr/pat.patterns')
    text=text.strip('\n')
    
    return text

    '''
    img=cropped_img
    img=cv2.resize(img, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting to grayscale
    kernel = ones((1, 1), uint8)
    dilated = cv2.dilate(gray_img, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    sharpened = unsharp_mask(eroded)

    text=pytesseract.image_to_string(sharpened, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11')
   
    return text

    
    '''

    '''
    img=cv2.imread('C:\\Users\\prana\\Desktop\\Minor Project\\ANPR\\ANPRyolov7\\yolov7-anpr\\test.jpg',0)
    img=cv2.resize(img, None, fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    kernel2=np.ones((2,2), np.float32)
    blur=cv2.GaussianBlur(img, (5,5), 0)

    img=cv2.medianBlur(img, 3)

    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    im2 = img.copy()

    plate_num = ""
        
    
    # loop through contours and find letters in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
                                    
        # if height of box is not a quarter of total height then skip
        if height / float(h) > 6: continue
        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue
        # if width is not more than 25 pixels skip
        if width / float(w) > 25: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 3)
        roi = cv2.filter2D(src=roi, ddepth=-1, kernel=kernel2)
        #cv2.imshow("ROI", roi)
        #cv2.waitKey(0)
        text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        #print(text)
        plate_num+=text
    final_num='' 
    l=plate_num.split()
    text=final_num.join(l)
    
    return text
    '''


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    #######################################################################################################
    '''
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    '''
    save_dir = None
    #########################################################################################################

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (
                            cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        # with open(txt_path + '.txt', 'a') as f:
                        #   f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=1)

                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())

                        confidence_score = float(conf)
                        class_index = cls
                        object_name = names[int(cls)]

                        #print('bounding box is ', x1, y1, x2, y2)
                        #print('detected object name is ', object_name)
                        #print("confidence = ",confidence_score)

                        original_img = im0
                        cropped_img = im0[y1:y2, x1:x2]
                        text= imgtotext(cropped_img)
                        if bool(re.match("[a-zA-Z][a-zA-Z]\d\d[a-zA-Z]{1,2}\d\d\d\d", text)):
                            print("Detected Number is:", text)
                            search(text)
                        #cv2.imwrite('test.jpg', cropped_img)

                        
            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            '''
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            
            '''
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov7.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    #parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
