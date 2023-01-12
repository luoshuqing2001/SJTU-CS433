import numpy as np

# for item in range(1, 21):
#     name = 'Conv_'+ str(item) +'_W'

#     detection_file = './npy_data/'+ name + '.npy'
#     detections = None
#     if detection_file is not None:
#         detections = np.load(detection_file)  # .npy文件
#         print(detections.shape)
#     # np.savetxt('./txt_data/Conv_1_W.txt', detections[0,0,:,:], fmt='%0.32f')
#     # print(detections)
#     file = open('./txt_data/'+ name +'.txt', 'w')
#     for i in range(detections.shape[0]):
#         for j in range(detections.shape[1]):
#             np.savetxt(file, detections[i,j,:,:], fmt='%0.32f')

# for item in range(1, 21):
#     name = 'Conv_'+ str(item) +'_B'

#     detection_file = './npy_data/'+ name + '.npy'
#     detections = None
#     if detection_file is not None:
#         detections = np.load(detection_file)  # .npy文件
#         print(detections.shape)
#     np.savetxt('./txt_data/'+ name +'.txt', detections, fmt='%0.32f')
    # print(detections)
    # file = open('./txt_data/'+ name +'.txt', 'w')
    # for i in range(detections.shape[0]):
    #     for j in range(detections.shape[1]):
    # np.savetxt(file, detections, fmt='%0.32f')
    
name = 'Fc_B'
detection_file = './npy_data/'+ name + '.npy'
detections = None
if detection_file is not None:
    detections = np.load(detection_file)  # .npy文件
    print(detections.shape)
np.savetxt('./txt_data/'+ name +'.txt', detections, fmt='%0.32f')