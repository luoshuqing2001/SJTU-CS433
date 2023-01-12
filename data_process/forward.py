import onnxruntime as ort
import numpy as np
import time
import copy
import onnx

model = onnx.load("./resnet18.onnx")

for name in range(1):
    print("Figure input %d" % (name + 1));
    # read the processed input
    with open('./input_data/'+ str(name + 1) +'.txt', 'r') as f:
        line=f.readline()
        data_array=[]
        while line:
            num=list(map(float,line.split(' ')))
            data_array.append(num)
            line=f.readline()
        data_array=np.array(data_array)

    in_data = np.random.randn(1, 3, 224, 224)
    for i in range(3):
        for j in range(224):
            for k in range(224):
                in_data[0][i][j][k] = data_array[i * 224 + j][k]

    ori_output = copy.deepcopy(model.graph.output)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    outputs = [x.name for x in model.graph.output]
    
    # time_start = time.time()
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_outs = ort_session.run(["input.4"], {"input": in_data.astype(np.float32)})
    # time_end = time.time()
    # time_c= time_end - time_start
    # with open('./time/'+ str(name + 1) +'.txt', 'w') as f:
    #     f.write(str(time_c) + '\n')


    # with open ('./result/'+ str(name + 1) +'.txt', 'w') as f:
    #     for i in range(ort_outs[0].shape[1]):
    #         f.write(str(ort_outs[0][0][i]) + '\n')
    
    # print(ort_outs[0][0][63][0])