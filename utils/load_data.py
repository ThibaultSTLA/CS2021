import time
import numpy as np

def load_data(path, input_size, output_size):
    # import the list
    print("Start loading Dataset")
    GoTime= time.time()
    InputFile = list(open(path,"r"))
    LengthData = len(InputFile)

    #convert the list into a numpy array
    InputNN = np.empty((LengthData,input_size))
    GroundTruth = np.empty((LengthData,output_size))
    for i in range(LengthData):
        TempRow = InputFile[i].split()
        for j in range (input_size):
            InputNN[i,j] = float(TempRow[j])
        for j in range(output_size):
            GroundTruth[i,j] = float(TempRow[j+input_size])
    print('Loading duration=', time.time() - GoTime,'s')

    return InputNN, GroundTruth, LengthData