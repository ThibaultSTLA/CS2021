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

def show_confusion_matrix(conf_mat):

    conf_mat = np.transpose(conf_mat)
    conf_mat = np.transpose((np.nan_to_num(conf_mat/np.sum(conf_mat, axis=0))))

    print("  {:<4}   {:<4}   {:<4}   {:<4}   {:<4}".format('Mod0','Mod1', 'Mod2', 'Mod3','Mod4'))
    print("  {:<4}   {:<4}   {:<4}   {:<4}   {:<4}".format('pred','pred', 'pred', 'pred','pred'),end="")

    print("\n____________________________________")
    for i, line in enumerate(conf_mat):
        print("|", end="")
        for sample in line:
            print(f" {sample:.2f} |",end="")
        print(f" Mod{i} truth", end="")
        print(f"\n|______|______|______|______|______|")