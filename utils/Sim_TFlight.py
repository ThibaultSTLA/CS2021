import pygame
import random
import numpy as np  
import math 

class Robot:
    def __init__(self, posx=0, vitesse=0, accel=0):
        self.posx = posx
        self.vitesse = vitesse
        self.accel = accel

class Sim_TFlight:
    def __init__(self, model=None, TFL_init=None):        

        self.ReplayedNN = model
        self.TFL_init = TFL_init
        self.fail_count = 0
        self.reset()

    def reset(self):
        
        self.init = True

        self.fail_on_red = False

        self.Slope = -(44-618)/100
        self.OriginOffset = 618
        self.BackgroundImage = pygame.image.load("data/images/ImageFond.JPG")
        self.VehicleImage = pygame.image.load('data/images/VeryTinyCar.JPG')

        self.VideoTimeStep = 50 # in 0.01 sec
        self.WidthPixelPosition = 0
        self.HeightPixelPosition = 0

        ##NWidthMax, NHeightMax =VehicleImage.size

        self.CarWidthMax, self.CarHeightMax = 33 , 16

        #Computation of the TrafficLightPosition
        self.TrafficLightSize = 10
        self.TrafficLightWidthPixel = self.OriginOffset - int(self.TrafficLightSize/2)
        self.TrafficLightHeightPixel = 120 + int(self.TrafficLightSize/2)

        self.DeltaT = 0.01 # en s

        self.NoiseMagnitude = 0.05

        self.DistanceInit = 200. # in m
        self.VitesseInit0 = 13 # en m/s
        self.AccelInit = 0 # en m/s²
        self.EmergencyBraking = -3 # in m/s²
        self.TuningFactor = 1.3 #to Compensate the first order expected lag
        self.UrbanMaxSpeed = 55/3.6 #in m/s

        self.DistStop = 0 # TrafficLightPosition
        self.DetectRangeInit = 40 # in m Traffic Light Detection Range
        self.TrafficLightPosition = 0 # in m

        self.MarginFactor = 1.1
        self.DeadZoneDuration = 0.7 # in s; JLAB03 pure delay before applying acceleration


        self.TrafficLightDuration = np.zeros([3])

        self.TauEngine = 0.4 # en s engine first order constant
        self.Alpha = self.DeltaT/(self.DeltaT + self.TauEngine)
        self.kv = 1/(2*(self.TauEngine + self.DeadZoneDuration))

        self.kd = self.kv / 2
        self.Epsilon = 0.0001

        self.DetectRange = self.DetectRangeInit - int(random.random()*30) # DetectRange between 40 and 10, average = 25m
        self.VitesseInit = self.VitesseInit0 - int(random.random()*699)/100 # speed between 13m/s and 6.1m/s

        if self.TFL_init == "GREEN":
            self.TrafficLightState = 0
            self.TrafficLightDuration[0] = 50
            self.TrafficLightDuration[1] = 4
            self.TrafficLightDuration[2] = 30
            self.TrafficLightLastChangeTime = 0
        elif self.TFL_init == "RED":
            self.TrafficLightState = 2
            self.TrafficLightDuration[0] = 30
            self.TrafficLightDuration[1] = 4
            self.TrafficLightDuration[2] = 50
            self.TrafficLightLastChangeTime = 0
        else:
            self.TrafficLightState = random.randint(0,2) # 0 = Green; 1 = Amber; 2 = Red

            self.TrafficLightDuration[1]= 4 # in sec
            self.TrafficLightDuration[2] = 30 + int(random.random()*30)  # in sec
            self.TrafficLightDuration[0] =  self.TrafficLightDuration[2] - self.TrafficLightDuration[1] -2 # in sec
        
            self.TrafficLightLastChangeTime = random.random()*self.TrafficLightDuration[self.TrafficLightState]

        self.TrafficLightNextChangeTime = self.TrafficLightDuration[self.TrafficLightState] - self.TrafficLightLastChangeTime
    
        self.Time = 0
        self.StandStill = 0
        self.NewDetect = 0
        self.IMode = 0
        #parameters for the Pure Time Delay management
        self.DelaySize = int (self.DeadZoneDuration // self.DeltaT) + 1
        self.AccelCArray = np.zeros([1, self.DelaySize])
        self.BarrelCounter = 0

        self.avacar = Robot(-self.DistanceInit,self.VitesseInit,self.AccelInit)

        self.SpeedC = 0
        self.AccelC = 0

        self.OldNeuron1=0
        self.OldNeuron2=0
        self.OldNeuron3=0

        self.OldIMode0 = 0
        self.OldIMode1 = 0
        self.OldIMode2 = 0
        self.OldIMode3 = 0
        self.OldIMode4 = 0

        self.InputNNWidth = 14 # posx, vitesse, accel, oldNeuron1,2,3, Neuron1,2,3, OldImode0..4, GT sous one hot encoding
        self.InputNN = np.empty((1,self.InputNNWidth))
        self.GroundTruthWidth = 5
        self.YPredictArray = np.zeros([1,1, self.GroundTruthWidth])

    def step(self):

    #----------------------------    
    #--Traffic Light Management
    #----------------------------

        if ( abs(self.TrafficLightNextChangeTime - self.Time) < self.DeltaT ):
            #need to change the traffic light color and reset the traffic light duration counter
            self.TrafficLightState +=1
            self.TrafficLightState = self.TrafficLightState %3
            self.TrafficLightNextChangeTime += self.TrafficLightDuration[self.TrafficLightState]
            self.TrafficLightChange = 1
        else:
            self.TrafficLightChange = 0

    #===================================================
    #=== Neural Network control
    #===================================================

    #----------------------------------------------------
    #-- Traffic light detection range
    #----------------------------------------------------)

        if ( self.avacar.posx < (self.TrafficLightPosition -self.DetectRange)) :
            #the sensor cannot see the traffic light, speed must be adapted 
            self.IDetect=0
        else:
            self.IDetect=1

    #----------------------------------------------------
    #-- Car control mode according to the traffic light state
    #----------------------------------------------------
        self.AnticipatedPosition = self.avacar.posx + 1.8*self.avacar.vitesse*self.DeadZoneDuration
        self.MaxSpeedCautious = math.sqrt(2*self.EmergencyBraking * 0.8* ( min(0,self.AnticipatedPosition)) ) / (self.MarginFactor * self.TuningFactor)

        if (self.avacar.posx < -80):
        #not in the learning base --> normal mode
            self.SpeedC = min(self.VitesseInit, self.MaxSpeedCautious)
            self.AccelC = self.kv * (self.SpeedC - self.avacar.vitesse)
            self.AutonomousMode = 0
        else:
        #within the learning database range --> autonomous mode
            self.AutonomousMode = 1
            self.SpeedC=0 #for the plot)
            # NN prediction
            self.InputNN[0,0] = self.avacar.posx #* (1 + random()*self.NoiseMagnitude * self.INoiseNormalInput)
            self.InputNN[0,1] = self.avacar.vitesse#*  (1 + random()*self.NoiseMagnitude * self.INoiseNormalInput)
            self.InputNN[0,2] = self.avacar.accel #* (1 + random()*self.NoiseMagnitude * self.INoiseNormalInput)
            self.InputNN[0,3] = self.OldNeuron1
            self.InputNN[0,4] = self.OldNeuron2
            self.InputNN[0,5] = self.OldNeuron3
            self.Neuron1 = 0
            self.Neuron2 = 0
            self.Neuron3 = 0
            if (self.IDetect ==1):
                if (self.TrafficLightState == 0):
                    self.Neuron1 = 1
                else:
                    if (self.TrafficLightState == 1):
                        self.Neuron2 = 1
                    else:
                        self.Neuron3 = 1  
            self.InputNN[0,6] = self.Neuron1
            self.InputNN[0,7] = self.Neuron2
            self.InputNN[0,8] = self.Neuron3

            self.InputNN[0,9] = self.OldIMode0
            self.InputNN[0,10] = self.OldIMode1
            self.InputNN[0,11] = self.OldIMode2
            self.InputNN[0,12] = self.OldIMode3
            self.InputNN[0,13] = self.OldIMode4

            self.YPredictArray [0,0,:] = self.ReplayedNN(self.InputNN, training=False)
            self.PredictedIMode = np.argmax(self.YPredictArray [0,0,:])


            self.OldNeuron1 = self.Neuron1
            self.OldNeuron2 = self.Neuron2
            self.OldNeuron3 = self.Neuron3

            self.OldIMode0 = self.YPredictArray [0,0,0]
            self.OldIMode1 = self.YPredictArray [0,0,1]
            self.OldIMode2 = self.YPredictArray [0,0,2]
            self.OldIMode3 = self.YPredictArray [0,0,3]
            self.OldIMode4 = self.YPredictArray [0,0,4]


    #------------------------------
    #-- Eventually computation of AccelC the same way as for state machine control
    #------------------------------
            self.AnticipatedPosition = self.avacar.posx + 1.8*self.avacar.vitesse*self.DeadZoneDuration
            self.MaxSpeedCautious = math.sqrt(2*self.EmergencyBraking * 0.8* ( min(0,self.AnticipatedPosition)) ) / (self.MarginFactor * self.TuningFactor)
            if (self.PredictedIMode ==0):
                self.SpeedC = min(self.VitesseInit, self.MaxSpeedCautious)
                self.AccelC = self.kv * (self.SpeedC - self.avacar.vitesse)
            else:
                if (self.PredictedIMode == 1) :
                    self.AccelC=0                       
                else:
                    if (self.PredictedIMode ==2):
                        self.AccelC = -self.kv * self.avacar.vitesse  - self.kd *self.kv*self.AnticipatedPosition
                    else:
                        if (self.PredictedIMode == 3):
                            # Mode 3: the light turns green --> accelerate to VitesseInit
                            self.SpeedC = self.VitesseInit
                            self.AccelC = self.kv *( self.SpeedC - self.avacar.vitesse)
                        else:
                            # Mode 4 --> to late to brake --> accelerate to VitesseInit
                            #AccelC = EmergencyBraking
                            self.SpeedC = self.VitesseInit
                            self.AccelC = self.kv *( self.SpeedC - self.avacar.vitesse)

    #==========================================
    #=== vehicle motion integration
    #==========================================


        #Pure Time Delay management
        self.BarrelIndex = (self.BarrelCounter % self.DelaySize)
        self.BarrelCounter +=1
        self.AccelCArray[0, self.BarrelIndex] = self.AccelC
        self.AccelCDelayedIndex = ((self.BarrelIndex + 1) % self.DelaySize)
        if (self.BarrelCounter >= self.DelaySize):
            self.AccelCDelayed = self.AccelCArray[0, self.AccelCDelayedIndex]
        else:
            #The barrel is not yet full for the first time --> no acceleration
            self.AccelCDelayed = 0

        #longitudinal acceleration saturation
        self.AccelCNonSat = self.AccelCDelayed
        self.AccelCSat = max(self.AccelCDelayed, self.EmergencyBraking)
        self.AccelCSat = min(self.AccelCDelayed, -self.EmergencyBraking)
        #first order engine response
        self.avacar.accel = self.Alpha * self.AccelCSat + (1-self.Alpha)*self.avacar.accel
        #longitudinal acceleration saturation
        self.avacar.accel = max(self.avacar.accel, self.EmergencyBraking)
        self.avacar.accel = min(self.avacar.accel, -self.EmergencyBraking)
        #motion integration
        self.avacar.vitesse += self.avacar.accel*self.DeltaT
        self.avacar.vitesse = max(0, self.avacar.vitesse)
        self.avacar.vitesse = min(self.avacar.vitesse, self.UrbanMaxSpeed)
        if ((self.avacar.vitesse < 0.3) and (self.TrafficLightState == 2) and (self.avacar.posx <0)):
            #Force standstill
            self.avacar.vitesse = 0
            self.AutonomousMode = 2

        #position integration
        self.avacar.posx += self.avacar.vitesse * self.DeltaT
        
        self.Time += self.DeltaT

        if ((self.TrafficLightState ==2) and (self.avacar.posx > 0.8) and (self.avacar.posx <2) and (self.avacar.vitesse > 0.1)):
            self.fail_on_red = True
        
        if self.avacar.posx > 40 or self.Time > 99:
            if self.fail_on_red:
                self.fail_count +=1
            return False
        else:
            return True

    def render(self):
        if self.init:
            self.init = False
            pygame.init()        
            self.screen = pygame.display.set_mode([self.BackgroundImage.get_width(), self.BackgroundImage.get_height()])

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit()
                self.reset()
                return False

        WidthPixelPosition = int ( self.Slope*self.avacar.posx + self.OriginOffset) - self.CarWidthMax
        HeightPixelPosition = 100

        self.screen.fill((0,0,0))
        self.screen.blit(self.BackgroundImage, (0,0))
        self.screen.blit(self.VehicleImage, (WidthPixelPosition,HeightPixelPosition))
        if (self.IDetect ==1):
            if (self.TrafficLightState ==0 ):
                RGBCode = (0,255,0)
            else:
                if (self.TrafficLightState ==1):
                    RGBCode = (255,220,104)
                else:
                    RGBCode = (255, 0, 0)
        else:
            RGBCode = (0,0,0)
        
        TrafficLightPixelPosition = (self.TrafficLightWidthPixel ,self.TrafficLightHeightPixel)

        pygame.draw.circle(self.screen, RGBCode,TrafficLightPixelPosition, self.TrafficLightSize)
        pygame.display.flip()
        if self.avacar.posx > 40 or self.Time > 99:
            pygame.quit()
            self.reset()
            return False
        else:
            return True