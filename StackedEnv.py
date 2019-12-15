from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
from past.utils import old_div
from collections import ChainMap

import MalmoPython
import json
import sys
import time
import numpy as np


Qtable = [0,0]     #one state, 5 actions
eTrace = [0,0]     #one state, 5 actions    how long ago did I do that action and state determines how much of total reward


class Neural_Network(object):
    def __init__(self):
     
        self.inputSize = 4 #grass, or carpet, or lapis, or fence
        self.outputSize = 2 # which action to take. - choose largest value to execute (move north, use 1)
        self.hiddenSize = 10
   
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
   
        np.loadtxt("w1Weights.txt", dtype = np.str)
        np.loadtxt("w2Weights.txt", dtype = np.str)

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))


    def saveWeights(self):
        np.savetxt("w1Weights.txt", self.W1, fmt="%s")
        np.savetxt("w2Weights.txt", self.W2, fmt="%s")
       
    def sigmoidPrime(self, s):
    #derivative of sigmoid
        return s * (1 - s)
   
    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
   
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
   
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
       
    def train (self, X, y):
        o = self.forward(X)
        #if global counter mod blah do backprop
        self.backward(X, y, o) #y = qtable(expected)    o is NN action(actual)  x = input
               
################################################################################################################################

def Menger(xorg, yorg, zorg, size, blocktype, variant, holetype):
    #draw solid chunk
    genstring = GenCuboidWithVariant(xorg,yorg,zorg,xorg+size-1,yorg+size-1,zorg+size-1,blocktype,variant) + "\n"
    #now remove holes
    unit = size
    while (unit >= 3):
        w=old_div(unit,3)
        for i in range(0, size, unit):
            for j in range(0, size, unit):
                x=xorg+i
                y=yorg+j
                genstring += GenCuboid(x+w,y+w,zorg,(x+2*w)-1,(y+2*w)-1,zorg+size-1,holetype) + "\n"
                y=yorg+i
                z=zorg+j
                genstring += GenCuboid(xorg,y+w,z+w,xorg+size-1, (y+2*w)-1,(z+2*w)-1,holetype) + "\n"
                genstring += GenCuboid(x+w,yorg,z+w,(x+2*w)-1,yorg+size-1,(z+2*w)-1,holetype) + "\n"
        unit = w
    return genstring

def GenCuboid(x1, y1, z1, x2, y2, z2, blocktype):
    return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

def GenCuboidWithVariant(x1, y1, z1, x2, y2, z2, blocktype, variant):
    return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '" variant="' + variant + '"/>'
   
missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Tyler Major's bot</Summary>
  </About>
 
  <ModSettings>
    <MsPerTick>1</MsPerTick>
  </ModSettings>

  <ServerSection>
      <ServerInitialConditions>
            <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>false</AllowSpawning>
      </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator forceReset="true" generatorString="4;8,3*4,3;13;,biome_5"/>
      <DrawingDecorator>
        <!-- coordinates for cuboid are inclusive -->
         <DrawCuboid type="air" x1="-24" x2="24" y1="4" y2="40" z1="-20" z2="20" />
               <!-- Draw world -->
               <DrawBlock type="spruce_fence" x="-10" y="4" z="7" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="5" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="3" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="2" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="1" />        <!--  left side of fence -->
               <DrawBlock type="spruce_fence" x="-10" y="4" z="4" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="7" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="8" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="9" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="10" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="11" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="12" />
                       
               <DrawBlock type="spruce_fence" x="-12" y="4" z="7" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="5" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="4" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="3" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="2" />        <!--  right side of fence -->
               <DrawBlock type="spruce_fence" x="-12" y="4" z="1" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="7" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="8" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="9" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="10" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="11" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="12" />
         
               <DrawBlock type="spruce_fence" x="0" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-1" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-2" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-3" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-4" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-5" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-6" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-7" y="4" z="0" />            <!-- Bottom fence  -->
               <DrawBlock type="spruce_fence" x="-8" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-9" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-11" y="4" z="0" />
                <DrawBlock type="spruce_fence" x="-12" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-13" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-14" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-15" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-16" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-17" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-18" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-19" y="4" z="0" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="0" />
               
               <DrawBlock type="spruce_fence" x="-11" y="4" z="8" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="8" />
               <DrawBlock type="spruce_fence" x="-12" y="4" z="8" />
               
                              <!-- top right of square -->
        <DrawBlock   x="-11"   y="3"  z="6" type="lapis_block" />                           <!-- right middle of square -->
        <DrawBlock   x="-11"   y="3"  z="7" type="lapis_block" />                           <!-- bottom right -->
                                         
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="100000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Tyler</Name>
    <AgentStart>
      <Placement x="-10.5" y="17.0" z="2.5" pitch="60" yaw="0"/>
      <Inventory>
          <InventoryBlock slot="0" type="carpet" quantity="2" />
      </Inventory>
    </AgentStart>
    <AgentHandlers>
     <ObservationFromFullInventory flat="false"/>
       <ObservationFromRay/>
       <ObservationFromHotBar/>
         <InventoryCommands/>
      <ObservationFromFullStats/>
                  <ObservationFromGrid>
                      <Grid name="floor3x3">
                        <min x="-1" y="-1" z="-1"/>
                        <max x="1" y="-1" z="1"/>
                      </Grid>
                  </ObservationFromGrid>
      <VideoProducer want_depth="false">
          <Width>640</Width>
          <Height>480</Height>
      </VideoProducer>
      <DiscreteMovementCommands>
          <ModifierList type="deny-list">
            <command>strafe</command>
          </ModifierList>
      </DiscreteMovementCommands>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

def getAction():
        print(largestValue,o[0])
        reward = 0
        if o[0][0]+0.00001 > largestValue > o[0][0]-0.00001:     #decimals used for rounding
            action = "movesouth 1"
            reward = -.01
            print(reward, " for moving north")
            time.sleep(.1)
            return action,reward
        elif o[0][1]+0.00001 > largestValue > o[0][1]-0.00001:
            if (botsEyes == ("lapis_block")):
                action = "use 1"
                reward = .05
                print(reward, " for placing on Lapis")
                return action,reward
            else:
                action = "use 1"
                reward = -.05
                print(reward, " for not placing on Lapis")
                return action,reward                                                            
                                                             
def currentValues():
    Qtable = [0,0]     #one state, 2 actions
    eTrace = [0,0]     #one state, 2 actions    ..how long ago did I do that action and state determines how much of total reward
    return Qtable,eTrace

NN = Neural_Network() #Remember only want to do this once...make sure this is outside loop to save weights


agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()
                                                           
actionHolder = []
inputHolder = []
counter = 0
#counter for total number of succssful
numOfSuccessful = 0

for i in range(1):
        carpet_remaining = 2
        newQ,newT = currentValues()  
        print(newQ)
        print(newT)
                                           
        # Attempt to start a mission:
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)
       
        # Loop until mission starts:
        print("Waiting for the mission to start ", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
       
        print()
        print("Mission running ", end=' ')
         
        agent_host.sendCommand("pitch 0") #Start looking downward slowly
        time.sleep(1)                      #Wait a second until we are looking in roughly the right direction
       
        #counter for 2 good rewards
        twoGoodRewards = 0
        while world_state.is_mission_running:   #may have to restart mission. look into it
            if(carpet_remaining > 0):
                print(".", end="")
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:",error.text)
                if world_state.number_of_observations_since_last_state > 0:
                    msg = world_state.observations[-1].text
                    observations = json.loads(msg).get("LineOfSight")
                    inventory = json.loads(msg).get("inventory")
                    amountleft = dict(ChainMap(*inventory))
                    finalamountleft = amountleft.get("quantity")
                   
                    new = json.loads(msg).get("Hotbar_0_size")
                    print("Number of carpet left",new)
                   
                    carpet_remaining = new
                   
                    # 1. Determine block type
                    #2. Make a switch case based on block type that determines to set input nodes.
                    #    for example if block type is 'grass' then grass node is 1, and all other nodes 0.
                    # 3. Run forward propagation.
                    # 4. Look at output nodes figure out node with highest value
                    # 5. execute action based on output nodes
                    botsEyes = observations.get("type")

                    if botsEyes == ("lapis_block"):
                        inputValues = [1, 0, 0, 0]
                        lapisNode = 1
                        grassNode = 0
                        fenceNode = 0
                        carpetNode = 0
                    elif botsEyes == ("grass"):
                        inputValues = [0, 1, 0, 0]
                        lapisNode = 0
                        grassNode = 1
                        fenceNode = 0
                        carpetNode = 0
                    elif botsEyes == ("spruce_fence"):
                        inputValues = [0, 0, 1, 0]
                        lapisNode = 0
                        grassNode = 0
                        fenceNode = 1
                        carpetNode = 0
                    elif botsEyes == ("carpet"):
                        inputValues = [0, 0, 0, 1]
                        lapisNode = 0
                        grassNode = 0
                        fenceNode = 0
                        carpetNode = 1
                    else:
                        inputValues = [0, 0, 0, 0]
                        lapisNode = 0
                        grassNode = 0
                        fenceNode = 0
                        carpetNode = 0
                       
                    print("Neural Net values: ",inputValues)
                    X = np.array((inputValues,), dtype = float)        
                    #print ("Actual Output: \n" + str(y))
                    #print ("Predicted Output: \n" + str(NN.forward(X)))
                    #print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
                    o = NN.forward(X)
                 
                   
                    #print ("Actual Output: \n" + str(o))
                    largestValue = np.amax(o)
                    print("Highest Value:" , largestValue)
                   
                    myaction,myreward = getAction()

                   
                    #Dr.Girard assisted
                    myAction = []
                    if myaction == "movesouth 1":  #move north
                        myAction = [1, 0]
                    elif myaction == "use 1":  #place
                        myAction = [0, 1]
                    '''
                    actionHolder.append(myAction)
                    if (len(actionHolder) > 3):
                        actionHolder = actionHolder[1:]
                   
                    inputHolder.append(inputValues)
                    if(len(inputHolder)>3):
                        inputHolder = inputHolder[1:]
                       
                    print(actionHolder)
                    print(inputHolder)
                    '''
 
                    agent_host.sendCommand(myaction)   #send bot actions from getAction method
                    print("Action Chosen is:",myaction)
                             
                    print("Current Reward: ", myreward)
                   
                    y = .5
                    lamd = .5
                    learningRate = .5
                   
                    #Choose action from a1 derived from s using policy from Q'  
                    sigma = myreward
                   
                    '''
                    #eTrace = eTrace + 1
                    if myaction == "movesouth 1":  #move north
                        eTrace[0] += 1
                    elif myaction == "use 1":  #place
                        eTrace[1] = 1
                    '''
                       

                    #For all s,a:
                    #Update Qtable
                    #Update eTrace
                    # print("Before: ", Qtable, eTrace)        
                    for i in range(2):
                        Qtable[i] = Qtable[i] + learningRate * sigma * eTrace[i]
                        eTrace[i] = y * lamd * eTrace[i]
                    #print("After: ", Qtable,eTrace)
                    counter += 1
                   
                   
                    if (myreward == -.05):
                        twoGoodRewards = 0
                 
                    #train good every time good action is done which is when reward is > 0
                    if(myreward > 0):
                       
                        twoGoodRewards += 1
                        print("\n\nTwo good rewards",twoGoodRewards)
                        if (twoGoodRewards % 2 == 0):
                            numOfSuccessful += 1
                            print("\n\nNum of success rewards: \n\n",numOfSuccessful)
                            if (numOfSuccessful == 3):
                                print("\n\n\n\n\nSTOP TRAINING HERE!!!!!!!!!!!!!!!!*******************************\n\n\n\n\n\n")
                                print("The bot has done the job 3 times in a row. Enough learning.")
                                break
                       
                       
                        '''
                        print("**A good reward was done***")
                        eX = np.array(tuple(inputHolder), dtype = float)
                             
                        beforeeX = NN.forward(eX)
                        print("eX Tuple before:\n",beforeeX)
                         
                        whY = np.array(tuple(actionHolder), dtype = float)
                             
                        b = NN.train(eX, whY)
                         
                        aftereX = NN.forward(eX)
                        print("eX Tuple after:\n",aftereX)
                             
                        m1 = NN.forward(X)
                        '''
                         
                        '''
                        #write positive data to w1
                        a = open('w1.txt', 'a')
                        a.write("------------------------Positive---------------------\n")
                        a.write("NN original: " + str(o))
                        a.write("\nQtable: " + str(Qtable))
                        a.write("\n New array: " + str(y))
                        a.write("\nNN after forward again: " + str(m1))
   
                        a.write("\n--------------------------------------------------------")
                        a.close()
                        '''

                    #train bad every 5 actions
                    elif(counter % 5 == 0) or (carpet_remaining > 0):
                        print("*** A bad reward was done ***")
                        '''
                        print("Old Q: ",Qtable) #print old Q tabled not scaled
                        newScaledQ = [x/2 for x in Qtable]
                       
                        largestQ = max(newScaledQ)
                       
                        if largestQ == newScaledQ[0]:
                            newScaledQ[0] = 1
                            newScaledQ[1] = 0
                        elif largestQ == newScaledQ[1]:
                            newScaledQ[0] = 0
                            newScaledQ[1] = 1
                        else:
                            newScaledQ = random.randint(Qtable[0],Qtable[1])
                           
                        print("New scaled Q", newScaledQ)
                       
                        y= np.array((newScaledQ,), dtype = float)  #use new array here
                        #print("After: ", Qtable,eTrace)
                        b = NN.train(X, y)
                        m2 = NN.forward(X)
                       
                        #write negative data to w1
                        a = open('w1.txt', 'a')
                        a.write("------------------------Negative---------------------\n")
                        a.write("NN original: " + str(o))
                        a.write("\nQtable: " + str(Qtable))
                        a.write("\n New array: " + str(y))
                        a.write("\nNN after forward again: "+ str(m2))
                        a.write("\nNew Scaled Q: " + str(newScaledQ))
                        a.write("\n--------------------------------------------------------")
                        a.close()
                        '''
       
                    print(counter)
            else:
                print("No carpet is left so no actions are being done") #<- Will print a lot if places fast.
                time.sleep(0.1)
                world_state = agent_host.getWorldState()   #this keeps timer going and pauses actions world
                                                   
                           
time.sleep(.1)  
time.sleep(0.5) # (let the Mod reset)
                 
print()
print("Mission ended")
