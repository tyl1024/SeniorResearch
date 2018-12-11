from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
from past.utils import old_div

import MalmoPython
import json
import logging
import os
import random
import sys
import time
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils
import numpy as np


###############################################################################################################################
#                                                    Neural Network                                                           #
###############################################################################################################################

# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
#X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)

#y = np.array(([92],    [86],   [89]), dtype=float)
#xPredicted = np.array(([4,8]), dtype=float)

# scale units
#X = X/np.amax(X, axis=0) # maximum of X array
#xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
#y = y/100 # max test score is 100


class Neural_Network(object):
    def __init__(self):
      
        self.inputSize = 4 #grass, or carpet, or lapis, or fence 4,5,10
        self.outputSize = 5 # which action to take. - choose largest value to execute (place, left, right, north, south)
        self.hiddenSize = 10
    
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    

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
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk


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
    <Summary>Cliff walking mission based on Sutton and Barto.</Summary>
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
      <FlatWorldGenerator generatorString="4;8,3*4,3;13;,biome_5"/>
      <DrawingDecorator>
        <!-- coordinates for cuboid are inclusive -->
         <DrawCuboid type="air" x1="-24" x2="24" y1="4" y2="40" z1="-20" z2="20" />
               <!-- Draw world -->
               <DrawBlock type="spruce_fence" x="1" y="4" z="4" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="5" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="1" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="2" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="3" />        <!--  left side of fence -->
               <DrawBlock type="spruce_fence" x="1" y="4" z="4" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="7" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="8" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="9" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="10" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="11" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="12" />
               
               
               <DrawBlock type="spruce_fence" x="-20" y="4" z="5" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="5" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="6" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="1" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="2" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="3" />        <!--  right side of fence -->
               <DrawBlock type="spruce_fence" x="-20" y="4" z="4" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="6" />
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
                
                <DrawBlock type="spruce_fence" x="0" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="1" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-1" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-2" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-3" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-4" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-5" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-6" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-7" y="4" z="12" />            <!-- Top fence  -->
               <DrawBlock type="spruce_fence" x="-8" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-9" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-10" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-11" y="4" z="12" />
                <DrawBlock type="spruce_fence" x="-12" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-13" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-14" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-15" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-16" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-17" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-18" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-19" y="4" z="12" />
               <DrawBlock type="spruce_fence" x="-20" y="4" z="12" />
               
               
        <DrawBlock x="-10"     y="3"  z="7" type="lapis_block" />                             <!-- top left of square -->
        <DrawBlock   x="-9"   y="3"  z="7" type="lapis_block" />                           <!-- top middle of square -->
        <DrawBlock   x="-8"   y="3"  z="7" type="lapis_block" />                           <!-- top right of square -->
        <DrawBlock   x="-8"   y="3"  z="6" type="lapis_block" />                           <!-- right middle of square -->
        <DrawBlock   x="-8"   y="3"  z="5" type="lapis_block" />                           <!-- bottom right --> 
        <DrawBlock   x="-9"   y="3"  z="5" type="lapis_block" />                           <!-- bottom middle of square -->  
        <DrawBlock   x="-10"   y="3"  z="5" type="lapis_block" />                           <!-- bottom left of square -->     
        <DrawBlock   x="-10"   y="3"  z="6" type="lapis_block" />                           <!-- bottom left of square -->  
        <DrawBlock   x="-8"   y="3"  z="6" type="lapis_block" />                           <!-- bottom left of square -->
                     
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="1000000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Tyler</Name>
    <AgentStart>
      <Placement x="-9.5" y="20.0" z="2.5" pitch="30" yaw="0"/>
      <Inventory>
          <InventoryBlock slot="0" type="carpet" quantity="8" />
      </Inventory>
    </AgentStart>
    <AgentHandlers>
     <ObservationFromFullInventory flat="false"/>
       <ObservationFromRay/>
         <InventoryCommands/>
      <ObservationFromFullStats/>
                  <ObservationFromGrid>
                      <Grid name="floor3x3">
                        <min x="-1" y="-1" z="-1"/>
                        <max x="1" y="-1" z="1"/>
                      </Grid>
                  </ObservationFromGrid>
     <ContinuousMovementCommands turnSpeedDegs="180"/>
      <VideoProducer want_depth="false">
          <Width>640</Width>
          <Height>480</Height>
      </VideoProducer>
      <DiscreteMovementCommands>
          <ModifierList type="deny-list">
            <command>strafe</command>
          </ModifierList>
      </DiscreteMovementCommands>
      <RewardForTouchingBlockType>
         <Block reward="-1.0" type="grass" behaviour="oncePerBlock"/> 
         <Block reward="100.0" type="lapis_block" behaviour="oncePerBlock"/>  
      </RewardForTouchingBlockType>
      <RewardForSendingCommand reward="-1"/>
      <AgentQuitFromTouchingBlockType>
      <!--      <Block type="lava" /> -->
          <Block type="lava" />
      </AgentQuitFromTouchingBlockType>
    </AgentHandlers>
  </AgentSection>

</Mission>'''

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

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
time.sleep(1)                        #Wait a second until we are looking in roughly the right direction
#agent_host.sendCommand("move .1")     #And start running...

#botsActions = ["moveeast 1", "movesouth 1", "movewest 1", "movenorth 1", "use 1"]
#randomAction = random.choice(botsActions)
#agent_host.sendCommand((randomAction))

jumping = False
NN = Neural_Network() #Remember only want to do this once

#NN = Neural_Network() #Remember only want to do this once
# Loop until mission ends:
while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    if world_state.number_of_observations_since_last_state > 0:
        msg = world_state.observations[-1].text
        observations = json.loads(msg).get("LineOfSight")
        #print(observations)
        # 1. Determine block type
        #2. Make a switch case based on block type that determines to set input nodes.
        #    for example if block type is 'grass' then grass node is 1, and all other nodes 0.
        # 3. Run forward propagation.
        # 4. Look at output nodes figure out node with highest value
        # 5. execute action based on output nodes
        #botsEyes = observations.get("LineOfSight")
        #print (botsEyes)
        botsEyes = observations.get("type")

        print(botsEyes)
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
            #inputValues = [0, 0, 0, 0]
            lapisNode = 0
            grassNode = 0
            fenceNode = 0 
            carpetNode = 0
            
            
        X = np.array((inputValues))
        #print ("Actual Output: \n" + str(y))
        #print ("Predicted Output: \n" + str(NN.forward(X)))
        #print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
        o = NN.forward(X)
        print ("Actual Output: \n" + str(o))
        largestValue = max((o))
        print("Highest Value:" , largestValue)
        if largestValue == o[0]:
                agent_host.sendCommand("movenorth 1")
                time.sleep(.1)
        elif largestValue == o[1]:
                agent_host.sendCommand("moveeast 1")
                time.sleep(.1)
        elif largestValue == o[2]:
                agent_host.sendCommand("movesouth 1")
                time.sleep(.1)
        elif largestValue == o[3]:
                agent_host.sendCommand("movewest 1")
                time.sleep(.1)
        else: 
                agent_host.sendCommand("use 1") 
                time.sleep(.1)  
    time.sleep(0.5) # (let the Mod reset)
          
             
print()
print("Mission ended")
# Mission has ended.
#print ("Actual Output: \n" + str(y))