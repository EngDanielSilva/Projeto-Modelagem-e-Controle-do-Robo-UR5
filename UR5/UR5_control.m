clear all
close all
clc


vrep=remApi('remoteApi'); % using the prototype file (remoteApiProto.m)
vrep.simxFinish(-1); % just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,true,true,5000,5);

if (clientID>-1)
    disp('Connected to remote API server');
      vrep.simxAddStatusbarMessage(clientID,'Matlab Connected',vrep.simx_opmode_oneshot);  
    %Handle
    [~, Joint1]   = vrep.simxGetObjectHandle(clientID,'UR5_joint1',vrep.simx_opmode_blocking);
    [~, Joint2]   = vrep.simxGetObjectHandle(clientID,'UR5_joint2',vrep.simx_opmode_blocking);
    [~, Joint3]   = vrep.simxGetObjectHandle(clientID,'UR5_joint3',vrep.simx_opmode_blocking);
    [~, Joint4]   = vrep.simxGetObjectHandle(clientID,'UR5_joint4',vrep.simx_opmode_blocking);
    [~, Joint5]   = vrep.simxGetObjectHandle(clientID,'UR5_joint5',vrep.simx_opmode_blocking);
    [~, Joint6]   = vrep.simxGetObjectHandle(clientID,'UR5_joint6',vrep.simx_opmode_blocking);
    [~, Gripper]   = vrep.simxGetObjectHandle(clientID,'ROBOTIQ_85',vrep.simx_opmode_blocking);
    [~, Cuboid]  = vrep.simxGetObjectHandle(clientID,'Cuboid',vrep.simx_opmode_blocking);
    [~, Copo]  = vrep.simxGetObjectHandle(clientID,'Cup',vrep.simx_opmode_blocking);
    [returnCode,camera]=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_blocking);
    [returnCode,resolution,image]=vrep.simxGetVisionSensorImage2(clientID,camera,1,vrep.simx_opmode_streaming);
     
     pause(.1);
       %tic
   % for i=1:50  % 5 seconds 
       vrep.simxSetJointTargetPosition(clientID,Joint1,degtorad(90),vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(clientID,Joint2,degtorad(0),vrep.simx_opmode_oneshot)
       vrep.simxSetJointTargetPosition(clientID,Joint3,degtorad(90),vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(clientID,Joint4,degtorad(0),vrep.simx_opmode_oneshot)
       vrep.simxSetJointTargetPosition(clientID,Joint5,degtorad(-90),vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(clientID,Joint6,degtorad(0),vrep.simx_opmode_oneshot)
       [returnCode,resolution,image]=vrep.simxGetVisionSensorImage2(clientID,camera,1,vrep.simx_opmode_buffer);
           
      
    
    imshow(image)
    pause(.1);
   % end
    
    
         vrep.simxAddStatusbarMessage(clientID,'Matlab DisConnected',vrep.simx_opmode_oneshot); 
        pause(.1);
        vrep.simxFinish(-1); % close all opened connections
    
    
end      
        



 vrep.delete(); % call the destructor!
    
 disp('Program ended');