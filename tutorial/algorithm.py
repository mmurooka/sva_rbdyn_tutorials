# Ref:
# https://nbviewer.jupyter.org/github/jorisv/sva_rbdyn_tutorials/blob/master/SomeAlgorithm.ipynb

import numpy as np
import eigen as e
import sva
import rbdyn as rbd

import sys
sys.path.append("../")
from robots import TutorialTree

#### Setup ####
print 'TutorialTree structure:'
print TutorialTree.__doc__

# create a robot with the same structure than the one in the MultiBody tutorial
mbg, mb, mbc = TutorialTree()


#### Forward Kinematics ####
# run the forward kinematics with q = 0
rbd.forwardKinematics(mb, mbc)

for body, pose in zip(mb.bodies(), mbc.bodyPosW):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())

# Change the articular configuration q
q = map(list, mbc.q)
quat = e.Quaterniond(np.pi/3., e.Vector3d(0.1, 0.5, 0.3).normalized())

mbc.q = [[],
         [np.pi/2.],
         [np.pi/3.],
         [-np.pi/2.],
         [0.5],
         [quat.w(), quat.x(), quat.y(), quat.z()]]

rbd.forwardKinematics(mb, mbc)

for body, pose in zip(mb.bodies(), mbc.bodyPosW):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())


#### Forward Velocity ####
alpha = map(list, mbc.alpha)

print 'alpha:', alpha

alpha[1] = [0.4] # first revolute joint will turn at 0.4 radian per second

mbc.alpha = alpha
rbd.forwardVelocity(mb, mbc)

for body, pose, velB, velW in zip(mb.bodies(), mbc.bodyPosW, mbc.bodyVelB, mbc.bodyVelW):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())
    print "velB: %s" % (velB)
    print "velW: %s" % (velW)


#### Forward Acceleration ####
alphaD = map(list, mbc.alphaD)

print 'alphaD:', alphaD

alphaD[1] = [-0.5] # first revolute joint will turn is deceleration at -1 radian per second^2

mbc.alphaD = alphaD
rbd.forwardAcceleration(mb, mbc) # no gravity compensation

for body, pose, velB, accB in zip(mb.bodies(), mbc.bodyPosW, mbc.bodyVelB, mbc.bodyAccB):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())
    print "velB: %s" % (velB)
    print "accB: %s" % (accB)

# we lower the default gravity because of display issues...
gravity = sva.MotionVecd(e.Vector3d.Zero(), mbc.gravity)*0.05

print 'gravity:', gravity
rbd.forwardAcceleration(mb, mbc, gravity) # with gravity compensation

for body, pose, velB, accB in zip(mb.bodies(), mbc.bodyPosW, mbc.bodyVelB, mbc.bodyAccB):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())
    print "velB: %s" % (velB)
    print "accB: %s" % (accB)


#### CoM Algorithms ####
mbcCoM = rbd.MultiBodyConfig(mbc)
# compute the acceleration without gravity compensation
rbd.forwardAcceleration(mb, mbcCoM)

com = rbd.computeCoM(mb, mbcCoM) # CoM
comVel = rbd.computeCoMVelocity(mb, mbcCoM) # CoM velocity
comAcc = rbd.computeCoMAcceleration(mb, mbcCoM) # CoM acceleration

print 'CoM:', com.transpose()
print 'CoM Velocity:', comVel.transpose()
print 'CoM Acceleration', comAcc.transpose()


#### Centroidal Momentum Algorithms ####
mbcCM = rbd.MultiBodyConfig(mbcCoM)

cm = rbd.computeCentroidalMomentum(mb, mbcCM, com) # com from computeCoM
cmDot = rbd.computeCentroidalMomentumDot(mb, mbcCM, com, comVel) # comVel from computeCoMVelocity

print 'CM:', cm
print 'CM Dot:', cmDot


#### Inverse Dynamics ####
# mbcID keep the mbc q vector but have a zero alpha and alphaD vector
mbcID = rbd.MultiBodyConfig(mb)
mbcID.zero(mb)
mbcID.q = mbc.q

rbd.forwardKinematics(mb, mbcID)
rbd.forwardVelocity(mb, mbcID)
ID = rbd.InverseDynamics(mb)
ID.inverseDynamics(mb, mbcID)

jointTorque = map(list, mbcID.jointTorque)
print 'jointTorque:', jointTorque


#### Forward Dynamics ####
# use the result of the ID to check if the acceleration computed
# by the FD are compatible with the ID torque.
mbcFD = rbd.MultiBodyConfig(mbcID)

FD = rbd.ForwardDynamics(mb)
FD.forwardDynamics(mb, mbcFD)

alphaDID = map(list, mbcID.alphaD)
alphaDFD = map(list, mbcFD.alphaD)

print 'input torque:', map(list, mbcFD.jointTorque)
print
print 'alphaDID:', alphaDID
print 'alphaDFD:', alphaDFD

alphaDIDVec = np.array(rbd.dofToVector(mb, alphaDID))
alphaDFDVec = np.array(rbd.dofToVector(mb, alphaDFD))
print 'residual:', np.linalg.norm(alphaDIDVec - alphaDFDVec)


#### Euler Integration ####
mbcEI = rbd.MultiBodyConfig(mb)
mbcEI.zero(mb)
mbcEI.q = mbc.q

mbcList = []

iterations = 100
timeStep = 0.01

for i in xrange(iterations):
    rbd.forwardKinematics(mb, mbcEI)
    rbd.forwardVelocity(mb, mbcEI)
    FD.forwardDynamics(mb, mbcEI)
    mbcList.append(rbd.MultiBodyConfig(mbcEI)) # store the robot state

    rbd.eulerIntegration(mb, mbcEI, timeStep)

step = 10
for mbcIter in mbcList[::step]:
    print mbcIter.q
