# Ref:
# https://nbviewer.jupyter.org/github/jorisv/sva_rbdyn_tutorials/blob/master/Jacobian.ipynb

import numpy as np
import eigen as e
import sva
import rbdyn as rbd

import sys
sys.path.append("../")
from robots import TutorialTree


print 'TutorialTree structure:'
print TutorialTree.__doc__

# create a robot with the same structure than the one in the MultiBody tutorial
mbg, mb, mbc = TutorialTree()


#### Kinematics Jacobian ####

## Classic use ##
mbc.zero(mb)
rbd.forwardKinematics(mb, mbc)
rbd.forwardVelocity(mb, mbc) # mandatory because jacobian need mbc.motionSubspace !

jac_b4 = rbd.Jacobian(mb, 'b4')
jacO = jac_b4.jacobian(mb, mbc)
jacB = jac_b4.bodyJacobian(mb, mbc)

print 'Dense Jacobian in Origin frame orientation'
print jacO
print
print 'Dense Jacobian in body frame orientation'
print jacB
print


quat = e.Quaterniond(np.pi/3., e.Vector3d(0.1, 0.5, 0.3).normalized())

mbc.q = [[],
         [np.pi/2.],
         [np.pi/3.],
         [-np.pi/2.],
         [0.5],
         [quat.w(), quat.x(), quat.y(), quat.z()]]

rbd.forwardKinematics(mb, mbc)
jacO = jac_b4.jacobian(mb, mbc)
jacB = jac_b4.bodyJacobian(mb, mbc)

print 'Dense Jacobian in Origin frame orientation'
print jacO
print
print 'Dense Jacobian in body frame orientation'
print jacB
print


# allocate sparse matrix
sparseJacO = e.MatrixXd(6, mb.nrDof())
sparseJacB = e.MatrixXd(6, mb.nrDof())

jac_b4.fullJacobian(mb, jacO, sparseJacO)
jac_b4.fullJacobian(mb, jacB, sparseJacB)

print 'Sparse Jacobian in Origin frame orientation'
print sparseJacO
print
print 'Sparse Jacobian in body frame orientation'
print sparseJacB
print

# 0 alpha vector
mbc.alpha = map(lambda j: j.zeroDof(), mb.joints())
rbd.forwardVelocity(mb, mbc) # run the forward velocity to compute bodyVelW and bodyVelB

# take back body velocity in Origin orientation frame and in body orientation frame
b4Index = mb.bodyIndexByName('b4')
bodyVelW = list(mbc.bodyVelW)
bodyVelB = list(mbc.bodyVelB)
V_b4_O = bodyVelW[b4Index]
V_b4 = bodyVelB[b4Index]

# convert the alpha articular parameter vector into a numpy vector
alphaVec = np.array(rbd.dofToVector(mb, mbc.alpha))

# compute velocity from jacobian
jacVelO = np.array(sparseJacO).dot(alphaVec)
jacVelB = np.array(sparseJacB).dot(alphaVec)

print 'alpha:', map(list, mbc.alpha)
print 'Residual in Origin orientation frame:', np.linalg.norm(jacVelO - np.array(V_b4_O.vector()))
print 'Residual in body orientation frame:', np.linalg.norm(jacVelB - np.array(V_b4.vector()))
print

# now we apply a new alpha vector
alphaVec = np.mat(np.random.rand(mb.nrDof(),1))

mbc.alpha = rbd.vectorToDof(mb, e.VectorXd(alphaVec))
rbd.forwardVelocity(mb, mbc) # run the forward velocity to compute bodyVelW and bodyVelB

bodyVelW = list(mbc.bodyVelW)
bodyVelB = list(mbc.bodyVelB)
V_b4_O = bodyVelW[b4Index]
V_b4 = bodyVelB[b4Index]

# compute velocity from jacobian
jacVelO = np.array(sparseJacO).dot(alphaVec)
jacVelB = np.array(sparseJacB).dot(alphaVec)

print 'alpha:', map(list, mbc.alpha)
print 'Residual in Origin orientation frame:', np.linalg.norm(jacVelO - np.array(V_b4_O.vector()))
print 'Residual in body orientation frame:', np.linalg.norm(jacVelB - np.array(V_b4.vector()))
print

## Modern use ##
bodyPosW = list(mbc.bodyPosW)
X_O_b = bodyPosW[b4Index]
X_b_p = sva.PTransformd(jac_b4.point())
X_O_p = X_b_p*X_O_b
X_O_p_O = sva.PTransformd(X_O_b.rotation()).inv()*X_O_p

jacO_modern = jac_b4.jacobian(mb, mbc, X_O_p_O)
jacB_modern = jac_b4.jacobian(mb, mbc, X_O_p)

print 'Residual of Origin orientation frame Jacobian:', np.linalg.norm(np.array(jacO) - np.array(jacO_modern))
print 'Residual of body frame Jacobian:', np.linalg.norm(np.array(jacB) - np.array(jacB_modern))

V_O_p_O_classic = jac_b4.velocity(mb, mbc)
V_O_p_classic = jac_b4.bodyVelocity(mb, mbc)
V_O_p = jac_b4.velocity(mb, mbc, X_b_p)

print 'Veloctiy in Origin orientation frame:', V_O_p_O_classic
print 'Velocity in body frame (classic):', V_O_p_classic
print 'Velocity in body frame (modern):', V_O_p


#### Center of Mass Jacobian ####
# create a random alpha vector
alphaVec = np.mat(np.random.rand(mb.nrDof(),1))

mbc.alpha = rbd.vectorToDof(mb, e.VectorXd(alphaVec))
rbd.forwardVelocity(mb, mbc) # run the forward velocity to compute bodyVelW and bodyVelB

# compute the jacobian
jac_com = rbd.CoMJacobian(mb)
jac_com_mat = jac_com.jacobian(mb, mbc)

# compute the velocity and the velocity from the CoM Jacobian matrix
vel_com = jac_com.velocity(mb, mbc)
vel_com_jac = np.array(jac_com_mat).dot(alphaVec)

print 'CoM velocity from velocity:', np.array(vel_com).T
print 'CoM velocity from Jacobian:', vel_com_jac.T
print 'Residual:', np.linalg.norm(np.array(vel_com).T - vel_com_jac.T)


#### Centroidal Momentum Matrix ####
# create a random alpha vector
alphaVec = np.mat(np.random.rand(mb.nrDof(),1))

mbc.alpha = rbd.vectorToDof(mb, e.VectorXd(alphaVec))
rbd.forwardVelocity(mb, mbc) # run the forward velocity to compute bodyVelW and bodyVelB
com = rbd.computeCoM(mb, mbc)

# compute the CM Matrix
CMM = rbd.CentroidalMomentumMatrix(mb)
CMM.computeMatrix(mb, mbc, com)
CMM_mat = CMM.matrix()

# compute the momentum and the momentum from the CM Matrix
h_c = CMM.momentum(mb, mbc, com)
h_c_jac = np.array(CMM_mat).dot(alphaVec)

print 'Centroidal Momentum from momentum:', np.array(h_c.vector()).T
print 'Centroidal Momentum from CM Matrix:', h_c_jac.T
print 'Residual:', np.linalg.norm(np.array(h_c.vector()).T - h_c_jac.T)
