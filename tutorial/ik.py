# Ref:
# https://nbviewer.jupyter.org/github/jorisv/sva_rbdyn_tutorials/blob/master/MyFirstIK.ipynb

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

# add visualization of the of the b4 and b5 body that don't have any successors
X_b4_ef = sva.PTransformd(sva.RotY(-np.pi/2.), e.Vector3d(0.2, 0., 0.))
X_b5_ef = sva.PTransformd(sva.RotX(-np.pi/2.), e.Vector3d(0., 0.2, 0.))


#### One Task IK ####
def oneTaskMin(mb, mbc, task, delta=1., maxIter=100, prec=1e-8):
    q = np.array(rbd.paramToVector(mb, mbc.q))
    iterate = 0
    minimizer = False
    while iterate < maxIter and not minimizer:
        # compute task data
        g = task.g(mb, mbc)
        J = task.J(mb, mbc)

        # compute alpha
        # print "g:"
        # print g
        # print "J:"
        # print J
        alpha = -np.mat(np.linalg.lstsq(J, g)[0])

        # integrate and run the forward kinematic
        mbc.alpha = rbd.vectorToDof(mb, e.VectorXd(alpha))
        rbd.eulerIntegration(mb, mbc, delta)
        rbd.forwardKinematics(mb, mbc)

        # take the new q vector
        q = np.array(rbd.paramToVector(mb, mbc.q))

        alphaInf = np.linalg.norm(alpha, np.inf)
        yield iterate, q, alpha, alphaInf # yield the current state

        # check if the current alpha is a minimizer
        if alphaInf < prec:
            minimizer = True
        iterate += 1


class BodyTask(object):
    def __init__(self, mb, bodyName, X_O_T, X_b_p=sva.PTransformd.Identity()):
        """
        Compute the error and the jacobian to target a static frame for a body.
        Parameters:
            - mb: MultiBody
            - bodyName: Name of the body that should move
            - X_0_T: targeted frame (PTransformd)
            - X_b_p: static frame on the body bodyName
        """
        self.bodyIndex = mb.bodyIndexByName(bodyName)
        self.X_O_T = X_O_T
        self.X_b_p = X_b_p
        self.jac = rbd.Jacobian(mb, bodyName)
        self.jac_mat_sparse = e.MatrixXd(6, mb.nrDof())

    def X_O_p(self, mbc):
        X_O_b = list(mbc.bodyPosW)[self.bodyIndex]
        return self.X_b_p*X_O_b

    def g(self, mb, mbc):
        X_O_p = self.X_O_p(mbc)
        g_body = sva.transformError(self.X_O_T, X_O_p)
        return np.array(g_body.vector())

    def J(self, mb, mbc):
        X_O_p = self.X_O_p(mbc)
        # set transformation in Origin orientation frame
        X_O_p_O = sva.PTransformd(X_O_p.rotation()).inv()*X_O_p
        jac_mat_dense = self.jac.jacobian(mb, mbc, X_O_p_O)
        self.jac.fullJacobian(mb, jac_mat_dense, self.jac_mat_sparse)
        return np.array(self.jac_mat_sparse)

# Set an initial configuration
mbcIK = rbd.MultiBodyConfig(mbc)
quat = e.Quaterniond(np.pi/3., e.Vector3d(0.1, 0.5, 0.3).normalized())
mbcIK.q = [[],
           [3.*np.pi/4.],
           [np.pi/3.],
           [-3.*np.pi/4.],
           [0.],
           [quat.w(), quat.x(), quat.y(), quat.z()]]

rbd.forwardKinematics(mb, mbcIK)
rbd.forwardVelocity(mb, mbcIK) # for motionSubspace

# target frame
X_O_T = sva.PTransformd(sva.RotY(np.pi/2.), e.Vector3d(1.5, 0.5, 1.))

# create the task
bodyTask = BodyTask(mb, "b5", X_O_T, X_b5_ef)

# copy the initial configuration to avoid the algorithm to change it
mbcIKSolve = rbd.MultiBodyConfig(mbcIK)
q_res = None
X_O_p_res = None
alphaInfList = []
for iterate, q, alpha, alphaInf in oneTaskMin(mb, mbcIKSolve, bodyTask, delta=1., maxIter=200, prec=1e-8):
    X_O_p = bodyTask.X_O_p(mbcIKSolve)
    q_res = q
    alphaInfList.append(alphaInf)

mbcIKResult = rbd.MultiBodyConfig(mbcIK)
mbcIKResult.q = rbd.vectorToParam(mb, e.VectorXd(q_res))
rbd.forwardKinematics(mb, mbcIKResult)

g_body = bodyTask.g(mb, mbcIKResult)
print 'g_body translation error:', g_body[3:].T
print 'g_body rotation error:', g_body[:3].T

import matplotlib.pyplot as plt
plt.plot(alphaInfList)
plt.ylabel('alphaInf')
plt.xlabel('iterate')
plt.pause(0.1)


#### Multi Task IK ####
def manyTaskMin(mb, mbc, tasks, delta=1., maxIter=100, prec=1e-8):
    q = np.array(rbd.paramToVector(mb, mbc.q))
    iterate = 0
    minimizer = False
    while iterate < maxIter and not minimizer:
        # compute task data
        gList = map(lambda (w, t): w*t.g(mb, mbc), tasks)
        JList = map(lambda (w, t): w*t.J(mb, mbc), tasks)

        g = np.concatenate(gList)
        J = np.concatenate(JList)

        # compute alpha
        alpha = -np.mat(np.linalg.lstsq(J, g)[0])

        # integrate and run the forward kinematic
        mbc.alpha = rbd.vectorToDof(mb, e.VectorXd(alpha))
        rbd.eulerIntegration(mb, mbc, delta)
        rbd.forwardKinematics(mb, mbc)

        # take the new q vector
        q = np.array(rbd.paramToVector(mb, mbc.q))

        alphaInf = np.linalg.norm(alpha, np.inf)
        yield iterate, q, alpha, alphaInf # yield the current state

        # check if the current alpha is a minimizer
        if alphaInf < prec:
            minimizer = True
        iterate += 1


class PostureTask(object):
    def __init__(self, mb, q_T):
        """
        Target a default configuration for the robot
        """
        self.q_T = q_T

        def isDefine(j):
            return j.type() in (rbd.Joint.Prism, rbd.Joint.Rev, rbd.Joint.Spherical)
        # take back joint and joint index that are define
        self.jointIndex = [i for i, j in enumerate(mb.joints()) if isDefine(j)]
        self.joints = [mb.joint(index) for index in self.jointIndex]
        nrDof = reduce(lambda dof, j: dof + j.dof(), self.joints, 0)

        # initialize g
        self.g_mat = np.mat(np.zeros((nrDof, 1)))

        # initialize the jacobian
        self.J_mat = np.mat(np.zeros((nrDof, mb.nrDof())))
        posInG = 0
        for jIndex, j in zip(self.jointIndex, self.joints):
            posInDof = mb.jointPosInDof(jIndex)
            self.J_mat[posInG:posInG+j.dof(), posInDof:posInDof+j.dof()] = np.eye(j.dof())
            posInG += j.dof()

    def g(self, mb, mbc):
        q = map(list, mbc.q)
        jointConfig = list(mbc.jointConfig)
        posInG = 0
        for jIndex, j in zip(self.jointIndex, self.joints):
            if j.type() in (rbd.Joint.Prism, rbd.Joint.Rev):
                self.g_mat[posInG:posInG+j.dof(),0] = q[jIndex][0] - self.q_T[jIndex][0]
            elif j.type() in (rbd.Joint.Spherical,):
                orid = e.Quaterniond(*self.q_T[jIndex]).inverse().matrix()
                self.g_mat[posInG:posInG+j.dof(),0] =\
                    np.array(sva.rotationError(orid, jointConfig[jIndex].rotation()))
            posInG += j.dof()
        return self.g_mat

    def J(self, mb, mbc):
        return self.J_mat


# Set an initial configuration
mbcMultiIK = rbd.MultiBodyConfig(mbcIK)

postureTask = PostureTask(mb, map(list, mbcMultiIK.q))

# copy the initial configuration to avoid the algorithm to change it
mbcMultiIKSolve = rbd.MultiBodyConfig(mbcMultiIK)
# set the weight of bodyTask to 10000 and the weight of the postureTask to 1
tasks = [(10000., bodyTask), (1., postureTask)]
q_res = None
X_O_p_res = None
alphaInfList = []
for iterate, q, alpha, alphaInf in manyTaskMin(mb, mbcMultiIKSolve, tasks, delta=1., maxIter=200, prec=1e-8):
    X_O_p = bodyTask.X_O_p(mbcMultiIKSolve)
    q_res = q
    alphaInfList.append(alphaInf)

mbcMultiIKResult = rbd.MultiBodyConfig(mbcMultiIK)
mbcMultiIKResult.q = rbd.vectorToParam(mb, e.VectorXd(q_res))
rbd.forwardKinematics(mb, mbcMultiIKResult)

g_body = bodyTask.g(mb, mbcMultiIKResult)
g_posture = postureTask.g(mb, mbcMultiIKResult)
print 'g_body translation error:', g_body[3:].T
print 'g_body rotation error:', g_body[:3].T
print 'g_posture error:', g_posture.T

monoQRes = np.array(rbd.paramToVector(mb, mbcIKResult.q))
multiQRes = np.array(rbd.paramToVector(mb, mbcMultiIKResult.q))
print 'residual between the two solution:', np.linalg.norm(monoQRes - multiQRes)

import matplotlib.pyplot as plt
plt.plot(alphaInfList)
plt.ylabel('alphaInf')
plt.xlabel('iterate')
plt.pause(0.1)
