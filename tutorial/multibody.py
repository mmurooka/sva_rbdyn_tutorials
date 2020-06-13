# Ref:
# https://nbviewer.jupyter.org/github/jorisv/sva_rbdyn_tutorials/blob/master/MultiBody.ipynb

import numpy as np
import eigen as e
import sva
import rbdyn as rbd



#### Body ####
b0m = 2. # 2kg
b0c = e.Vector3d(0.5, 0.5, 0.) # center of mass at [0.5, 0.5, 0.] from the body origin
b0rI = e.Matrix3d.Identity() # rotational inertia at body origin
# here the second argument is the body first moment of mass (h = m c)
b0I = sva.RBInertiad(b0m, b0m*b0c, b0rI)
# first body constructor that take sva.RBInertia as firt param
b0 = rbd.Body(b0I, 'b0')

b1m = 4. # 4kg
b1c = e.Vector3d(0., 0.5, 0.) # center of mass at [0., 0.5, 0.] from the body origin
b1rI = e.Matrix3d.Identity() # rotational inertia at body origin
# here the second argument is the body center of mass (not the first moment of mass)
b1 = rbd.Body(b1m, b1c, b1rI, 'b1')

b2 = rbd.Body(1., e.Vector3d.Zero(), e.Matrix3d.Identity(), 'b2')
b3 = rbd.Body(1., e.Vector3d.Zero(), e.Matrix3d.Identity(), 'b3')
b4 = rbd.Body(1., e.Vector3d.Zero(), e.Matrix3d.Identity(), 'b4')
b5 = rbd.Body(1., e.Vector3d.Zero(), e.Matrix3d.Identity(), 'b5')


#### Joint ####
def jointResume(j):
    print 'P =', j.params()
    print 'A =', j.dof()
    print 'qZero =', list(j.zeroParam())
    print 'alphaZero =', list(j.zeroDof())
    print 'motion subspace ='
    if j.dof() > 0:
        print np.array(j.motionSubspace())

jFix = rbd.Joint(rbd.Joint.Fixed, True, 'jFix')
jointResume(jFix)

# Revolute joint around X axis
jRev = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitX(), True, 'jRev')
jointResume(jRev)
print 'rotation:'
print jRev.pose([np.pi/2.]).rotation()
print 'angular motion:', jRev.motion([1.]).angular().transpose()

# Prismatic joint at Y axis
jPrism = rbd.Joint(rbd.Joint.Prism, e.Vector3d.UnitY(), True, 'jPrism')
jointResume(jPrism)
print 'translation:', jPrism.pose([1.]).translation().transpose()
print 'linear motion:', jPrism.motion([0.1]).linear().transpose()

jSph = rbd.Joint(rbd.Joint.Spherical, True, 'jSph')
jointResume(jSph)
q = e.Quaterniond(np.pi/4., e.Vector3d(1., 0., 0.5).normalized())
qParam = [q.w(), q.x(), q.y(), q.z()]
print 'rotation:'
print jSph.pose(qParam).rotation()
print 'angular motion:', jSph.motion([1., 0.2, 0.5]).angular().transpose()

jPlan = rbd.Joint(rbd.Joint.Planar, True, 'jPlan')
jointResume(jPlan)
qParam = [np.pi/2., 0.2, 0.1]
print 'translation:', jPlan.pose(qParam).translation().transpose()
print 'rotation:'
print jPlan.pose(qParam).rotation()
print 'motion:', jPlan.motion([0.2, 0.5, -0.5])

jCyl = rbd.Joint(rbd.Joint.Cylindrical, e.Vector3d.UnitZ(), True, 'jCyl')
jointResume(jCyl)
qParam = [np.pi/2., 0.4]
print 'translation:', jCyl.pose(qParam).translation().transpose()
print 'rotation:'
print jCyl.pose(qParam).rotation()
print 'motion:', jCyl.motion([0.4, -0.7])

jFree = rbd.Joint(rbd.Joint.Free, True, 'jFree')
jointResume(jFree)
q = e.Quaterniond(-np.pi/3., e.Vector3d(0.5, 1., 0.2).normalized())
qParam = [q.w(), q.x(), q.y(), q.z(), 0.2, -0.4, 0.7]
print 'translation:', jFree.pose(qParam).translation().transpose()
print 'rotation:'
print jFree.pose(qParam).rotation()
print 'motion:', jFree.motion([0.3, -0.2, 0.1, 1., 2.5, -3.])


#### MultiBodyGraph ####
mbg = rbd.MultiBodyGraph()

# first we add all the bodies previously created
mbg.addBody(b0)
mbg.addBody(b1)
mbg.addBody(b2)
mbg.addBody(b3)
mbg.addBody(b4)
mbg.addBody(b5)

# create and add joints

# revolute joint around the X axis
j0 = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitX(), True, "j0")
# revolute joint around the Y axis
j1 = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitY(), True, "j1")
# revolute joint around the Z axis
j2 = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitZ(), True, "j2")
# spherical joint
j3 = rbd.Joint(rbd.Joint.Spherical, True, "j3")
# prismatic joint on the Y axis
j4 = rbd.Joint(rbd.Joint.Prism, e.Vector3d.UnitY(), True, "j4")

mbg.addJoint(j0)
mbg.addJoint(j1)
mbg.addJoint(j2)
mbg.addJoint(j3)
mbg.addJoint(j4)

# Link the bodies to have this tree structure

#           b4
#        j3 | Spherical
#      j0   |   j1     j2     j4
#  b0 ---- b1 ---- b2 ----b3 ----b5
#     RevX   RevY    RevZ   PrismZ

to = sva.PTransformd(e.Vector3d(0., 0.5, 0.))
fro = sva.PTransformd.Identity()

# link b0 to b1 with j0
mbg.linkBodies("b0", to, "b1", fro, "j0")
# link b1 to b2 with j1
mbg.linkBodies("b1", to, "b2", fro, "j1")
# link b2 to b3 with j2
mbg.linkBodies("b2", to, "b3", fro, "j2")
# link b1 to b4 with j3
mbg.linkBodies("b1", sva.PTransformd(e.Vector3d(0.5, 0., 0.)), "b4", fro, "j3")
# link b3 to b5 with j4
mbg.linkBodies("b3", to, "b5", fro, "j4")

# create the MultiBody with a fixed base and b0 has root
mb = mbg.makeMultiBody("b0", True)


#### MultiBody ####
def printMultiBody(m):
    print 'number of bodies:', m.nrBodies()
    print 'number of joints:', m.nrJoints()
    print

    print 'bodies:'
    for bi, b in enumerate(m.bodies()):
        print 'body index: %s, body name: %s' % (bi, b.name())
    print

    print 'joints:'
    for ji, j in enumerate(m.joints()):
        print 'joint index: %s, joint name: %s' % (ji, j.name())
    print

    bodies = list(m.bodies())
    pred = list(m.predecessors())
    succ = list(m.successors())
    trans = list(m.transforms())
    print 'joints predecessors and successors'
    for ji, j in enumerate(m.joints()):
        predBi = pred[ji] # body INDEX of the joint predecessor
        succBi = succ[ji] # body INDEX of the joint successors
        predBName = bodies[predBi].name() if predBi != -1 else "Origin"
        succBName = bodies[succBi].name()
        print 'the joint %s is supported by the body %s and support the body %s' % (j.name(), predBName, succBName)
        print 'the static translation between the body %s and the joint %s is %s' %\
        (predBName, j.name(), trans[ji].translation().transpose())
        print


print 'MultiBody: mb'
printMultiBody(mb)
print

# now if we create a new MultiBody with a different root (b4)
# we will see that body and joint index will change
mb2 = mbg.makeMultiBody("b4", True)
print 'MultiBody: mb2'
printMultiBody(mb2)


#### MultiBodyConfig ####
# create the MultiBodyConfig
mbc = rbd.MultiBodyConfig(mb)
mbc.zero(mb)

# take the q vector
q = map(list, mbc.q)
print 'mbc.q:', q

# apply the forward kinematics
rbd.forwardKinematics(mb, mbc)

for body, pose in zip(mb.bodies(), mbc.bodyPosW):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())

# update the q vector
quat = e.Quaterniond(np.pi/3., e.Vector3d(0.1, 0.5, 0.3).normalized())
q[1] = [np.pi/2.]
q[2] = [np.pi/3.]
q[3] = [-np.pi/2.]
q[4] = [0.5]
q[5] = [quat.w(), quat.x(), quat.y(), quat.z()]
print 'mbc.q:', q

# you can also do like this
# mbc.q = [[], [np.pi/2.], [np.pi/3.], [-np.pi/2.], [0.5], [quat.w(), quat.x(), quat.y(), quat.z()]]
mbc.q = q

# compute the forward kinematics
rbd.forwardKinematics(mb, mbc)

for body, pose in zip(mb.bodies(), mbc.bodyPosW):
    print "== %s ==" % (body.name())
    print "translation: %s" % (pose.translation().transpose())
    print "rotation:\n%s" % (pose.rotation())
