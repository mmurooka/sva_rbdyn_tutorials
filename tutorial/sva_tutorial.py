# Ref:
# https://nbviewer.jupyter.org/github/jorisv/sva_rbdyn_tutorials/blob/master/SpaceVecAlg.ipynb

import numpy as np
import eigen as e
import sva


#### Ptransform ####
X_O = sva.PTransformd.Identity()

X_O_foot = sva.PTransformd(sva.RotZ(np.pi)*sva.RotX(-np.pi/2.), e.Vector3d(0.5, -0.3, 0.))
X_O_CoM = sva.PTransformd(e.Vector3d(-0.2, 0.8, 0.))

print 'X_O_foot translation:', X_O_foot.translation().transpose()
print 'X_O_foot rotation matrix:'
print X_O_foot.rotation()


X_foot_CoM = X_O_CoM*X_O_foot.inv()
X_O_CoM_prime = X_foot_CoM*X_O_foot

print 'Error must be near 0:', (X_O_CoM_prime.matrix() - X_O_CoM.matrix()).norm()


#### MotionVec ####
V_l = sva.MotionVecd(e.Vector3d.Zero(), e.Vector3d(0.2, 0.2, 0.))
V_a = sva.MotionVecd(e.Vector3d(-0.4, 0., 0.), e.Vector3d.Zero())

print 'linear velocity:', V_l.linear().transpose()
print 'angular velocity:', V_a.angular().transpose()

V_O = V_l + V_a

print 'linear velocity:', V_O.linear().transpose()
print 'angular velocity:', V_O.angular().transpose()

V_CoM = X_O_CoM*V_O
V_O_prime = X_O_CoM.invMul(V_CoM)

print 'Error must be near 0:', (V_O_prime - V_O).vector().norm()

V_O_foot = sva.PTransformd(X_O_foot.rotation())*V_O # just keep the rotational part of the PTransformd


#### ForceVec ####
F_f = sva.ForceVecd(e.Vector3d.Zero(), e.Vector3d(0.1, 0., 0.4))
F_t = sva.ForceVecd(e.Vector3d(-0.2, 0., 0.), e.Vector3d.Zero())

print 'force:', F_f.force().transpose()
print 'torque:', F_t.couple().transpose()

F_foot = F_f + F_t

print 'force:', F_foot.force().transpose()
print 'torque:', F_foot.couple().transpose()

X_foot_CoM = X_O_CoM*X_O_foot.inv()

F_CoM = X_foot_CoM.dualMul(F_foot)
F_foot_prime = X_foot_CoM.transMul(F_CoM)

print 'Error must be near 0:', (F_foot_prime - F_foot).vector().norm()

F_foot_O = sva.PTransformd(X_O_foot.rotation()).dualMul(F_foot) # just keep the rotational part of the PTransformd


#### RBInertia ####
m = 4.# mass
c = e.Vector3d(0.1, -0.2, 0.02) # center of mass
h = m*c # first moment of mass
I_foot_c = e.Matrix3d.Identity() # rotational inertia at center of mass
I_foot = sva.inertiaToOrigin(I_foot_c, m, c, e.Matrix3d.Identity()) # rotational inertia at body base

# create the Rigid Body Inertia of the foot body
RBI_foot = sva.RBInertiad(m, c, I_foot) # !!!! Maybe the second argument should be h instead of c.

# transform foot inertia in the CoM frame and do the reverse
RBI_foot_CoM = X_foot_CoM.dualMul(RBI_foot)
RBI_foot_prime = X_foot_CoM.transMul(RBI_foot_CoM)

print 'Error must be near 0:', (RBI_foot - RBI_foot_prime).matrix().norm()
print

RBI_CoM = sva.RBInertiad(10., e.Vector3d.Zero(), e.Matrix3d.Identity())

RBI_CoM_c = RBI_CoM + RBI_foot
print 'mass:', RBI_CoM_c.mass()
print 'center of mass:', (RBI_CoM_c.momentum()/RBI_CoM_c.mass()).transpose()
print 'inertia:'
print RBI_CoM_c.inertia()


#### Transformation error ####
Err_foot_CoM_O = sva.transformError(X_O_foot, X_O_CoM)
print 'error between foot and CoM frame in O frame'
print 'translation error', Err_foot_CoM_O.linear().transpose()
print 'orientation error', Err_foot_CoM_O.angular().transpose()
print

Err_foot_CoM = sva.transformVelocity(X_foot_CoM)
print 'error between foot and CoM frame in foot frame'
print 'translation error', Err_foot_CoM.linear().transpose()
print 'orientation error', Err_foot_CoM.angular().transpose()
print

Err_foot_CoM_O_prime = sva.PTransformd(X_O_foot.rotation()).invMul(Err_foot_CoM)
print 'We can compute the Err_foot_CoM_O error from Err_foot_CoM by applying a pure rotation from foot to O'
print 'translation error', Err_foot_CoM_O_prime.linear().transpose()
print 'orientation error', Err_foot_CoM_O_prime.angular().transpose()
