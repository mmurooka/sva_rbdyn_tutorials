import eigen as e
import sva
import rbdyn as rbd


def TutorialTree():
  """
  Return the MultiBodyGraph, MultiBody and the zeroed MultiBodyConfig with the
  following tree structure:

                b4
             j3 | Spherical
  Root     j0   |   j1     j2     j4
  ---- b0 ---- b1 ---- b2 ----b3 ----b5
  Fixed    RevX   RevY    RevZ   PrismZ
  """

  mbg = rbd.MultiBodyGraph()

  mass = 1.
  I = e.Matrix3d.Identity()
  h = e.Vector3d.Zero()

  rbi = sva.RBInertiad(mass, h, I)

  b0 = rbd.Body(rbi, "b0")
  b1 = rbd.Body(rbi, "b1")
  b2 = rbd.Body(rbi, "b2")
  b3 = rbd.Body(rbi, "b3")
  b4 = rbd.Body(rbi, "b4")
  b5 = rbd.Body(rbi, "b5")

  mbg.addBody(b0)
  mbg.addBody(b1)
  mbg.addBody(b2)
  mbg.addBody(b3)
  mbg.addBody(b4)
  mbg.addBody(b5)

  j0 = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitX(), True, "j0")
  j1 = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitY(), True, "j1")
  j2 = rbd.Joint(rbd.Joint.Rev, e.Vector3d.UnitZ(), True, "j2")
  j3 = rbd.Joint(rbd.Joint.Spherical, True, "j3")
  j4 = rbd.Joint(rbd.Joint.Prism, e.Vector3d.UnitY(), True, "j4")

  mbg.addJoint(j0)
  mbg.addJoint(j1)
  mbg.addJoint(j2)
  mbg.addJoint(j3)
  mbg.addJoint(j4)

  to = sva.PTransformd(e.Vector3d(0., 0.5, 0.))
  fro = sva.PTransformd.Identity()

  mbg.linkBodies("b0", to, "b1", fro, "j0")
  mbg.linkBodies("b1", to, "b2", fro, "j1")
  mbg.linkBodies("b2", to, "b3", fro, "j2")
  mbg.linkBodies("b1", sva.PTransformd(e.Vector3d(0.5, 0., 0.)),
                 "b4", fro, "j3")
  mbg.linkBodies("b3", to, "b5", fro, "j4")

  mb = mbg.makeMultiBody("b0", True)
  mbc = rbd.MultiBodyConfig(mb)
  mbc.zero(mb)

  return mbg, mb, mbc
