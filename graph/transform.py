def vtkTransform(X):
  """
  Transform a PTranform into a vtk homogeneous transformation matrix.
  """
  R = X.rotation()
  T = X.translation()
  return (R[0, 0], R[1, 0], R[2, 0], T[0],
          R[0, 1], R[1, 1], R[2, 1], T[1],
          R[0, 2], R[1, 2], R[2, 2], T[2],
          0.,   0.,   0.,   1.)


def setActorTransform(actor, X):
  """
  Set an actor user_transform with a PTransform
  """
  actor.user_transform.set_matrix(vtkTransform(X))
