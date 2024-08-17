import numpy as np

from ..homegeneous_projection import HomVectors, HomMatrix

def test_create_homogeneous_vectors_from_scalars():
    x = 1.0
    y = 2.0
    z = 3.0
    v = HomVectors(x=x, y=y, z=z)
    assert len(v) == 1
    assert np.array_equal(v.x, np.array([x]))
    assert np.array_equal(v.y, np.array([y]))
    assert np.array_equal(v.z, np.array([z]))
    assert np.array_equal(v.w, np.array([1.0]))
    assert np.array_equal(v.vec3, np.array([[[x], [y], [z]]]))
    assert np.array_equal(v.data, np.stack([[[x], [y], [z], [1]]]))

def test_create_homogeneous_vectors_from_vectors():
    x = np.array([1, 2, 3, 4], dtype=np.float64)
    y = np.array([5, 6, 7, 8], dtype=np.float64)
    z = np.array([9, 10, 11, 12], dtype=np.float64)
    v = HomVectors(x=x, y=y, z=z)
    assert len(v) == 4
    assert np.array_equal(v.x, x)
    assert np.array_equal(v.y, y)
    assert np.array_equal(v.z, z)
    assert np.array_equal(v.w, np.ones_like(x))
    assert np.array_equal(v.vec3, np.expand_dims(np.stack([x, y, z], axis=1), axis=2))
    assert np.array_equal(v.data, np.stack([x, y, z, np.ones_like(x)], axis=1).reshape(-1, 4, 1))

def test_multiply_homogeneous_vectors_by_scalar():
    a = HomVectors(x=1, y=2, z=3)
    A = HomVectors(x=[1, 2], y=[3, 4], z=[5, 6])
    
    # positive scalar
    assert np.array_equal((2 * a).data, np.array([[[2], [4], [6], [1]]]))
    assert np.array_equal((2 * A).data, np.array([[[2], [6], [10], [1]], [[4], [8], [12], [1]]]))

    # negative scalar
    assert np.array_equal((-2 * a).data, np.array([[[-2], [-4], [-6], [1]]]))
    assert np.array_equal((-2 * A).data, np.array([[[-2], [-6], [-10], [1]], [[-4], [-8], [-12], [1]]]))

    # float scalar
    assert np.array_equal((2.5 * a).data, np.array([[[2.5], [5], [7.5], [1]]]))
    assert np.array_equal((2.5 * A).data, np.array([[[2.5], [7.5], [12.5], [1]], [[5], [10], [15], [1]]]))

    # zero scalar
    assert np.array_equal((0 * a).data, np.array([[[0], [0], [0], [1]]]))
    assert np.array_equal((0 * A).data, np.array([[[0], [0], [0], [1]], [[0], [0], [0], [1]]]))

def test_add_homogeneous_vectors():
    # add single vectors
    a = HomVectors(x=1, y=2, z=3)
    b = HomVectors(x=4, y=5, z=6)
    c = a + b
    assert len(c) == 1
    assert np.array_equal(c.data, np.array([[[5], [7], [9], [1]]]))
    assert np.array_equal((a + a).data, HomVectors(x=2, y=4, z=6).data)
    assert np.array_equal((a + (-1)*a).data, HomVectors(x=0, y=0, z=0).data)

    # add multiple vectors
    A = HomVectors(x=[1, 2, 3, 4], y=[5, 6, 7, 8], z=[9, 10, 11, 12])
    B = HomVectors(x=[13, 14, 15, 16], y=[17, 18, 19, 20], z=[21, 22, 23, 24])
    C = A + B
    assert len(C) == 4
    assert np.array_equal(C.data, np.array([
        [[14], [22], [30], [1]],
        [[16], [24], [32], [1]],
        [[18], [26], [34], [1]],
        [[20], [28], [36], [1]]
    ], dtype=np.float64))

    # Left broadcasting
    cC = a + B
    assert len(cC) == 4
    assert np.array_equal(cC.data, np.array([
        [[14], [19], [24], [1]],
        [[15], [20], [25], [1]],
        [[16], [21], [26], [1]],
        [[17], [22], [27], [1]]
    ], dtype=np.float64))

    # Right broadcasting
    Cc = B + a
    assert len(cC) == 4
    assert np.array_equal(Cc.data, cC.data)

def test_subtract_homogeneous_vectors():
    # Subtracting single vectors
    a = HomVectors(x=1, y=2, z=3)
    b = HomVectors(x=4, y=5, z=6)
    c = a - b
    assert len(c) == 1
    assert np.array_equal(c.data, np.array([[[-3], [-3], [-3], [1]]]))
    assert np.array_equal((a - a).data, HomVectors(x=0, y=0, z=0).data)

    # Subtracting multiple vectors
    A = HomVectors(x=[1, 2, 3, 4], y=[5, 6, 7, 8], z=[9, 10, 11, 12])
    B = HomVectors(x=[13, 14, 15, 16], y=[17, 18, 19, 20], z=[21, 22, 23, 24])
    C = A - B
    assert len(C) == 4
    assert np.array_equal(C.data, np.array([
        [[-12], [-12], [-12], [1]],
        [[-12], [-12], [-12], [1]],
        [[-12], [-12], [-12], [1]],
        [[-12], [-12], [-12], [1]]
    ]))

    # Left broadcasting
    cC = a - B
    assert len(cC) == 4
    assert np.array_equal(cC.data, np.array([
        [[-12], [-15], [-18], [1]],
        [[-13], [-16], [-19], [1]],
        [[-14], [-17], [-20], [1]],
        [[-15], [-18], [-21], [1]]
    ], dtype=np.float64))

    # Right broadcasting (should be the same as left broadcasting but negative)
    Cc = B - a
    assert len(cC) == 4
    assert np.array_equal(Cc.data, (-1 * cC).data)

def test_indexing_homogeneous_vectors():
    x = np.array([1, 2, 3, 4], dtype=np.float64)
    y = np.array([5, 6, 7, 8], dtype=np.float64)
    z = np.array([9, 10, 11, 12], dtype=np.float64)
    v = HomVectors(x=x, y=y, z=z)

    # Scalar index
    assert np.array_equal(v[0].data, np.array([[[x[0]], [y[0]], [z[0]], [1.0]]]))
    assert np.array_equal(v[1].data, np.array([[[x[1]], [y[1]], [z[1]], [1.0]]]))
    assert np.array_equal(v[2].data, np.array([[[x[2]], [y[2]], [z[2]], [1.0]]]))
    assert np.array_equal(v[3].data, np.array([[[x[3]], [y[3]], [z[3]], [1.0]]]))
    
    # Slice first half
    assert len(v[0:2]) == 2
    assert np.array_equal(
        v[0:2].data,
        np.expand_dims(
            np.stack(
                [x[0:2], y[0:2], z[0:2], np.ones_like(x[0:2])],
                axis=1
            ),
            axis=2
        )
    )

    # Slice middle
    assert len(v[1:3]) == 2
    assert np.array_equal(
        v[1:3].data,
        np.expand_dims(
            np.stack(
                [x[1:3], y[1:3], z[1:3], np.ones_like(x[1:3])],
                axis=1
            ),
            axis=2
        )
    )

    # Slice all
    assert len(v[:]) == 4
    assert np.array_equal(v.data, v[:].data)

    # Reverse slice
    assert np.array_equal(
        v[::-1].data,
        np.array(
            [
                [[x[3]], [y[3]], [z[3]], [1.0]],
                [[x[2]], [y[2]], [z[2]], [1.0]],
                [[x[1]], [y[1]], [z[1]], [1.0]],
                [[x[0]], [y[0]], [z[0]], [1.0]]
            ]
        )
    )

    # Reverse slice vec3
    assert np.array_equal(
        v[::-1].vec3,
        np.array(
            [
                [[x[3]], [y[3]], [z[3]]],
                [[x[2]], [y[2]], [z[2]]],
                [[x[1]], [y[1]], [z[1]]],
                [[x[0]], [y[0]], [z[0]]],
            ]
        )
    )

def test_concatenating_homogeneous_vectors():
    u = HomVectors(x=1, y=2, z=3)
    v = HomVectors(x=4, y=5, z=6)
    a = u.concatenate(v)
    b = v.concatenate(u)

    assert len(a) == 2
    assert len(b) == 2

    assert np.array_equal(a.data, np.array([[[1], [2], [3], [1]], [[4], [5], [6], [1]]]))
    assert np.array_equal(b.data, np.array([[[4], [5], [6], [1]], [[1], [2], [3], [1]]]))

    assert np.array_equal(a[::-1].data, b.data)
