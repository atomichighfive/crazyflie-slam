import numpy as np
from typing import Self, Optional, Union, Any
from numbers import Number

class HomMatrix:
    def __init__(self, data: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None, T: Optional[np.ndarray] = None) -> None:
        assert data is not None or (R is not None and T is not None), "Must either provide the whole matrix as data or R and T"
        if data is not None:
            assert data.shape == (4, 4), "Matrix must be 4x4"
            self._data = data.astype(np.float64)
        else:
            self._data = np.eye(4, dtype=np.float64)
            self._data[:3, :3] = R
            self._data[:3, 3] = T
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def R(self) -> np.ndarray:
        return self._data[:3, :3]
    
    @property
    def T(self) -> np.ndarray:
        return self._data[:3, 3]
    
    @property
    def W(self) -> np.ndarray:
        return self._data[3, :]
    
    @property
    def inv(self) -> Self:
        return HomMatrix(data=np.linalg.inv(self.data))


class HomVectors:
    def __init__(
            self,
            data: Optional[np.ndarray] = None,
            x: Optional[Union[Number, np.float64, np.ndarray]] = None,
            y: Optional[Union[Number, np.float64, np.ndarray]] = None,
            z: Optional[Union[Number, np.float64, np.ndarray]] = None
    ) -> None:
        assert data is not None or (x is not None and y is not None and z is not None), "Must either provide the whole matrix as data or x, y, z"
        if data is not None:
            assert data.ndim in (2, 3) and data.shape[-2] == 4 and data.shape[-1] == 1, "Data must be of shape (4, 1) or (N, 4, 1)"
            if data.ndim == 2:
                data = data.reshape(-1, 4, 1)
            self._data = data.astype(np.float64)
        else:
            if isinstance(x, (Number, np.float64)):
                self._data = np.array([x, y, z, 1], dtype=np.float64).reshape(1, 4, 1).astype(np.float64)
            else:
                self._data = np.expand_dims(np.stack([x, y, z, np.ones_like(x)], axis=1), 2).astype(np.float64)

    @property
    def x(self) -> np.float64:
        return self._data[:, 0, 0]
    
    @property
    def y(self) -> np.float64:
        return self._data[:, 1, 0]
    
    @property
    def z(self) -> np.float64:
        return self._data[:, 2, 0]
    
    @property
    def w(self) -> np.float64:
        return self._data[:, 3, 0]
    
    @property
    def vec3(self) -> np.ndarray:
        return self.data[:, :3, :] / self.w[:, np.newaxis, np.newaxis]
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    def __len__(self) -> int:
        return self._data.shape[0]
    
    def __getitem__(self, key: Union[slice, int]) -> Self:
        if isinstance(key, slice):
            return HomVectors(data=self.data[key])
        return HomVectors(data=np.expand_dims(self.data[key], axis=0))
    
    def __add__(self, other: Self) -> Self:
        assert isinstance(other, HomVectors), "Only addition with another HomVectors is supported. If you want to add a scalar, create a HomVector the same value for all elements."
        assert len(self) == 1 or len(other) in (1, len(self)), "Can only add vectors with the same length, or with len(1), in which case broadcasting is used."
        new_data = self.data + other.data
        new_data[:, 3, :] = 1
        return HomVectors(data=new_data)
    
    def __rmul__(self, other: Union[Number, np.float64]) -> Self:
        assert isinstance(other, (Number, np.float64)), "Only scalar multiplication is supported. If scaling is needed, use a scaling matrix."
        new_data = other * self.data
        new_data[:, 3, :] = 1
        return HomVectors(data=new_data)
    
    def __sub__(self, other: Self) -> Self:
        assert isinstance(other, HomVectors), "Only subtraction with another HomVectors is supported. If you want to subtract a scalar, create a HomVector the same value for all elements."
        assert len(self) == 1 or len(other) in (1, len(self)), "Can only subtract vectors with the same length, or with len(1), in which case broadcasting is used."
        new_data = self.data + (-1.0 * other.data)
        new_data[:, 3, :] = 1
        return HomVectors(data=new_data)
    
    def __mul__(self, other: Any) -> Self:
        raise NotImplementedError("Only scalar multiplication is supported. By convension, scalar multiplication should use the scalar on the left side.")
    
    def __div__(self, other: Any) -> Self:
        raise NotImplementedError("Division is not supported. Use scalar multiplication with reciprocal instead.")
    
    def __truediv__(self, other: Any) -> Self:
        raise NotImplementedError("True division is not supported. Use scalar multiplication with reciprocal instead.")

    def transform(self, matrix: HomMatrix) -> Self:
        result = np.dot(matrix, self.data)
        result = result / result.w
        return result
    
    def concatenate(self, other: Self) -> Self:
        return HomVectors(np.concatenate([self.data, other.data], axis=0))
        