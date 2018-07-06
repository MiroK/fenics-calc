# This code is taken from PyDMD https://github.com/mathLab/PyDMD
# It is included here for the sake of simplifying the installation
# on UiO machines

from __future__ import division
from os.path import splitext
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np


class DMDBase(object):
    """
    Dynamic Mode Decomposition base class.
    :param int svd_rank: rank truncation in SVD. If 0, the method computes the
        optimal rank and uses it for truncation; if positive number, the method
        uses the argument for the truncation; if -1, the method does not
        compute truncation.
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    :cvar dict original_time: dictionary that contains information about the
        time window where the system is sampled:
           - `t0` is the time of the first input snapshot;
           - `tend` is the time of the last input snapshot;
           - `dt` is the delta time between the snapshots.
    :cvar dict dmd_time: dictionary that contains information about the time
        window where the system is reconstructed:
            - `t0` is the time of the first approximated solution;
            - `tend` is the time of the last approximated solution;
            - `dt` is the delta time between the approximated solutions.
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False):
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        self.original_time = None
        self.dmd_time = None

        self._eigs = None
        self._Atilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None

    @property
    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.
        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(self.dmd_time['t0'],
                         self.dmd_time['tend'] + self.dmd_time['dt'],
                         self.dmd_time['dt'])

    @property
    def original_timesteps(self):
        """
        Get the timesteps of the original snapshot.
        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(self.original_time['t0'],
                         self.original_time['tend'] + self.original_time['dt'],
                         self.original_time['dt'])

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.
        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return self._modes

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called A tilde.
        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.
        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return self._eigs

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.
        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray
        """
        omega = old_div(np.log(self.eigs), self.original_time['dt'])
        vander = np.exp(np.multiply(*np.meshgrid(omega, self.dmd_timesteps)))
        return (vander * self._b).T

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        return self.modes.dot(self.dynamics)

    @property
    def snapshots(self):
        """
        Get the original input data.
        :return: the matrix that contains the original snapshots.
        :rtype: numpy.ndarray
        """
        return self._snapshots

    def fit(self, X):
        """
        Abstract method to fit the snapshots matrices.
        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.fit'.format(
                self.__class__.__name__))

    @staticmethod
    def _col_major_2darray(X):
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.
        :param X: the input snapshots.
        :type X: int or numpy.ndarray
        :return: the 2D matrix that contains the flatten snapshots, the shape
            of original snapshots.
        :rtype: numpy.ndarray, tuple
        """

        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return X, None

        input_shapes = [np.asarray(x).shape for x in X]

        if len(set(input_shapes)) is not 1:
            raise ValueError('Snapshots have not the same dimension.')

        snapshots_shape = input_shapes[0]
        snapshots = np.transpose([np.asarray(x).flatten() for x in X])
        return snapshots, snapshots_shape

    @staticmethod
    def _compute_tlsq(X, Y, tlsq_rank):
        """
        Compute Total Least Square.
        :param numpy.ndarray X: the first matrix;
        :param numpy.ndarray Y: the second matrix;
        :param int tlsq_rank: the rank for the truncation; If 0, the method
            does not compute any noise reduction; if positive number, the
            method uses the argument for the SVD truncation used in the TLSQ
            method.
        :return: the denoised matrix X, the denoised matrix Y
        :rtype: numpy.ndarray, numpy.ndarray
        References:
        https://arxiv.org/pdf/1703.11004.pdf
        https://arxiv.org/pdf/1502.03854.pdf
        """
        # Do not perform tlsq
        if tlsq_rank is 0:
            return X, Y

        V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
        rank = min(tlsq_rank, V.shape[0])
        VV = V[:rank, :].conj().T.dot(V[:rank, :])

        return X.dot(VV), Y.dot(VV)

    @staticmethod
    def _compute_svd(X, svd_rank):
        """
        Truncated Singular Value Decomposition.
        :param numpy.ndarray X: the matrix to decompose.
        :param svd_rank: the rank for the truncation; If 0, the method computes
            the optimal rank and uses it for truncation; if positive interger,
            the method uses the argument for the truncation; if float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`; if -1,
            the method does not compute truncation.
        :type svd_rank: int or float
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s, V = np.linalg.svd(X, full_matrices=False)
        V = V.conj().T

        if svd_rank is 0:
            omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif svd_rank > 0 and svd_rank < 1:
            cumulative_energy = np.cumsum(s / s.sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        U = U[:, :rank]
        V = V[:, :rank]
        s = s[:rank]

        return U, s, V

    @staticmethod
    def _build_lowrank_op(U, s, V, Y):
        """
        Private method that computes the lowrank operator from the singular
        value decomposition of matrix X and the matrix Y.
        .. math::
            \\mathbf{\\tilde{A}} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{X}^\\dagger \\mathbf{U} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{V} \\mathbf{S}^{-1}
        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param numpy.ndarray Y: input matrix Y.
        :return: the lowrank operator
        :rtype: numpy.ndarray
        """
        return U.T.conj().dot(Y).dot(V) * np.reciprocal(s)

    @staticmethod
    def _eig_from_lowrank_op(Atilde, Y, U, s, V, exact):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator from the low-dimensional operator and the
        input matrix.
        :param numpy.ndarray Atilde: the lowrank operator.
        :param numpy.ndarray Y: input matrix Y.
        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param bool exact: if True, the exact modes are computed; otherwise,
            the projected ones are computed.
        :return: eigenvalues, eigenvectors
        :rtype: numpy.ndarray, numpy.ndarray
        """
        lowrank_eigenvalues, lowrank_eigenvectors = np.linalg.eig(Atilde)

        # Compute the eigenvectors of the high-dimensional operator
        if exact:
            eigenvectors = ((
                Y.dot(V) * np.reciprocal(s)).dot(lowrank_eigenvectors))
        else:
            eigenvectors = U.dot(lowrank_eigenvectors)

        # The eigenvalues are the same
        eigenvalues = lowrank_eigenvalues

        return eigenvalues, eigenvectors

    @staticmethod
    def _compute_amplitudes(modes, snapshots, eigs, opt):
        """
        Compute the amplitude coefficients. If `opt` is False the amplitudes
        are computed by minimizing the error between the modes and the first
        snapshot; if `opt` is True the amplitudes are computed by minimizing
        the error between the modes and all the snapshots, at the expense of
        bigger computational cost.
        :param numpy.ndarray modes: 2D matrix that contains the modes, stored
            by column.
        :param numpy.ndarray snapshots: 2D matrix that contains the original
            snapshots, stored by column.
        :param numpy.ndarray eigs: array that contains the eigenvalues of the
            linear operator.
        :param bool opt: flag for optimized dmd.
        :return: the amplitudes array
        :rtype: numpy.ndarray
        """
        if opt:
            L = np.concatenate(
                [
                    modes.dot(np.diag(eigs**i))
                    for i in range(snapshots.shape[1])
                ],
                axis=0)
            b = np.reshape(snapshots, (-1, ), order='F')

            a = np.linalg.lstsq(L, b)[0]
        else:
            a = np.linalg.lstsq(modes, snapshots.T[0])[0]

        return a


class DMD(DMDBase):
    """
    Dynamic Mode Decomposition
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    """

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.
        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]

        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        U, s, V = self._compute_svd(X, self.svd_rank)

        self._Atilde = self._build_lowrank_op(U, s, V, Y)

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, Y, U, s, V, self.exact)

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        return self
