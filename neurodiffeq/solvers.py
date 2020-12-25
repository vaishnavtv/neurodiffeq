import torch
import warnings
import torch.nn as nn
from abc import ABC, abstractmethod
from itertools import chain
from copy import deepcopy
from torch.optim import Adam
from neurodiffeq.networks import FCNN
from neurodiffeq._version_utils import deprecated_alias


class BaseSolver(ABC):
    """A class for solving ODE/PDE systems.

    :param diff_eqs:
        The PDE system to solve, which maps a tuple of three coordinates to a tuple of PDE residuals.
        Both the coordinates and PDE residuals must have shape (-1, 1).
    :type diff_eqs: callable
    :param conditions:
        List of boundary conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length must equal that of conditions.
    :type nets: list[`torch.nn.Module`], optional
    :param train_generator:
        A generator for sampling training points.
        It must provide a `.get_examples()` method and a `.size` field.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, required
    :param valid_generator:
        A generator for sampling validation points.
        It must provide a `.get_examples()` method and a `.size` field.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, required
    :param analytic_solutions:
        The analytical solutions to be compared with neural net solutions.
        It maps a tuple of three coordinates to a tuple of function values.
        The output shape should match that of networks.
    :type analytic_solutions: callable, optional
    :param optimizer:
        The optimizer to be used for training.
    :type optimizer: `torch.nn.optim.Optimizer`, optional
    :param criterion:
        A function that maps a PDE residual vector (torch tensor with shape (-1, 1)) to a scalar loss.
    :type criterion: callable, optional
    :param n_batches_train:
        The number of batches to train in every epoch, where batch-size equals `train_generator.size`.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        The number of batches to valid in every epoch, where batch-size equals `valid_generator.size`.
    :type n_batches_valid: int, optional
    :param n_input_units:
        Number of input units for each neural network. Ignored if ``nets`` is specified.
    :type n_input_units: int, required
    :param n_output_units:
        Number of output units for each neural network. Ignored if ``nets`` is specified.
    :type n_output_units: int, required
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify n_batches_train and n_batches_valid instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    def __init__(self, diff_eqs, conditions,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
                 optimizer=None, criterion=None, n_batches_train=1, n_batches_valid=4,
                 n_input_units=None, n_output_units=None,
                 # deprecated arguments are listed below
                 shuffle=False, batch_size=None):
        # deprecate argument `shuffle`
        if shuffle:
            warnings.warn(
                "param `shuffle` is deprecated and ignored; shuffling should be performed by generators",
                FutureWarning,
            )
        # deprecate argument `batch_size`
        if batch_size is not None:
            warnings.warn(
                "param `batch_size` is deprecated and ignored; specify n_batches_train and n_batches_valid instead",
                FutureWarning,
            )

        self.diff_eqs = diff_eqs
        self.conditions = conditions
        self.n_funcs = len(conditions)
        if nets is None:
            self.nets = [
                FCNN(n_input_units=n_input_units, n_output_units=n_output_units, hidden_units=(32, 32), actv=nn.Tanh)
                for _ in range(self.n_funcs)
            ]
        else:
            self.nets = nets

        if train_generator is None:
            raise ValueError("train_generator must be specified")

        if valid_generator is None:
            raise ValueError("valid_generator must be specified")

        self.analytic_solutions = analytic_solutions
        self.optimizer = optimizer if optimizer else Adam(chain.from_iterable(n.parameters() for n in self.nets))
        self.criterion = criterion if criterion else lambda r: (r ** 2).mean()

        def make_pair_dict(train=None, valid=None):
            return {'train': train, 'valid': valid}

        self.generator = make_pair_dict(train=train_generator, valid=valid_generator)
        # loss history
        self.loss = make_pair_dict(train=[], valid=[])
        # analytic MSE history
        self.analytic_mse = make_pair_dict(train=[], valid=[])
        # number of batches for training / validation;
        self.n_batches = make_pair_dict(train=n_batches_train, valid=n_batches_valid)
        # current batch of samples, kept for additional_loss term to use
        self._batch_examples = make_pair_dict()
        # current network with lowest loss
        self.best_nets = None
        # current lowest loss
        self.lowest_loss = None
        # local epoch in a `.fit` call, should only be modified inside self.fit()
        self.local_epoch = 0
        # maximum local epochs to run in a `.fit()` call, should only set by inside self.fit()
        self._max_local_epoch = 0
        # controls early stopping, should be set to False at the beginning of a `.fit()` call
        # and optionally set to False by `callbacks` in `.fit()` to support early stopping
        self._stop_training = False
        # the _phase variable is registered for callback functions to access
        self._phase = None

    @property
    def global_epoch(self):
        """Global epoch count, always equal to the length of train loss history.

        :return: Number of training epochs that have been run.
        :rtype: int
        """
        return len(self.loss['train'])

    def compute_func_val(self, net, cond, *coordinates):
        """Compute the function value evaluated on the points specified by ``coordinates``.

        :param net: The network to be parameterized and evaluated.
        :type net: torch.nn.Module
        :param cond: The condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of coordinate components, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at the sampled points.
        :rtype: torch.Tensor
        """
        return cond.enforce(net, *coordinates)

    def _update_history(self, value, metric_type, key):
        """Append a value to corresponding history list.

        :param value: Value to be appended.
        :type value: float
        :param metric_type: {'loss', 'analytic_mse'}; Type of history metrics.
        :type metric_type: str
        :param key: {'train', 'valid'}; Dict key in ``self.loss`` or ``self.analytic_mse``.
        :type key: str
        """
        self._phase = key
        if metric_type == 'loss':
            self.loss[key].append(value)
        elif metric_type == 'analytic_mse':
            self.analytic_mse[key].append(value)
        else:
            raise KeyError(f'history type = {metric_type} not understood')

    def _update_train_history(self, value, metric_type):
        """Append a value to corresponding training history list."""
        self._update_history(value, metric_type, key='train')

    def _update_valid_history(self, value, metric_type):
        """Append a value to corresponding validation history list."""
        self._update_history(value, metric_type, key='valid')

    def _generate_batch(self, key):
        """Generate the next batch, register in self._batch_examples and return the batch.

        :param key:
            {'train', 'valid'};
            Dict key in ``self._examples``, ``self._batch_examples``, or ``self._batch_start``
        :type key: str
        :return: The generated batch of points.
        :type: List[`torch.Tensor`]
        """
        # the following side effects are helpful for future extension,
        # especially for additional loss term that depends on the coordinates
        self._phase = key
        self._batch_examples[key] = [v.reshape(-1, 1) for v in self.generator[key].get_examples()]
        return self._batch_examples[key]

    def _generate_train_batch(self):
        """Generate the next training batch, register in ``self._batch_examples`` and return."""
        return self._generate_batch('train')

    def _generate_valid_batch(self):
        """Generate the next validation batch, register in ``self._batch_examples`` and return."""
        return self._generate_batch('valid')

    def _do_optimizer_step(self):
        r"""Optimization procedures after gradients have been computed. Usually ``self.optimizer.step()`` is sufficient.
        At times, users can overwrite this method to perform gradient clipping, etc. Here is an example::

            import itertools
            class MySolver(Solver)
                def _do_optimizer_step(self):
                    nn.utils.clip_grad_norm_(itertools.chain([net.parameters() for net in self.nets]), 1.0, 'inf')
                    self.optimizer.step()
        """
        self.optimizer.step()

    def _run_epoch(self, key):
        """Run an epoch on train/valid points, update history, and perform an optimization step if key=='train'.

        :param key: {'train', 'valid'}; phase of the epoch
        :type key: str

        .. note::
            The optimization step is only performed after all batches are run.
        """
        self._phase = key
        epoch_loss = 0.0
        epoch_analytic_mse = 0

        # perform forward pass for all batches: a single graph is created and release in every iteration
        # see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/17
        for batch_id in range(self.n_batches[key]):
            batch = self._generate_batch(key)
            funcs = [
                self.compute_func_val(n, c, *batch) for n, c in zip(self.nets, self.conditions)
            ]

            if self.analytic_solutions is not None:
                funcs_true = self.analytic_solutions(*batch)
                for f_pred, f_true in zip(funcs, funcs_true):
                    epoch_analytic_mse += ((f_pred - f_true) ** 2).mean().item()

            residuals = self.diff_eqs(*funcs, *batch)
            residuals = torch.cat(residuals, dim=1)
            loss = self.criterion(residuals) + self.additional_loss(funcs, key)

            # normalize loss across batches
            loss /= self.n_batches[key]

            # accumulate gradients before the current graph is collected as garbage
            if key == 'train':
                loss.backward()
            epoch_loss += loss.item()

        # calculate mean loss of all batches and register to history
        self._update_history(epoch_loss, 'loss', key)

        # perform optimization step when training
        if key == 'train':
            self._do_optimizer_step()
            self.optimizer.zero_grad()
        # update lowest_loss and best_net when validating
        else:
            self._update_best()

        # calculate mean analytic mse of all batches and register to history
        if self.analytic_solutions is not None:
            epoch_analytic_mse /= self.n_batches[key]
            epoch_analytic_mse /= self.n_funcs
            self._update_history(epoch_analytic_mse, 'analytic_mse', key)

    def run_train_epoch(self):
        """Run a training epoch, update history, and perform gradient descent."""
        self._run_epoch('train')

    def run_valid_epoch(self):
        """Run a validation epoch and update history."""
        self._run_epoch('valid')

    def _update_best(self):
        """Update ``self.lowest_loss`` and ``self.best_nets``
        if current validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.loss['valid'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)

    def fit(self, max_epochs, callbacks=None, monitor=None):
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training and validating and updating best model but before monitor checking

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param monitor:
            **[DEPRECATED]** use a MonitorCallback instance instead.
            The monitor for visualizing solution and metrics.
        :rtype monitor: `neurodiffeq.pde_spherical.MonitorSpherical`
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback function `cb(solver)` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        if monitor:
            warnings.warn("Monitor is deprecated, use a MonitorCallback instead")

        for local_epoch in range(max_epochs):
            # stops training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register local epoch so it can be accessed by callbacks
            self.local_epoch = local_epoch
            self.run_train_epoch()
            self.run_valid_epoch()

            if callbacks:
                for cb in callbacks:
                    cb(self)

            if monitor:
                if (local_epoch + 1) % monitor.check_every == 0 or local_epoch == max_epochs - 1:
                    monitor.check(
                        self.nets,
                        self.conditions,
                        loss_history=self.loss,
                        analytic_mse_history=self.analytic_mse,
                    )

    @abstractmethod
    def get_solution(self, copy=True):
        """Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: callable
        """
        pass

    def _get_internal_variables(self):
        """Get a dict of all available internal variables.

        :return:
            All available interal parameters,
            where keys are variable names and values are the corresponding variables.
        :rtype: dict

        .. note::
            Children classes should inherit all items and optionally include new ones.
        """

        return {
            "analytic_mse": self.analytic_mse,
            "analytic_solutions": self.analytic_solutions,
            "n_batches": self.n_batches,
            "best_nets": self.best_nets,
            "criterion": self.criterion,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch,
            "loss": self.loss,
            "lowest_loss": self.lowest_loss,
            "n_funcs": self.n_funcs,
            "nets": self.nets,
            "optimizer": self.optimizer,
            "pdes": self.diff_eqs,
            "generator": self.generator,
        }

    @deprecated_alias(param_names='var_names')
    def get_internals(self, var_names, return_type='list'):
        """Return internal variable(s) of the solver

        - If var_names == 'all', return all internal variables as a dict.
        - If var_names is single str, return the corresponding variables.
        - If var_names is a list and return_type == 'list', return corresponding internal variables as a list.
        - If var_names is a list and return_type == 'dict', return a dict with keys in var_names.

        :param var_names: An internal variable name or a list of internal variable names.
        :type var_names: str or list[str]
        :param return_type: {'list', 'dict'}; Ignored if ``var_names`` is a string.
        :type return_type: str
        :return: A single variable, or a list/dict of internal variables as indicated above.
        :rtype: list or dict or any
        """

        available_variables = self._get_internal_variables()

        if var_names == "all":
            return available_variables

        if isinstance(var_names, str):
            return available_variables[var_names]

        if return_type == 'list':
            return [available_variables[name] for name in var_names]
        elif return_type == "dict":
            return {name: available_variables[name] for name in var_names}
        else:
            raise ValueError(f"unrecognized return_type = {return_type}")

    def additional_loss(self, funcs, key):
        r"""Additional loss terms for training. This method is to be overridden by subclasses.
        This method can use any of the internal variables: the current batch, the nets, the conditions, etc.

        :param funcs: Outputs of the networks after parameterization.
        :type funcs: list[torch.Tensor]
        :param key: {'train', 'valid'}; Phase of the epoch, used to access the sample batch, etc.
        :type key: str
        :return: Additional loss. Must be a ``torch.Tensor`` of empty shape (scalar).
        :rtype: torch.Tensor
        """
        return 0.0


