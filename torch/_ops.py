import contextlib
import ctypes
import inspect
import sys
import types
from abc import ABC
from typing import Any, Dict

import torch._C

import torch.jit
from torch import _utils_internal
from torch._C import DispatchKey  # type: ignore[attr-defined]
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

# Query `hasattr` only once.
_SET_GLOBAL_FLAGS = hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags")


@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    if _SET_GLOBAL_FLAGS:
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    if _SET_GLOBAL_FLAGS:
        sys.setdlopenflags(old_flags)


# TODO(voz) We are missing an entire axis of registration - Modes for the python key
class PyOperatorABC(ABC):
    def __call__(self, *args, **kwargs):
        pass

    def py_impl(self, dispatch_key, fn):
        pass

    def name(self):
        pass

    def dispatch(self, dispatch_key, *args, **kwargs):
        pass


class PyOperator(PyOperatorABC):
    def __init__(self, name):
        self._name = name
        self.table = {}
        self.python_key_mode_table = {}

        # Make _OPNamespace not scream, this whole name based association needs a good hard look
        self.__name__ = "pyop." + name
        pyop_namespace.py_ops[name] = self

    def fallthrough(self, dispatch_key):
        self.table[dispatch_key] = self._fallthrough_fn(self, dispatch_key)

    def py_impl(self, dispatch_key_or_mode):
        def inner(fn):
            if inspect.isclass(dispatch_key_or_mode) and issubclass(
                dispatch_key_or_mode, TorchDispatchMode
            ):
                mode = dispatch_key_or_mode
                assert mode not in self.python_key_mode_table
                # TODO(voz): Should we replace setting torch._C.DispatchKey.Python entirely with setting mode keys?
                self.python_key_mode_table[mode] = fn
                return fn

            dispatch_key = dispatch_key_or_mode
            assert (
                dispatch_key != torch._C.DispatchKey.Python  # type: ignore[attr-defined]
            ), "Please register a mode for the torch._C.DispatchKey.Python key instead."
            assert isinstance(dispatch_key, torch._C.DispatchKey)  # type: ignore[attr-defined]
            assert dispatch_key not in self.table
            self.table[dispatch_key] = fn
            return fn

        return inner

    def dispatch(self, dispatch_key, *args, **kwargs):
        if dispatch_key == torch._C.DispatchKey.Python:  # type: ignore[attr-defined]
            # TODO(voz): We should walk all the nodes here / turn it into a list, topmode is ok for now.
            curr_mode = type(torch._C._get_torch_dispatch_mode())
            assert (
                curr_mode is not None
            ), "Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode."
            assert (
                curr_mode in self.python_key_mode_table
            ), f"Current active mode {curr_mode} not registered"
            # TODO(voz): The idea behind this is that we do not yet support dispatch by key + mode, only key.
            return self.python_key_mode_table[curr_mode](*args, **kwargs)

        assert dispatch_key in self.table
        return self.table[dispatch_key](*args, **kwargs)

    def __call__(self, *args, **kwargs):
        flat_args = _to_flat_tuple(args, kwargs)
        if has_torch_function(flat_args):
            return handle_torch_function(self, flat_args, *args, **kwargs)

        dispatch_key_set = _compute_keyset(args, kwargs)
        return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)

    def name(self):
        return self.name

    # TODO(voz): Should rewrite fallthrough register as the impl for keys we do not specify
    # as opposed to being this sort of explicit thing where ops are a little too key aware...
    def _fallthrough_fn(self, operator, dispatch_key):
        def inner(*args, **kwargs):
            all_keys_after_current = torch._C._dispatch_keyset_full_after(dispatch_key)  # type: ignore[attr-defined]
            all_keys_after_current_masked = all_keys_after_current & _compute_keyset(
                args, kwargs
            )  # type: ignore[attr-defined]
            return self.dispatch(
                all_keys_after_current_masked.highestPriorityTypeId(), *args, **kwargs
            )

        return inner


def _to_flat_tuple(args, kwargs):
    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    flat_all = flat_args + flat_kwargs
    return flat_all


def _compute_keyset(args, kwargs):
    tensors = _get_tensors(args, kwargs)
    return key_extractor(tensors)


def _get_tensors(args, kwargs):
    flat_all = _to_flat_tuple(args, kwargs)
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)


# Note - this should maintain identical impl to the C++ dispatcher key extraction logic
# at ATen/core/dispatch/DispatchKeyExtractor.h
def key_extractor(tensors):
    key_set = torch._C._dispatch_tls_local_include_set()  # type: ignore[attr-defined]
    for tensor in tensors:
        key_set = key_set | torch._C._dispatch_keys(tensor)  # type: ignore[attr-defined]
    key_set = key_set - torch._C._dispatch_tls_local_exclude_set()  # type: ignore[attr-defined]
    return key_set


# Each OpOverload object contains pointer to a a specific operator overload, a pointer to the parent `OpOverloadPacket` object.
# You can obtain an OpOverload object through attribute query on OpOverloadPacket.
class OpOverload(PyOperatorABC):
    def __init__(self, overloadpacket, op, op_dk, schema, tags):
        self._op = op
        self._op_dk = op_dk
        self._schema = schema
        self._overloadpacket = overloadpacket
        self._tags = tags
        self._overloadname = (
            "default" if schema.overload_name == "" else schema.overload_name
        )
        self._name = self._schema.name
        if schema.overload_name:
            self._name += "." + schema.overload_name
        self.py_kernels: Dict[DispatchKey, Any] = {}
        self.__name__ = "{}.{}".format(
            self._schema.name.split("::")[1], self._overloadname
        )
        # TODO(voz): Lots of shared logic around python_key_mode_table, maybe pull into base...
        self.python_key_mode_table = {}
        self.__module__ = overloadpacket.__module__
        op.__module__ = overloadpacket.__module__

    # it's a no-op since OpOverload object is immutable and must be unique for a given op overload.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverload(op='{}.{}', overload='{}')>".format(
            *self._schema.name.split("::"), self._overloadname
        )

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs or {})

    def __getattr__(self, key):
        return getattr(self._op, key)

    def __hash__(self):
        return hash(self._op)

    # `my_namespace.my_op_name.overload_name`
    def __str__(self):
        return "{}.{}.{}".format(*self._schema.name.split("::"), self._overloadname)

    def decompose(self, *args, **kwargs):
        dk = torch._C.DispatchKey.CompositeImplicitAutograd  # type: ignore[attr-defined]
        if (
            torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk)
            or dk in self.py_kernels
        ):
            return self.dispatch(dk, *args, **kwargs)
        else:
            return NotImplemented

    def py_impl(self, dispatch_key_or_mode):
        def inner(fn):
            if inspect.isclass(dispatch_key_or_mode) and issubclass(
                dispatch_key_or_mode, TorchDispatchMode
            ):
                mode = dispatch_key_or_mode
                assert mode not in self.python_key_mode_table
                # TODO(voz): Should we replace setting torch._C.DispatchKey.Python entirely with setting mode keys?
                self.python_key_mode_table[mode] = fn
                return fn

            assert isinstance(dispatch_key_or_mode, torch._C.DispatchKey)  # type: ignore[attr-defined]
            assert (
                dispatch_key_or_mode != torch._C.DispatchKey.Python  # type: ignore[attr-defined]
            ), "Please register a mode for the torch._C.DispatchKey.Python key instead."

            self.py_kernels[dispatch_key_or_mode] = fn
            return fn

        return inner

    def dispatch(self, dispatch_key, *args, **kwargs):
        if dispatch_key == torch._C.DispatchKey.Python:  # type: ignore[attr-defined]
            # TODO(voz): We should walk all the nodes here / turn it into a list, topmode is ok for now.
            curr_mode = type(torch._C._get_torch_dispatch_mode())
            assert (
                curr_mode is not None
            ), "Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode."
            if curr_mode not in self.python_key_mode_table:
                return self._op_dk(dispatch_key, *args, **kwargs)
            # TODO(voz): The idea behind this is that we do not yet support dispatch by key + mode, only key.
            return self.python_key_mode_table[curr_mode](*args, **kwargs)

        if dispatch_key in self.py_kernels:
            return self.py_kernels[dispatch_key](*args, **kwargs)
        else:
            return self._op_dk(dispatch_key, *args, **kwargs)

    def name(self):
        return self._name

    @property
    def overloadpacket(self):
        return self._overloadpacket

    @property
    def op(self):
        return self._op

    @property
    def tags(self):
        return self._tags

    # TODO: add more methods to expose information about input and output arguments


# OpOverloadPacket class contains pointer to a base unresolved operator that doesn't correspond to a specific operator
# You can obtain an OpOverload object through attribute query.
class OpOverloadPacket:
    def __init__(self, qualified_op_name, op_name, op, overload_names):
        # These attributes are accessible on the object through the properties
        # defined below but are immutable
        self._qualified_op_name = qualified_op_name
        self.__name__ = op_name
        self._op = op
        self._overload_names = overload_names

    # it's a no-op since OpOverloadPacket object is immutable and must be unique for a given op.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverloadPacket(op='{}.{}')>".format(
            *self._qualified_op_name.split("::")
        )

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return "{}.{}".format(*self._qualified_op_name.split("::"))

    @property
    def op(self):
        return self._op

    def __getattr__(self, key):
        # It is not a valid op_name when __file__ is passed in
        if key == "__file__":
            return "torch.ops"

        # ensure that query for dunder attributes that does not exist on
        # opoverloadpacket but instead exists on the self._op object does not unnecessarily call
        # `_get_operation_overload` (which is an expensive operation).
        # This is done to prevent any potential slowdown. This list can be extended
        # if there exists other attributes like `__name__` that only exist on self._op and not on the
        # opoverloadpacket.
        # This is ok since we are guaranteed that an overload name for an aten op can't start with '__'
        try:
            if key.startswith("__"):
                return getattr(self._op, key)
        except AttributeError:
            # for consistency because it seems weird to
            # throw an attribute error with a message containing
            # an object name different from the one the attribute
            # query was performed on.
            raise AttributeError(
                "'{}' can't have an overload name beginning with '__' and the "
                "underlying op {} has no attribute {} either.".format(
                    str(self), str(self._op), key
                )
            ) from None

        try:
            # This is ok since we are guaranteed that an overload name for an aten op can't be 'default'
            use_key = "" if key == "default" else key
            # TODO: disallow access to overloads registered by JIT
            op_, op_dk_, tags = torch._C._get_operation_overload(
                self._qualified_op_name, use_key
            )
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            overload = OpOverload(self, op_, op_dk_, schema, tags)
            # cache the overload object
            setattr(self, key, overload)
            return overload
        except RuntimeError:
            raise AttributeError(
                "The underlying op of '{}' has no overload name '{}'".format(
                    str(self), key
                )
            ) from None

    def __call__(self, *args, **kwargs):
        # overloading __call__ to ensure torch.ops.foo.bar()
        # is still callable from JIT
        # We save the function ptr as the `op` attribute on
        # OpOverloadPacket to access it here.
        return self._op(*args, **kwargs or {})

    # TODO: use this to make a __dir__
    def overloads(self):
        return [n if n else "default" for n in self._overload_names]


# Resolution of torch.fn is different from torch.ops.aten.fn
# torch.fn uses the Python argparser, matches with the
# appropriate schema, and calls into the unboxed version of the method
# torch.ops.aten.fn resolution is done via the mechanism defined in JIT.
# JIT creates a stack of all the overloads and then tries to match the
# correct one at runtime and always calls into the boxed version of the method
# Autograd codegen creates VariableType, TracerType,
# inplace or view type and python bindings.
# Aten codegen generates tensor methods for the the tensor class.

# _OpNamespace is a subclass of ModuleType because the torch script
# allows attribute lookups on modules only. Since we want torch.ops.foo.bar()
# to work from script, we need to ensure ops and foo are modules


class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """

    def __init__(self, name):
        super(_OpNamespace, self).__init__("torch.ops." + name)
        self.name = name
        if self.name == "pyop":
            self.pyops = pyop_namespace
        else:
            self.pyops = None  # type: ignore[assignment]

    def __getattr__(self, op_name):
        pyops = object.__getattribute__(self, "pyops")
        if pyops is not None:
            return pyops.py_ops[op_name]
        # It is not a valid op_name when __file__ is passed in
        if op_name == "__file__":
            return "torch.ops"
        elif op_name == "__origin__":
            raise AttributeError()

        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        namespace_name = self.name
        qualified_op_name = "{}::{}".format(namespace_name, op_name)
        try:
            op, overload_names = torch._C._jit_get_operation(qualified_op_name)
        except RuntimeError as e:
            # Turn this into AttributeError so getattr(obj, key, default)
            # works (this is called by TorchScript with __origin__)
            raise AttributeError(
                f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
            ) from e

        # let the script frontend know that op is identical to the builtin op
        # with qualified_op_name
        torch.jit._builtins._register_builtin(op, qualified_op_name)
        op.__module__ = self.__module__ + "." + namespace_name
        opoverloadpacket = OpOverloadPacket(
            qualified_op_name, op_name, op, overload_names
        )
        opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        setattr(self, op_name, opoverloadpacket)
        return opoverloadpacket


class _PyOpNamespace(_OpNamespace):
    def __init__(self):
        super(_PyOpNamespace, self).__init__("torch.ops.pyop")
        self.py_ops = {}


pyop_namespace = _PyOpNamespace()


class _Ops(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        super(_Ops, self).__init__("torch.ops")
        self.loaded_libraries = set()

    def __getattr__(self, name):
        # Here we are creating `torch.ops.my_namespace`
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        return namespace

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        if sys.executable == "torch_deploy":
            return

        path = _utils_internal.resolve_library_path(path)
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)


# The ops "namespace"
ops = _Ops()
