# https://github.com/fastai/fastcore
from types import MethodType, FunctionType
import functools, builtins, copy
def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f,FunctionType): return copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    fn.__qualname__ = f.__qualname__
    return fn
def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple,list)): cls=(cls,)
    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS: setattr(nf, o, getattr(f,o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))
    return _inner
#####################################################################
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging

logger = logging.getLogger()

@patch_to(paddle.Tensor)
def float(self):
    return self.astype("float32")

@patch_to(paddle.Tensor)
def narrow(self, axis, start, length):
    return paddle.slice(self, input, [axis], [start], [start+length])

@patch_to(paddle.Tensor)
def half(self):
    return self.astype("float16")

@patch_to(nn.Layer)
def half(self):
    self.to(dtype="float16")
    return self

@patch_to(paddle.Tensor)
def long(self):
    return self.astype("int64")

@patch_to(paddle.Tensor)
def int(self):
    return self.astype("int32")

@patch_to(paddle.Tensor, as_prop=True)
def device(self):
    return self.place

@patch_to(paddle.Tensor)
def softmax(self, axis: int = -1, dtype = None, name = None):
    return F.softmax(self, axis=axis, dtype=dtype, name=name)

@patch_to(paddle.Tensor)
def to(self, dtype="float32", device=None):
    if paddle.is_tensor(dtype):
        dtype = dtype.dtype
    return self.astype(dtype)

@patch_to(paddle.Tensor, as_prop=True)
def data(self):
    tensor = self.clone()
    tensor.stop_gradient = True
    return tensor

@patch_to(paddle.Tensor)
def copy_(self, data):
    self.set_value(data)

@patch_to(paddle.Tensor)
def size(self, i=None):
    if i is None:
        return self.shape
    return self.shape[i]

@patch_to(paddle.Tensor)
def clamp(self, min=None, max=None, name=None):
    return paddle.clip(self, min=min, max=max, name=name)

paddle.clamp = paddle.clip

# scatter
def scatter_torch(tensor, dim, index, src):
    assert dim == 0 or dim == 1
    assert tensor.ndim == index.ndim == src.ndim == 2
    index = index.astype("int64")
    i, j = index.shape
    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    if dim == 0:
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    else:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(src, index=updates_index)
    res = paddle.scatter_nd_add(tensor, index, updates)
    return res
paddle.scatter_torch = scatter_torch
scatter_torch = patch_to(paddle.Tensor)(scatter_torch)

@patch_to(nn.Layer)
def load_state_dict(self, *args, **kwargs):
    strict = kwargs.pop("strict", None)
    # print(f"load_state_dict?? {strict}?")
    return self.set_dict(*args, **kwargs)

# device
def get_parameter_device(parameter: nn.Layer):
    try:
        return next(parameter.named_parameters())[1].device
    except StopIteration:
        return paddle.get_device()


def get_parameter_dtype(parameter: nn.Layer):
    try:
        return next(parameter.named_parameters())[1].dtype
    except StopIteration:
        return paddle.get_default_dtype()
    
@patch_to(nn.Layer, as_prop=True)
def device(self):
    return get_parameter_device(self)

@patch_to(nn.Layer, as_prop=True)
def dtype(self):
    return get_parameter_dtype(self)
    
@patch_to(nn.Layer)
def named_layers(self):
    return self.named_sublayers(include_self=True)

@patch_to(nn.Layer)
def stop_gradient(self):
    for paramater in self.paramaters():
        paramater.stop_gradient = True

def pad_new(x, pad, mode="constant", value=0):
    # 如果所有的pad值全为0，则不进行任何操作。
    if not any(pad): return x
    new_pad = []
    for _ in range(x.ndim * 2 - len(pad)):
        new_pad.append(0)
    ndim = list(range(x.ndim - 1, 0, -1))
    axes_start = {}
    for i, _pad in enumerate(pad):
        if _pad < 0:
            new_pad.append(0)
            zhengshu, yushu = divmod(i, 2)
            if yushu == 0:
                axes_start[ndim[zhengshu]] = -_pad
        else:
            new_pad.append(_pad)

    padded = paddle.nn.functional.pad(x, new_pad, mode=mode, value=value)
    padded_shape = paddle.shape(padded)
    axes = []
    starts = []
    ends = []
    for k, v in axes_start.items():
        axes.append(k)
        starts.append(v)
        ends.append(padded_shape[k])
        assert v < padded_shape[k]

    if axes:
        return padded.slice(axes=axes, starts=starts, ends=ends)
    else:
        return padded
    
F.pad_new = pad_new

# # BatchNorm2D不转成fp16
# @patch_to(nn.Layer)
# def _apply(self, func, device, dtype, blocking, include_sublayers=True):
#     if isinstance(self, nn.BatchNorm2D):
#         return 
#     if include_sublayers:
#         for layer in self.children():
#             layer._apply(func, device, dtype, blocking, include_sublayers)

#     for key, param in self._parameters.items():
#         if param is not None:
#             with paddle.no_grad():
#                 param_applied = func(param, device, dtype, blocking)

#             if param.grad is not None:
#                 with paddle.no_grad():
#                     grad_applied = func(param._grad_ivar(), device, dtype,
#                                         blocking)

#     for key, buf in self._buffers.items():
#         if buf is not None:
#             self._buffers[key] = func(buf, device, dtype, blocking)

#     self._dtype = dtype
##########################################################CLIPFeatureExtractor patch
from typing import Any, Dict, Tuple, Union
import os
import numpy as np
import json
import copy
from paddlenlp.transformers.clip.feature_extraction import CLIPFeatureExtractor


@patch_to(CLIPFeatureExtractor)
def to_json_file(self, json_file_path: Union[str, os.PathLike]):
    with open(json_file_path, "w", encoding="utf-8") as writer:
        writer.write(self.to_json_string())

@patch_to(CLIPFeatureExtractor)
def to_dict(self) -> Dict[str, Any]:
    output = copy.deepcopy(self.__dict__)
    output["feature_extractor_type"] = self.__class__.__name__
    return output

@patch_to(CLIPFeatureExtractor)
def to_json_string(self) -> str:
    dictionary = self.to_dict()

    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            dictionary[key] = value.tolist()

    # make sure private name "_processor_class" is correctly
    # saved as "processor_class"
    _processor_class = dictionary.pop("_processor_class", None)
    if _processor_class is not None:
        dictionary["processor_class"] = _processor_class

    return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

@patch_to(CLIPFeatureExtractor, cls_method=True)
def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
    feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
    return cls.from_dict(feature_extractor_dict, **kwargs)

@patch_to(CLIPFeatureExtractor)
def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
    if os.path.isfile(save_directory):
        raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

    os.makedirs(save_directory, exist_ok=True)

    # # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
    # # loaded from the Hub.
    # if self._auto_class is not None:
    #     custom_object_save(self, save_directory, config=self)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)

    self.to_json_file(output_feature_extractor_file)
    logger.info(f"Feature extractor saved in {output_feature_extractor_file}")

    return [output_feature_extractor_file]


@patch_to(CLIPFeatureExtractor, cls_method=True)
def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs):
    return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

    feature_extractor = cls(**feature_extractor_dict)

    # Update feature_extractor with kwargs if needed
    to_remove = []
    for key, value in kwargs.items():
        if hasattr(feature_extractor, key):
            setattr(feature_extractor, key, value)
            to_remove.append(key)
    for key in to_remove:
        kwargs.pop(key, None)

    logger.info(f"Feature extractor {feature_extractor}")
    if return_unused_kwargs:
        return feature_extractor, kwargs
    else:
        return feature_extractor


################################################get_feature_extractor_dict
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"


def is_offline_mode():
    return _is_offline_mode

@patch_to(CLIPFeatureExtractor, cls_method=True)
def get_feature_extractor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
    feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.
    Parameters:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
    Returns:
        `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
    """
    # cache_dir = kwargs.pop("cache_dir", None)
    # force_download = kwargs.pop("force_download", False)
    # resume_download = kwargs.pop("resume_download", False)
    # proxies = kwargs.pop("proxies", None)
    # use_auth_token = kwargs.pop("use_auth_token", None)
    local_files_only = kwargs.pop("local_files_only", False)
    # revision = kwargs.pop("revision", None)

    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)

    user_agent = {"file_type": "feature extractor", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
    if os.path.isfile(pretrained_model_name_or_path):
        resolved_feature_extractor_file = pretrained_model_name_or_path
        is_local = True
    else:
        feature_extractor_file = FEATURE_EXTRACTOR_NAME
        try:
            resolved_feature_extractor_file = os.path.join(pretrained_model_name_or_path, feature_extractor_file)
        except EnvironmentError:
            # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
            # the original exception.
            raise
        except Exception:
            # For any other exception, we throw a generic error.
            raise EnvironmentError(
                f"Can't load feature extractor for '{pretrained_model_name_or_path}'. If you were trying to load"
                " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                f" directory containing a {FEATURE_EXTRACTOR_NAME} file"
            )

    try:
        # Load feature_extractor dict
        with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)

    except json.JSONDecodeError:
        raise EnvironmentError(
            f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file."
        )

    if is_local:
        logger.info(f"loading configuration file {resolved_feature_extractor_file}")
    else:
        logger.info(
            f"loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
        )

    return feature_extractor_dict, kwargs

from paddlenlp.transformers import CLIPTextModel, PretrainedModel
@patch_to(CLIPTextModel)
def get_input_embeddings(self):
    return self.text_model.token_embedding

@patch_to(CLIPTextModel)
def set_input_embeddings(self, value):
    self.text_model.token_embedding = value

@patch_to(CLIPTextModel)
def get_model_config(self):
    # If init_config contains a Layer, use the layer's init_config to save
    def get_config(model):
        model_config = model.init_config
        for key, value in model_config.items():
            if key == "init_args":
                args = []
                for arg in value:
                    args.append(
                        get_config(arg) if isinstance(arg, PretrainedModel
                                                        ) else arg)
                model_config[key] = tuple(args)
            elif isinstance(value, PretrainedModel):
                model_config[key] = value.init_config
        return model_config

    model_config = get_config(self)
    model_config['vocab_size'] = self.base_model.config['vocab_size']
    return model_config