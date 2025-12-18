import numpy as np

def _cast_to_bytes(value):
    """
    Decodes a value of type bytearray to bytes, with the goal of making it immutable. If value is of type ndarray,
    and its elements are objects of python (not numpy objects), it flattens it to an iterator (the assumption is that
    we don't lose info because the shape is (,1) or (1,1)), and gets the first element.

    :param value: np.ndarray | (bytes, bytearray) | Any
         input value
    :return: an array of bytes of the value
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.size != 1:
                raise ValueError(f"Expected a single bytes object, got {value.size}")
            return bytes(value.flat[0])
        raise TypeError("normalize_to_bytes called on numeric ndarray")
    else:
        raise TypeError(f"Unexpected type: {type(value)}")


def _reshape_with_inferred_dim(arr, shape_spec):
    """
    Reshape arr according to shape_spec, where at most one entry may be -1 (to be inferred from arr.size).

    :param arr: np.ndarray
    :param shape_spec

    :return arr.reshape(shape)
        reshaped np.ndarray
    """
    shape = list(shape_spec)
    # sanity check
    if shape.count(-1) > 1:
        raise ValueError("At most one -1 is allowed in shape_spec")
    # infer missing shape
    if -1 in shape:
        known = int(np.prod([d for d in shape if d != -1])) or 1
        if arr.size % known != 0:
            raise ValueError(f"Cannot infer missing dimension: {arr.size} elements "
                             f"is not divisible by known product {known}")
        inferred = arr.size // known
        shape[shape.index(-1)] = inferred
    # sanity check
    if np.prod(shape) != arr.size:
        raise ValueError(f"Cannot reshape array of size {arr.size} into shape {tuple(shape)}")
    # Avoid unnecessary reshape if shape already matches
    if tuple(arr.shape) == tuple(shape):
        return arr

    return arr.reshape(shape)


def _cast_raw_array(value, dtype, shape_spec):
    """
    Cast an array to a np.ndarray.
    It splits in two cases, one if (value is np.ndarray AND value.dtype != object), the other one is the else case.
    For both do: 1. Cast to dtype 2. Reshape

    :param value:
        input value that I want to decode
    :param dtype:
        dtype I want to convert value in.
    :param shape_spec:
        shape I want to convert value in.

    :return converted_arr: np.ndarray
        array converted to desired
    """
    # Cast to dtype
    if isinstance(value, np.ndarray) and value.dtype != object:
        # If it's already a numpy array and has numpy values/elements
        converted_arr = value.astype(dtype)
    else:
        # If it's not a numpy array or it has python values/elements
        raw_bytes = _cast_to_bytes(value)
        # Re-read raw bytes as for the desired dtype
        converted_arr = np.frombuffer(raw_bytes, dtype=dtype)
    # Reshape
    converted_arr = _reshape_with_inferred_dim(converted_arr, shape_spec)
    return converted_arr


def cast_trajectory_from_record(record, meta):
    """
    Casts the TFRecord into numpy arrays.
    The feature meta["features"] is a dict that contains info about the features and their shape.

    :param record:
        A TFRecord object
    :param meta:
        json file

    :return trajectory_dict: Dict
        A dict containing all trajectory features
    """

    shapes = meta["features"]
    world_pos = _cast_raw_array(record["world_pos"], np.float32, shapes["world_pos"]["shape"])
    stress = _cast_raw_array(record["stress"], np.float32, shapes["stress"]["shape"])
    node_type = _cast_raw_array(record["node_type"], np.int32, shapes["node_type"]["shape"])
    mesh_pos = _cast_raw_array(record["mesh_pos"], np.float32, shapes["mesh_pos"]["shape"])
    mesh_cells = _cast_raw_array(record["cells"], np.int32, shapes["cells"]["shape"])

    # remove extra dims
    if node_type.shape[0] == 1:
        node_type = node_type[0]
    if mesh_pos.shape[0] == 1:
        mesh_pos = mesh_pos[0]
    if mesh_cells.shape[0] == 1:
        mesh_cells = mesh_cells[0]

    trajectory_dict = {
        "world_pos": world_pos.astype(np.float32),
        "stress": stress.astype(np.float32),
        "node_type": node_type.astype(np.int32),
        "mesh_pos": mesh_pos.astype(np.float32),
        "cells": mesh_cells.astype(np.int32),
    }
    return trajectory_dict
