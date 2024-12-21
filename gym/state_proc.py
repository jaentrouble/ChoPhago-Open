import numpy as np
import numba as nb

@nb.njit(parallel=True)
def parallel_step(gamestates, actions):
    """
    !Note: Use np.typed.List or numpy array as input
    """
    dummy_state, dummy_action_mask = gamestates[0].state()
    output_state_array = np.zeros((len(gamestates), len(dummy_state)), dtype=np.int32)
    output_mask_array = np.zeros((len(gamestates), len(dummy_action_mask)), dtype=np.bool_)
    output_finished_array = np.zeros(len(gamestates), dtype=np.bool_)
    output_successed_array = np.zeros(len(gamestates), dtype=np.bool_)
    for i in nb.prange(len(gamestates)):
        gamestates[i].act(actions[i])
        output_state_array[i], output_mask_array[i] = gamestates[i].state()
        output_finished_array[i] = gamestates[i].finished()
        output_successed_array[i] = gamestates[i].successed()
    return output_state_array, output_mask_array, output_finished_array, output_successed_array

@nb.njit(parallel=True)
def parallel_step_ignore_finished(gamestates, actions):
    """
    !Note: Use np.typed.List or numpy array as input
    """
    dummy_state, dummy_action_mask = gamestates[0].state()
    output_state_array = np.zeros((len(gamestates), len(dummy_state)), dtype=np.int32)
    output_mask_array = np.zeros((len(gamestates), len(dummy_action_mask)), dtype=np.bool_)
    output_finished_array = np.zeros(len(gamestates), dtype=np.bool_)
    output_successed_array = np.zeros(len(gamestates), dtype=np.bool_)
    for i in nb.prange(len(gamestates)):
        output_finished_array[i] = gamestates[i].finished()
        output_successed_array[i] = gamestates[i].successed()
        if not gamestates[i].finished():
            gamestates[i].act(actions[i])
            output_state_array[i], output_mask_array[i] = gamestates[i].state()
            output_finished_array[i] = gamestates[i].finished()
            output_successed_array[i] = gamestates[i].successed()
    return output_state_array, output_mask_array, output_finished_array, output_successed_array

@nb.njit(parallel=True)
def parallel_step_ignore_finished_and_mask(gamestates, actions, ignore_mask):
    """
    !Note: Use np.typed.List or numpy array as input
    ignore true = ignore
    """
    dummy_state, dummy_action_mask = gamestates[0].state()
    output_state_array = np.zeros((len(gamestates), len(dummy_state)), dtype=np.int32)
    output_mask_array = np.zeros((len(gamestates), len(dummy_action_mask)), dtype=np.bool_)
    output_finished_array = np.zeros(len(gamestates), dtype=np.bool_)
    output_successed_array = np.zeros(len(gamestates), dtype=np.bool_)
    for i in nb.prange(len(gamestates)):
        output_finished_array[i] = gamestates[i].finished() or ignore_mask[i]
        output_successed_array[i] = gamestates[i].successed()
        if (not gamestates[i].finished()) and (not ignore_mask[i]):
            gamestates[i].act(actions[i])
            output_state_array[i], output_mask_array[i] = gamestates[i].state()
            output_finished_array[i] = gamestates[i].finished()
            output_successed_array[i] = gamestates[i].successed()
    return output_state_array, output_mask_array, output_finished_array, output_successed_array

@nb.njit(parallel=True)
def parallel_state(gamestates):
    """
    !Note: Use np.typed.List or numpy array as input
    """
    dummy_state, dummy_action_mask = gamestates[0].state()
    output_state_array = np.zeros((len(gamestates), len(dummy_state)), dtype=np.int32)
    output_mask_array = np.zeros((len(gamestates), len(dummy_action_mask)), dtype=np.bool_)
    output_finished_array = np.zeros(len(gamestates), dtype=np.bool_)
    output_successed_array = np.zeros(len(gamestates), dtype=np.bool_)
    for i in nb.prange(len(gamestates)):
        output_state_array[i], output_mask_array[i] = gamestates[i].state()
        output_finished_array[i] = gamestates[i].finished()
        output_successed_array[i] = gamestates[i].successed()
    return output_state_array, output_mask_array, output_finished_array, output_successed_array


def parallel_state_debug(gamestates):
    """
    !Note: Use np.typed.List or numpy array as input
    """
    dummy_state, dummy_action_mask = gamestates[0].state()
    output_state_array = np.zeros((len(gamestates), len(dummy_state)), dtype=np.int32)
    output_mask_array = np.zeros((len(gamestates), len(dummy_action_mask)), dtype=np.bool_)
    output_finished_array = np.zeros(len(gamestates), dtype=np.bool_)
    output_successed_array = np.zeros(len(gamestates), dtype=np.bool_)
    for i in range(len(gamestates)):
        output_state_array[i], output_mask_array[i] = gamestates[i].state()
        output_finished_array[i] = gamestates[i].finished()
        output_successed_array[i] = gamestates[i].successed()
    return output_state_array, output_mask_array, output_finished_array, output_successed_array


@nb.njit(parallel=True)
def parallel_reset(gamestates):
    for i in nb.prange(len(gamestates)):
        gamestates[i].reset()
        
@nb.njit(parallel=True)
def parallel_reset_failed(gamestates, chogibaek: int):
    for i in nb.prange(len(gamestates)):
        if not gamestates[i].successed():
            gamestates[i].chogibaek = chogibaek
            gamestates[i].reset()