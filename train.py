if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # select gpu
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#! DEBUG
# tf.config.run_functions_eagerly(True)

from gym import GameState, parallel_step_ignore_finished, parallel_state, parallel_reset
import models
import numba as nb
import numpy as np
from pathlib import Path
from tqdm import trange, tqdm
import psutil
import copy
import contextlib

MEMORY = 30

# If tensorflow > 2.15, create a new Layer
# class GatherLayer(tf.keras.layers.Layer):
#     def call(self, inputs):
#         return tf.gather(inputs[0], inputs[1], batch_dims=1)

def wrap_model(model:tf.keras.Model):
    inputs = model.inputs
    all_logits = model.outputs[0]
    action_inputs = tf.keras.Input(shape=(), dtype=tf.int32)
    
    # Comment this line if tensorflow > 2.15
    chosen_logits = tf.gather(all_logits, action_inputs, batch_dims=1)
    
    # Uncomment this line if tensorflow > 2.15
    # chosen_logits = GatherLayer()([all_logits, action_inputs])
    
    return tf.keras.Model(inputs=inputs+[action_inputs], outputs=chosen_logits)


class Trainer:
    def __init__(
        self,
        model_name,
        model_kwargs,
        lr,
        train_batch_size,
        infer_batch_size,
        gamestate_kwarg_list,
        epochs,
        train_epochs_per_epoch,
        random_action_ratio_start,
        random_action_ratio_end,
        random_action_ratio_decay_n,
        name,
        checkpoint_restore_path=None,
        tb_epoch_start=0,
        multi_gpu=False,
        teacher_path=None,
        teacher_model_name=None,
        teacher_model_kwargs=None,
        teacher_epochs=None,
        teacher_random_ratio=0,
        **kwargs
    ):
        self.strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.ReductionToOneDevice()
        )
        self.multi_gpu = multi_gpu

        cm = self.strategy.scope() if self.multi_gpu else contextlib.nullcontext()
        with cm:
            self.model = getattr(models, model_name)(**model_kwargs)
            self.wrapped_model = wrap_model(self.model)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.wrapped_model.compile(
                optimizer=self.optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            )
            if teacher_path is not None:
                self.teach = True
                self.teacher_path = teacher_path
                self.teacher_model = getattr(models, teacher_model_name)(**teacher_model_kwargs)
                self.teacher_epochs = teacher_epochs
                self.teacher_random_ratio = teacher_random_ratio
            else:
                self.teach = False
                self.teacher_model = None
                self.teacher_epochs = 0
                self.teacher_random_ratio = 0

                
        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.gamestate_kwarg_list = gamestate_kwarg_list
        self.gamestate_kwarg_full_list = []
        self.gamestate_name_list = []
        self.gamestate_name_full_list = []
        self.gamestate_kwarg_idx_full_list = []
        
        for k_i, kwarg_raw in enumerate(gamestate_kwarg_list):
            kwarg = copy.deepcopy(kwarg_raw)
            if len(kwarg["immutable_idx_list"]) > 0:
                kwarg["immutable_idx_list"] = np.array(
                    kwarg["immutable_idx_list"], dtype=np.int32
                )
            else:
                kwarg["immutable_idx_list"] = None
            if len(kwarg["chaos_idx_list"]) > 0:
                kwarg["chaos_idx_list"] = np.array(
                    kwarg["chaos_idx_list"], dtype=np.int32
                )
            else:
                kwarg["chaos_idx_list"] = None

            self.gamestate_name_list.append(kwarg.pop("name"))
            n = kwarg.pop("N")

            for _ in range(n):
                #! need to copy arrays or they are all shared
                kwarg['immutable_idx_list'] = np.copy(kwarg['immutable_idx_list'])
                kwarg['chaos_idx_list'] = np.copy(kwarg['chaos_idx_list'])
                self.gamestate_kwarg_full_list.append(kwarg)
                self.gamestate_name_full_list.append(self.gamestate_name_list[-1])
                self.gamestate_kwarg_idx_full_list.append(k_i)

        self.gamestate_list = nb.typed.List([GameState(**kwarg) for kwarg in self.gamestate_kwarg_full_list])
        
        self.epochs = epochs
        self.train_epochs_per_epoch = train_epochs_per_epoch
        self.random_action_ratio_start = random_action_ratio_start
        self.random_action_ratio_end = random_action_ratio_end
        self.random_action_ratio_decay_n = random_action_ratio_decay_n
        self.tb_epoch_start = tb_epoch_start

        self.name = name
        self._build_checkpoint_writer(checkpoint_restore_path)

        self.success_rate_memory = np.zeros(MEMORY, dtype=np.float32)

        # self.generator = Generator()

    def _get_random_action_ratio(self, epoch):
        # Linear decay
        return max(
            self.random_action_ratio_end,
            self.random_action_ratio_start
            - (self.random_action_ratio_start - self.random_action_ratio_end)
            * (epoch - self.teacher_epochs)
            / self.random_action_ratio_decay_n,
        )

    def _build_checkpoint_writer(self, checkpoint_restore_path=None):
        self.log_dir = Path("logs") / self.name
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_prefix = self.checkpoint_dir / "ckpt"
        cm = self.strategy.scope() if self.multi_gpu else contextlib.nullcontext()
        with cm:
            self.checkpoint = tf.train.Checkpoint(
                optimizer=self.optimizer, model=self.model
            )
            if checkpoint_restore_path is not None:
                self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_restore_path))
                print("Checkpoint restored from {}".format(checkpoint_restore_path))

            self.writer = tf.summary.create_file_writer(str(self.log_dir))
            self.writer.set_as_default()
            
            if self.teach:
                self.teacher_checkpoint = tf.train.Checkpoint(
                    model=self.teacher_model
                )
                self.teacher_checkpoint.restore(tf.train.latest_checkpoint(self.teacher_path)).expect_partial()
                print("Teacher checkpoint restored from {}".format(self.teacher_path))

    def reset_gamestate(self):
        parallel_reset(self.gamestate_list)

    def train_epoch(self, current_epoch):
        self.reset_gamestate()
        collect_tqdm = tqdm(ncols=100, desc="Collecting data", leave=False)
        (
            output_state_array,
            output_mask_array,
            output_finished_array,
            output_successed_array,
        ) = parallel_state(self.gamestate_list)

        while not np.all(output_finished_array):
            # Memory usage safety check
            if psutil.virtual_memory().percent > 90:
                raise Exception("Memory usage too high")
            
            collect_tqdm.update(1)
            
            action_logits = self.model.predict(
                output_state_array, batch_size=self.infer_batch_size, verbose=0
            )
            action_logits[np.logical_not(output_mask_array)] = -np.inf
            random_action_logits = np.random.rand(*action_logits.shape)
            random_action_logits[np.logical_not(output_mask_array)] = -np.inf
            actions = np.argmax(action_logits, axis=1)
            random_actions = np.argmax(random_action_logits, axis=1)
            random_mask = np.random.rand(
                *actions.shape
            ) < self._get_random_action_ratio(current_epoch)
            actions[random_mask] = random_actions[random_mask]
            (
                output_state_array,
                output_mask_array,
                output_finished_array,
                output_successed_array,
            ) = parallel_step_ignore_finished(self.gamestate_list, actions)
            collect_tqdm.set_postfix({
                "finished": f'{np.sum(output_finished_array)/len(output_finished_array)*100:.2f}%',
                "successed": f'{np.sum(output_successed_array)/len(output_successed_array)*100:.2f}%',
            })

        collect_tqdm.close()
        # train
        all_success_state_histories = []
        all_success_action_histories = []
        all_fail_state_histories = []
        all_fail_action_histories = []
        total_count = np.zeros(len(self.gamestate_name_list), dtype=int)
        success_count = np.zeros(len(self.gamestate_name_list), dtype=int)
        np.add.at(total_count, self.gamestate_kwarg_idx_full_list, 1)
        np.add.at(success_count, self.gamestate_kwarg_idx_full_list, output_successed_array)
        max_fail_N = []
        for k_i, kwargs in enumerate(self.gamestate_kwarg_list):
            min_s_ratio = kwargs.setdefault("min_s_ratio", None)
            if min_s_ratio is not None:
                max_fail_N.append(int(
                    (1-min_s_ratio) * success_count[k_i] / min_s_ratio
                ))
            else:
                max_fail_N.append(total_count[k_i] - success_count[k_i])

        for i in range(len(self.gamestate_list)):
            kwarg_idx = self.gamestate_kwarg_idx_full_list[i]
            state_history, action_history = self.gamestate_list[i].get_history()
            if output_successed_array[i]:
                all_success_state_histories.extend(state_history)
                all_success_action_histories.extend(action_history)
            elif max_fail_N[kwarg_idx] > 0:
                all_fail_state_histories.extend(state_history)
                all_fail_action_histories.extend(action_history)
                max_fail_N[kwarg_idx] -= 1

        state_histories = all_success_state_histories + all_fail_state_histories
        action_histories = all_success_action_histories + all_fail_action_histories
        labels = np.concatenate(
            [np.ones(len(all_success_state_histories)), np.zeros(len(all_fail_state_histories))]
        )
        indices = np.arange(len(state_histories))
        np.random.shuffle(indices)
        # drop leftover -> Weird error happens if exactly 1 is left
        indices = indices[:len(indices) - (len(indices) % self.train_batch_size)]
        state_histories = np.array(state_histories, dtype=np.int32)[indices]
        action_histories = np.array(action_histories, dtype=np.int32)[indices]
        labels = np.array(labels[indices], dtype=np.float32)
        # save 'before' training
        for i in range(len(self.gamestate_name_list)):
            success_rate = success_count[i] / total_count[i]
            tf.summary.scalar(self.gamestate_name_list[i], 
                              success_rate,
                              step=current_epoch + self.tb_epoch_start)

        current_total_success_rate = np.sum(success_count) / np.sum(total_count)
        best_of_last_30 = np.max(self.success_rate_memory)
        if current_total_success_rate >= best_of_last_30:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.success_rate_memory[current_epoch % MEMORY] = current_total_success_rate
        tf.summary.scalar("success_rate", current_total_success_rate, step=current_epoch + self.tb_epoch_start)
        tf.summary.scalar("success_rate_max_30", np.max(self.success_rate_memory), step=current_epoch + self.tb_epoch_start)

        self.wrapped_model.fit(
            x=[state_histories, action_histories],
            y=labels,
            batch_size=self.train_batch_size,
            epochs=self.train_epochs_per_epoch,
            verbose=1,
        )

    def teach_epoch(self, current_epoch):
        assert self.teach, "No teacher model"
        self.reset_gamestate()
        collect_tqdm = tqdm(ncols=100, desc="Collecting teacher data", leave=False)
        (
            output_state_array,
            output_mask_array,
            output_finished_array,
            output_successed_array,
        ) = parallel_state(self.gamestate_list)

        while not np.all(output_finished_array):
            # Memory usage safety check
            if psutil.virtual_memory().percent > 90:
                raise Exception("Memory usage too high")
            
            collect_tqdm.update(1)
            
            action_logits = self.teacher_model.predict(
                output_state_array, batch_size=self.infer_batch_size, verbose=0
            )
            action_logits[np.logical_not(output_mask_array)] = -np.inf
            random_action_logits = np.random.rand(*action_logits.shape)
            random_action_logits[np.logical_not(output_mask_array)] = -np.inf
            actions = np.argmax(action_logits, axis=1)
            random_actions = np.argmax(random_action_logits, axis=1)
            random_mask = np.random.rand(
                *actions.shape
            ) < self.teacher_random_ratio
            actions[random_mask] = random_actions[random_mask]
            (
                output_state_array,
                output_mask_array,
                output_finished_array,
                output_successed_array,
            ) = parallel_step_ignore_finished(self.gamestate_list, actions)
            collect_tqdm.set_postfix({
                "finished": f'{np.sum(output_finished_array)/len(output_finished_array)*100:.2f}%',
                "successed": f'{np.sum(output_successed_array)/len(output_successed_array)*100:.2f}%',
            })

        collect_tqdm.close()

        # train
        all_success_state_histories = []
        all_success_action_histories = []
        all_fail_state_histories = []
        all_fail_action_histories = []
        total_count = np.zeros(len(self.gamestate_name_list), dtype=int)
        success_count = np.zeros(len(self.gamestate_name_list), dtype=int)
        np.add.at(total_count, self.gamestate_kwarg_idx_full_list, 1)
        np.add.at(success_count, self.gamestate_kwarg_idx_full_list, output_successed_array)
        max_fail_N = []
        for k_i, kwargs in enumerate(self.gamestate_kwarg_list):
            min_s_ratio = kwargs.setdefault("min_s_ratio", None)
            if min_s_ratio is not None:
                max_fail_N.append(int(
                    (1-min_s_ratio) * success_count[k_i] / min_s_ratio
                ))
            else:
                max_fail_N.append(total_count[k_i] - success_count[k_i])

        for i in range(len(self.gamestate_list)):
            kwarg_idx = self.gamestate_kwarg_idx_full_list[i]
            state_history, action_history = self.gamestate_list[i].get_history()
            if output_successed_array[i]:
                all_success_state_histories.extend(state_history)
                all_success_action_histories.extend(action_history)
            elif max_fail_N[kwarg_idx] > 0:
                all_fail_state_histories.extend(state_history)
                all_fail_action_histories.extend(action_history)
                max_fail_N[kwarg_idx] -= 1

        state_histories = all_success_state_histories + all_fail_state_histories
        action_histories = all_success_action_histories + all_fail_action_histories
        labels = np.concatenate(
            [np.ones(len(all_success_state_histories)), np.zeros(len(all_fail_state_histories))]
        )
        indices = np.arange(len(state_histories))
        np.random.shuffle(indices)
        # drop leftover -> Weird error happens if exactly 1 is left
        indices = indices[:len(indices) - (len(indices) % self.train_batch_size)]
        state_histories = np.array(state_histories, dtype=np.int32)[indices]
        action_histories = np.array(action_histories, dtype=np.int32)[indices]
        labels = np.array(labels[indices], dtype=np.float32)

        self.wrapped_model.fit(
            x=[state_histories, action_histories],
            y=labels,
            batch_size=self.train_batch_size,
            epochs=self.train_epochs_per_epoch,
            verbose=1,
        )
    
    def eval_model(self, current_epoch):
        self.reset_gamestate()
        collect_tqdm = tqdm(ncols=100, desc="Evaluating", leave=False)
        (
            output_state_array,
            output_mask_array,
            output_finished_array,
            output_successed_array,
        ) = parallel_state(self.gamestate_list)

        while not np.all(output_finished_array):
            # Memory usage safety check
            if psutil.virtual_memory().percent > 90:
                raise Exception("Memory usage too high")
            
            collect_tqdm.update(1)
            
            action_logits = self.model.predict(
                output_state_array, batch_size=self.infer_batch_size, verbose=0
            )
            action_logits[np.logical_not(output_mask_array)] = -np.inf
            actions = np.argmax(action_logits, axis=1)
            (
                output_state_array,
                output_mask_array,
                output_finished_array,
                output_successed_array,
            ) = parallel_step_ignore_finished(self.gamestate_list, actions)
            collect_tqdm.set_postfix({
                "finished": f'{np.sum(output_finished_array)/len(output_finished_array)*100:.2f}%',
                "successed": f'{np.sum(output_successed_array)/len(output_successed_array)*100:.2f}%',
            })

        collect_tqdm.close()
        
        total_count = np.zeros(len(self.gamestate_name_list), dtype=int)
        success_count = np.zeros(len(self.gamestate_name_list), dtype=int)
        np.add.at(total_count, self.gamestate_kwarg_idx_full_list, 1)
        np.add.at(success_count, self.gamestate_kwarg_idx_full_list, output_successed_array)

        for i in range(len(self.gamestate_name_list)):
            success_rate = success_count[i] / total_count[i]
            tf.summary.scalar(self.gamestate_name_list[i], 
                              success_rate,
                              step=current_epoch + self.tb_epoch_start)
            
        current_total_success_rate = np.sum(success_count) / np.sum(total_count)
        best_of_last_30 = np.max(self.success_rate_memory)
        if current_total_success_rate >= best_of_last_30:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.success_rate_memory[current_epoch % MEMORY] = current_total_success_rate
        tf.summary.scalar("success_rate", current_total_success_rate, step=current_epoch + self.tb_epoch_start)
        tf.summary.scalar("success_rate_max_30", np.max(self.success_rate_memory), step=current_epoch + self.tb_epoch_start)



    def train(self):
        if self.teach:
            for e in trange(
                self.teacher_epochs, 
                ncols=100, 
                desc="Teacher Epoch", 
                leave=True
            ):
                self.teach_epoch(e)
                self.eval_model(e)
                self.writer.flush()
            
        for e in trange(
            self.teacher_epochs, 
            self.epochs, 
            ncols=100, 
            desc="Epoch", 
            leave=True
        ):
            # self.train_epoch(e)
            self.eval_model(e)
            self.writer.flush()


if __name__ == "__main__":
    import json

    with open("configs/config.json", "r") as f:
        config = json.load(f)
    trainer = Trainer(**config)
    with open(trainer.log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    trainer.train()
