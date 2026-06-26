"""
Фітнес-функція: тренування CNN моделі та оцінка точності
"""

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Приховати TensorFlow warnings

import numpy as np
import tensorflow as tf

# Налаштування для оптимального використання пам'яті (Metal на Mac)
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
except:
    pass

# Оптимізація для Apple Silicon
# XLA compilation вимкнено - конфліктує з Metal на Mac
# tf.config.optimizer.set_jit(True)

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import Tuple, Optional
import config
from chromosome import Chromosome
from dataset_loader import load_dataset as load_military_vehicles_dataset


def build_model(chromosome: Chromosome, input_shape: tuple) -> keras.Model:
    """
    Будує CNN Keras модель з хромосоми
    """
    model = models.Sequential()

    first_layer = True

    for i, layer in enumerate(chromosome.layers):
        l2 = keras.regularizers.l2(layer.l2_reg) if getattr(layer, 'l2_reg', 0.0) > 0 else None

        if layer.layer_type == 'conv2d':
            if first_layer:
                model.add(layers.Conv2D(
                    layer.filters,
                    (layer.kernel_size, layer.kernel_size),
                    activation=layer.activation,
                    padding='same',
                    kernel_regularizer=l2,
                    input_shape=input_shape
                ))
                first_layer = False
            else:
                model.add(layers.Conv2D(
                    layer.filters,
                    (layer.kernel_size, layer.kernel_size),
                    activation=layer.activation,
                    padding='same',
                    kernel_regularizer=l2,
                ))

        elif layer.layer_type == 'depthwise_conv':
            if first_layer:
                model.add(layers.SeparableConv2D(
                    layer.filters,
                    (layer.kernel_size, layer.kernel_size),
                    activation=layer.activation,
                    padding='same',
                    depthwise_regularizer=l2,
                    pointwise_regularizer=l2,
                    input_shape=input_shape
                ))
                first_layer = False
            else:
                model.add(layers.SeparableConv2D(
                    layer.filters,
                    (layer.kernel_size, layer.kernel_size),
                    activation=layer.activation,
                    padding='same',
                    depthwise_regularizer=l2,
                    pointwise_regularizer=l2,
                ))

        elif layer.layer_type == 'maxpool':
            model.add(layers.MaxPooling2D(
                pool_size=(layer.pool_size, layer.pool_size)
            ))

        elif layer.layer_type == 'flatten':
            model.add(layers.Flatten())

        elif layer.layer_type == 'global_avg_pool':
            model.add(layers.GlobalAveragePooling2D())

        elif layer.layer_type == 'dense':
            model.add(layers.Dense(
                layer.neurons,
                activation=layer.activation,
                kernel_regularizer=l2,
            ))

        elif layer.layer_type == 'dropout':
            model.add(layers.Dropout(layer.rate))

        elif layer.layer_type == 'batch_norm':
            model.add(layers.BatchNormalization())

    return model


def transfer_weights(parent_model: keras.Model, child_model: keras.Model) -> int:
    """
    Ламаркіанський теплий старт: переносить ваги сумісних шарів від батька до нащадка.
    Сумісний шар = однаковий тип Keras-шару + однакова форма тензорів ваг.
    
    Returns:
        Кількість шарів, для яких успішно передано ваги.
    """
    transferred = 0
    parent_layers = [l for l in parent_model.layers if l.get_weights()]
    child_layers  = [l for l in child_model.layers  if l.get_weights()]

    for p_layer, c_layer in zip(parent_layers, child_layers):
        if type(p_layer) is not type(c_layer):
            break  # Зупиняємось при першому розходженні типів
        p_weights = p_layer.get_weights()
        c_weights = c_layer.get_weights()
        if len(p_weights) != len(c_weights):
            break
        if all(pw.shape == cw.shape for pw, cw in zip(p_weights, c_weights)):
            c_layer.set_weights(p_weights)
            transferred += 1
        else:
            break  # Розмір не збігається — зупиняємось
    return transferred


def get_optimizer(chromosome: Chromosome):
    lr = chromosome.learning_rate
    
    if chromosome.optimizer == 'adam':
        return optimizers.Adam(learning_rate=lr)
    elif chromosome.optimizer == 'sgd':
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif chromosome.optimizer == 'rmsprop':
        return optimizers.RMSprop(learning_rate=lr)
    else:
        return optimizers.Adam(learning_rate=lr)


def evaluate_fitness(chromosome: Chromosome, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     epochs: int,
                     verbose: int = 1,
                     return_model: bool = False) -> Tuple[float, keras.Model]:
    """
    Оцінює фітнес хромосоми через тренування CNN моделі
    Повертає точність на валідаційному наборі та (опціонально) натреновану модель
    
    Args:
        return_model: Якщо True, повертає (fitness, model), інакше тільки fitness
    """
    
    try:
        # Будуємо модель
        input_shape = X_train.shape[1:]  # (64, 64, 3)
        model = build_model(chromosome, input_shape)
        
        # Warm start setup
        warm_layers = 0
        actual_epochs = epochs
        if (config.WARM_START_ENABLED and
                chromosome.parent_weights_file and
                os.path.exists(chromosome.parent_weights_file)):
            try:
                parent_model = tf.keras.models.load_model(chromosome.parent_weights_file)
                warm_layers = transfer_weights(parent_model, model)
                del parent_model
                if warm_layers > 0:
                    actual_epochs = max(1, int(epochs * config.WARM_START_EPOCH_REDUCTION))
                    print(f"   🔥 Warm start: {warm_layers} шарів успадковано, епох: {actual_epochs} (замість {epochs})")
            except Exception as warm_err:
                print(f"   ⚠️  Warm start не вдався: {warm_err}")
                warm_layers = 0
                actual_epochs = epochs
        chromosome.warm_layers_count = warm_layers

        # Компілюємо модель
        model.compile(
            optimizer=get_optimizer(chromosome),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Збираємо callbacks
        scheduler = getattr(chromosome, 'lr_scheduler', 'none')
        fit_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                mode='max',
                restore_best_weights=True,
                min_delta=0.001
            ),
        ]
        if scheduler == 'step':
            fit_callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5,
                    min_lr=1e-6, verbose=0
                )
            )
        elif scheduler == 'cosine':
            total_steps = actual_epochs * max(1, len(X_train) // chromosome.batch_size)
            cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=chromosome.learning_rate,
                decay_steps=total_steps,
                alpha=1e-6
            )
            # Перекомпілюємо з cosine schedule
            opt_name = chromosome.optimizer
            if opt_name == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate=cosine_schedule)
            elif opt_name == 'sgd':
                opt = tf.keras.optimizers.SGD(learning_rate=cosine_schedule, momentum=0.9)
            else:
                opt = tf.keras.optimizers.RMSprop(learning_rate=cosine_schedule)
            model.compile(optimizer=opt,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        # 'none' - не додаємо scheduler callback

        # Тренуємо модель
        train_start = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=chromosome.batch_size,
            epochs=actual_epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=fit_callbacks
        )
        
        # NOTE: Перевірка weights_validator видалена
        
        # Беремо НАЙКРАЩУ val_accuracy з усіх епох (не фінальну!)
        best_val_accuracy = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_accuracy) + 1
        total_epochs = len(history.history['val_accuracy'])

        # Зберігаємо метрики складності в хромосому
        chromosome.num_params = int(model.count_params())
        chromosome.training_time = round(time.time() - train_start, 2)

        if config.HARDWARE_BENCHMARK_ENABLED:
            from hardware_benchmark import benchmark_model, apply_metrics_to_chromosome
            metrics = benchmark_model(model, input_shape)
            apply_metrics_to_chromosome(chromosome, metrics)
            print(
                f"   HW: latency={chromosome.inference_latency_ms:.2f}ms  "
                f"size={chromosome.model_size_bytes:,}B  "
                f"ram={chromosome.peak_ram_mb:.2f}MB"
            )
        
        # Інформативне логування - показуємо на якій епосі була найкраща accuracy
        if best_epoch < total_epochs:
            # Найкраща не в кінці - є overfitting
            final_acc = history.history['val_accuracy'][-1]
            print(f"   📊 Best: epoch {best_epoch}/{total_epochs} ({best_val_accuracy:.4f}), final: {final_acc:.4f} ⚠️")
        else:
            # Найкраща в кінці - модель ще вчиться
            print(f"   📊 Best: epoch {best_epoch}/{total_epochs} ({best_val_accuracy:.4f}) ✓")
        
        val_accuracy = best_val_accuracy
        
        # Повертаємо результат
        if return_model:
            return val_accuracy, model
        else:
            # Очищуємо пам'ять якщо модель не потрібна
            del model
            tf.keras.backend.clear_session()
            return val_accuracy
    
    except Exception as e:
        print(f"Помилка при тренуванні моделі: {e}")
        # Повертаємо низький фітнес у випадку помилки
        if return_model:
            return 0.0, None
        else:
            return 0.0


def load_dataset(subset_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Завантажує Military Vehicles датасет
    
    Args:
        subset_size: Розмір вибірки (для fast mode)
    """
    return load_military_vehicles_dataset(subset_size=subset_size)


def get_dataset_info() -> Tuple[tuple, int]:
    """Повертає input_shape та output_dim для Military Vehicles"""
    input_shape = (64, 64, 3)  # Military Vehicles: 64x64 RGB
    output_dim = 6  # 6 класів
    return input_shape, output_dim
