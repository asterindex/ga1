"""
Хромосома для генетичного алгоритму NAS
Кодує структуру мережі + параметри тренування
"""

import random
import copy
from typing import List, Dict, Any
import config


class Layer:
    """Представлення одного шару мережі"""
    
    def __init__(self, layer_type: str, neurons: int = None,
                 activation: str = None, rate: float = None,
                 filters: int = None, kernel_size: int = None,
                 pool_size: int = None, l2_reg: float = 0.0):
        self.layer_type = layer_type
        # Dense параметри
        self.neurons = neurons
        self.activation = activation
        # Dropout параметри
        self.rate = rate
        # Conv2D / depthwise_conv параметри
        self.filters = filters
        self.kernel_size = kernel_size
        # MaxPool параметри
        self.pool_size = pool_size
        # L2 regularization (для conv2d, depthwise_conv, dense)
        self.l2_reg: float = l2_reg
    
    def to_dict(self) -> Dict:
        return {
            'type': self.layer_type,
            'neurons': self.neurons,
            'activation': self.activation,
            'rate': self.rate,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'l2_reg': self.l2_reg,
        }
    
    def __repr__(self):
        if self.layer_type == 'conv2d':
            return f"Conv2D({self.filters}, {self.kernel_size}x{self.kernel_size}, {self.activation})"
        elif self.layer_type == 'depthwise_conv':
            return f"DepthwiseConv({self.filters}, {self.kernel_size}x{self.kernel_size}, {self.activation})"
        elif self.layer_type == 'maxpool':
            return f"MaxPool({self.pool_size}x{self.pool_size})"
        elif self.layer_type == 'dense':
            return f"Dense({self.neurons}, {self.activation})"
        elif self.layer_type == 'dropout':
            return f"Dropout({self.rate:.2f})"
        elif self.layer_type == 'batch_norm':
            return "BatchNorm()"
        elif self.layer_type == 'flatten':
            return "Flatten()"
        elif self.layer_type == 'global_avg_pool':
            return "GlobalAvgPool()"
        return self.layer_type


class Chromosome:
    """Хромосома: структура мережі + гіперпараметри"""
    
    def __init__(self):
        # Структурна частина
        self.layers: List[Layer] = []
        
        # Параметрична частина
        self.learning_rate: float = 0.001
        self.batch_size: int = 32
        self.optimizer: str = 'adam'
        
        # Фітнес
        self.fitness: float = 0.0
        self.trained: bool = False

        # Метрики складності (для Pareto front)
        self.num_params: int = 0        # Кількість параметрів моделі
        self.training_time: float = 0.0  # Час тренування в секундах

        # LR scheduler (еволюціонується як гіперпараметр)
        self.lr_scheduler: str = 'none'  # 'none' | 'step' | 'cosine'

        # Ламаркіанський теплий старт
        self.parent_weights_file: str | None = None  # Шлях до .h5 батьківської моделі
        self.warm_layers_count: int = 0              # Кількість успадкованих шарів
    
    def validate_architecture(self) -> bool:
        """
        Перевірити чи архітектура валідна для Keras

        Правила:
        1. BatchNorm тільки після Conv2D/depthwise_conv або Dense
        2. Після Flatten/GlobalAvgPool тільки Dense/Dropout
        3. Не більше 2 однакових utility-шарів підряд (BatchNorm, MaxPool)
        4. Не більше 3 MaxPool для 32x32 зображень
        """
        if not self.layers:
            return False

        has_flatten = False
        consecutive_same = 0
        prev_type = None
        maxpool_count = 0

        for i, layer in enumerate(self.layers):
            layer_type = layer.layer_type

            # Правило 1: BatchNorm тільки після Conv2D/depthwise_conv або Dense
            if layer_type == 'batch_norm':
                if i == 0:
                    print(f"❌ Валідація: BatchNorm не може бути першим шаром")
                    return False
                prev_layer = self.layers[i-1]
                if prev_layer.layer_type not in ['conv2d', 'depthwise_conv', 'dense']:
                    print(f"❌ Валідація: BatchNorm після {prev_layer.layer_type} (має бути після conv2d/depthwise_conv/dense)")
                    return False

            # Правило 2: Після Flatten/GlobalAvgPool тільки Dense/Dropout
            if has_flatten:
                if layer_type in ['conv2d', 'depthwise_conv', 'maxpool', 'batch_norm',
                                  'flatten', 'global_avg_pool']:
                    print(f"❌ Валідація: {layer_type} після Flatten/GlobalAvgPool (заборонено)")
                    return False

            if layer_type in ('flatten', 'global_avg_pool'):
                has_flatten = True

            # Правило 3: Не більше 2 однакових utility-шарів підряд
            if layer_type in ['batch_norm', 'maxpool', 'dropout']:
                if layer_type == prev_type:
                    consecutive_same += 1
                    if consecutive_same >= 2:
                        return False
                else:
                    consecutive_same = 0
            else:
                consecutive_same = 0

            # Правило 4: Лічильник MaxPool
            if layer_type == 'maxpool':
                maxpool_count += 1
                if maxpool_count > 3:
                    return False

            prev_type = layer_type

        return True
    
    @staticmethod
    def random(input_shape: tuple, output_dim: int) -> 'Chromosome':
        """Створює випадкову CNN хромосому"""
        chromo = Chromosome()

        # Генеруємо випадкову CNN структуру
        num_layers = random.randint(config.MIN_LAYERS, config.MAX_LAYERS)

        has_flatten = False
        has_conv = False

        for i in range(num_layers):
            # ПЕРШИЙ ШАР ЗАВЖДИ Conv2D!
            if i == 0:
                layer_type = 'conv2d'
            # Спочатку conv шари, потім dense
            elif not has_flatten:
                # Не дозволяємо MaxPool/Dropout якщо ще немає Conv2D
                if not has_conv:
                    layer_type = random.choice(['conv2d', 'batch_norm', 'flatten'])
                else:
                    layer_type = random.choice([
                        'conv2d', 'depthwise_conv', 'maxpool', 'batch_norm',
                        'flatten', 'global_avg_pool'
                    ])
            else:
                # Після flatten тільки dense/dropout (НЕ batch_norm!)
                layer_type = random.choice(['dense', 'dropout'])

            if layer_type in ('conv2d', 'depthwise_conv'):
                filters = random.choice([32, 64, 128])
                kernel_size = random.choice(config.KERNEL_SIZES)
                activation = random.choice(config.ACTIVATION_FUNCTIONS)
                l2_reg = random.choice(config.L2_REG_VALUES)
                chromo.layers.append(Layer(layer_type, filters=filters,
                                           kernel_size=kernel_size,
                                           activation=activation,
                                           l2_reg=l2_reg))
                # ЗАВЖДИ додаємо BatchNorm після Conv шару
                chromo.layers.append(Layer('batch_norm'))
                has_conv = True

            elif layer_type == 'maxpool':
                pool_size = random.choice(config.POOL_SIZES)
                chromo.layers.append(Layer('maxpool', pool_size=pool_size))

            elif layer_type == 'flatten':
                chromo.layers.append(Layer('flatten'))
                has_flatten = True

            elif layer_type == 'global_avg_pool':
                chromo.layers.append(Layer('global_avg_pool'))
                has_flatten = True

            elif layer_type == 'dense':
                neurons = random.choice([32, 64, 128, 256])
                activation = random.choice(config.ACTIVATION_FUNCTIONS)
                l2_reg = random.choice(config.L2_REG_VALUES)
                chromo.layers.append(Layer('dense', neurons=neurons,
                                           activation=activation, l2_reg=l2_reg))
                # Додаємо Dropout після Dense для боротьби з overfitting
                dropout_rate = random.uniform(*config.DROPOUT_RATE_RANGE)
                chromo.layers.append(Layer('dropout', rate=dropout_rate))

            elif layer_type == 'dropout':
                rate = random.uniform(*config.DROPOUT_RATE_RANGE)
                chromo.layers.append(Layer('dropout', rate=rate))

            elif layer_type == 'batch_norm':
                chromo.layers.append(Layer('batch_norm'))

        # Переконуємось що є flatten/global_avg_pool перед dense шарами
        if not has_flatten:
            chromo.layers.append(Layer('flatten'))

        # Додаємо вихідний шар
        chromo.layers.append(Layer('dense', neurons=output_dim, activation='softmax'))

        # Генеруємо гіперпараметри
        chromo.learning_rate = random.uniform(*config.LEARNING_RATE_RANGE)
        chromo.batch_size = random.choice([16, 32])
        chromo.optimizer = random.choice(config.OPTIMIZERS)
        chromo.lr_scheduler = random.choice(config.LR_SCHEDULERS)

        # Перевіряємо валідність архітектури
        if not chromo.validate_architecture():
            return Chromosome.random(input_shape, output_dim)

        return chromo
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертує хромосому в словник"""
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'fitness': self.fitness,
            'num_params': self.num_params,
            'training_time': self.training_time,
            'warm_layers_count': self.warm_layers_count
        }
    
    def copy(self) -> 'Chromosome':
        """Створює глибоку копію хромосоми"""
        return copy.deepcopy(self)
    
    def validate_and_fix(self) -> None:
        """Валідує і виправляє архітектуру CNN"""
        if len(self.layers) < 2:
            return

        # Видаляємо вихідний шар
        output_layer = self.layers[-1]
        layers = self.layers[:-1]

        # Знаходимо позицію flatten/global_avg_pool
        flatten_idx = None
        for i, layer in enumerate(layers):
            if layer.layer_type in ('flatten', 'global_avg_pool'):
                flatten_idx = i
                break

        fixed_layers = []

        conv_types = {'conv2d', 'depthwise_conv'}

        if flatten_idx is not None:
            # До flatten - тільки CNN шари
            for i, layer in enumerate(layers[:flatten_idx + 1]):
                if layer.layer_type in conv_types | {'maxpool', 'dropout',
                                                      'batch_norm', 'flatten',
                                                      'global_avg_pool'}:
                    if layer.layer_type == 'batch_norm':
                        if i > 0 and fixed_layers and fixed_layers[-1].layer_type in conv_types:
                            fixed_layers.append(layer)
                    else:
                        fixed_layers.append(layer)

            # Після flatten - тільки Dense/Dropout
            for layer in layers[flatten_idx + 1:]:
                if layer.layer_type in ['dense', 'dropout']:
                    fixed_layers.append(layer)
        else:
            # Немає flatten - додаємо його
            for i, layer in enumerate(layers):
                if layer.layer_type in conv_types | {'maxpool', 'dropout', 'batch_norm'}:
                    if layer.layer_type == 'batch_norm':
                        if fixed_layers and fixed_layers[-1].layer_type in conv_types:
                            fixed_layers.append(layer)
                    else:
                        fixed_layers.append(layer)

            fixed_layers.append(Layer('flatten'))

            for layer in layers:
                if layer.layer_type in ['dense', 'dropout']:
                    fixed_layers.append(layer)

        # Переконуємось що є хоч один Conv2D
        has_conv = any(l.layer_type in conv_types for l in fixed_layers)
        if not has_conv:
            fixed_layers.insert(0, Layer('conv2d', filters=32, kernel_size=3, activation='relu'))

        # Повертаємо вихідний шар
        fixed_layers.append(output_layer)
        self.layers = fixed_layers
    
    def __repr__(self):
        layers_str = ' -> '.join([str(layer) for layer in self.layers])
        return f"Chromosome(fitness={self.fitness:.4f}, lr={self.learning_rate:.4f}, " \
               f"opt={self.optimizer}, layers=[{layers_str}])"
