"""
Конфігурація генетичного алгоритму для Hardware-Aware NAS (ga2).
Оптимізація архітектури CNN з урахуванням inference latency, model size та RAM.
"""

# Параметри популяції (оптимізовано для швидкості)
POPULATION_SIZE_FAST = 4  # 4 моделі для швидкості
NUM_GENERATIONS_FAST = 3  # 3 покоління для швидких тестів

POPULATION_SIZE_FULL = 12  # Збільшено для ширшого дослідження
NUM_GENERATIONS_FULL = 15  # Збільшено з 10 до 15

# За замовчуванням (будуть перевизначені в main.py)
POPULATION_SIZE = POPULATION_SIZE_FAST
NUM_GENERATIONS = NUM_GENERATIONS_FAST

# Параметри генетичних операторів
CROSSOVER_RATE = 0.7  # Зменшено для більшої різноманітності
MUTATION_RATE = 0.4   # Збільшено з 0.2 до 0.4 для більших змін
TOURNAMENT_SIZE = 2   # Зменшено з 3 до 2 (менший тиск селекції)

# Параметри тренування (оптимізовано для швидкості)
TRAINING_EPOCHS_FULL = 50  # Зменшено з 100 до 50 для швидкості
TRAINING_EPOCHS_FAST = 5   # 5 епох для кращої оцінки
BATCH_SIZE = 128  # Збільшено з 64 до 128 для стабільності
VALIDATION_SPLIT = 0.2

# Розмір вибірки для режимів
DATASET_SUBSET_FAST = 10000  # 10k зображень для fast mode (замість 50k)
DATASET_SUBSET_FULL = None   # Весь датасет для full mode

# Обмеження на структуру мережі (оптимізовано для GPU L4 + 128x128 зображень)
MIN_LAYERS = 3
MAX_LAYERS = 6  # Збільшено для глибших архітектур
MIN_NEURONS = 32
MAX_NEURONS = 512  # Збільшено для більшої ємності

# Параметри CNN (оптимізовано для GPU пам'яті та 64x64 зображень)
MIN_FILTERS = 16
MAX_FILTERS = 128  # Відновлено: дозволяємо 16/32/64/128 фільтрів
KERNEL_SIZES = [3, 5]  # Додано 5x5 для більшого receptive field
POOL_SIZES = [2]

# Можливі типи шарів
LAYER_TYPES = ['conv2d', 'depthwise_conv', 'maxpool', 'dense', 'dropout',
               'batch_norm', 'flatten', 'global_avg_pool']
ACTIVATION_FUNCTIONS = ['relu', 'elu']  # Прибрав selu - менш стабільний
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']  # Генетичний алгоритм вибирає найкращий

# Діапазони гіперпараметрів
LEARNING_RATE_RANGE = (0.0001, 0.001)  # Зменшено ще більше для стабільності
DROPOUT_RATE_RANGE = (0.2, 0.5)  # Збільшено для боротьби з overfitting

# LR scheduler (вибирається генетичним алгоритмом)
LR_SCHEDULERS = ['none', 'step', 'cosine']

# L2 regularization для Conv/Dense шарів
L2_REG_VALUES = [0.0, 0.0001, 0.001]

# Параметри датасету (оптимізовано для швидкості та GPU пам'яті)
DATASET_PATH = 'data'
IMAGE_SIZE = 64  # Зменшено з 128 до 64 для економії GPU пам'яті (4x швидше!)
NUM_CLASSES = 6  # Military Vehicles: 6 класів

# Режими роботи
MODE_FULL = 'full'
MODE_FAST = 'fast'

# Ламаркіанський теплий старт
WARM_START_ENABLED = True       # Вмикач для порівняльного експерименту
WARM_START_EPOCH_REDUCTION = 0.5  # При warm start: epochs = epochs * 0.5

# Hardware-aware benchmark (TFLite на сервері)
HARDWARE_BENCHMARK_ENABLED = False
BENCHMARK_WARMUP = 10
BENCHMARK_RUNS = 50
TFLITE_DELEGATE = 'cpu'  # 'cpu' | 'gpu' (optional later)

# Penalty values when benchmark fails
BENCHMARK_FAIL_LATENCY_MS = 9999.0
BENCHMARK_FAIL_SIZE_BYTES = 10**9
BENCHMARK_FAIL_RAM_MB = 9999.0

# NSGA-II objective modes
OBJECTIVE_MODE_STANDARD = 'standard'   # accuracy, num_params, training_time
OBJECTIVE_MODE_HARDWARE = 'hardware'   # accuracy, latency, size, ram
