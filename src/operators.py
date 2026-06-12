"""
Генетичні оператори: селекція, кросовер, мутація
"""

import random
import copy
from typing import List, Tuple
import config
from chromosome import Chromosome, Layer


def tournament_selection(population: List[Chromosome], 
                        tournament_size: int = config.TOURNAMENT_SIZE) -> Chromosome:
    """Турнірна селекція"""
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda x: x.fitness)
    return winner.copy()


def roulette_wheel_selection(population: List[Chromosome]) -> Chromosome:
    """Рулеточна селекція"""
    total_fitness = sum(ind.fitness for ind in population)
    
    if total_fitness == 0:
        return random.choice(population).copy()
    
    pick = random.uniform(0, total_fitness)
    current = 0
    
    for individual in population:
        current += individual.fitness
        if current > pick:
            return individual.copy()
    
    return population[-1].copy()


def _count_compatible_layers(child: Chromosome, parent: Chromosome) -> int:
    """
    Підраховує кількість шарів, сумісних для передачі ваг.
    Сумісний шар = однаковий тип на тій самій позиції.
    """
    count = 0
    for c_layer, p_layer in zip(child.layers, parent.layers):
        if c_layer.layer_type == p_layer.layer_type:
            count += 1
        else:
            break  # Зупиняємось при першому розходженні (prefix-match)
    return count


def _pick_closer_parent(child: Chromosome, parent1: Chromosome, parent2: Chromosome) -> 'str | None':
    """
    Повертає weights_file батька, чия архітектура ближча до нащадка.
    Якщо обидва батьки не мають збережених ваг — повертає None.
    """
    score1 = _count_compatible_layers(child, parent1) if parent1.parent_weights_file else -1
    score2 = _count_compatible_layers(child, parent2) if parent2.parent_weights_file else -1

    if score1 <= 0 and score2 <= 0:
        return None
    if score1 >= score2:
        return parent1.parent_weights_file
    return parent2.parent_weights_file



def crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """
    Кросовер двох батьків
    - Структура: одноточковий кросовер для шарів
    - Параметри: усереднення або випадковий вибір
    """
    child1 = Chromosome()
    child2 = Chromosome()
    
    # Кросовер структури (шарів)
    if len(parent1.layers) > 1 and len(parent2.layers) > 1:
        # Виключаємо останній шар (вихідний)
        p1_layers = parent1.layers[:-1]
        p2_layers = parent2.layers[:-1]
        
        # Одноточковий кросовер
        if len(p1_layers) > 0 and len(p2_layers) > 0:
            point1 = random.randint(0, len(p1_layers))
            point2 = random.randint(0, len(p2_layers))
            
            child1.layers = copy.deepcopy(p1_layers[:point1] + p2_layers[point2:])
            child2.layers = copy.deepcopy(p2_layers[:point2] + p1_layers[point1:])
        else:
            child1.layers = copy.deepcopy(p1_layers)
            child2.layers = copy.deepcopy(p2_layers)
    else:
        child1.layers = copy.deepcopy(parent1.layers[:-1])
        child2.layers = copy.deepcopy(parent2.layers[:-1])
    
    # Додаємо вихідний шар назад
    child1.layers.append(copy.deepcopy(parent1.layers[-1]))
    child2.layers.append(copy.deepcopy(parent2.layers[-1]))
    
    # Кросовер параметрів (усереднення)
    child1.learning_rate = (parent1.learning_rate + parent2.learning_rate) / 2
    child2.learning_rate = (parent1.learning_rate + parent2.learning_rate) / 2

    child1.batch_size = random.choice([parent1.batch_size, parent2.batch_size])
    child2.batch_size = random.choice([parent1.batch_size, parent2.batch_size])

    child1.optimizer = random.choice([parent1.optimizer, parent2.optimizer])
    child2.optimizer = random.choice([parent1.optimizer, parent2.optimizer])

    child1.lr_scheduler = random.choice([parent1.lr_scheduler, parent2.lr_scheduler])
    child2.lr_scheduler = random.choice([parent1.lr_scheduler, parent2.lr_scheduler])

    # Валідуємо і виправляємо архітектуру після кросоверу
    child1.validate_and_fix()
    child2.validate_and_fix()

    # Якщо після виправлення все ще невалідна - замінюємо на копію кращого батька
    better_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
    if not child1.validate_architecture():
        child1 = better_parent.copy()
        child1.fitness = 0.0
        child1.trained = False
    if not child2.validate_architecture():
        child2 = better_parent.copy()
        child2.fitness = 0.0
        child2.trained = False

    # Ламаркіанський теплий старт: вибираємо батька з найближчою архітектурою
    child1.parent_weights_file = _pick_closer_parent(child1, parent1, parent2)
    child2.parent_weights_file = _pick_closer_parent(child2, parent2, parent1)

    return child1, child2


def mutate_structure(chromosome: Chromosome) -> None:
    """Мутація структури CNN мережі"""
    mutation_type = random.choice(['add', 'remove', 'modify'])

    # Виключаємо вихідний шар з мутацій
    layers_without_output = chromosome.layers[:-1]
    output_layer = chromosome.layers[-1]

    if mutation_type == 'add' and len(layers_without_output) < config.MAX_LAYERS:
        # Додаємо новий шар
        position = random.randint(0, len(layers_without_output))

        # Визначаємо чи після flatten/global_avg_pool
        has_flatten = any(l.layer_type in ('flatten', 'global_avg_pool')
                          for l in layers_without_output[:position])

        if has_flatten:
            layer_type = random.choice(['dense', 'dropout', 'batch_norm'])
        else:
            layer_type = random.choice([
                'conv2d', 'depthwise_conv', 'maxpool', 'dropout',
                'batch_norm', 'flatten', 'global_avg_pool'
            ])

        l2_reg = random.choice(config.L2_REG_VALUES)

        if layer_type in ('conv2d', 'depthwise_conv'):
            filters = 2 ** int(random.uniform(4, 7))
            kernel_size = random.choice(config.KERNEL_SIZES)
            activation = random.choice(config.ACTIVATION_FUNCTIONS)
            new_layer = Layer(layer_type, filters=filters, kernel_size=kernel_size,
                              activation=activation, l2_reg=l2_reg)
        elif layer_type == 'maxpool':
            pool_size = random.choice(config.POOL_SIZES)
            new_layer = Layer('maxpool', pool_size=pool_size)
        elif layer_type in ('flatten', 'global_avg_pool'):
            new_layer = Layer(layer_type)
        elif layer_type == 'dense':
            neurons = 2 ** int(random.uniform(4, 9))
            activation = random.choice(config.ACTIVATION_FUNCTIONS)
            new_layer = Layer('dense', neurons=neurons, activation=activation, l2_reg=l2_reg)
        elif layer_type == 'dropout':
            rate = random.uniform(*config.DROPOUT_RATE_RANGE)
            new_layer = Layer('dropout', rate=rate)
        else:  # batch_norm
            new_layer = Layer('batch_norm')

        layers_without_output.insert(position, new_layer)

    elif mutation_type == 'remove' and len(layers_without_output) > config.MIN_LAYERS:
        # Видаляємо випадковий шар (крім flatten/global_avg_pool)
        removable = [i for i, l in enumerate(layers_without_output)
                     if l.layer_type not in ('flatten', 'global_avg_pool')]
        if removable:
            position = random.choice(removable)
            layers_without_output.pop(position)

    elif mutation_type == 'modify' and len(layers_without_output) > 0:
        # Змінюємо параметри випадкового шару
        position = random.randint(0, len(layers_without_output) - 1)
        layer = layers_without_output[position]

        if layer.layer_type in ('conv2d', 'depthwise_conv'):
            choice = random.choice(['filters', 'kernel', 'l2'])
            if choice == 'filters':
                layer.filters = 2 ** int(random.uniform(4, 7))
            elif choice == 'kernel':
                layer.kernel_size = random.choice(config.KERNEL_SIZES)
            else:
                layer.l2_reg = random.choice(config.L2_REG_VALUES)

        elif layer.layer_type == 'dense':
            choice = random.choice(['neurons', 'activation', 'l2'])
            if choice == 'neurons':
                layer.neurons = 2 ** int(random.uniform(4, 9))
            elif choice == 'activation':
                layer.activation = random.choice(config.ACTIVATION_FUNCTIONS)
            else:
                layer.l2_reg = random.choice(config.L2_REG_VALUES)

        elif layer.layer_type == 'dropout':
            layer.rate = random.uniform(*config.DROPOUT_RATE_RANGE)

        elif layer.layer_type == 'maxpool':
            layer.pool_size = random.choice(config.POOL_SIZES)

    # Збираємо мережу назад
    chromosome.layers = layers_without_output + [output_layer]


def mutate_parameters(chromosome: Chromosome) -> None:
    """Мутація гіперпараметрів"""
    mutation_choice = random.choice(['lr', 'batch', 'optimizer', 'scheduler'])

    if mutation_choice == 'lr':
        if random.random() < 0.5:
            chromosome.learning_rate *= random.uniform(0.5, 2.0)
            chromosome.learning_rate = max(config.LEARNING_RATE_RANGE[0],
                                           min(config.LEARNING_RATE_RANGE[1],
                                               chromosome.learning_rate))
        else:
            chromosome.learning_rate = random.uniform(*config.LEARNING_RATE_RANGE)

    elif mutation_choice == 'batch':
        chromosome.batch_size = random.choice([16, 32])

    elif mutation_choice == 'optimizer':
        chromosome.optimizer = random.choice(config.OPTIMIZERS)

    elif mutation_choice == 'scheduler':
        chromosome.lr_scheduler = random.choice(config.LR_SCHEDULERS)


def mutate(chromosome: Chromosome, max_attempts: int = 20) -> Chromosome:
    """
    Загальна мутація: структура або параметри.
    Повторює спроби до max_attempts разів поки архітектура не стане валідною.
    """
    do_structure = random.random() < 0.5

    for _ in range(max_attempts):
        mutated = copy.deepcopy(chromosome)

        if do_structure:
            mutate_structure(mutated)
            if not mutated.validate_architecture():
                continue  # Спробуємо ще раз
            mutated.parent_weights_file = None
        else:
            mutate_parameters(mutated)

        mutated.fitness = 0.0
        mutated.trained = False
        return mutated

    # Якщо за max_attempts спроб не вдалось - повертаємо незмінений оригінал
    return chromosome
