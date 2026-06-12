"""
Validate Models - перевірка точності записаних моделей

Зчитує all_models.json та перевіряє валідаційну точність кожної моделі
для підтвердження збережених результатів.
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
from tensorflow import keras

from dataset_loader import load_dataset
from chromosome import Chromosome
from config import DATASET_SUBSET_FAST, DATASET_SUBSET_FULL


def load_models_history(history_file: str = "output/model_history/all_models.json"):
    """Завантажити історію моделей з JSON"""
    if not os.path.exists(history_file):
        print(f"❌ Файл не знайдено: {history_file}")
        return None
    
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Завантажено історію:")
    print(f"   Поколінь: {data['metadata']['total_generations']}")
    print(f"   Моделей: {data['metadata']['total_models']}")
    
    return data


def rebuild_model_from_history(model_data: dict, input_shape=(32, 32, 3), num_classes=10):
    """Відновити модель з JSON даних"""
    from chromosome import Layer
    import tensorflow as tf

    # Створити chromosome з даних
    chromosome = Chromosome()
    chromosome.optimizer = model_data['hyperparameters']['optimizer']
    chromosome.learning_rate = model_data['hyperparameters']['learning_rate']
    chromosome.batch_size = model_data['hyperparameters']['batch_size']
    chromosome.fitness = model_data['fitness']

    # Відновити layers
    chromosome.layers = []
    for layer_data in model_data['architecture']['layers']:
        layer = Layer(layer_data['type'])

        if 'neurons' in layer_data:
            layer.neurons = layer_data['neurons']
        if 'activation' in layer_data:
            layer.activation = layer_data['activation']
        # підтримка обох імен поля для dropout rate
        if 'rate' in layer_data:
            layer.rate = layer_data['rate']
        elif 'dropout_rate' in layer_data:
            layer.rate = layer_data['dropout_rate']
        if 'filters' in layer_data:
            layer.filters = layer_data['filters']
        if 'kernel_size' in layer_data:
            layer.kernel_size = layer_data['kernel_size']
        if 'pool_size' in layer_data:
            layer.pool_size = layer_data['pool_size']
        if 'l2_reg' in layer_data:
            layer.l2_reg = layer_data['l2_reg']

        chromosome.layers.append(layer)

    # Побудувати Keras модель
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    for layer in chromosome.layers:
        l2 = (keras.regularizers.l2(layer.l2_reg)
              if getattr(layer, 'l2_reg', 0.0) > 0 else None)

        if layer.layer_type == 'conv2d':
            model.add(keras.layers.Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding='same',
                kernel_regularizer=l2,
            ))
        elif layer.layer_type == 'depthwise_conv':
            model.add(keras.layers.SeparableConv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding='same',
                depthwise_regularizer=l2,
                pointwise_regularizer=l2,
            ))
        elif layer.layer_type == 'maxpool':
            model.add(keras.layers.MaxPooling2D(pool_size=layer.pool_size))
        elif layer.layer_type == 'batch_norm':
            model.add(keras.layers.BatchNormalization())
        elif layer.layer_type == 'flatten':
            model.add(keras.layers.Flatten())
        elif layer.layer_type == 'global_avg_pool':
            model.add(keras.layers.GlobalAveragePooling2D())
        elif layer.layer_type == 'dense':
            model.add(keras.layers.Dense(
                layer.neurons,
                activation=layer.activation,
                kernel_regularizer=l2,
            ))
        elif layer.layer_type == 'dropout':
            model.add(keras.layers.Dropout(layer.rate))

    # Compile
    optimizer_map = {
        'adam': keras.optimizers.Adam(learning_rate=chromosome.learning_rate),
        'sgd': keras.optimizers.SGD(learning_rate=chromosome.learning_rate),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=chromosome.learning_rate)
    }

    model.compile(
        optimizer=optimizer_map.get(chromosome.optimizer, 'adam'),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load full model if available (preferred method)
    if 'weights_file' in model_data and model_data['weights_file']:
        model_path = os.path.join("output/model_history/weights", model_data['weights_file'])
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path, compile=True)
                print(f"   ✅ Завантажено повну модель з: {model_path}")
            except Exception as e:
                print(f"   ⚠️  Не вдалося завантажити модель: {e}")
        else:
            print(f"   ⚠️  Файл моделі не знайдено: {model_path}")

    return model, chromosome


def validate_model(model, X_val, y_val, model_id: str, original_fitness: float, batch_size: int = 128):
    """Перевірити модель на валідаційних даних"""
    print(f"\n🔍 Валідація {model_id}...")
    print(f"   Оригінальна val_accuracy: {original_fitness:.4f}")
    
    try:
        # Evaluate
        loss, accuracy = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
        
        print(f"   Поточна val_accuracy:     {accuracy:.4f}")
        
        # Порівняння
        diff = accuracy - original_fitness
        if abs(diff) < 0.001:
            status = "✅ ІДЕНТИЧНО"
        elif abs(diff) < 0.01:
            status = "⚠️  БЛИЗЬКО"
        else:
            status = "❌ ВІДРІЗНЯЄТЬСЯ"
        
        print(f"   Різниця: {diff:+.4f} {status}")
        
        return {
            'model_id': model_id,
            'original_fitness': original_fitness,
            'validated_accuracy': accuracy,
            'difference': diff,
            'status': status
        }
        
    except Exception as e:
        print(f"   ❌ Помилка: {e}")
        return {
            'model_id': model_id,
            'original_fitness': original_fitness,
            'validated_accuracy': None,
            'difference': None,
            'status': '❌ ПОМИЛКА',
            'error': str(e)
        }


def validate_all_models(history_file: str = "output/model_history/all_models.json",
                       max_models: int = None,
                       generation: int = None):
    """Валідувати всі моделі з історії"""
    
    # Setup logging to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"output/validation_{timestamp}.log"
    
    class TeeOutput:
        """Дублює вивід в консоль та файл"""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Відкрити лог-файл
    log_handle = open(log_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_handle)
    
    try:
        print("\n" + "="*70)
        print("🔬 VALIDATE MODELS")
        print("="*70 + "\n")
        
        # Завантажити історію моделей
        data = load_models_history(history_file)
        if data is None:
            return
        
        # Визначити режим з першого покоління
        mode = 'full'  # За замовчуванням
        if data['generations'] and len(data['generations']) > 0:
            mode = data['generations'][0].get('mode', 'full')
        
        # Вибрати правильний subset_size
        subset_size = DATASET_SUBSET_FAST if mode == 'fast' else DATASET_SUBSET_FULL
        
        # Завантажити дані з правильним subset_size
        print(f"📊 Завантаження Military Vehicles (режим: {mode})...")
        X_train, y_train, X_val, y_val = load_dataset(subset_size=subset_size)
        print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"   (використовується той самий subset, що й під час тренування)\n")
        
        # Вибрати моделі для валідації
        models_to_validate = []
        
        if generation is not None:
            # Валідувати тільки конкретне покоління
            for gen_data in data['generations']:
                if gen_data['generation'] == generation:
                    models_to_validate.extend(gen_data['models'])
                    break
            print(f"📌 Вибрано покоління {generation}: {len(models_to_validate)} моделей\n")
        else:
            # Валідувати всі моделі
            for gen_data in data['generations']:
                models_to_validate.extend(gen_data['models'])
        
        # Обмежити кількість моделей якщо потрібно
        if max_models and len(models_to_validate) > max_models:
            models_to_validate = models_to_validate[:max_models]
            print(f"📌 Обмежено до {max_models} моделей\n")
        
        # Валідувати моделі
        results = []
        total = len(models_to_validate)
        
        print(f"🚀 Початок валідації {total} моделей...\n")
        print("-"*70)
        
        for i, model_data in enumerate(models_to_validate, 1):
            model_id = model_data['model_id']
            original_fitness = model_data.get('fitness')
            
            if original_fitness is None:
                print(f"\n⏭️  {i}/{total}: {model_id} - не тренована модель, пропускаємо")
                continue
            
            try:
                # Відновити модель з правильним input_shape з датасету
                input_shape = X_val.shape[1:]
                model, chromosome = rebuild_model_from_history(model_data, input_shape=input_shape)
                
                # Валідувати
                result = validate_model(
                    model, X_val, y_val, 
                    model_id, original_fitness,
                    batch_size=chromosome.batch_size
                )
                results.append(result)
                
                # Очистити пам'ять
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"\n❌ {i}/{total}: {model_id} - помилка побудови: {e}")
                results.append({
                    'model_id': model_id,
                    'original_fitness': original_fitness,
                    'validated_accuracy': None,
                    'difference': None,
                    'status': '❌ BUILD ERROR',
                    'error': str(e)
                })
        
        # Підсумок
        print("\n" + "="*70)
        print("📊 ПІДСУМОК ВАЛІДАЦІЇ")
        print("="*70 + "\n")
        
        successful = [r for r in results if r['validated_accuracy'] is not None]
        
        if successful:
            print(f"✅ Успішно валідовано: {len(successful)}/{len(results)}")
            
            differences = [abs(r['difference']) for r in successful]
            avg_diff = np.mean(differences)
            max_diff = np.max(differences)
            
            print(f"\n📈 Статистика різниць:")
            print(f"   Середня різниця: {avg_diff:.4f}")
            print(f"   Максимальна різниця: {max_diff:.4f}")
            
            # Топ відмінності
            print(f"\n🔝 Топ-5 моделей з найбільшими відмінностями:")
            sorted_results = sorted(successful, key=lambda x: abs(x['difference']), reverse=True)
            for i, r in enumerate(sorted_results[:5], 1):
                print(f"   {i}. {r['model_id']}: {r['difference']:+.4f} "
                      f"(orig: {r['original_fitness']:.4f}, val: {r['validated_accuracy']:.4f})")
        
        failed = len(results) - len(successful)
        if failed > 0:
            print(f"\n❌ Помилок: {failed}")
        
        # Зберегти результати
        output_file = f"output/validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'validation_date': datetime.now().isoformat(),
                'total_models': len(results),
                'successful': len(successful),
                'failed': failed,
                'results': results
            }, f, indent=2, ensure_ascii=False)
    
        print(f"\n💾 Результати збережено: {output_file}")
        print(f"📄 Лог збережено: {log_file}")
        print("\n" + "="*70)
    
    finally:
        # Відновити stdout та закрити лог-файл
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description='Validate Models')
    parser.add_argument('--history', type=str, default='output/model_history/all_models.json',
                       help='Шлях до all_models.json')
    parser.add_argument('--max-models', type=int, default=None,
                       help='Максимальна кількість моделей для валідації')
    parser.add_argument('--generation', type=int, default=None,
                       help='Валідувати тільки конкретне покоління')
    
    args = parser.parse_args()
    
    validate_all_models(
        history_file=args.history,
        max_models=args.max_models,
        generation=args.generation
    )


if __name__ == '__main__':
    main()
