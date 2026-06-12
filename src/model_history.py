"""
System for saving complete history of all models during evolution
"""

import json
import os
from typing import List, Dict, Any, Optional
from chromosome import Chromosome
from datetime import datetime


class ModelHistoryTracker:
    """Tracks and saves all models evaluated during evolution"""

    def __init__(self, history_dir: str = "model_history",
                 method: str = 'baseline', dataset: str = 'data', seed: int = None):
        self.history_dir = history_dir
        self.history_file = os.path.join(history_dir, "all_models.json")
        self.weights_dir = os.path.join(history_dir, "weights")
        self.method = method
        self.dataset = dataset
        self.seed = seed

        # Create directories for history and weights
        os.makedirs(history_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        # Load existing history if file exists (for --resume)
        self.models_history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    self.models_history = existing_data.get('generations', [])
                    print(f"✅ Завантажено існуючу історію: {len(self.models_history)} поколінь")
            except Exception as e:
                print(f"⚠️ Помилка завантаження історії: {e}, починаємо з нуля")
                self.models_history = []
    
    def add_generation(self, generation: int, individuals: List[Chromosome], mode: str, 
                      trained_models: Optional[Dict[int, Any]] = None):
        """
        Add all models from a generation to history
        
        Args:
            generation: Generation number
            individuals: List of all chromosomes in the generation
            mode: Mode (full/fast)
            trained_models: Optional dict mapping index to trained Keras model
        """
        generation_data = {
            'generation': generation,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'models': []
        }
        
        for idx, individual in enumerate(individuals):
            model_id = f"gen{generation}_model{idx}"
            weights_file = None
            
            # Save FULL model (not just weights) if provided and trained
            if trained_models and idx in trained_models and individual.trained:
                keras_model = trained_models[idx]
                if keras_model is not None:
                    model_file = f"{model_id}.h5"  # H5 формат (сумісний з legacy optimizers)
                    model_path = os.path.join(self.weights_dir, model_file)
                    try:
                        keras_model.save(model_path, save_format='h5')  # Зберігаємо ВСЮ модель у H5
                        weights_file = model_file
                        print(f"   💾 Збережено модель: {model_file}")
                    except Exception as e:
                        print(f"   ⚠️  Не вдалося зберегти модель {model_id}: {e}")
                        weights_file = None
            
            # Elite model або копія: використовуємо ваги з попереднього покоління
            elif individual.trained and generation > 0:
                # Шукаємо модель з ІДЕНТИЧНОЮ архітектурою і fitness
                current_arch = [self._layer_to_dict(layer) for layer in individual.layers]
                
                for prev_gen in reversed(self.models_history):
                    for prev_model in prev_gen['models']:
                        # Перевіряємо fitness І архітектуру
                        if (abs(prev_model['fitness'] - individual.fitness) < 1e-6 and 
                            prev_model.get('weights_file') and
                            prev_model['architecture']['layers'] == current_arch):
                            weights_file = prev_model['weights_file']
                            if idx == 0:
                                print(f"   🔗 Elite використовує ваги: {weights_file}")
                            else:
                                print(f"   🔗 {model_id} використовує ваги: {weights_file} (копія)")
                            break
                    if weights_file:
                        break
            
            model_data = {
                'model_id': model_id,
                'generation': generation,
                'index_in_generation': idx,
                'fitness': individual.fitness,
                'validation_accuracy': individual.fitness,
                'trained': individual.trained,
                'weights_file': weights_file,
                'warm_layers_count': individual.warm_layers_count,
                
                # Architecture
                'architecture': {
                    'num_layers': len(individual.layers),
                    'layers': [self._layer_to_dict(layer) for layer in individual.layers]
                },
                
                # Hyperparameters
                'hyperparameters': {
                    'optimizer': individual.optimizer,
                    'learning_rate': individual.learning_rate,
                    'batch_size': individual.batch_size,
                    'lr_scheduler': getattr(individual, 'lr_scheduler', 'none'),
                }
            }
            
            generation_data['models'].append(model_data)
        
        self.models_history.append(generation_data)
        
        # Clean up trained models from memory after saving weights
        if trained_models:
            import tensorflow as tf
            for model in trained_models.values():
                if model is not None:
                    del model
            tf.keras.backend.clear_session()
        
        # Повертаємо словник {idx: full_path_to_h5} для оновлення parent_weights_file
        saved_paths = {}
        for model_data in generation_data['models']:
            if model_data['weights_file']:
                saved_paths[model_data['index_in_generation']] = os.path.join(
                    self.weights_dir, model_data['weights_file']
                )
        return saved_paths
    
    def save_history(self):
        """Save complete history to JSON file"""
        history_data = {
            'metadata': {
                'total_generations': len(self.models_history),
                'total_models': sum(len(gen['models']) for gen in self.models_history),
                'created_at': datetime.now().isoformat(),
                'method': self.method,
                'dataset': self.dataset,
                'seed': self.seed,
            },
            'generations': self.models_history
        }
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        return self.history_file
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about saved models"""
        if not self.models_history:
            return {}
        
        all_fitnesses = []
        for gen in self.models_history:
            for model in gen['models']:
                if model['fitness'] is not None:
                    all_fitnesses.append(model['fitness'])
        
        return {
            'total_generations': len(self.models_history),
            'total_models': sum(len(gen['models']) for gen in self.models_history),
            'best_fitness': max(all_fitnesses) if all_fitnesses else None,
            'worst_fitness': min(all_fitnesses) if all_fitnesses else None,
            'avg_fitness': sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else None
        }
    
    def _layer_to_dict(self, layer) -> Dict[str, Any]:
        """Convert layer to dictionary"""
        layer_dict = {
            'type': layer.layer_type
        }

        if hasattr(layer, 'neurons') and layer.neurons is not None:
            layer_dict['neurons'] = layer.neurons
        if hasattr(layer, 'activation') and layer.activation is not None:
            layer_dict['activation'] = layer.activation
        if hasattr(layer, 'rate') and layer.rate is not None:
            layer_dict['rate'] = layer.rate
        if hasattr(layer, 'filters') and layer.filters is not None:
            layer_dict['filters'] = layer.filters
        if hasattr(layer, 'kernel_size') and layer.kernel_size is not None:
            layer_dict['kernel_size'] = layer.kernel_size
        if hasattr(layer, 'pool_size') and layer.pool_size is not None:
            layer_dict['pool_size'] = layer.pool_size
        if hasattr(layer, 'l2_reg') and layer.l2_reg:
            layer_dict['l2_reg'] = layer.l2_reg

        return layer_dict
