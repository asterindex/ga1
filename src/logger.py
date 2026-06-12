"""
Logging system for genetic algorithm
"""

import sys
from datetime import datetime
from typing import Optional


class Colors:
    """ANSI color codes for terminal"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'


class Logger:
    """Logger with color support and different levels"""
    
    def __init__(self, verbose: bool = True, log_file: Optional[str] = None):
        self.verbose = verbose
        self.log_file = log_file
        self.start_time = datetime.now()
        
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Genetic Algorithm Log ===\n")
                f.write(f"Start time: {self.start_time}\n\n")
    
    def _log(self, message: str, color: str = '', prefix: str = ''):
        """Internal logging method"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Консольний вивід
        if self.verbose:
            console_msg = f"{Colors.GRAY}[{timestamp}]{Colors.ENDC} {color}{prefix}{message}{Colors.ENDC}"
            print(console_msg)
            sys.stdout.flush()
        
        # Файловий вивід (без кольорів)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {prefix}{message}\n")
    
    def header(self, message: str):
        """Section header"""
        border = "=" * 70
        self._log(f"\n{border}", Colors.HEADER)
        self._log(message, Colors.HEADER + Colors.BOLD)
        self._log(f"{border}\n", Colors.HEADER)
    
    def subheader(self, message: str):
        """Subheader"""
        border = "-" * 70
        self._log(f"\n{border}", Colors.OKBLUE)
        self._log(message, Colors.OKBLUE + Colors.BOLD)
        self._log(f"{border}", Colors.OKBLUE)
    
    def info(self, message: str):
        """Information message"""
        self._log(message, Colors.OKCYAN, "ℹ️  ")
    
    def success(self, message: str):
        """Success message"""
        self._log(message, Colors.OKGREEN, "✅ ")
    
    def warning(self, message: str):
        """Warning message"""
        self._log(message, Colors.WARNING, "⚠️  ")
    
    def error(self, message: str):
        """Error message"""
        self._log(message, Colors.FAIL, "❌ ")
    
    def debug(self, message: str):
        """Debug information"""
        self._log(message, Colors.GRAY, "🔍 ")
    
    def progress(self, current: int, total: int, message: str = ""):
        """Progress bar"""
        percent = (current / total) * 100
        filled = int(50 * current / total)
        bar = '█' * filled + '-' * (50 - filled)
        
        msg = f"[{bar}] {percent:.1f}% ({current}/{total})"
        if message:
            msg += f" - {message}"
        
        # Use \r to rewrite the line
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\r{Colors.GRAY}[{timestamp}]{Colors.ENDC} {Colors.OKCYAN}{msg}{Colors.ENDC}", end='', flush=True)
            
            if current == total:
                print()  # New line after completion
    
    def model_info(self, model_num: int, total: int, chromosome_info: dict):
        """Detailed model information"""
        self.debug(f"Модель {model_num}/{total}:")
        self.debug(f"  └─ Оптимізатор: {chromosome_info.get('optimizer', 'N/A')}")
        self.debug(f"  └─ Learning rate: {chromosome_info.get('learning_rate', 0):.5f}")
        self.debug(f"  └─ Batch size: {chromosome_info.get('batch_size', 0)}")
        self.debug(f"  └─ Кількість шарів: {chromosome_info.get('num_layers', 0)}")
    
    def training_progress(self, epoch: int, total_epochs: int, loss: float, accuracy: float):
        """Training progress"""
        self.debug(f"  Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def generation_summary(self, generation: int, best_fitness: float, avg_fitness: float, worst_fitness: float):
        """Generation summary"""
        self.subheader(f"📊 Покоління {generation} - Підсумок")
        self.info(f"Найкраща точність:  {best_fitness:.4f} {'🏆' if best_fitness > 0.95 else ''}")
        self.info(f"Середня точність:   {avg_fitness:.4f}")
        self.info(f"Найгірша точність:  {worst_fitness:.4f}")
    
    def best_model_details(self, chromosome):
        """Detailed information about the best model"""
        self.info("🔝 Найкраща модель покоління:")
        self.info(f"  ├─ Точність: {chromosome.fitness:.4f}")
        self.info(f"  ├─ Learning rate: {chromosome.learning_rate:.5f}")
        self.info(f"  ├─ Batch size: {chromosome.batch_size}")
        self.info(f"  ├─ Оптимізатор: {chromosome.optimizer}")
        self.info(f"  └─ Кількість шарів: {len(chromosome.layers)}")
        
        # Show structure
        self.debug("  Структура мережі:")
        for i, layer in enumerate(chromosome.layers, 1):
            self.debug(f"    {i}. {layer}")
    
    def genetic_operation(self, operation: str, details: str = ""):
        """Log genetic operations"""
        emoji_map = {
            'selection': '🎯',
            'crossover': '🧬',
            'mutation': '🔀',
            'elitism': '👑'
        }
        emoji = emoji_map.get(operation, '⚙️')
        msg = f"{emoji} {operation.capitalize()}"
        if details:
            msg += f": {details}"
        self.debug(msg)
    
    def checkpoint_saved(self, generation: int, path: str):
        """Checkpoint save message"""
        self.success(f"💾 Checkpoint збережено - Покоління {generation}")
        self.debug(f"  └─ Шлях: {path}")
    
    def checkpoint_loaded(self, generation: int, fitness: float):
        """Checkpoint load message"""
        self.success(f"📂 Checkpoint завантажено - Покоління {generation}")
        self.info(f"  └─ Найкраща точність: {fitness:.4f}")
    
    def elapsed_time(self):
        """Print elapsed time"""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        seconds = elapsed.seconds % 60
        
        time_str = f"{hours}год {minutes}хв {seconds}с" if hours > 0 else f"{minutes}хв {seconds}с"
        self.info(f"⏱️  Час виконання: {time_str}")
    
    def final_summary(self, best_fitness: float, total_generations: int):
        """Final summary"""
        self.header("🎉 ЕВОЛЮЦІЯ ЗАВЕРШЕНА")
        self.success(f"Найкраща досягнута точність: {best_fitness:.4f}")
        self.info(f"Завершено поколінь: {total_generations}")
        self.elapsed_time()


# Global logger instance
_logger = None


def get_logger(verbose: bool = True, log_file: Optional[str] = None) -> Logger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = Logger(verbose=verbose, log_file=log_file)
    return _logger


def set_logger(logger: Logger):
    """Set global logger"""
    global _logger
    _logger = logger
